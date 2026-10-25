from __future__ import annotations

"""Run a blinded, small-scale VLM inference ablation on labeled development images.

This runner is deliberately a pilot: the clinical label is used only after the model
response is written, never in the prompt.  It is intended to quantify whether adding
an image-level VLM decision changes the numerical SAE baseline, not to replace a
clinician-validated diagnostic study.
"""

import argparse
import base64
import csv
import hashlib
import io
import json
import mimetypes
import os
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image, ImageOps
from sklearn.metrics import average_precision_score, roc_auc_score

EXTERNAL_ROOT = Path("/Volumes/Elements/Retina-LanSE")

PROMPT = """You are evaluating one retinal fundus photograph for research. The reference
label is hidden. Estimate the severity of visible diabetic-retinopathy-like findings,
but do not assume that every bright spot is a lesion and explicitly state uncertainty.
Return JSON only:
{
  "predicted_grade": 0,
  "referable_probability": 0.0,
  "visible_findings": ["..."],
  "uncertainty": "low/medium/high",
  "image_quality": "adequate/limited/poor"
}
Use predicted_grade 0--4 as a visual estimate only. Referable probability is the
model's probability that the underlying reference grade is >=2; it is not a calibrated
clinical diagnosis. Do not mention or guess a hidden ground-truth label.
"""


def require_external(path: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(EXTERNAL_ROOT)
    except ValueError as exc:
        raise RuntimeError(f"{label} must stay under {EXTERNAL_ROOT}: {resolved}") from exc
    return resolved


def image_bytes(path: Path, max_side: int) -> bytes:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90, optimize=False)
        return buffer.getvalue()


def call_ollama(base_url: str, model: str, prompt: str, image: Path, timeout: int, context_length: int, max_side: int) -> str:
    payload = {
        "model": model,
        "stream": False,
        "options": {"temperature": 0, "num_ctx": context_length},
        "messages": [{"role": "user", "content": prompt, "images": [
            base64.b64encode(image_bytes(image, max_side)).decode("ascii")
        ]}],
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return str(body["message"]["content"])


def parse_json(text: str) -> dict | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def load_labels(path: Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            labels[row["image"].strip()] = int(row["level"])
    return labels


def stable_key(seed: int, patient: str, relative_path: str) -> str:
    return hashlib.sha256(f"{seed}:{patient}:{relative_path}".encode()).hexdigest()


def select_sample(records: list[dict], labels: dict[str, int], seed: int, max_images: int) -> list[dict]:
    eligible = []
    for record in records:
        image_key = Path(record["relative_path"]).stem
        if image_key not in labels:
            continue
        item = dict(record)
        item["grade"] = labels[image_key]
        item["target"] = int(item["grade"] >= 2)
        item["_order"] = stable_key(seed, item["patient_id"], item["relative_path"])
        eligible.append(item)
    # Balance the pilot by endpoint and keep patients disjoint in the actual split.
    chosen = []
    for target in (0, 1):
        group = sorted((x for x in eligible if x["target"] == target), key=lambda x: x["_order"])
        chosen.extend(group[: max_images // 2])
    return sorted(chosen[:max_images], key=lambda x: x["_order"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Blinded VLM image-level inference ablation pilot.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--max-images", type=int, default=16)
    parser.add_argument("--seed", type=int, default=4106)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default="qwen2.5vl:3b")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--max-image-side", type=int, default=560)
    args = parser.parse_args()

    manifest_path = require_external(Path(args.manifest), "manifest")
    labels_path = require_external(Path(args.labels), "labels")
    dataset_root = require_external(Path(args.dataset_root), "dataset")
    output = require_external(Path(args.output), "output")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "vlm_predictions.partial.jsonl"
    base_url = args.base_url or os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")

    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [x for x in records if x.get("split") == args.split]
    sample = select_sample(records, load_labels(labels_path), args.seed, args.max_images)
    if len(sample) < args.max_images:
        raise RuntimeError(f"only {len(sample)} labeled images available for requested {args.max_images}")

    result_path = output / "vlm_predictions.jsonl"
    results: list[dict] = []
    # Checkpoint every response so a stalled or interrupted request cannot erase
    # evidence from earlier images. The final JSONL remains the canonical output.
    checkpoint = checkpoint_path.open("w", encoding="utf-8")
    for item in sample:
        image = dataset_root / item["relative_path"]
        if not image.is_file():
            result = {"relative_path": item["relative_path"], "status": "image_missing"}
            results.append(result)
            checkpoint.write(json.dumps(result, sort_keys=True) + "\n")
            checkpoint.flush()
            continue
        prompt = PROMPT
        try:
            raw = call_ollama(base_url, args.model, prompt, image, args.timeout, args.context_length, args.max_image_side)
            parsed = parse_json(raw)
            prediction = None
            if isinstance(parsed, dict):
                probability = float(parsed.get("referable_probability"))
                if not 0 <= probability <= 1:
                    raise ValueError("referable_probability outside [0,1]")
                prediction = probability
            result = {
                "relative_path": item["relative_path"],
                "patient_id": item["patient_id"],
                "status": "complete" if prediction is not None else "unparsed",
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "raw_response": raw,
                "parsed_response": parsed,
                "target": item["target"],
                "grade_used_after_response": item["grade"],
            }
            results.append(result)
            checkpoint.write(json.dumps(result, sort_keys=True) + "\n")
            checkpoint.flush()
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError) as exc:
            result = {"relative_path": item["relative_path"], "status": "request_failed", "error": repr(exc), "target": item["target"], "grade_used_after_response": item["grade"]}
            results.append(result)
            checkpoint.write(json.dumps(result, sort_keys=True) + "\n")
            checkpoint.flush()
    checkpoint.close()

    with result_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, sort_keys=True) + "\n")
    valid = [x for x in results if x.get("status") == "complete" and isinstance(x.get("parsed_response"), dict)]
    metrics: dict[str, float] = {}
    if len({x["target"] for x in valid}) == 2:
        scores = [float(x["parsed_response"]["referable_probability"]) for x in valid]
        targets = [int(x["target"]) for x in valid]
        metrics = {"auroc": float(roc_auc_score(targets, scores)), "auprc": float(average_precision_score(targets, scores))}
    summary = {
        "schema_version": "1.0",
        "status": "complete_vlm_inference_pilot" if len(valid) == len(sample) else "partial_vlm_inference_pilot",
        "provider": "ollama",
        "model": args.model,
        "split": args.split,
        "images_requested": len(sample),
        "images_completed": len(valid),
        "positive_images": int(sum(x["target"] for x in sample)),
        "labels_hidden_from_prompt": True,
        "image_preprocessing": f"in_memory_exif_transpose_rgb_resize_longest_side_{args.max_image_side}_jpeg_quality90",
        "metrics": metrics,
        "output_jsonl": str(result_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "labels_sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest(),
    }
    (output / "vlm_inference_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
