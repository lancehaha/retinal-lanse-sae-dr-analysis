from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image, ImageOps

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_global_sae_search import require_external


PROMPT = """You are independently checking whether a proposed visual description is
actually visible in one retinal fundus photograph. The image may or may not show the
pattern. Do not infer diabetic-retinopathy grade, prevalence, or hidden labels.
Return JSON only:
{
  "pattern_present": true,
  "confidence": "low/medium/high",
  "visible_evidence": "one short sentence",
  "alternative_explanation": "one short sentence or none",
  "nuisance_flag": "blur/illumination/camera/border/artifact/none"
}
"""


def image_bytes(path: Path, max_side: int) -> bytes:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90, optimize=False)
        return buffer.getvalue()


def call_ollama(base_url: str, model: str, prompt: str, image: Path, timeout: int, context_length: int, max_side: int) -> str:
    payload = {"model": model, "stream": False, "options": {"temperature": 0, "num_ctx": context_length}, "messages": [{"role": "user", "content": prompt, "images": [base64.b64encode(image_bytes(image, max_side)).decode("ascii")]}]}
    request = urllib.request.Request(base_url.rstrip("/") + "/api/chat", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return str(json.loads(response.read().decode())["message"]["content"])


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Blinded VLM consistency check for SAE feature hypotheses.")
    parser.add_argument("--packet", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default="qwen2.5vl:3b")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--max-image-side", type=int, default=560)
    args = parser.parse_args()
    packet_path = require_external(Path(args.packet), "blinded review packet")
    dataset_root = require_external(Path(args.dataset_root), "dataset")
    output = require_external(Path(args.output), "verification output")
    output.mkdir(parents=True, exist_ok=True)
    base_url = args.base_url or os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
    rows = [json.loads(line) for line in packet_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    results = []
    for row in rows:
        image = dataset_root / row["relative_path"]
        prompt = PROMPT + "\nProposed feature hypothesis: " + row["hypothesis"]
        try:
            raw = call_ollama(base_url, args.model, prompt, image, args.timeout, args.context_length, args.max_image_side)
            parsed = parse_json(raw)
            valid = isinstance(parsed, dict) and isinstance(parsed.get("pattern_present"), bool)
            results.append({"review_id": row["review_id"], "feature_id": row["feature_id"], "relative_path": row["relative_path"], "status": "complete" if valid else "unparsed", "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "raw_response": raw, "parsed_response": parsed})
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError) as exc:
            results.append({"review_id": row["review_id"], "feature_id": row["feature_id"], "relative_path": row["relative_path"], "status": "request_failed", "error": repr(exc)})
    result_path = output / "hypothesis_verification.jsonl"
    with result_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, sort_keys=True) + "\n")
    valid = [x for x in results if x.get("status") == "complete"]
    by_feature = {}
    for feature_id in sorted({int(x["feature_id"]) for x in valid}):
        subset = [x for x in valid if int(x["feature_id"]) == feature_id]
        by_feature[str(feature_id)] = {"images": len(subset), "pattern_present_count": int(sum(x["parsed_response"]["pattern_present"] for x in subset)), "pattern_present_rate": float(sum(x["parsed_response"]["pattern_present"] for x in subset) / len(subset))}
    summary = {"schema_version": "1.0", "status": "complete_blinded_hypothesis_verification" if len(valid) == len(rows) else "partial_blinded_hypothesis_verification", "model": args.model, "images_requested": len(rows), "images_completed": len(valid), "cards": len(by_feature), "labels_hidden": True, "packet_sha256": hashlib.sha256(packet_path.read_bytes()).hexdigest(), "image_preprocessing": f"in_memory_exif_transpose_rgb_resize_longest_side_{args.max_image_side}_jpeg_quality90", "overall_pattern_present_rate": float(sum(x["parsed_response"]["pattern_present"] for x in valid) / len(valid)) if valid else None, "by_feature": by_feature, "output_jsonl": str(result_path), "caveat": "This is same-model consistency checking, not independent ophthalmologist semantic validation."}
    (output / "hypothesis_verification_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
