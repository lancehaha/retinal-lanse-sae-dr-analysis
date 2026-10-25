from __future__ import annotations

"""Evaluate a blinded VLM with SAE-derived feature-card context.

The same image sample used by the image-only pilot is augmented with descriptions
for grounded cards whose SAE features are active in that image. Ground-truth DR
labels are read only after a response is stored. Cards without a grounded
hypothesis are not invented; the audit therefore exposes coverage explicitly.
"""

import argparse
import base64
import hashlib
import io
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image, ImageOps
from scipy import sparse
from sklearn.metrics import average_precision_score, roc_auc_score

from retina_lanse.data.embedding_shards import load_embedding_run
from retina_lanse.sae.checkpoints import load_checkpoint
from scripts.run_global_sae_search import require_external
from scripts.run_single_sae_feasibility import encode_csr


PROMPT = """You are evaluating one retinal fundus photograph for research. Estimate the
severity of visible diabetic-retinopathy-like findings. The reference label is hidden.
Below are optional descriptions of sparse-autoencoder features active in this image;
they are uncertain hypotheses and may be nuisance patterns, so use them only as
additional context and disagree when the image does not support them.

Return JSON only:
{
  "predicted_grade": 0,
  "referable_probability": 0.0,
  "visible_findings": ["..."],
  "uncertainty": "low/medium/high",
  "image_quality": "adequate/limited/poor"
}
Use predicted_grade 0--4 as a visual estimate only. Referable probability is the
model's probability that the hidden reference grade is >=2; it is not a calibrated
clinical diagnosis. Do not mention or guess a hidden ground-truth label.
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


def load_cards(path: Path) -> dict[int, dict]:
    cards = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            parsed = item.get("parsed_response")
            if item.get("status") == "complete" and isinstance(parsed, dict) and parsed.get("hypothesis"):
                cards[int(item["feature_id"])] = parsed
    return cards


def select_context_cards(active_ids: list[int], cards: dict[int, dict], selection_mode: str, seed: int, relative_path: str) -> list[dict]:
    """Return active or deterministic matched-count random card contexts."""
    active_cards = [{"feature_id": feature_id, **cards[feature_id]} for feature_id in active_ids if feature_id in cards]
    if selection_mode == "active" or not active_cards:
        return active_cards
    all_card_ids = sorted(cards)
    active_set = set(active_ids)
    candidates = [feature_id for feature_id in all_card_ids if feature_id not in active_set]
    if len(candidates) < len(active_cards):
        candidates = all_card_ids
    order = sorted(candidates, key=lambda feature_id: hashlib.sha256(f"{seed}:{relative_path}:{feature_id}".encode()).hexdigest())
    return [{"feature_id": feature_id, **cards[feature_id]} for feature_id in order[: len(active_cards)]]


def main() -> int:
    parser = argparse.ArgumentParser(description="Blinded VLM inference with active SAE feature-card context.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--grounding", required=True)
    parser.add_argument("--validation-run", required=True)
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default="qwen2.5vl:3b")
    parser.add_argument("--top-features", type=int, default=8)
    parser.add_argument("--selection-mode", choices=("active", "random"), default="active")
    parser.add_argument("--prompt-style", choices=("current", "neutral"), default="current")
    parser.add_argument("--seed", type=int, default=4106)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--context-length", type=int, default=32768)
    parser.add_argument("--max-image-side", type=int, default=560)
    args = parser.parse_args()

    prediction_path = require_external(Path(args.predictions), "image-only predictions")
    grounding_path = require_external(Path(args.grounding), "grounding JSONL")
    validation_root = require_external(Path(args.validation_run), "validation embeddings")
    normalization_path = require_external(Path(args.normalization), "normalization")
    checkpoint_path = require_external(Path(args.checkpoint), "SAE checkpoint")
    dataset_root = require_external(Path(args.dataset_root), "dataset")
    output = require_external(Path(args.output), "output")
    output.mkdir(parents=True, exist_ok=True)
    partial_path = output / "vlm_featurecard_predictions.partial.jsonl"
    base_url = args.base_url or os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")

    prior = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    prior = [x for x in prior if x.get("status") == "complete"]
    if not prior:
        raise RuntimeError("no complete image-only predictions")
    run = load_embedding_run(validation_root, expected_split="validation", verify_sha256=True)
    lookup = {path: index for index, path in enumerate(run.relative_paths)}
    indices = [lookup[x["relative_path"]] for x in prior if x["relative_path"] in lookup]
    if len(indices) != len(prior):
        raise RuntimeError("prediction image is missing from validation embedding run")
    normalization = torch.load(normalization_path, map_location="cpu", weights_only=False)
    normalized = (run.embeddings - normalization["mean"]) / normalization["std"]
    model, payload = load_checkpoint(checkpoint_path)
    codes = encode_csr(model, normalized, batch_size=1024)
    cards = load_cards(grounding_path)
    results = []
    checkpoint = partial_path.open("w", encoding="utf-8")
    for item, index in zip(prior, indices):
        row = codes.getrow(index)
        order = np.argsort(row.data)[::-1]
        active_ids = [int(row.indices[pos]) for pos in order[: args.top_features]]
        selected_cards = select_context_cards(active_ids, cards, args.selection_mode, args.seed, item["relative_path"])
        context = "\n".join(f"- SAE feature {card['feature_id']}: {card['hypothesis']} (uncertainty={card.get('uncertain')})" for card in selected_cards)
        if context:
            if args.prompt_style == "neutral":
                heading = "Feature-card hypotheses provided as additional context:"
            else:
                heading = "Active feature-card hypotheses:" if args.selection_mode == "active" else "Randomly sampled feature-card distractors (not selected from this image):"
            prompt = PROMPT + "\n" + heading + "\n" + context
        else:
            prompt = PROMPT + "\nNo grounded feature-card hypothesis was available for this image."
        image = dataset_root / item["relative_path"]
        try:
            raw = call_ollama(base_url, args.model, prompt, image, args.timeout, args.context_length, args.max_image_side)
            parsed = parse_json(raw)
            probability = float(parsed["referable_probability"]) if isinstance(parsed, dict) else None
            if probability is not None and not 0 <= probability <= 1:
                raise ValueError("probability outside [0,1]")
            result = {"relative_path": item["relative_path"], "patient_id": item.get("patient_id"), "active_feature_ids": active_ids, "grounded_cards": selected_cards, "cards_available": len(selected_cards), "context_selection_mode": args.selection_mode, "prompt_style": args.prompt_style, "status": "complete" if probability is not None else "unparsed", "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "raw_response": raw, "parsed_response": parsed, "target": item["target"], "grade_used_after_response": item["grade_used_after_response"]}
            results.append(result)
            checkpoint.write(json.dumps(result, sort_keys=True) + "\n")
            checkpoint.flush()
            continue
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError) as exc:
            result = {"relative_path": item["relative_path"], "active_feature_ids": active_ids, "cards_available": len(selected_cards), "context_selection_mode": args.selection_mode, "prompt_style": args.prompt_style, "status": "request_failed", "error": repr(exc), "target": item["target"], "grade_used_after_response": item["grade_used_after_response"]}
            results.append(result)
            checkpoint.write(json.dumps(result, sort_keys=True) + "\n")
            checkpoint.flush()

    checkpoint.close()

    result_path = output / "vlm_featurecard_predictions.jsonl"
    with result_path.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    valid = [x for x in results if x.get("status") == "complete"]
    metrics = {}
    if len({x["target"] for x in valid}) == 2:
        metrics = {"auroc": float(roc_auc_score([x["target"] for x in valid], [x["parsed_response"]["referable_probability"] for x in valid])), "auprc": float(average_precision_score([x["target"] for x in valid], [x["parsed_response"]["referable_probability"] for x in valid]))}
    summary = {"schema_version": "1.0", "status": "complete_vlm_featurecard_pilot" if len(valid) == len(prior) else "partial_vlm_featurecard_pilot", "provider": "ollama", "model": args.model, "images_requested": len(prior), "images_completed": len(valid), "grounding_cards_available": len(cards), "context_selection_mode": args.selection_mode, "prompt_style": args.prompt_style, "seed": args.seed, "mean_grounded_cards_per_image": float(np.mean([x.get("cards_available", 0) for x in results])), "labels_hidden_from_prompt": True, "image_preprocessing": f"in_memory_exif_transpose_rgb_resize_longest_side_{args.max_image_side}_jpeg_quality90", "metrics": metrics, "output_jsonl": str(result_path), "grounding_sha256": hashlib.sha256(grounding_path.read_bytes()).hexdigest()}
    (output / "vlm_featurecard_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
