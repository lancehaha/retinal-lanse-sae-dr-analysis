from __future__ import annotations

"""Experiment 5 stage 2: label-blind K=1024 feature cards and AutoResearch."""

import argparse
import base64
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw
import torch
from scipy import sparse
from sklearn.metrics import average_precision_score, roc_auc_score

from retina_lanse.data.embedding_shards import load_embedding_run
from retina_lanse.data.manifest import sha256_path
from retina_lanse.sae.checkpoints import load_checkpoint
from scripts.run_activation_interaction_autoresearch import feature_stats, split_patients
from scripts.run_global_sae_search import require_external
from scripts.run_label_efficiency_external_benchmark import (
    collect_eyepacs,
    collect_idrid,
    load_idrid_manifest,
    load_eyepacs_labels,
)
from scripts.run_single_sae_feasibility import encode_csr


GROUNDING_PROMPT = """You are describing one sparse-autoencoder feature in retinal fundus photographs.
The images were selected because this feature has high activation. Clinical labels are hidden.
Describe only visible shared visual patterns; do not infer diabetic-retinopathy grade or disease
prevalence. Mention heterogeneity and nuisance explanations when appropriate.
Return JSON only:
{
  \"hypothesis\": \"one concise visual description\",
  \"anatomical_region\": \"retina/vessel/optic-disc/macula/global/uncertain\",
  \"evidence_strength\": \"high/medium/low\",
  \"alternative_explanations\": [\"...\"],
  \"nuisance_flags\": [\"blur/brightness/camera/border/compression/none/other\"],
  \"uncertain\": true
}
"""
SEED = 4106
MAX_FEATURES = 256
TOP_IMAGES = 32


def parse_json(text: str) -> dict | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                cleaned = re.sub(r",\s*([}\]])", r"\1", text[start : end + 1])
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    return None
    return None


def call_ollama(base_url: str, model: str, prompt: str, images: list[Path], timeout: int, context_length: int, max_image_side: int) -> str:
    # Keep all 32 examples but pack them into one numbered 4x8 contact sheet
    # so the multimodal request stays below the model context limit.
    tile = max(64, max_image_side // 8)
    columns, rows = 8, 4
    sheet = Image.new("RGB", (columns * tile, rows * tile), "white")
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(images[: columns * rows]):
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((tile - 4, tile - 4), Image.Resampling.LANCZOS)
            x = (index % columns) * tile + (tile - image.width) // 2
            y = (index // columns) * tile + (tile - image.height) // 2
            sheet.paste(image, (x, y))
        draw.rectangle(((index % columns) * tile, (index // columns) * tile, (index % columns) * tile + 18, (index // columns) * tile + 16), fill="white")
        draw.text(((index % columns) * tile + 2, (index // columns) * tile), str(index + 1), fill="black")
    buffer = BytesIO()
    sheet.save(buffer, format="JPEG", quality=80, optimize=True)
    encoded_image = base64.b64encode(buffer.getvalue()).decode("ascii")
    content = prompt + "\nThe examples are shown in a numbered 4x8 contact sheet (left-to-right, top-to-bottom)."
    payload = {"model": model, "stream": False, "options": {"temperature": 0, "num_ctx": context_length}, "messages": [{"role": "user", "content": content, "images": [encoded_image]}]}
    request = urllib.request.Request(base_url.rstrip("/") + "/api/chat", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return str(json.loads(response.read().decode())["message"]["content"])


def make_cards(codes: sparse.csr_matrix, paths: list[str], patients: list[str], *, max_features: int, top_images: int) -> list[dict]:
    active = codes.copy()
    active.data = np.ones_like(active.data)
    counts = np.asarray(active.sum(axis=0)).ravel().astype(np.int64)
    mass = np.asarray(codes.sum(axis=0)).ravel().astype(np.float64)
    eligible = counts >= 50
    ranked = np.argsort(np.where(eligible, mass, -np.inf))[::-1]
    cards = []
    for rank, feature in enumerate([int(x) for x in ranked[:max_features] if eligible[x]], start=1):
        column = codes.getcol(feature).tocoo()
        order = np.argsort(column.data)[::-1][:top_images]
        examples = [{"relative_path": paths[int(column.row[pos])], "patient_id": str(patients[int(column.row[pos])]), "activation": float(column.data[pos])} for pos in order]
        cards.append({"feature_id": feature, "global_prominence_rank": rank, "activation_count": int(counts[feature]), "activation_mass": float(mass[feature]), "mean_activation": float(mass[feature] / codes.shape[0]), "hypothesis_status": "pending_llm", "top_high_activation_examples": examples})
    return cards


def feature_verification(codes: sparse.csr_matrix, grades: np.ndarray, cards: list[dict]) -> list[dict]:
    active = codes.copy()
    active.data = np.ones_like(active.data)
    labeled = grades >= 0
    positive, negative = labeled & (grades >= 2), labeled & (grades < 2)
    records = []
    for card in cards:
        feature = int(card["feature_id"])
        col = active.getcol(feature)
        pos_count = int(col[positive].sum())
        neg_count = int(col[negative].sum())
        pos_rate = (pos_count + 0.5) / (max(int(positive.sum()), 1) + 1.0)
        neg_rate = (neg_count + 0.5) / (max(int(negative.sum()), 1) + 1.0)
        records.append({"feature_id": feature, "all_image_activation_count": int(card["activation_count"]), "labeled_dr_active_count": pos_count, "labeled_non_dr_active_count": neg_count, "labeled_dr_activation_rate": float(pos_rate), "labeled_non_dr_activation_rate": float(neg_rate), "log_activation_rate_ratio": float(np.log(pos_rate / neg_rate)), "non_dr_occurrence_rate": float(neg_count / max(int(negative.sum()), 1)), "hypothesis_is_not_a_clinical_label": True})
    return records


def recipe_scores(codes: sparse.csr_matrix, targets: np.ndarray, cards: list[dict], hypotheses: dict[int, dict], *, fit_mask: np.ndarray) -> list[dict]:
    candidate_ids = np.asarray([int(card["feature_id"]) for card in cards], dtype=np.int64)
    stats = feature_stats(codes[fit_mask][:, candidate_ids], targets[fit_mask])
    order = np.argsort(stats["mean_difference"])[::-1][:16]
    ids = candidate_ids[order]
    diffs = stats["mean_difference"][order].astype(np.float32)
    codes_ids = codes[:, ids]
    non_dr_rates = np.asarray((codes[fit_mask][:, ids] > 0).mean(axis=0)).ravel().astype(np.float32)
    recipes = []
    weights = diffs / max(float(np.abs(diffs).sum()), 1e-8)
    recipes.append({"name": "weighted_activation_sum_top16", "feature_ids": ids.tolist(), "weights": weights.tolist(), "score": np.asarray(codes_ids.dot(weights)).ravel()})
    penalty = diffs * (1.0 - non_dr_rates)
    penalty_weights = penalty / max(float(np.abs(penalty).sum()), 1e-8)
    recipes.append({"name": "non_dr_penalized_activation_sum_top16", "feature_ids": ids.tolist(), "weights": penalty_weights.tolist(), "score": np.asarray(codes_ids.dot(penalty_weights)).ravel()})
    vote_weights = (diffs > 0).astype(np.float32) / max(float((diffs > 0).sum()), 1.0)
    recipes.append({"name": "top16_positive_evidence_vote", "feature_ids": ids.tolist(), "weights": vote_weights.tolist(), "score": np.asarray((codes_ids > 0).dot(vote_weights)).ravel()})
    llm_multiplier = np.asarray([{"high": 1.0, "medium": 0.75, "low": 0.5}.get(str(hypotheses.get(int(feature), {}).get("evidence_strength", "low")), 0.5) for feature in ids], dtype=np.float32)
    llm_weights = diffs * llm_multiplier
    llm_weights /= max(float(np.abs(llm_weights).sum()), 1e-8)
    recipes.append({"name": "llm_evidence_weighted_activation_sum_top16", "feature_ids": ids.tolist(), "weights": llm_weights.tolist(), "score": np.asarray(codes_ids.dot(llm_weights)).ravel()})
    return recipes


def apply_recipe(codes: sparse.csr_matrix, recipe: dict) -> np.ndarray:
    ids = np.asarray(recipe["feature_ids"], dtype=np.int64)
    weights = np.asarray(recipe["weights"], dtype=np.float32)
    return np.asarray(codes[:, ids].dot(weights)).ravel()


def main() -> int:
    parser = argparse.ArgumentParser(description="K=1024 SAE feature meaning AutoResearch")
    parser.add_argument("--train-run", required=True)
    parser.add_argument("--validation-run", required=True)
    parser.add_argument("--internal-test-run", required=True)
    parser.add_argument("--idrid-run", required=True)
    parser.add_argument("--idrid-manifest", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="qwen2.5-vl:3b")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-features", type=int, default=MAX_FEATURES)
    parser.add_argument("--top-images", type=int, default=TOP_IMAGES)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--context-length", type=int, default=32768)
    parser.add_argument("--max-image-side", type=int, default=1024)
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.max_features != MAX_FEATURES or args.top_images != TOP_IMAGES:
        parser.error("the preregistered feature search requires 256 cards and 32 examples per card")
    roots = {name: require_external(Path(value), name) for name, value in {"train run": args.train_run, "validation run": args.validation_run, "internal test run": args.internal_test_run, "IDRiD run": args.idrid_run, "IDRiD manifest": args.idrid_manifest, "labels": args.labels, "normalization": args.normalization, "checkpoint": args.checkpoint, "dataset root": args.dataset_root, "output": args.output}.items()}
    output = roots["output"]
    output.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        smoke = {"status": "smoke_passed", "cards": MAX_FEATURES, "examples_per_card": TOP_IMAGES, "labels_in_hypothesis_prompt": False}
        (output / "stage2_smoke.json").write_text(json.dumps(smoke, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(smoke, indent=2))
        return 0

    labels = load_eyepacs_labels(roots["labels"])
    idrid_labels = load_idrid_manifest(roots["IDRiD manifest"])
    train_run = load_embedding_run(roots["train run"], expected_split="train", verify_sha256=True)
    val_run = load_embedding_run(roots["validation run"], expected_split="validation", verify_sha256=True)
    test_run = load_embedding_run(roots["internal test run"], expected_split="internal_test", verify_sha256=True)
    idrid_run = load_embedding_run(roots["IDRiD run"], expected_split="external", verify_sha256=True)
    normalization = torch.load(roots["normalization"], map_location="cpu", weights_only=False)
    mean, std = normalization["mean"].float(), normalization["std"].float()
    model, payload = load_checkpoint(roots["checkpoint"])
    if payload["model_config"] != {"input_dim": 1024, "dictionary_size": 16384, "k": 1024}:
        raise RuntimeError("semantic checkpoint must be K=1024, input=1024, dictionary=16384")
    train_dense, train_grades, train_patients, train_paths = collect_eyepacs(train_run, labels)
    validation_dense, validation_grades, validation_patients, validation_paths = collect_eyepacs(val_run, labels)
    test_dense, test_grades, test_patients, test_paths = collect_eyepacs(test_run, labels)
    idrid_dense, idrid_grades, idrid_patients, idrid_paths = collect_idrid(idrid_run, idrid_labels)
    train_all = (train_run.embeddings - mean) / std
    train_codes = encode_csr(model, train_all, batch_size=args.batch_size)
    cards = make_cards(train_codes, train_run.relative_paths, train_run.patient_ids, max_features=args.max_features, top_images=args.top_images)
    cards_path = output / "feature_cards.jsonl"
    with cards_path.open("w", encoding="utf-8") as handle:
        for card in cards:
            handle.write(json.dumps(card, sort_keys=True) + "\n")

    hypotheses_path = output / "feature_hypotheses.jsonl"
    prior = {}
    if hypotheses_path.is_file():
        for line in hypotheses_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("status") == "complete":
                    prior[int(row["feature_id"])] = row
    hypotheses = {}
    base_url = args.base_url or os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
    with hypotheses_path.open("w", encoding="utf-8") as handle:
        for card in cards:
            feature = int(card["feature_id"])
            if feature in prior:
                row = prior[feature]
            elif args.skip_llm:
                row = {"feature_id": feature, "status": "skipped_llm", "parsed_response": None, "labels_hidden_from_prompt": True}
            else:
                images = [roots["dataset root"] / item["relative_path"] for item in card["top_high_activation_examples"] if (roots["dataset root"] / item["relative_path"]).is_file()]
                prompt = GROUNDING_PROMPT + f"\nThis card contains {len(images)} examples. Feature identity is not a clinical label."
                try:
                    raw = call_ollama(base_url, args.model, prompt, images, args.timeout, args.context_length, args.max_image_side)
                    parsed = parse_json(raw)
                    row = {"feature_id": feature, "status": "complete" if isinstance(parsed, dict) and parsed.get("hypothesis") else "unparsed", "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "raw_response": raw, "parsed_response": parsed, "labels_hidden_from_prompt": True}
                except (urllib.error.URLError, TimeoutError, OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
                    row = {"feature_id": feature, "status": "request_failed", "error": repr(exc), "labels_hidden_from_prompt": True}
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            if row.get("status") == "complete":
                hypotheses[feature] = row.get("parsed_response") or {}
    verification = feature_verification(train_codes, np.asarray([labels.get(Path(path).stem, -1) for path in train_run.relative_paths], dtype=np.int64), cards)
    verification_path = output / "feature_verification.jsonl"
    with verification_path.open("w", encoding="utf-8") as handle:
        for row in verification:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    train_lookup = {Path(path).stem: i for i, path in enumerate(train_run.relative_paths)}
    labeled_codes = train_codes[np.asarray([train_lookup[Path(path).stem] for path in train_paths], dtype=np.int64)]
    train_targets = (train_grades >= 2).astype(np.int64)
    dev_mask = split_patients([str(x) for x in train_patients], SEED)
    recipes = recipe_scores(labeled_codes, train_targets, cards, hypotheses, fit_mask=~dev_mask)
    for recipe in recipes:
        recipe["development"] = {"auroc": float(roc_auc_score(train_targets[dev_mask], recipe["score"][dev_mask])), "auprc": float(average_precision_score(train_targets[dev_mask], recipe["score"][dev_mask]))}
        recipe.pop("score", None)
    selected = max(recipes, key=lambda row: (row["development"]["auroc"], row["development"]["auprc"]))
    validation_codes = encode_csr(model, (torch.from_numpy(validation_dense) - mean) / std, batch_size=args.batch_size)
    test_codes = encode_csr(model, (torch.from_numpy(test_dense) - mean) / std, batch_size=args.batch_size)
    idrid_codes = encode_csr(model, (idrid_run.embeddings - mean) / std, batch_size=args.batch_size)
    selected_scores = {"validation": apply_recipe(validation_codes, selected), "locked_test": apply_recipe(test_codes, selected), "idrid": apply_recipe(idrid_codes, selected)}
    validation_targets = (validation_grades >= 2).astype(np.int64)
    test_targets = (test_grades >= 2).astype(np.int64)
    idrid_targets = (idrid_grades >= 2).astype(np.int64)
    report = {"schema_version": "1.0", "status": "complete_feature_meaning_autoresearch" if len(hypotheses) == len(cards) else "partial_feature_meaning_autoresearch", "autoresearch_used": True, "semantic_checkpoint": str(roots["checkpoint"]), "semantic_checkpoint_sha256": sha256_path(roots["checkpoint"]), "model": args.model, "cards_requested": len(cards), "hypotheses_completed": len(hypotheses), "examples_per_card": args.top_images, "llm_image_preprocessing": {"max_side_px": args.max_image_side, "format": "JPEG", "quality": 75}, "dictionary_size": 16384, "images_indexed": len(train_run.embeddings), "patients_indexed": len(set(train_run.patient_ids)), "labels_hidden_from_hypothesis_prompt": True, "recipe_candidates": [{key: value for key, value in recipe.items() if key != "weights"} for recipe in recipes], "selected_recipe": {key: value for key, value in selected.items() if key != "weights"}, "selected_recipe_weights": selected["weights"], "selected_recipe_evaluation": {"validation": {"auroc": float(roc_auc_score(validation_targets, selected_scores["validation"])), "auprc": float(average_precision_score(validation_targets, selected_scores["validation"]))}, "locked_test": {"auroc": float(roc_auc_score(test_targets, selected_scores["locked_test"])), "auprc": float(average_precision_score(test_targets, selected_scores["locked_test"]))}, "idrid": {"auroc": float(roc_auc_score(idrid_targets, selected_scores["idrid"])), "auprc": float(average_precision_score(idrid_targets, selected_scores["idrid"]))}}, "outputs": {"cards": str(cards_path), "hypotheses": str(hypotheses_path), "verification": str(verification_path)}, "provenance": {"train_evidence_sha256": sha256_path(roots["train run"] / "extraction_evidence.json"), "validation_evidence_sha256": sha256_path(roots["validation run"] / "extraction_evidence.json"), "internal_test_evidence_sha256": sha256_path(roots["internal test run"] / "extraction_evidence.json"), "idrid_evidence_sha256": sha256_path(roots["IDRiD run"] / "extraction_evidence.json"), "labels_sha256": sha256_path(roots["labels"]), "normalization_sha256": sha256_path(roots["normalization"])}, "caveat": "Feature hypotheses describe visual patterns in high-activation examples; they are not lesion or clinical-validity claims."}
    result_path = output / "stage2_feature_meaning_autoresearch.json"
    result_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
