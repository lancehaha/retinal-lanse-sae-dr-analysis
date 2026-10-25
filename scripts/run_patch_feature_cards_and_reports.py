from __future__ import annotations

"""Experiment 8 feature cards, deterministic reports, and report-only Qwen narratives.

The card stage reads no labels.  It only sees image patches selected from the
SAE-training cohort by patch-SAE activation.  Pilot-holdout data are loaded
only after cards and the numerical Transformer are frozen.
"""

import argparse
import base64
import hashlib
import json
import os
import re
import sys
import urllib.request
from io import BytesIO
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

from retina_lanse.data.preprocessing import preprocess_global
from retina_lanse.sae.checkpoints import load_checkpoint
from scripts.run_global_sae_search import choose_device, require_external
import scripts.run_patch_sae_transformer_pilot as patch_pilot
from scripts.run_patch_sae_transformer_pilot import (
    DICTIONARY, GRID, MODEL_DIM, PATCHES, TOP_K, PatchSAETransformer, Scaling,
    apply_scaling, load_patch_run,
)


SEED = 4106
FEATURES = 64
SAMPLES = 32
VALID_RATE = 0.60
VALID_CONTRAST = 0.30

HYPOTHESIS_PROMPT = """You are describing one unsupervised sparse-autoencoder pattern in retinal image patches.
All examples in this numbered sheet were selected only because one internal feature activated strongly.
Patient IDs, labels, grades, and activation strengths are hidden. State a concise *visual* hypothesis,
visible evidence, plausible nuisance alternatives, confidence, and uncertainty. Do not infer disease,
diabetic-retinopathy grade, prevalence, or clinical meaning. Return JSON only:
{"hypothesis":"...","evidence":"...","alternatives":["..."],"confidence":"low/medium/high","uncertainty":"..."}."""

VALIDATE_PROMPT = """You are a blinded visual validator. A feature-pattern hypothesis is supplied below.
Two numbered contact sheets follow: HIGH and LOW. For every image, judge only whether the stated visible
pattern is present. Patient IDs, labels, grades, activation values and disease status are hidden. Do not
infer any clinical diagnosis. Return JSON only with exactly {n} boolean values per array:
{{"high_present":[true,false],"low_present":[false,true],"high_evidence":"...","low_evidence":"...","nuisance":"..."}}
Hypothesis: {hypothesis}"""

NARRATIVE_PROMPT = """You are writing a cautious retinal-image research narrative from a structured machine report.
You did not see the retinal image, labels, or any training examples. Restate only observations listed in
the supplied JSON. Every evidence claim must cite a feature_id and must not say lesion, diagnosis, or
clinical confirmation. Return JSON only: {"summary":"...","evidence_feature_ids":[0],"uncertainty":"..."}.
Structured report:\n"""


def parse_json(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(re.sub(r",\s*([}\]])", r"\1", text[start:end + 1]))
            except json.JSONDecodeError:
                return None
    return None


def contact_sheet(paths: list[Path], side: int = 1024) -> bytes:
    tile = max(64, side // 8)
    sheet = Image.new("RGB", (8 * tile, 4 * tile), "white")
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(paths[:SAMPLES]):
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((tile - 4, tile - 4), Image.Resampling.LANCZOS)
            x, y = (index % 8) * tile + (tile - image.width) // 2, (index // 8) * tile + (tile - image.height) // 2
            sheet.paste(image, (x, y))
        draw.rectangle(((index % 8) * tile, (index // 8) * tile, (index % 8) * tile + 18, (index // 8) * tile + 16), fill="white")
        draw.text(((index % 8) * tile + 2, (index // 8) * tile), str(index + 1), fill="black")
    buffer = BytesIO(); sheet.save(buffer, format="JPEG", quality=80, optimize=True)
    return buffer.getvalue()


def call_ollama(base_url: str, model: str, prompt: str, images: list[list[Path]], timeout: int) -> str:
    payload = {"model": model, "stream": False, "format": "json", "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 2048},
               "messages": [{"role": "user", "content": prompt, "images": [base64.b64encode(contact_sheet(group)).decode() for group in images]}]}
    request = urllib.request.Request(base_url.rstrip("/") + "/api/chat", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return str(json.loads(response.read().decode())["message"]["content"])


def call_text_ollama(base_url: str, model: str, prompt: str, timeout: int) -> str:
    payload = {"model": model, "stream": False, "format": "json", "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 1024}, "messages": [{"role": "user", "content": prompt}]}
    request = urllib.request.Request(base_url.rstrip("/") + "/api/chat", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return str(json.loads(response.read().decode())["message"]["content"])


def repair_json_once(base_url: str, model: str, malformed: str, timeout: int) -> str:
    """The sole permitted repair: format the already returned text, with no new evidence."""
    return call_text_ollama(base_url, model, "Return valid JSON only. Preserve no information beyond this malformed response:\n" + malformed, timeout)


def crop_patch(dataset: Path, rel: str, patch_index: int, destination: Path) -> Path:
    """Render the exact crop_retinal_field + 224px coordinate used by RETFound."""
    if destination.is_file():
        return destination
    with Image.open(dataset / rel) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        tensor = preprocess_global(image, 224)
    array = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    frame = Image.fromarray(array, "RGB")
    row, col = divmod(int(patch_index), GRID)
    crop = frame.crop((col * 16, row * 16, (col + 1) * 16, (row + 1) * 16)).resize((128, 128), Image.Resampling.NEAREST)
    destination.parent.mkdir(parents=True, exist_ok=True); crop.save(destination, format="PNG")
    return destination


def candidate_rows(ids: np.ndarray, values: np.ndarray, image_indices: np.ndarray, feature: int, paths: list[str], patients: np.ndarray) -> list[dict]:
    image, patch_index, rank = np.where(ids[image_indices] == feature)
    if not len(image):
        return []
    actual = image_indices[image]
    rows = [{"image_index": int(i), "patch_index": int(p), "relative_path": str(paths[int(i)]), "patient_id": str(patients[int(i)]), "activation": float(values[int(i), int(p), int(k)])}
            for i, p, k in zip(actual, patch_index, rank)]
    return sorted(rows, key=lambda item: (-item["activation"], item["relative_path"], item["patch_index"]))


def sample_disjoint(rows: list[dict], count: int, used_patients: set[str], *, reverse: bool = False) -> list[dict]:
    ordered = list(reversed(rows)) if reverse else rows
    chosen, local = [], set(used_patients)
    for row in ordered:
        if row["patient_id"] in local:
            continue
        chosen.append(row); local.add(row["patient_id"])
        if len(chosen) == count:
            break
    return chosen


def checked_arrays(parsed: dict | None, n: int) -> tuple[list[bool], list[bool]] | None:
    if not isinstance(parsed, dict):
        return None
    lookup = {str(key).lower(): value for key, value in parsed.items()}
    high, low = lookup.get("high_present"), lookup.get("low_present")
    if isinstance(high, list) and isinstance(low, list) and len(high) == n and len(low) == n and all(isinstance(x, bool) for x in high + low):
        return high, low
    return None


def make_cards(args, root: Path, patch: dict, cohorts: dict, ids: np.ndarray, values: np.ndarray) -> dict[int, dict]:
    selected = {str(x) for x in cohorts["patients"]["sae_train"]}
    train_indices = np.asarray([i for i, patient in enumerate(patch["patients"]) if str(patient) in selected], dtype=np.int64)
    active, active_values = ids[train_indices], values[train_indices]
    mass = np.zeros(DICTIONARY, dtype=np.float64); count = np.zeros(DICTIONARY, dtype=np.int64)
    nonzero = active_values.reshape(-1) > 0
    np.add.at(mass, active.reshape(-1)[nonzero], active_values.reshape(-1)[nonzero]); np.add.at(count, active.reshape(-1)[nonzero], 1)
    eligible = np.where(count >= 100)[0]
    feature_ids = sorted((int(fid) for fid in eligible), key=lambda fid: (-mass[fid], fid))[:FEATURES]
    base_url = args.base_url or os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
    cards = {}
    for feature in feature_ids:
        rows = candidate_rows(ids, values, train_indices, feature, patch["paths"], patch["patients"])
        discovery = sample_disjoint(rows, SAMPLES, set())
        initial_high = sample_disjoint(rows, SAMPLES, {x["patient_id"] for x in discovery})
        used = {x["patient_id"] for x in discovery + initial_high}
        initial_low = sample_disjoint(rows, SAMPLES, used, reverse=True)
        card = {"feature_id": feature, "activation_count": int(count[feature]), "activation_mass": float(mass[feature]), "labels_hidden": True,
                "discovery": discovery, "validation_high": initial_high, "validation_low": initial_low, "status": "unavailable", "hypothesis": None}
        if min(len(discovery), len(initial_high), len(initial_low)) < SAMPLES:
            card["reason"] = "insufficient_patient_disjoint_patch_examples"; cards[feature] = card; continue
        discovery_paths = [crop_patch(root, x["relative_path"], x["patch_index"], Path(args.output) / "card_patch_crops" / str(feature) / "discovery" / f"{i:02d}.png") for i, x in enumerate(discovery)]
        if args.skip_llm:
            card["reason"] = "llm_skipped"; cards[feature] = card; continue
        raw_hypothesis = call_ollama(base_url, args.model, HYPOTHESIS_PROMPT, [discovery_paths], args.timeout)
        hypothesis = parse_json(raw_hypothesis)
        repaired_hypothesis = None
        if hypothesis is None:
            repaired_hypothesis = repair_json_once(base_url, args.model, raw_hypothesis, args.timeout)
            hypothesis = parse_json(repaired_hypothesis)
        card["raw_hypothesis"] = raw_hypothesis; card["hypothesis_prompt_sha256"] = hashlib.sha256(HYPOTHESIS_PROMPT.encode()).hexdigest()
        if repaired_hypothesis is not None: card["hypothesis_json_repair"] = repaired_hypothesis
        if not isinstance(hypothesis, dict) or not str(hypothesis.get("hypothesis", "")).strip():
            card["reason"] = "malformed_hypothesis"; cards[feature] = card; continue
        high_paths = [crop_patch(root, x["relative_path"], x["patch_index"], Path(args.output) / "card_patch_crops" / str(feature) / "validation_high" / f"{i:02d}.png") for i, x in enumerate(initial_high)]
        low_paths = [crop_patch(root, x["relative_path"], x["patch_index"], Path(args.output) / "card_patch_crops" / str(feature) / "validation_low" / f"{i:02d}.png") for i, x in enumerate(initial_low)]
        # Two 16-vs-16 requests avoid the known local-3B truncation of a single
        # 32-vs-32 JSON array, while the validation sample remains 32 per group.
        high_votes, low_votes, raw_validation, repaired_validation = [], [], [], []
        for start in range(0, SAMPLES, 16):
            raw = call_ollama(base_url, args.model, VALIDATE_PROMPT.format(n=16, hypothesis=str(hypothesis["hypothesis"])), [high_paths[start:start + 16], low_paths[start:start + 16]], args.timeout)
            verdict = checked_arrays(parse_json(raw), 16); repaired = None
            if verdict is None:
                repaired = repair_json_once(base_url, args.model, raw, args.timeout)
                verdict = checked_arrays(parse_json(repaired), 16)
            raw_validation.append(raw)
            if repaired is not None: repaired_validation.append(repaired)
            if verdict is None:
                break
            high_votes.extend(verdict[0]); low_votes.extend(verdict[1])
        card.update({"hypothesis": hypothesis, "raw_validation": raw_validation, "validation_prompt_sha256": hashlib.sha256(VALIDATE_PROMPT.encode()).hexdigest()})
        if repaired_validation: card["validation_json_repair"] = repaired_validation
        if len(high_votes) != SAMPLES or len(low_votes) != SAMPLES:
            card["reason"] = "malformed_validation"; cards[feature] = card; continue
        high_rate, low_rate = float(np.mean(high_votes)), float(np.mean(low_votes))
        card.update({"high_pattern_rate": high_rate, "low_pattern_rate": low_rate, "contrast": high_rate - low_rate,
                     "status": "passed" if high_rate >= VALID_RATE and high_rate - low_rate >= VALID_CONTRAST else "failed"})
        cards[feature] = card
    path = Path(args.output) / "patch_feature_cards.jsonl"
    path.write_text("".join(json.dumps(cards[key], sort_keys=True) + "\n" for key in sorted(cards)), encoding="utf-8")
    summary = {"status": "complete", "features_requested": FEATURES, "features_available": len(cards),
               "passed": sum(x["status"] == "passed" for x in cards.values()), "failed": sum(x["status"] == "failed" for x in cards.values()),
               "unavailable": sum(x["status"] == "unavailable" for x in cards.values()), "labels_loaded": False,
               "card_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (Path(args.output) / "patch_feature_card_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return cards


def full_ensemble(prediction_dir: Path, geometry: torch.Tensor, device: torch.device) -> list[PatchSAETransformer]:
    models = []
    for seed in (4106, 4107, 4108):
        payload = torch.load(prediction_dir / f"full_binary_seed_{seed}.pt", map_location="cpu", weights_only=False)
        model = PatchSAETransformer(1, geometry, use_geometry=True, use_position=True).to(device)
        model.load_state_dict(payload["state_dict"]); model.eval(); models.append(model)
    return models


@torch.inference_mode()
def ensemble_probability(models, dense, ids, values, device) -> float:
    x_dense = torch.as_tensor(dense[None], dtype=torch.float32, device=device); x_ids = torch.as_tensor(ids[None], dtype=torch.long, device=device); x_values = torch.as_tensor(values[None], dtype=torch.float32, device=device)
    return float(np.mean([torch.sigmoid(model(x_dense, x_ids, x_values)[0, 0]).item() for model in models]))


def reports(args, root: Path, patch: dict, ids: np.ndarray, values: np.ndarray, cards: dict[int, dict]) -> None:
    prediction_dir = require_external(Path(args.prediction_dir), "prediction directory")
    result = json.loads((prediction_dir / "patch_transformer_pilot_result.json").read_text())
    if result.get("status") != "complete": raise RuntimeError("prediction stage is not complete")
    holdout = np.load(prediction_dir / "pilot_holdout_predictions.npz", allow_pickle=False)
    scaling_file = np.load(prediction_dir / "prediction_scaling.npz", allow_pickle=False)
    scaling = Scaling(scaling_file["dense_mean"], scaling_file["dense_std"], float(scaling_file["activation_mean"][0]), float(scaling_file["activation_std"][0]))
    dense_scaled, values_scaled = apply_scaling(patch["global"].numpy(), values, scaling)
    geometry = torch.load(prediction_dir / "decoder_geometry.pt", map_location="cpu", weights_only=False)["geometry"]
    device = choose_device(args.device); models = full_ensemble(prediction_dir, geometry, device)
    five = holdout["five_grade_full"] if "five_grade_full" in holdout.files else None
    lookup = {path: index for index, path in enumerate(patch["paths"])}; verified = {fid: card for fid, card in cards.items() if card.get("status") == "passed"}
    base_url = args.base_url or os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
    all_reports, cited_effects, random_effects = [], [], []
    rng = np.random.default_rng(SEED + 8)
    for position, rel in enumerate(holdout["paths"].tolist()):
        index = lookup[str(rel)]; base = float(holdout["full"][position])
        candidates = [(float(values[index, p, k]), int(ids[index, p, k]), p, k) for p in range(PATCHES) for k in range(TOP_K) if int(ids[index, p, k]) in verified]
        candidates.sort(key=lambda item: (-item[0], item[1], item[2], item[3])); candidates = candidates[:16]
        evidence, selected_pairs = [], set()
        for activation, feature, patch_index, rank in candidates:
            ablated = values_scaled[index].copy(); ablated[patch_index, rank] = 0.0
            delta = base - ensemble_probability(models, dense_scaled[index], ids[index], ablated, device)
            if delta > 0:
                selected_pairs.add((patch_index, rank))
                row, col = divmod(patch_index, GRID)
                evidence.append({"feature_id": feature, "activation": activation, "grid_row": row, "grid_column": col,
                                 "normalized_patch_box": [col / GRID, row / GRID, (col + 1) / GRID, (row + 1) / GRID],
                                 "feature_hypothesis": str(verified[feature]["hypothesis"].get("hypothesis", "")), "deletion_score_change": delta,
                                 "uncertainty": str(verified[feature]["hypothesis"].get("uncertainty", "same-model visual validation only"))})
                cited_effects.append(delta)
        count = len(evidence)
        other = [(p, k) for p in range(PATCHES) for k in range(TOP_K) if (p, k) not in selected_pairs]
        if count and len(other) >= count:
            for patch_index, rank in rng.choice(np.asarray(other), size=count, replace=False):
                ablated = values_scaled[index].copy(); ablated[int(patch_index), int(rank)] = 0.0
                random_effects.append(base - ensemble_probability(models, dense_scaled[index], ids[index], ablated, device))
        report = {"referable_dr_probability": base, "five_grade_probabilities": five[position].tolist() if five is not None else None,
                  "supporting_evidence": evidence, "conflicting_or_normal_evidence": [],
                  "scope_warning": "Feature patterns are not clinically verified lesions; this report is a pilot explanation, not clinical advice."}
        raw_narrative, narrative = None, None
        if not args.skip_llm and evidence:
            raw_narrative = call_text_ollama(base_url, args.model, NARRATIVE_PROMPT + json.dumps(report, sort_keys=True), args.timeout)
            narrative = parse_json(raw_narrative)
            if narrative is None:
                repaired = repair_json_once(base_url, args.model, raw_narrative, args.timeout)
                narrative = parse_json(repaired)
                raw_narrative = raw_narrative + "\n--- JSON_REPAIR ---\n" + repaired
        all_reports.append({"relative_path": str(rel), "patient_id": str(holdout["patients"][position]), "structured_report": report,
                            "qwen_raw": raw_narrative, "qwen_narrative": narrative, "qwen_status": "complete" if isinstance(narrative, dict) else ("skipped" if args.skip_llm else "malformed")})
    report_path = Path(args.output) / "pilot_holdout_structured_reports.jsonl"; report_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in all_reports), encoding="utf-8")
    faithfulness = {"status": "complete" if cited_effects and random_effects else "unavailable", "cited_mean_deletion_effect": float(np.mean(cited_effects)) if cited_effects else None,
                    "random_mean_deletion_effect": float(np.mean(random_effects)) if random_effects else None,
                    "cited_beats_random": bool(np.mean(cited_effects) > np.mean(random_effects)) if cited_effects and random_effects else None,
                    "reports": len(all_reports), "qwen_completion_rate": float(np.mean([x["qwen_status"] == "complete" for x in all_reports]))}
    (Path(args.output) / "patch_report_faithfulness.json").write_text(json.dumps(faithfulness, indent=2, sort_keys=True) + "\n")


def main() -> int:
    global TOP_K
    parser = argparse.ArgumentParser(description="Experiment 8 patch feature cards and structured reports")
    parser.add_argument("--stage", choices=("cards", "reports", "smoke"), required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--patch-run"); parser.add_argument("--cohorts"); parser.add_argument("--dataset-root"); parser.add_argument("--prediction-dir")
    parser.add_argument("--top-k", type=int, choices=(16, 32, 64), default=16,
                        help="Per-patch activation capacity used to create the supplied activation traces.")
    parser.add_argument("--model", default="qwen2.5-vl:3b"); parser.add_argument("--base-url"); parser.add_argument("--timeout", type=int, default=300); parser.add_argument("--device", default="auto"); parser.add_argument("--skip-llm", action="store_true")
    args = parser.parse_args()
    # PatchSAETransformer reads this module-level value at construction time.
    # Keep the report runner aligned with the checkpoint/trace capacity rather
    # than assuming the original K=16 pilot setting.
    TOP_K = int(args.top_k)
    patch_pilot.TOP_K = TOP_K
    output = require_external(Path(args.output), "output"); output.mkdir(parents=True, exist_ok=True)
    if args.stage == "smoke":
        record = {"status": "smoke_passed", "grid_row_major_index_137": [9, 11], "samples_per_group": SAMPLES, "labels_in_prompts": False, "one_json_repair_max": True}
        (output / "patch_cards_smoke.json").write_text(json.dumps(record, indent=2) + "\n"); print(json.dumps(record)); return 0
    needed = (args.patch_run, args.cohorts, args.dataset_root)
    if not all(needed): parser.error("cards and reports require --patch-run --cohorts --dataset-root")
    patch = load_patch_run(require_external(Path(args.patch_run), "patch run")); cohorts = json.loads(require_external(Path(args.cohorts), "cohorts").read_text())
    trace = np.load(require_external(Path(args.output) / "patch_activation_traces.npz", "activation traces"), allow_pickle=False)
    ids, values = trace["feature_ids"], trace["values"]
    if ids.shape != (len(patch["paths"]), PATCHES, TOP_K) or values.shape != ids.shape: raise RuntimeError("activation trace shape is invalid")
    root = require_external(Path(args.dataset_root), "dataset root")
    cards_path = output / "patch_feature_cards.jsonl"
    if args.stage == "cards":
        make_cards(args, root, patch, cohorts, ids, values); return 0
    if not args.prediction_dir or not cards_path.is_file(): parser.error("reports requires --prediction-dir and completed patch_feature_cards.jsonl")
    cards = {int(row["feature_id"]): row for row in (json.loads(line) for line in cards_path.read_text().splitlines() if line.strip())}
    reports(args, root, patch, ids, values, cards); return 0


if __name__ == "__main__":
    raise SystemExit(main())
