"""Join the neutral-prompt layer map to released RetinaVLM validation errors.

This is an exploratory patient-held-out association analysis. It neither
selects an arm-D lens nor turns correlations into causal or lesion claims.
The test split is not read.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from fit_retinavlm_fluid_lens_candidate import CANDIDATE_STATUSES, NEUTRAL_QUESTION
from retinavlm_released_model import SPECIALIST_REVISION
from retinavlm_verified_images import load_verified_manifest, select_split, sha256
from validate_retinavlm_fluid_lens import auc, percentile


def error_category(label: int, answer: str) -> str:
    if label == 1:
        return "correct_positive" if answer == "present" else "false_negative"
    return "false_positive" if answer == "present" else "correct_negative"


def optional_auc(pairs: list[tuple[int, float]]) -> float | None:
    return auc(pairs) if len({label for label, _ in pairs}) == 2 else None


def auc_pairs(rows: list[dict], layer: str, outcome: str) -> list[tuple[int, float]]:
    if outcome == "fluid":
        return [(row["fluid"], row["scores"][layer]) for row in rows]
    if outcome == "false_positive":
        return [(int(row["category"] == "false_positive"), row["scores"][layer])
                for row in rows if row["fluid"] == 0]
    if outcome == "false_negative":
        return [(int(row["category"] == "false_negative"), -row["scores"][layer])
                for row in rows if row["fluid"] == 1]
    raise ValueError("Unknown neutral-map outcome")


def patient_bootstrap(rows: list[dict], layer: str, outcome: str,
                      replicates: int, seed: int) -> dict:
    if replicates < 1:
        raise ValueError("Bootstrap needs at least one replicate")
    by_patient = defaultdict(list)
    by_trial = defaultdict(set)
    for row in rows:
        patient_key = (row["trial"], row["Patient_ID"])
        by_patient[patient_key].append(row)
        by_trial[row["trial"]].add(patient_key)
    rng = random.Random(seed)
    values = []
    for _ in range(replicates):
        sampled = []
        for trial in sorted(by_trial):
            patients = sorted(by_trial[trial])
            for _ in patients:
                sampled.extend(by_patient[rng.choice(patients)])
        value = optional_auc(auc_pairs(sampled, layer, outcome))
        if value is not None:
            values.append(value)
    return {
        "ci95": [percentile(values, 0.025), percentile(values, 0.975)] if values else None,
        "valid_replicates": len(values), "requested_replicates": replicates,
        "trial_patient_counts": {trial: len(group) for trial, group in sorted(by_trial.items())},
    }


def analyze_rows(rows: list[dict], layers: list[int], replicates: int,
                 seed: int) -> dict:
    categories = Counter(row["category"] for row in rows)
    per_layer = {}
    for layer in layers:
        key = str(layer)
        per_layer[key] = {}
        for outcome in ("fluid", "false_positive", "false_negative"):
            pairs = auc_pairs(rows, key, outcome)
            per_layer[key][outcome] = {
                "auc": optional_auc(pairs),
                "positive_cases": sum(label == 1 for label, _ in pairs),
                "negative_cases": sum(label == 0 for label, _ in pairs),
                "patient_bootstrap": patient_bootstrap(rows, key, outcome,
                                                       replicates, seed + layer),
            }
    return {"error_counts": dict(sorted(categories.items())), "per_layer": per_layer}


def paired_auc_difference(lens_rows: list[dict], probe_rows: list[dict],
                          layer: str, outcome: str, replicates: int, seed: int) -> dict:
    if len(lens_rows) != len(probe_rows):
        raise ValueError("Lens and probe maps cover different numbers of cases")
    pairs_by_patient = defaultdict(list)
    by_trial = defaultdict(set)
    for lens, probe in zip(lens_rows, probe_rows, strict=True):
        if lens["source_image_path"] != probe["source_image_path"]:
            raise ValueError("Lens and probe case order differs")
        patient_key = (lens["trial"], lens["Patient_ID"])
        pairs_by_patient[patient_key].append((lens, probe))
        by_trial[lens["trial"]].add(patient_key)
    lens_auc = optional_auc(auc_pairs(lens_rows, layer, outcome))
    probe_auc = optional_auc(auc_pairs(probe_rows, layer, outcome))
    rng = random.Random(seed)
    differences = []
    for _ in range(replicates):
        sampled = []
        for trial in sorted(by_trial):
            patients = sorted(by_trial[trial])
            for _ in patients:
                sampled.extend(pairs_by_patient[rng.choice(patients)])
        sampled_lens = [pair[0] for pair in sampled]
        sampled_probe = [pair[1] for pair in sampled]
        sampled_lens_auc = optional_auc(auc_pairs(sampled_lens, layer, outcome))
        sampled_probe_auc = optional_auc(auc_pairs(sampled_probe, layer, outcome))
        if sampled_lens_auc is not None and sampled_probe_auc is not None:
            differences.append(sampled_lens_auc - sampled_probe_auc)
    return {
        "point_estimate": lens_auc - probe_auc if lens_auc is not None and probe_auc is not None else None,
        "patient_bootstrap_ci95": ([percentile(differences, 0.025),
                                    percentile(differences, 0.975)] if differences else None),
        "valid_replicates": len(differences),
        "requested_replicates": replicates,
    }


def run(*, manifest: Path, image_audit: Path, neutral_candidate: Path,
        released_prediction_dir: Path, output: Path,
        bootstrap_replicates: int = 2000,
        probe_receipt: Path | None = None) -> dict:
    if output.exists():
        raise FileExistsError("Neutral-map analysis already exists")
    all_rows = load_verified_manifest(manifest, image_audit)
    validation_rows = select_split(all_rows, "validation")
    official = {row["source_image_path"]: row for row in validation_rows}
    manifest_digest, audit_digest = sha256(manifest), sha256(image_audit)
    candidate = json.loads(neutral_candidate.read_text(encoding="utf-8"))
    layers = candidate.get("source_layers")
    if (candidate.get("status") != CANDIDATE_STATUSES["neutral"]
            or candidate.get("readout_mode") != "neutral"
            or candidate.get("question") != NEUTRAL_QUESTION
            or candidate.get("checkpoint_revision") != SPECIALIST_REVISION
            or candidate.get("manifest_sha256") != manifest_digest
            or candidate.get("image_audit_sha256") != audit_digest
            or candidate.get("test_split_accessed") is not False
            or not isinstance(layers, list) or not layers
            or len(set(layers)) != len(layers)
            or any(not isinstance(layer, int) or layer < 0 for layer in layers)):
        raise ValueError("Neutral candidate is not bound to this validation cohort")
    candidate_scores = candidate.get("validation_score_rows")
    if (not isinstance(candidate_scores, list)
            or len(candidate_scores) != len(official)
            or {row["source_image_path"] for row in candidate_scores} != set(official)):
        raise ValueError("Neutral candidate lacks complete validation scores")
    scores_by_path = {}
    for row in candidate_scores:
        source = row["source_image_path"]
        gold = official[source]
        if (row["Patient_ID"] != gold["Patient_ID"] or row["trial"] != gold["trial"]
                or row["fluid"] != int(gold["fluid"])
                or set(row["scores"]) != {str(layer) for layer in layers}):
            raise ValueError("Neutral scores differ from expert validation rows")
        scores_by_path[source] = row["scores"]
    candidates = candidate.get("candidates", {})
    if set(candidates) != {str(layer) for layer in layers}:
        raise ValueError("Neutral calibration is missing a selected layer")

    prediction_path = released_prediction_dir / "predictions.csv"
    receipt_path = released_prediction_dir / "prediction_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (receipt.get("status") != "complete_verified_split_predictions"
            or receipt.get("arm") != "A" or receipt.get("split") != "validation"
            or receipt.get("manifest_sha256") != manifest_digest
            or receipt.get("image_audit_sha256") != audit_digest
            or receipt.get("released_revision") != SPECIALIST_REVISION
            or receipt.get("checkpoint_sha256") is not None
            or receipt.get("scans") != len(official)
            or receipt.get("predictions_sha256") != sha256(prediction_path)):
        raise ValueError("Released A predictions are not bound to this validation cohort")
    with prediction_path.open(newline="", encoding="utf-8") as handle:
        predictions = list(csv.DictReader(handle))
    if len(predictions) != len(official):
        raise ValueError("Released A predictions are incomplete")
    joined = []
    seen = set()
    for prediction in predictions:
        source = prediction["source_image_path"]
        if (source in seen or source not in official
                or prediction["image_sha256"] != official[source]["sha256"]
                or prediction["arm"] != "A"
                or prediction["seed"] != str(receipt["seed"])
                or prediction["answer"] not in {"present", "absent"}):
            raise ValueError("Released A prediction is repeated or mismatched")
        seen.add(source)
        gold = official[source]
        label = int(gold["fluid"])
        calibrated = {}
        for layer in layers:
            key = str(layer)
            raw = float(scores_by_path[source][key])
            scale = float(candidates[key]["scale"])
            intercept = float(candidates[key]["intercept"])
            value = scale * raw + intercept
            if not all(map(math.isfinite, (raw, scale, intercept, value))) or scale == 0:
                raise ValueError("Neutral map contains nonfinite or degenerate scores")
            calibrated[key] = value
        joined.append({"source_image_path": source, "Patient_ID": gold["Patient_ID"],
                       "trial": gold["trial"], "fluid": label,
                       "answer": prediction["answer"],
                       "category": error_category(label, prediction["answer"]),
                       "scores": calibrated})
    if seen != set(official):
        raise ValueError("Released A validation predictions omit official images")
    joined.sort(key=lambda row: row["source_image_path"])
    analysis = analyze_rows(joined, layers, bootstrap_replicates, int(candidate["seed"]))
    probe_comparison = None
    if probe_receipt is not None:
        probe = json.loads(probe_receipt.read_text(encoding="utf-8"))
        probe_scores = probe.get("validation_score_rows")
        if (probe.get("status") != "frozen_neutral_linear_probe_validation"
                or probe.get("checkpoint_revision") != SPECIALIST_REVISION
                or probe.get("manifest_sha256") != manifest_digest
                or probe.get("image_audit_sha256") != audit_digest
                or probe.get("neutral_candidate_sha256") != sha256(neutral_candidate)
                or probe.get("question") != NEUTRAL_QUESTION
                or probe.get("source_layers") != layers
                or probe.get("validation_scans") != len(joined)
                or probe.get("test_split_accessed") is not False
                or not isinstance(probe_scores, list)
                or len(probe_scores) != len(joined)
                or {row["source_image_path"] for row in probe_scores} != set(official)):
            raise ValueError("Neutral probe is not bound to the same held-out cases")
        probe_by_path = {row["source_image_path"]: row for row in probe_scores}
        probe_rows = []
        for lens_row in joined:
            source = lens_row["source_image_path"]
            probe_row = probe_by_path[source]
            if (probe_row["Patient_ID"] != lens_row["Patient_ID"]
                    or probe_row["trial"] != lens_row["trial"]
                    or probe_row["fluid"] != lens_row["fluid"]
                    or set(probe_row["scores"]) != {str(layer) for layer in layers}):
                raise ValueError("Probe validation row differs from lens gold row")
            calibrated_scores = {key: float(value) for key, value in probe_row["scores"].items()}
            if any(not math.isfinite(value) for value in calibrated_scores.values()):
                raise ValueError("Probe validation score is nonfinite")
            probe_rows.append({**lens_row, "scores": calibrated_scores})
            lens_row["probe_scores"] = calibrated_scores
        probe_analysis = analyze_rows(probe_rows, layers, bootstrap_replicates,
                                      int(candidate["seed"]))
        paired = {}
        for layer in layers:
            key = str(layer)
            paired[key] = {outcome: paired_auc_difference(
                joined, probe_rows, key, outcome, bootstrap_replicates,
                int(candidate["seed"]) + layer)
                for outcome in ("fluid", "false_positive", "false_negative")}
        probe_comparison = {
            "probe_receipt_sha256": sha256(probe_receipt),
            "probe_per_layer": probe_analysis["per_layer"],
            "lens_minus_probe_auc": paired,
        }
    result = {
        "status": "exploratory_neutral_validation_map_no_test_access",
        "manifest_sha256": manifest_digest,
        "image_audit_sha256": audit_digest,
        "candidate_sha256": sha256(neutral_candidate),
        "released_prediction_receipt_sha256": sha256(receipt_path),
        "released_predictions_sha256": sha256(prediction_path),
        "scans": len(joined),
        "patients": len({row["Patient_ID"] for row in joined}),
        "question": NEUTRAL_QUESTION,
        "source_layers": layers,
        **analysis,
        "probe_comparison": probe_comparison,
        "case_map_rows": joined,
        "test_split_accessed": False,
        "interpretation_boundary": "Association with output errors only; no lesion localization or causal claim",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-image-manifest", type=Path, required=True)
    parser.add_argument("--image-audit", type=Path, required=True)
    parser.add_argument("--neutral-candidate", type=Path, required=True)
    parser.add_argument("--released-prediction-dir", type=Path, required=True)
    parser.add_argument("--probe-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    result = run(manifest=args.official_image_manifest, image_audit=args.image_audit,
                 neutral_candidate=args.neutral_candidate,
                 released_prediction_dir=args.released_prediction_dir,
                 output=args.output, bootstrap_replicates=args.bootstrap_replicates,
                 probe_receipt=args.probe_receipt)
    print(json.dumps({key: value for key, value in result.items() if key != "case_map_rows"}, indent=2))
