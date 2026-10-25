"""Audit a fitted aligned-model J-Lens map before causal intervention checks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


SPLIT_SHA = "28690d3ced5cdd3349100448b8f471cb56bb1690b0502c28148d055a266068fe"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def validate_v3_plan(summary: dict, scores: list[dict], plan_path: Path,
                     split_path: Path) -> None:
    if sha256(plan_path) != summary["selection_plan_sha256"]:
        raise ValueError("Selection-plan hash mismatch")
    plan = json.loads(plan_path.read_text())
    if (plan.get("protocol") != "retinavlm_jlens_train_gate_v3"
            or plan.get("finding") != summary["finding"]
            or plan.get("split_sha256") != sha256(split_path)
            or plan.get("split_sha256") != summary["split_sha256"]):
        raise ValueError("Selection plan does not match the frozen split")
    split_rows = read_csv(split_path)
    train = {row["source_image_path"]: row for row in split_rows
             if row["split"] == "train"}
    if len(train) != sum(row["split"] == "train" for row in split_rows):
        raise ValueError("Duplicate training scan paths")
    cohorts = []
    finding = summary["finding"]
    for key, per_class in (("direction_paths", summary["direction_per_class"]),
                           ("calibration_paths", summary["calibration_per_class"]),
                           ("gate_paths", None)):
        paths = plan[key]
        if len(paths) != len(set(paths)) or any(path not in train for path in paths):
            raise ValueError(f"Invalid {key} in selection plan")
        rows = [train[path] for path in paths]
        if per_class is not None and any(
                sum(row[finding] == str(label) for row in rows) != per_class
                for label in (0, 1)):
            raise ValueError(f"Per-class sample budget mismatch in {key}")
        cohorts.append(rows)
    patients = [{row["patient_key"] for row in rows} for rows in cohorts]
    if any(patients[i] & patients[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Selection plan shares patients between cohorts")
    if len(set().union(*(set(row["source_image_path"] for row in rows)
                         for rows in cohorts))) != sum(map(len, cohorts)):
        raise ValueError("Selection plan shares scans between cohorts")
    if (summary["direction_contexts"] != len(cohorts[0])
            or summary["calibration_scans"] != len(cohorts[1])
            or summary["validation_scans"] != len(cohorts[2])
            or summary["validation_patients"] != len(patients[2])):
        raise ValueError("Map summary cohort counts differ from selection plan")
    expected = {(row["source_image_path"], row["patient_key"], row[finding])
                for row in cohorts[2]}
    for layer in summary["source_layers"]:
        actual = {(row["source_image_path"], row["patient_key"], row["label"])
                  for row in scores if int(row["layer"]) == int(layer)}
        if actual != expected:
            raise ValueError(f"Gate scores at layer {layer} differ from selection plan")


def audit(directory: Path, output: Path, selection_plan: Path | None = None,
          split_csv: Path | None = None):
    summary_path = directory / "summary.json"
    score_path = directory / "validation_map_scores.csv"
    control_path = directory / "image_controls.csv"
    map_path = directory / "map_and_probe.pt"
    if any(not path.is_file() for path in
           (summary_path, score_path, control_path, map_path)):
        raise FileNotFoundError("Incomplete map candidate bundle")
    summary = json.loads(summary_path.read_text())
    if (summary.get("map_eval_cohort") == "held-out original training patients"
            and not summary.get("selection_plan_sha256")):
        raise ValueError("Training-only gate requires a selection-plan hash")
    if (summary.get("status") != "candidate_map_fitted_validation_pending_gate"
            or summary.get("split_sha256") != SPLIT_SHA
            or summary.get("test_accessed") is not False
            or sha256(map_path) != summary.get("map_file_sha256")
            or sha256(score_path) != summary.get("validation_rows_sha256")
            or sha256(control_path) != summary.get("controls_sha256")):
        raise ValueError("Map candidate provenance mismatch")
    finding = summary["finding"]
    floor = 50 if finding == "SRF" else 100
    if summary.get("map_eval_cohort") == "held-out original training patients" and (
            int(summary.get("direction_per_class", -1)) < floor
            or int(summary.get("calibration_per_class", -1)) < floor
            or summary["direction_contexts"] != 2 * summary["direction_per_class"]
            or summary["calibration_scans"] != 2 * summary["calibration_per_class"]):
        raise ValueError("v3 fit reduced or misreported the per-class sample budget")
    layers = [int(layer) for layer in summary["source_layers"]]
    scores = read_csv(score_path)
    controls = read_csv(control_path)
    if summary.get("map_eval_cohort") == "held-out original training patients":
        if selection_plan is None or split_csv is None:
            raise ValueError("v3 audit requires selection plan and split CSV")
        validate_v3_plan(summary, scores, selection_plan, split_csv)
    by_layer = defaultdict(list)
    for row in scores:
        layer = int(row["layer"])
        if row["finding"] != finding or layer not in layers:
            raise ValueError("Unexpected validation score row")
        if int(row["label"]) not in (0, 1):
            raise ValueError("Invalid validation label")
        calibration = summary["calibration"][str(layer)]
        expected_logit = (float(calibration["scale"]) * float(row["raw_score"])
                          + float(calibration["intercept"]))
        if abs(float(row["calibrated_logit"]) - expected_logit) > 1e-5:
            raise ValueError("Calibrated score does not match map parameters")
        by_layer[layer].append(row)
    if any(len(by_layer[layer]) != summary["validation_scans"] for layer in layers):
        raise ValueError("Missing validation scans for a layer")
    keys = [{(r["source_image_path"], r["patient_key"], r["label"]) for r in by_layer[layer]}
            for layer in layers]
    if any(len(keyset) != summary["validation_scans"] for keyset in keys) or any(
            keyset != keys[0] for keyset in keys[1:]):
        raise ValueError("Layer validation cohorts differ or contain duplicates")
    score_lookup = {(int(r["layer"]), r["source_image_path"]): float(r["raw_score"])
                    for r in scores}
    control_deltas = defaultdict(list)
    for row in controls:
        layer = int(row["layer"])
        key = (layer, row["source_image_path"])
        if (row["finding"] != finding or layer not in layers
                or key not in score_lookup
                or abs(float(row["real_score"]) - score_lookup[key]) > 1e-5):
            raise ValueError("Control row does not match validation score")
        scale = float(summary["calibration"][str(layer)]["scale"])
        control_deltas[layer].append(
            abs(scale * (float(row["real_score"]) - float(row["blank_score"]))))
    decisions = {}
    for layer in layers:
        result = summary["layers"][str(layer)]
        if len(control_deltas[layer]) != int(result["control_scans"]):
            raise ValueError("Missing blank-image controls")
        gradient = summary["actual_model_gradient_checks"][str(layer)]
        auc = float(result["validation_auc"])
        lower = float(result["patient_bootstrap_auc_ci95"][0])
        pairs = int(result["within_patient_opposite_label_pairs"])
        pair_fraction = result["positive_higher_pair_fraction"]
        pair_fraction = float(pair_fraction) if pair_fraction is not None else float("nan")
        analytic = float(gradient["analytic"])
        numeric_error = float(gradient["absolute_error"])
        blank_median = statistics.median(control_deltas[layer]) if control_deltas[layer] else 0.0
        checks = {
            "validation_auc_at_least_0_60": auc >= 0.60,
            "patient_bootstrap_lower_above_0_50": lower > 0.50,
            "six_opposite_label_patients": pairs >= 6,
            "positive_higher_pair_fraction_at_least_0_60": pair_fraction >= 0.60,
            "calibrated_blank_change_at_least_0_10": blank_median >= 0.10,
            "finite_difference_error_within_tolerance": (
                math.isfinite(numeric_error)
                and numeric_error <= max(0.10, 0.50 * abs(analytic))),
        }
        decisions[str(layer)] = {
            "checks": checks, "statistical_gate_passed": all(checks.values()),
            "validation_auc": auc, "patient_bootstrap_auc_lower": lower,
            "same_patient_pair_fraction": pair_fraction,
            "calibrated_median_blank_change": blank_median,
            "gradient_absolute_error": numeric_error,
            "linear_probe_validation_auc": float(result["linear_probe_validation_auc"]),
            "diagnostic_readability_threshold_logit": 0.0,
        }
    selected = sorted(
        (layer for layer in layers if decisions[str(layer)]["statistical_gate_passed"]),
        key=lambda layer: (-decisions[str(layer)]["validation_auc"], layer))[:4]
    selected.sort()
    status = ("candidate_statistical_gate_passed_needs_intervention"
              if len(selected) >= 2 else "candidate_statistical_gate_failed")
    receipt = {
        "status": status, "finding": finding, "selected_layers": selected,
        "all_diagnostic_layers": layers,
        "selection_rule": "up to four eligible layers with highest validation AUC, at least two required; fixed zero calibrated-logit diagnostic threshold",
        "map_sha256": sha256(map_path), "summary_sha256": sha256(summary_path),
        "validation_scores_sha256": sha256(score_path),
        "image_controls_sha256": sha256(control_path),
        "validation_patients": summary["validation_patients"],
        "validation_scans": summary["validation_scans"],
        "map_eval_cohort": summary.get("map_eval_cohort", "original validation patients"),
        "selection_plan_sha256": summary.get("selection_plan_sha256"),
        "test_images_accessed": False, "test_targets_used": False,
        "layers": decisions,
        "limitation": "statistical/image controls only; actual state intervention and report-claim effects still required",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": status, "finding": finding,
                      "selected_layers": selected, "receipt": str(output)}), flush=True)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-plan", type=Path)
    parser.add_argument("--split-csv", type=Path)
    args = parser.parse_args()
    audit(args.map_dir, args.output, args.selection_plan, args.split_csv)
