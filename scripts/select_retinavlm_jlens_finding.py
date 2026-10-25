"""Select one J-Lens pilot finding using validation predictions only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

FINDINGS = ("B1", "B2", "B3", "B4", "B5", "B6", "SRF")
TIE_PRIORITY = ("B5", "B4", "SRF", "B1", "B2", "B3", "B6")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def auc(labels, scores):
    ordered = sorted(zip(scores, labels))
    positive = sum(labels)
    negative = len(labels) - positive
    if positive == 0 or negative == 0:
        raise ValueError("AUC requires both classes")
    positive_rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2
        positive_rank_sum += average_rank * sum(label for _, label in ordered[index:end])
        index = end
    return (positive_rank_sum - positive * (positive + 1) / 2) / (positive * negative)


def select(predictions: Path, output: Path):
    with predictions.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows or {r["split"] for r in rows} != {"validation"}:
        raise ValueError("Selector must receive validation-only predictions")
    if {r["variant"] for r in rows} != {"adapted_byol"}:
        raise ValueError("Expected aligned BYOL model predictions")
    candidates = {}
    for finding in FINDINGS:
        group = [r for r in rows if r["task"] == finding and r["label"] in {"0", "1"}]
        if not group or len({r["source_image_path"] for r in group}) != len(group):
            raise ValueError(f"Missing or duplicate validation rows for {finding}")
        labels = [int(r["label"]) for r in group]
        scores = [float(r["score"]) for r in group]
        if set(labels) != {0, 1} or not all(map(math.isfinite, scores)):
            raise ValueError(f"Invalid two-class scores for {finding}")
        per_patient = defaultdict(lambda: {0: [], 1: []})
        for row, score in zip(group, scores):
            per_patient[row["patient_key"]][int(row["label"])].append(float(score))
        pair_differences = [sum(classes[1]) / len(classes[1]) - sum(classes[0]) / len(classes[0])
                            for classes in per_patient.values() if classes[0] and classes[1]]
        area = float(auc(labels, scores))
        pair_fraction = (sum(delta > 0 for delta in pair_differences) / len(pair_differences)
                         if pair_differences else None)
        eligible = (area >= 0.60 and len(pair_differences) >= 6
                    and pair_fraction is not None and pair_fraction >= 0.60)
        candidates[finding] = {"validation_auc": area, "validation_scans": len(group),
                               "patients": len(per_patient), "opposite_label_patient_pairs": len(pair_differences),
                               "positive_higher_patient_fraction": pair_fraction,
                               "pilot_eligible": eligible}
    eligible_findings = [finding for finding, result in candidates.items() if result["pilot_eligible"]]
    chosen = min(eligible_findings, key=lambda finding: (
        -candidates[finding]["validation_auc"],
        -candidates[finding]["positive_higher_patient_fraction"],
        TIE_PRIORITY.index(finding))) if eligible_findings else None
    result = {"status": "validation_pilot_finding_selected" if chosen else "no_validation_signal_gate_passed",
              "chosen_finding": chosen, "selection_rule": "AUC >= 0.60, >=6 patients with both labels, >=0.60 positive-higher patient fraction; highest AUC wins, then pair fraction, then prespecified B5/B4/SRF/B1/B2/B3/B6 tie order",
              "validation_predictions_sha256": sha256(predictions),
              "test_predictions_accessed": False, "candidates": candidates}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    if chosen:
        print(chosen)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = select(args.validation_predictions, args.output)
    if result["chosen_finding"] is None:
        raise SystemExit(3)
