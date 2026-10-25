"""Exploratory score stratification on the frozen training-patient map gate.

This does not change candidate eligibility or supply independent validation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def auc(rows: list[dict]) -> float | None:
    ordered = sorted((float(row["calibrated_logit"]), int(row["label"])) for row in rows)
    n_pos = sum(label for _, label in ordered)
    n_neg = len(ordered) - n_pos
    if not n_pos or not n_neg:
        return None
    rank = 1
    positive_rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        positive_rank_sum += sum(label for _, label in ordered[index:end]) * (
            rank + (end - index - 1) / 2
        )
        rank += end - index
        index = end
    return (positive_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def summarize(rows: list[dict]) -> dict:
    return {
        "scans": len(rows),
        "patients": len({row["patient_key"] for row in rows}),
        "positive_scans": sum(row["label"] == "1" for row in rows),
        "auc": auc(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--b2-scores", type=Path, required=True)
    parser.add_argument("--b6-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.split.open(newline="") as handle:
        split = {row["source_image_path"]: row for row in csv.DictReader(handle)}
    result = {
        "status": "exploratory_same_gate_cohort_no_eligibility_change",
        "inputs_sha256": {
            "split": sha256(args.split),
            "b2_scores": sha256(args.b2_scores),
            "b6_scores": sha256(args.b6_scores),
        },
        "findings": {},
    }
    for finding, layer, score_path, strata in (
        ("B2", "20", args.b2_scores, ("B3",)),
        ("B6", "28", args.b6_scores, ("B1", "B2", "B3", "B4", "B5")),
    ):
        with score_path.open(newline="") as handle:
            rows = [row for row in csv.DictReader(handle) if row["layer"] == layer]
        if not rows or len({row["source_image_path"] for row in rows}) != len(rows):
            raise ValueError(f"Incomplete or duplicate {finding} score table")
        if any(row["finding"] != finding or row["source_image_path"] not in split
               or row["label"] != split[row["source_image_path"]][finding]
               or split[row["source_image_path"]]["split"] != "train" for row in rows):
            raise ValueError(f"{finding} score table disagrees with the official training split")
        out = {"layer": int(layer), "overall": summarize(rows), "strata": {}}
        for other in strata:
            out["strata"][other] = {
                value: summarize([row for row in rows
                                  if split[row["source_image_path"]][other] == value])
                for value in ("0", "1")
            }
        result["findings"][finding] = out
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
