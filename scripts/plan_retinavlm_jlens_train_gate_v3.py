"""Freeze patient-disjoint map fitting and gate cohorts from OLIVES training patients.

The old validation patients are not used for map selection. The output contains
patient-linked scan paths and belongs in the protected local data directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


FINDINGS = ("B1", "B2", "B3", "B4", "B5", "B6", "SRF")
PATIENT_COUNTS = (20, 20, 21)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def balanced_paths(rows: list[dict], finding: str, per_class: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    chosen = []
    for label in ("0", "1"):
        by_patient = defaultdict(list)
        for row in rows:
            if row[finding] == label:
                by_patient[row["patient_key"]].append(row["source_image_path"])
        for values in by_patient.values():
            rng.shuffle(values)
        patients = sorted(by_patient)
        while len([path for path in chosen if path[0] == label]) < per_class:
            available = [patient for patient in patients if by_patient[patient]]
            if not available:
                raise ValueError(f"Insufficient distinct {finding}={label} scans")
            rng.shuffle(available)
            for patient in available:
                if len([path for path in chosen if path[0] == label]) == per_class:
                    break
                chosen.append((label, by_patient[patient].pop()))
    return [path for _, path in chosen]


def make_plan(rows: list[dict], finding: str, split_sha: str, seed: int,
              trials: int) -> dict:
    patients = sorted({row["patient_key"] for row in rows})
    if len(patients) != sum(PATIENT_COUNTS):
        raise ValueError("Expected the frozen 61-patient training cohort")
    per_class = 50 if finding == "SRF" else 100
    by_patient = defaultdict(list)
    for row in rows:
        by_patient[row["patient_key"]].append(row)
    counts = {patient: Counter(row[finding] for row in values)
              for patient, values in by_patient.items()}
    rng = random.Random(seed)
    best = None
    for trial in range(trials):
        shuffled = patients.copy()
        rng.shuffle(shuffled)
        a, b = PATIENT_COUNTS[:2]
        groups = (shuffled[:a], shuffled[a:a + b], shuffled[a + b:])
        positives = [sum(counts[patient]["1"] for patient in group) for group in groups]
        negatives = [sum(counts[patient]["0"] for patient in group) for group in groups]
        mixed = [sum(counts[patient]["0"] > 0 and counts[patient]["1"] > 0
                     for patient in group) for group in groups]
        min_fit_mixed = 4 if finding == "SRF" else 6
        min_gate_mixed = 8 if finding == "SRF" else 6
        if (min(positives[:2]) < per_class or min(negatives[:2]) < per_class
                or min(mixed[:2]) < min_fit_mixed or mixed[2] < min_gate_mixed
                or positives[2] < 16 or negatives[2] < 16):
            continue
        # Prefer independent gate patients, then diversity in fitting/calibration.
        score = (mixed[2], min(mixed[:2]), min(positives[:2]), trial * -1)
        if best is None or score > best[0]:
            best = (score, groups, positives, negatives, mixed, trial)
    if best is None:
        raise ValueError(f"No feasible patient-disjoint plan for {finding}")
    _, groups, positives, negatives, mixed, trial = best
    direction_rows = [row for patient in groups[0] for row in by_patient[patient]]
    calibration_rows = [row for patient in groups[1] for row in by_patient[patient]]
    gate_rows = [row for patient in groups[2] for row in by_patient[patient]
                 if row[finding] in ("0", "1")]
    direction_paths = balanced_paths(direction_rows, finding, per_class, seed + 1)
    calibration_paths = balanced_paths(calibration_rows, finding, per_class, seed + 2)
    gate_paths = [row["source_image_path"] for row in gate_rows]
    if len(set(direction_paths + calibration_paths + gate_paths)) != sum(
            map(len, (direction_paths, calibration_paths, gate_paths))):
        raise ValueError("Cohort scan overlap")
    return {
        "protocol": "retinavlm_jlens_train_gate_v3",
        "finding": finding,
        "split_sha256": split_sha,
        "seed": seed,
        "trial": trial,
        "gate_cohort": "held-out patients from original training split",
        "direction_per_class": per_class,
        "calibration_per_class": per_class,
        "direction_paths": direction_paths,
        "calibration_paths": calibration_paths,
        "gate_paths": gate_paths,
        "patient_counts": dict(zip(("direction", "calibration", "gate"),
                                   map(len, groups))),
        "positive_scan_pool_counts": dict(zip(("direction", "calibration", "gate"),
                                               positives)),
        "negative_scan_pool_counts": dict(zip(("direction", "calibration", "gate"),
                                               negatives)),
        "mixed_label_patient_counts": dict(zip(("direction", "calibration", "gate"),
                                               mixed)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=4106)
    parser.add_argument("--trials", type=int, default=50000)
    args = parser.parse_args()
    split_sha = sha256(args.split_csv)
    with args.split_csv.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "train"]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary = {"protocol": "retinavlm_jlens_train_gate_v3", "split_sha256": split_sha,
               "findings": {}}
    for offset, finding in enumerate(FINDINGS):
        plan = make_plan(rows, finding, split_sha, args.seed + offset, args.trials)
        path = args.output_dir / f"{finding}_selection_plan.json"
        path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        summary["findings"][finding] = {
            "plan_sha256": sha256(path),
            "direction_per_class": plan["direction_per_class"],
            "calibration_per_class": plan["calibration_per_class"],
            "patient_counts": plan["patient_counts"],
            "mixed_label_patient_counts": plan["mixed_label_patient_counts"],
            "positive_scan_pool_counts": plan["positive_scan_pool_counts"],
        }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
