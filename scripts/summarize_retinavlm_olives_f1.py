"""Patient-clustered uncertainty and simple baselines for the frozen F1 run."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def f1(tp, fp, fn):
    den = 2 * tp + fp + fn
    return 2 * tp / den if den else 0.0


def quantile(values, q):
    ordered = sorted(values)
    x = q * (len(ordered) - 1)
    lower = int(x)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] * (upper - x) + ordered[upper] * (x - lower)


def main(args):
    metrics = json.loads(args.metrics.read_text())
    with args.predictions.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "test"]
    findings = list(metrics["findings"])
    encoders = ("released", "adapted_byol")
    counts = defaultdict(lambda: Counter())
    baseline = {}
    for finding in findings:
        block = metrics["findings"][finding]["encoders"]["released"]["test"]
        positives, negatives = block["positive"], block["n"] - block["positive"]
        baseline[finding] = {"always_positive_F1": f1(positives, negatives, 0),
                             "positive_scans": positives, "total_scans": block["n"]}
    for row in rows:
        if row["task"] not in findings:
            continue
        if row["label"] not in {"0", "1"}:
            continue
        pred = float(row["score"]) >= float(row["threshold"])
        label = row["label"] == "1"
        category = "TP" if pred and label else "FP" if pred else "FN" if label else "TN"
        counts[row["patient_key"], row["task"], row["encoder"]][category] += 1
    patients = sorted({r["patient_key"] for r in rows if r["task"] in findings})
    if len(patients) != 13:
        raise ValueError("Expected 13 test patients")
    observed = {model: sum(metrics["findings"][f]["encoders"][model]["test"]["F1"] for f in findings) / len(findings)
                for model in encoders}
    for finding in findings:
        for model in encoders:
            total = sum((counts[p, finding, model] for p in patients), Counter())
            computed = f1(total["TP"], total["FP"], total["FN"])
            reported = metrics["findings"][finding]["encoders"][model]["test"]["F1"]
            if abs(computed - reported) > 1e-10:
                raise ValueError(f"Prediction/metric mismatch: {finding}, {model}")
    rng = random.Random(42)
    deltas = []
    for _ in range(10000):
        sample = rng.choices(patients, k=len(patients))
        macro = {}
        for model in encoders:
            label_f1 = []
            for finding in findings:
                total = sum((counts[p, finding, model] for p in sample), Counter())
                label_f1.append(f1(total["TP"], total["FP"], total["FN"]))
            macro[model] = sum(label_f1) / len(label_f1)
        deltas.append(macro["adapted_byol"] - macro["released"])
    summary = {"status": "olives_f1_patient_bootstrap_summary", "test_patients": len(patients),
               "finding_labels": findings, "finding_test_macro_F1": observed,
               "adapted_minus_released_finding_macro_F1": observed["adapted_byol"] - observed["released"],
               "paired_patient_bootstrap_95pct": [quantile(deltas, 0.025), quantile(deltas, 0.975)],
               "simple_baselines": baseline,
               "DRSS_note": "Only 12 test visits from 6 patients; report descriptive F1 and confusion, not a strong population estimate."}
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    main(p.parse_args())
