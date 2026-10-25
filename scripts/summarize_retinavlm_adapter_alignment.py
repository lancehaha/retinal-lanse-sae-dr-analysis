"""Audit and tabulate matched aligned-adapter OLIVES evaluation artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


FINDINGS = ("B1", "B2", "B3", "B4", "B5", "B6", "SRF")
VARIANTS = ("released", "adapted_byol")
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


def verify_split_predictions(split_csv, predictions):
    if sha256(split_csv) != SPLIT_SHA:
        raise ValueError("Frozen patient split hash mismatch")
    official = read_csv(split_csv)
    for split in ("validation", "test"):
        group = [row for row in official if row["split"] == split]
        expected = {
            (finding, row["source_image_path"]):
                (row["patient_key"], row[finding] if row[finding] in {"0", "1"} else "-1")
            for row in group for finding in FINDINGS
        }
        if len(expected) != len(group) * len(FINDINGS):
            raise ValueError(f"Duplicate official {split} image")
        for variant in VARIANTS:
            observed = {}
            for row in predictions[variant][split]:
                if row["task"] not in FINDINGS:
                    continue
                key = (row["task"], row["source_image_path"])
                if key in observed:
                    raise ValueError(f"Duplicate {variant} {split} finding prediction")
                observed[key] = (row["patient_key"], row["label"])
            if observed != expected:
                raise ValueError(f"{variant} {split} predictions differ from frozen split")


def f1(rows):
    tp = sum(r["label"] == 1 and r["pred"] == 1 for r in rows)
    fp = sum(r["label"] == 0 and r["pred"] == 1 for r in rows)
    fn = sum(r["label"] == 1 and r["pred"] == 0 for r in rows)
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def quantile(values, p):
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def run(args):
    if args.bootstrap_replicates < 1:
        raise ValueError("At least one bootstrap replicate is required")
    metrics = {}
    predictions = {}
    receipts = {}
    table = []
    for variant in VARIANTS:
        root = args.checkpoint_root / variant
        receipt_path = root / "training_receipt.json"
        checkpoint_path = root / "adapter_final.pt"
        eval_dir = args.evaluation_root / variant
        metric_path = eval_dir / "f1_metrics.json"
        if not all(path.is_file() for path in (receipt_path, checkpoint_path, metric_path)):
            raise FileNotFoundError(f"Incomplete {variant} adapter/evaluation bundle")
        receipt = json.loads(receipt_path.read_text())
        metric = json.loads(metric_path.read_text())
        if (receipt.get("status") != "adapter_alignment_complete"
                or receipt.get("variant") != variant
                or receipt.get("split_sha256") != SPLIT_SHA
                or receipt.get("adapter_sha256") != sha256(checkpoint_path)
                or metric.get("status") != "adapter_aligned_vlm_olives_f1_complete"
                or metric.get("variant") != variant
                or metric.get("split_sha256") != SPLIT_SHA
                or metric.get("adapter_sha256") != sha256(checkpoint_path)):
            raise ValueError(f"Provenance mismatch for {variant}")
        if (args.expected_batch_size is not None
                and metric.get("inference_batch_size") != args.expected_batch_size):
            raise ValueError(f"{variant} inference batch size differs from report protocol")
        split_rows = {}
        for split in ("validation", "test"):
            path = eval_dir / f"{split}_predictions.csv"
            if sha256(path) != metric[f"{split}_predictions_sha256"]:
                raise ValueError(f"{variant} {split} prediction hash mismatch")
            rows = read_csv(path)
            if len(rows) != metric[f"{split}_prediction_rows"]:
                raise ValueError(f"{variant} {split} prediction row count mismatch")
            split_rows[split] = rows
        metrics[variant] = metric
        predictions[variant] = split_rows
        receipts[variant] = receipt
        for finding in FINDINGS:
            for split in ("validation", "test"):
                value = metric["findings"][finding][split]
                relevant = [r for r in split_rows[split]
                            if r["task"] == finding and r["label"] in {"0", "1"}]
                audit_rows = [{"label": int(r["label"]),
                               "pred": int(float(r["score"]) >= float(r["threshold"]))}
                              for r in relevant]
                if len(audit_rows) != value["n"] or abs(f1(audit_rows) - value["F1"]) > 1e-9:
                    raise ValueError(f"{variant} {split} {finding} F1 differs from saved predictions")
                table.append({
                    "variant": variant, "split": split, "finding": finding,
                    "patients": metric[f"{split}_patients"],
                    "scans": value["n"], "positive_scans": value["positive"],
                    "TN": value["TN"], "FP": value["FP"],
                    "FN": value["FN"], "TP": value["TP"],
                    "precision": value["TP"] / (value["TP"] + value["FP"])
                        if value["TP"] + value["FP"] else 0.0,
                    "sensitivity": value["TP"] / (value["TP"] + value["FN"])
                        if value["TP"] + value["FN"] else 0.0,
                    "F1": value["F1"], "validation_selected_threshold": value["threshold"],
                    "inference_batch_size": metric.get("inference_batch_size"),
                    "adapter_sha256": metric["adapter_sha256"],
                })
    batch_sizes = {metrics[variant].get("inference_batch_size") for variant in VARIANTS}
    if len(batch_sizes) != 1:
        raise ValueError("Compared variants use different inference batch sizes")
    verify_split_predictions(args.split_csv, predictions)
    aligned = {}
    for split in ("validation", "test"):
        by_variant = {}
        for variant in VARIANTS:
            rows = [r for r in predictions[variant][split] if r["task"] in FINDINGS]
            expected = {(r["task"], r["source_image_path"]): r for r in rows}
            if len(expected) != len(rows):
                raise ValueError(f"{variant} duplicate prediction rows")
            by_variant[variant] = {
                key: {"patient": row["patient_key"], "label": int(row["label"]),
                      "pred": int(float(row["score"]) >= float(row["threshold"]))}
                for key, row in expected.items()
            }
        if set(by_variant["released"]) != set(by_variant["adapted_byol"]):
            raise ValueError(f"{split} variants scored different images")
        for key in by_variant["released"]:
            a, b = (by_variant[v][key] for v in VARIANTS)
            if a["patient"] != b["patient"] or a["label"] != b["label"]:
                raise ValueError(f"{split} patient/label pairing changed")
        aligned[split] = by_variant
    test = aligned["test"]
    patients = sorted({r["patient"] for r in test["released"].values()})
    keys_by_patient = defaultdict(list)
    for key, row in test["released"].items():
        keys_by_patient[row["patient"]].append(key)
    rng = random.Random(args.seed)
    macro_differences = []
    finding_differences = {finding: [] for finding in FINDINGS}
    for _ in range(args.bootstrap_replicates):
        sampled_patients = [rng.choice(patients) for _ in patients]
        sampled_keys = [key for patient in sampled_patients for key in keys_by_patient[patient]]
        per_finding = {}
        for finding in FINDINGS:
            selected = [key for key in sampled_keys if key[0] == finding]
            scores = {variant: f1([test[variant][key] for key in selected])
                      for variant in VARIANTS}
            difference = scores["adapted_byol"] - scores["released"]
            finding_differences[finding].append(difference)
            per_finding[finding] = difference
        macro_differences.append(sum(per_finding.values()) / len(FINDINGS))
    summary = {
        "status": "aligned_adapter_evaluation_tabulated",
        "metric_type": "VLM forced-choice final-answer finding F1; separate DRSS representation probe",
        "split_sha256": SPLIT_SHA, "test_patients": len(patients),
        "inference_batch_size": next(iter(batch_sizes)),
        "paired_patient_bootstrap_replicates": args.bootstrap_replicates,
        "macro_F1_test": {v: metrics[v]["finding_macro_F1_test"] for v in VARIANTS},
        "adapted_minus_released_macro_F1": (
            metrics["adapted_byol"]["finding_macro_F1_test"]
            - metrics["released"]["finding_macro_F1_test"]),
        "adapted_minus_released_macro_F1_ci95": [
            quantile(macro_differences, 0.025), quantile(macro_differences, 0.975)],
        "per_finding_adapted_minus_released_F1_ci95": {
            finding: [quantile(values, 0.025), quantile(values, 0.975)]
            for finding, values in finding_differences.items()},
        "DRSS_visit_probe_macro_F1_test": {
            v: metrics[v]["DRSS"]["test"]["macro_F1_all_train_classes"] for v in VARIANTS},
        "adapter_checkpoint_sha256": {v: receipts[v]["adapter_sha256"] for v in VARIANTS},
    }
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Adapter report output directory must be empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_path = args.output_dir / "finding_f1_table.csv"
    with table_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    summary["finding_table_sha256"] = sha256(table_path)
    (args.output_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--expected-batch-size", type=int)
    parser.add_argument("--seed", type=int, default=42)
    run(parser.parse_args())
