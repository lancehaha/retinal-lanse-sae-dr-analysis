"""Select A/B/C checkpoints and finding thresholds using validation only."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from retinavlm_abc_artifact_paths import local_artifact
from train_retinavlm_adapter_olives import QUESTIONS, SPLIT_SHA, load_rows, sha256


ARMS = ("A", "B", "C")


def f1_counts(labels, predictions):
    tp = sum(y == 1 and p for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p for y, p in zip(labels, predictions))
    fn = sum(y == 1 and not p for y, p in zip(labels, predictions))
    tn = sum(y == 0 and not p for y, p in zip(labels, predictions))
    return {"TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "F1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def choose_threshold(labels, margins):
    if len(labels) != len(margins) or not labels or set(labels) != {0, 1}:
        raise ValueError("Threshold selection needs both verified classes")
    candidates = sorted({0.0, *(margin for y, margin in zip(labels, margins) if y == 1)})
    choices = []
    for threshold in candidates:
        metrics = f1_counts(labels, [margin >= threshold for margin in margins])
        choices.append((metrics["F1"], -abs(threshold), -threshold, threshold, metrics))
    _, _, _, threshold, metrics = max(choices)
    return threshold, metrics


def read_candidate(entry, arm, official, map_sha, remote_root=None, local_root=None):
    checkpoint = local_artifact(entry["checkpoint"], remote_root, local_root)
    root = local_artifact(entry["evaluation_dir"], remote_root, local_root)
    if not checkpoint.is_file() or not root.is_dir():
        raise FileNotFoundError("Missing validation candidate checkpoint/evaluation")
    checkpoint_sha = sha256(checkpoint)
    parts = sorted(root.glob("validation_*.receipt.json"))
    if not parts:
        raise ValueError("No validation segment receipts")
    by_index = {}
    receipts = []
    for receipt_path in parts:
        receipt = json.loads(receipt_path.read_text())
        rows_path = local_artifact(receipt["rows_file"], remote_root, local_root)
        expected_rows_path = root / (f"validation_{receipt['start']:05d}_"
                                     f"{receipt['stop']:05d}.jsonl")
        if (receipt.get("status") != "complete_abc_inference_segment"
                or rows_path != expected_rows_path
                or receipt.get("arm") != arm or receipt.get("split") != "validation"
                or receipt.get("checkpoint_sha256") != checkpoint_sha
                or receipt.get("checkpoint_step") != entry["step"]
                or receipt.get("map_set_sha256") != map_sha
                or receipt.get("split_sha256") != SPLIT_SHA
                or receipt.get("rows_sha256") != sha256(rows_path)
                or receipt.get("rows") != receipt["stop"] - receipt["start"]):
            raise ValueError("Validation segment receipt mismatch")
        with rows_path.open(encoding="utf-8") as file:
            segment = [json.loads(line) for line in file]
        if len(segment) != receipt["rows"]:
            raise ValueError("Validation segment row count mismatch")
        for offset, row in enumerate(segment):
            index = receipt["start"] + offset
            if (row.get("index") != index or index in by_index
                    or row.get("arm") != arm or row.get("split") != "validation"
                    or row.get("checkpoint_sha256") != checkpoint_sha
                    or row.get("checkpoint_step") != entry["step"]
                    or row.get("map_set_sha256") != map_sha
                    or index >= len(official)
                    or row.get("source_image_path") != official[index]["source_image_path"]
                    or row.get("patient_key") != official[index]["patient_key"]):
                raise ValueError("Validation scan order, identity, or provenance mismatch")
            predictions = row.get("finding_predictions", {})
            if set(predictions) != set(QUESTIONS):
                raise ValueError("Validation scan missing finding predictions")
            for finding in QUESTIONS:
                expected = official[index][finding]
                label = int(expected) if expected in {"0", "1"} else None
                if predictions[finding]["expert_label"] != label:
                    raise ValueError("Validation finding label mismatch")
            by_index[index] = row
        receipts.append({"path": str(receipt_path), "sha256": sha256(receipt_path),
                         "rows_sha256": receipt["rows_sha256"]})
    if set(by_index) != set(range(len(official))):
        raise ValueError("Validation candidate does not cover every scan exactly once")
    rows = [by_index[index] for index in range(len(official))]
    metrics = {}
    for finding in QUESTIONS:
        labeled = [row["finding_predictions"][finding] for row in rows
                   if row["finding_predictions"][finding]["expert_label"] is not None]
        labels = [item["expert_label"] for item in labeled]
        margins = [float(item["answer_margin"]) for item in labeled]
        if any(not math.isfinite(margin) for margin in margins):
            raise ValueError("Nonfinite validation answer margin")
        threshold, score = choose_threshold(labels, margins)
        zero = f1_counts(labels, [margin >= 0 for margin in margins])
        metrics[finding] = {"threshold_margin": threshold, "validation": score,
                            "zero_margin": zero, "labeled_scans": len(labels)}
    macro_f1 = sum(item["validation"]["F1"] for item in metrics.values()) / len(metrics)
    macro_recall = sum(item["validation"]["recall"] for item in metrics.values()) / len(metrics)
    return {"checkpoint": entry["checkpoint"],
            "local_checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha,
            "step": int(entry["step"]), "evaluation_dir": entry["evaluation_dir"],
            "local_evaluation_dir": str(root),
            "segment_receipts": receipts, "finding_metrics": metrics,
            "macro_F1": macro_f1, "macro_recall": macro_recall,
            "validation_patients": len({row["patient_key"] for row in rows}),
            "validation_scans": len(rows)}


def run(args):
    if args.output.exists() or not 0 <= args.max_recall_drop <= 1:
        raise ValueError("Selection output exists or recall-drop rule invalid")
    groups = load_rows(args.split_csv)
    official = groups["validation"]
    manifest = json.loads(args.candidates.read_text())
    map_sha = sha256(args.map_set)
    if (manifest.get("status") != "prespecified_abc_validation_candidates"
            or set(manifest.get("arms", {})) != set(ARMS)
            or manifest.get("map_set_sha256") != map_sha):
        raise ValueError("Candidate manifest is incomplete or map version changed")
    chosen, all_candidates = {}, {}
    schedules = {}
    for arm in ARMS:
        entries = manifest["arms"][arm]
        if not entries:
            raise ValueError(f"No validation candidates for arm {arm}")
        candidates = [read_candidate(entry, arm, official, map_sha,
                                     getattr(args, "remote_root", None),
                                     getattr(args, "local_root", None)) for entry in entries]
        schedule = [candidate["step"] for candidate in candidates]
        if len(set(schedule)) != len(schedule) or schedule != sorted(schedule):
            raise ValueError("Candidate steps must be distinct and ordered")
        schedules[arm] = schedule
        all_candidates[arm] = candidates
        selected = max(candidates, key=lambda candidate: (candidate["macro_F1"],
                                                           -candidate["step"]))
        chosen[arm] = {
            "checkpoint": selected["checkpoint"],
            "checkpoint_sha256": selected["checkpoint_sha256"],
            "step": selected["step"],
            "thresholds_margin": {finding: block["threshold_margin"]
                                  for finding, block in selected["finding_metrics"].items()},
            "validation_macro_F1": selected["macro_F1"],
            "validation_macro_recall": selected["macro_recall"],
        }
    if len({tuple(schedule) for schedule in schedules.values()}) != 1:
        raise ValueError("A/B/C checkpoint selection schedules differ")
    sensitivity_floor = max(chosen["A"]["validation_macro_recall"],
                            chosen["B"]["validation_macro_recall"]) - args.max_recall_drop
    result = {
        "status": "locked_abc_validation_selection", "split_sha256": SPLIT_SHA,
        "candidate_manifest_sha256": sha256(args.candidates),
        "map_set_sha256": map_sha, "arms": chosen,
        "all_validation_candidates": all_candidates,
        "selection_rule": "highest seven-finding validation macro-F1, tie earliest prespecified step; each finding threshold maximizes validation F1, tie nearest zero margin",
        "prespecified_steps": schedules["A"],
        "maximum_acceptable_C_macro_recall_drop": args.max_recall_drop,
        "C_validation_recall_gate_passed": chosen["C"]["validation_macro_recall"] >= sensitivity_floor,
        "test_accessed": False, "test_targets_used": False,
        "test_cohort_status": "patient-disjoint reused from prior Stage 3 analysis, not pristine",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_rows, finding_rows = [], []
    for arm in ARMS:
        for candidate in all_candidates[arm]:
            common = {"arm": arm, "step": candidate["step"],
                      "checkpoint_sha256": candidate["checkpoint_sha256"],
                      "validation_patients": candidate["validation_patients"],
                      "validation_scans": candidate["validation_scans"]}
            checkpoint_rows.append({**common, "macro_F1": candidate["macro_F1"],
                                    "macro_recall": candidate["macro_recall"],
                                    "selected": int(chosen[arm]["step"] == candidate["step"])})
            for finding, block in candidate["finding_metrics"].items():
                finding_rows.append({**common, "finding": finding,
                                     "labeled_scans": block["labeled_scans"],
                                     "threshold_margin": block["threshold_margin"],
                                     **block["validation"],
                                     "zero_margin_F1": block["zero_margin"]["F1"],
                                     "zero_margin_recall": block["zero_margin"]["recall"]})
    table_paths = {
        "checkpoint_metrics": args.output.with_name("validation_checkpoint_metrics.csv"),
        "finding_metrics": args.output.with_name("validation_finding_metrics.csv"),
    }
    if any(path.exists() for path in table_paths.values()):
        raise FileExistsError("Validation report table already exists")
    for name, rows in (("checkpoint_metrics", checkpoint_rows),
                       ("finding_metrics", finding_rows)):
        with table_paths[name].open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    result["validation_table_sha256"] = {
        name: sha256(path) for name, path in table_paths.items()
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "steps": result["prespecified_steps"],
                      "C_recall_gate": result["C_validation_recall_gate_passed"],
                      "output": str(args.output)}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--map-set", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-recall-drop", type=float, default=0.05)
    parser.add_argument("--remote-root", type=Path)
    parser.add_argument("--local-root", type=Path)
    run(parser.parse_args())
