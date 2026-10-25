"""Summarize one complete, training-selected C validation pass on all OLIVES patients."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from collect_retinavlm_abc_validation_segments import receipt_verified
from retinavlm_olives_report import parse_report_claims
from train_retinavlm_adapter_olives import QUESTIONS, SPLIT_SHA, load_rows, sha256


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Empty table: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_complete(args, official, lock, map_sha):
    if len(args.manifest) != 4 or len(args.transfer_receipt) != 4:
        raise ValueError("Four frozen validation segments and transfer receipts required")
    by_index = {}
    provenance = []
    for manifest_path, transfer_path in zip(args.manifest, args.transfer_receipt, strict=True):
        manifest = json.loads(manifest_path.read_text())
        if not receipt_verified(manifest_path, transfer_path):
            raise ValueError(f"Unverified transfer: {manifest_path}")
        start, stop = manifest["start"], manifest["stop"]
        if (manifest.get("arm") != "C" or manifest.get("checkpoint_step") != lock["step"]
                or manifest.get("split") != "validation"):
            raise ValueError("Validation manifest identity mismatch")
        artifacts = {Path(entry["remote"]).name: Path(entry["local"])
                     for entry in manifest["artifacts"]}
        base = f"validation_{start:05d}_{stop:05d}"
        part = artifacts[base + ".jsonl"]
        receipt_path = artifacts[base + ".receipt.json"]
        segment = json.loads(receipt_path.read_text())
        if (segment.get("status") != "complete_abc_inference_segment"
                or segment.get("arm") != "C" or segment.get("split") != "validation"
                or segment.get("start") != start or segment.get("stop") != stop
                or segment.get("rows") != stop - start
                or segment.get("checkpoint_step") != lock["step"]
                or segment.get("checkpoint_sha256") != lock["checkpoint_sha256"]
                or segment.get("map_set_sha256") != map_sha
                or segment.get("split_sha256") != SPLIT_SHA
                or segment.get("rows_sha256") != sha256(part)):
            raise ValueError(f"Segment provenance mismatch: {base}")
        count = 0
        with part.open(encoding="utf-8") as file:
            for count, line in enumerate(file, 1):
                row = json.loads(line)
                index = start + count - 1
                if index >= stop or index in by_index:
                    raise ValueError("Duplicate or excess validation row")
                source = official[index]
                if (row.get("index") != index or row.get("arm") != "C"
                        or row.get("split") != "validation"
                        or row.get("source_image_path") != source["source_image_path"]
                        or row.get("patient_key") != source["patient_key"]
                        or row.get("checkpoint_sha256") != lock["checkpoint_sha256"]
                        or row.get("map_set_sha256") != map_sha
                        or row.get("parsed_report_claims") != parse_report_claims(
                            row.get("generated_report", ""))
                        or set(row.get("finding_predictions", {})) != set(QUESTIONS)):
                    raise ValueError(f"Scan identity or report mismatch at {index}")
                by_index[index] = row
        if count != stop - start:
            raise ValueError(f"Incomplete segment {base}")
        provenance.append({"manifest": str(manifest_path),
                           "manifest_sha256": sha256(manifest_path),
                           "transfer_receipt": str(transfer_path),
                           "transfer_receipt_sha256": sha256(transfer_path),
                           "inference_receipt_sha256": sha256(receipt_path),
                           "rows_sha256": sha256(part),
                           "start": start, "stop": stop,
                           "job_id": manifest["job_id"]})
    if set(by_index) != set(range(len(official))) or len(official) != 1470:
        raise ValueError("Full 1470-scan validation coverage was not achieved")
    if len({source["patient_key"] for source in official}) != 13:
        raise ValueError("Expected 13 validation patients")
    return [by_index[i] for i in range(len(official))], provenance


def counts(pairs):
    tp = sum(label == 1 and pred == 1 for label, pred in pairs)
    fp = sum(label == 0 and pred == 1 for label, pred in pairs)
    fn = sum(label == 1 and pred == 0 for label, pred in pairs)
    tn = sum(label == 0 and pred == 0 for label, pred in pairs)
    total = tp + fp + fn + tn
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "accuracy": (tp + tn) / total if total else None,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "F1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    return values[int((len(values) - 1) * fraction)]


def bootstrap(rows, metric, seed, replicates):
    by_patient = defaultdict(list)
    for row in rows:
        by_patient[row["patient_key"]].append(row)
    patients = sorted(by_patient)
    rng = random.Random(seed)
    samples = []
    for _ in range(replicates):
        selected = [row for _ in patients for row in by_patient[rng.choice(patients)]]
        value = metric(selected)
        if value is not None:
            samples.append(value)
    return percentile(samples, 0.025), percentile(samples, 0.975), len(samples)


def run(args):
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Result directory must be new or empty")
    official = load_rows(args.split_csv)["validation"]
    lock = json.loads(args.selection_lock.read_text())
    map_set = json.loads(args.map_set.read_text())
    map_sha = sha256(args.map_set)
    if (lock.get("status") != "locked_c_training_only_hard_cap"
            or lock.get("step") != 3000 or lock.get("selection_basis") != "frozen_training_only_hard_cap_3000"
            or lock.get("validation_used_for_checkpoint_selection") is not False
            or lock.get("test_used") is not False
            or lock.get("split_sha256") != SPLIT_SHA
            or lock.get("map_set_sha256") != map_sha
            or sha256(Path(lock["checkpoint_local"])) != lock["checkpoint_sha256"]
            or map_set.get("status") != "validated_retinavlm_olives_map_set"
            or set(map_set["findings"]) != {"B1", "B4", "B5"}):
        raise ValueError("Frozen training selection or map set mismatch")
    rows, provenance = load_complete(args, official, lock, map_sha)
    predictions, claims, reports, map_traces, per_finding, uncertainty = [], [], [], [], [], []
    for row, source in zip(rows, official, strict=True):
        common = {"scan_index": row["index"], "patient_key": row["patient_key"],
                  "source_image_path": row["source_image_path"], "checkpoint_step": lock["step"],
                  "checkpoint_sha256": lock["checkpoint_sha256"]}
        reports.append({**common, "generated_report": row["generated_report"],
                        "parsed_report_claims_json": json.dumps(row["parsed_report_claims"], sort_keys=True)})
        for finding in QUESTIONS:
            pred = row["finding_predictions"][finding]
            expected = source[finding]
            label = int(expected) if expected in {"0", "1"} else None
            if pred["expert_label"] != label:
                raise ValueError("Finding label mismatch")
            answer = int(float(pred["answer_margin"]) >= 0)
            claim = row["parsed_report_claims"][finding]
            predictions.append({**common, "finding": finding, "expert_label": label,
                                "answer_margin": float(pred["answer_margin"]),
                                "zero_margin_answer": pred["zero_margin_answer"],
                                "zero_margin_predicted_present": answer,
                                "answer_correct": int(answer == label) if label is not None else None})
            claims.append({**common, "finding": finding, "expert_label": label,
                           "parsed_claim": claim,
                           "report_correct": int(claim == label) if label is not None else None,
                           "explicit_false_positive": int(label == 0 and claim == 1),
                           "explicit_false_negative": int(label == 1 and claim == 0),
                           "missing_or_ambiguous": int(claim is None)})
            definition = map_set["findings"].get(finding)
            layers = pred["map_layers"]
            if [item["layer"] for item in layers] != (definition["layers"] if definition else []):
                raise ValueError("Map layer sequence mismatch")
            if definition:
                trace = [bool(item["diagnostic_present"]) for item in layers]
                for item in layers:
                    if item["map_sha256"] != definition["map_sha256"]:
                        raise ValueError("Map version mismatch")
                map_traces.append({**common, "finding": finding, "expert_label": label,
                                   "answer_predicted_present": answer,
                                   "selected_layers": ",".join(map(str, definition["layers"])),
                                   "early_present": int(trace[0]), "late_present": int(trace[-1]),
                                   "persistent_positive_signal": int(label == 1 and all(trace)),
                                   "persistent_false_signal": int(label == 0 and all(trace)),
                                   "positive_signal_lost": int(label == 1 and trace[0] and not trace[-1]),
                                   "positive_signal_emerged": int(label == 1 and not trace[0] and trace[-1]),
                                   "false_signal_emerged": int(label == 0 and not trace[0] and trace[-1]),
                                   "early_logit": float(layers[0]["calibrated_logit"]),
                                   "late_logit": float(layers[-1]["calibrated_logit"])})
    for finding in QUESTIONS:
        answer_rows = [row for row in predictions if row["finding"] == finding
                       and row["expert_label"] is not None]
        report_rows = [row for row in claims if row["finding"] == finding
                       and row["expert_label"] is not None]
        answer_pairs = [(row["expert_label"], row["zero_margin_predicted_present"])
                        for row in answer_rows]
        score = counts(answer_pairs)
        missing = sum(row["missing_or_ambiguous"] for row in report_rows)
        explicit_fp = sum(row["explicit_false_positive"] for row in report_rows)
        explicit_fn = sum(row["explicit_false_negative"] for row in report_rows)
        report_accuracy = sum(row["report_correct"] for row in report_rows) / len(report_rows)
        per_finding.append({"finding": finding, "patients": len({r["patient_key"] for r in answer_rows}),
                            "labeled_scans": len(answer_rows), "positive_scans": sum(r["expert_label"] for r in answer_rows),
                            **{f"answer_{key}": value for key, value in score.items()},
                            "report_accuracy": report_accuracy,
                            "report_explicit_FP": explicit_fp, "report_explicit_FN": explicit_fn,
                            "report_missing_or_ambiguous": missing,
                            "report_positive_missing": sum(r["expert_label"] == 1 and r["missing_or_ambiguous"] for r in report_rows),
                            "report_negative_missing": sum(r["expert_label"] == 0 and r["missing_or_ambiguous"] for r in report_rows),
                            "report_FP_rate_per_negative": explicit_fp / sum(r["expert_label"] == 0 for r in report_rows),
                            "report_FN_rate_per_positive": explicit_fn / sum(r["expert_label"] == 1 for r in report_rows)})
        for name, metric_rows, metric in [
            ("answer_F1", answer_rows, lambda sample: counts([(r["expert_label"], r["zero_margin_predicted_present"]) for r in sample])["F1"]),
            ("answer_accuracy", answer_rows, lambda sample: counts([(r["expert_label"], r["zero_margin_predicted_present"]) for r in sample])["accuracy"]),
            ("report_accuracy", report_rows, lambda sample: sum(r["report_correct"] for r in sample) / len(sample) if sample else None),
        ]:
            low, high, valid = bootstrap(metric_rows, metric, args.seed + sum(map(ord, finding + name)), args.bootstrap_replicates)
            uncertainty.append({"finding": finding, "metric": name, "ci95_low": low,
                                "ci95_high": high, "valid_replicates": valid,
                                "resampling_unit": "patient"})
    persistence = []
    for finding in map_set["findings"]:
        subset = [row for row in map_traces if row["finding"] == finding and row["expert_label"] is not None]
        positive = [row for row in subset if row["expert_label"] == 1]
        negative = [row for row in subset if row["expert_label"] == 0]
        persistence.append({"finding": finding, "selected_layers": subset[0]["selected_layers"],
                            "positive_scans": len(positive), "negative_scans": len(negative),
                            "positive_persistent_count": sum(r["persistent_positive_signal"] for r in positive),
                            "positive_persistent_rate": sum(r["persistent_positive_signal"] for r in positive) / len(positive),
                            "positive_lost_count": sum(r["positive_signal_lost"] for r in positive),
                            "positive_emerged_count": sum(r["positive_signal_emerged"] for r in positive),
                            "negative_persistent_false_count": sum(r["persistent_false_signal"] for r in negative),
                            "negative_persistent_false_rate": sum(r["persistent_false_signal"] for r in negative) / len(negative),
                            "negative_false_emergence_count": sum(r["false_signal_emerged"] for r in negative),
                            "negative_late_signal_count": sum(r["late_present"] for r in negative),
                            "negative_late_signal_rate": sum(r["late_present"] for r in negative) / len(negative),
                            "limitation": "persistence measured only between the two selected decoder layers"})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tables = {"per_finding": per_finding, "patient_uncertainty": uncertainty,
              "map_persistence": persistence, "scan_answers": predictions,
              "scan_report_claims": claims, "scan_reports": reports,
              "scan_map_traces": map_traces}
    hashes = {}
    for name, data in tables.items():
        path = args.output_dir / f"{name}.csv"
        write_csv(path, data)
        hashes[name] = sha256(path)
    summary = {"status": "exploratory_full_validation_c_7finding_v2_complete",
               "split": "validation", "scans": len(rows), "patients": 13,
               "checkpoint_step": lock["step"], "checkpoint_sha256": lock["checkpoint_sha256"],
               "selection_lock_sha256": sha256(args.selection_lock), "map_set_sha256": map_sha,
               "mapped_findings": sorted(map_set["findings"]),
               "unmapped_findings": sorted(set(QUESTIONS) - set(map_set["findings"])),
               "map_selection_reused_validation_patients": True,
               "interpretation": "exploratory validation; new maps were selected using validation patients; no test accessed",
               "decoder_limit": "two selected layers per map show between-layer persistence, not full decoder stability",
               "training_stop": "hard cap 3000; frozen convergence criterion not met",
               "files_sha256": hashes, "segment_provenance": provenance,
               "bootstrap_replicates": args.bootstrap_replicates, "bootstrap_seed": args.seed}
    path = args.output_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": summary["status"], "output": str(args.output_dir)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--map-set", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--transfer-receipt", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=4106)
    run(parser.parse_args())
