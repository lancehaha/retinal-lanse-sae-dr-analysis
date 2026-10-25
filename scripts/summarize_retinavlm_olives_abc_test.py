"""Audit the one locked A/B/C test pass and export report-ready tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

from lock_retinavlm_olives_abc_validation import ARMS, f1_counts
from retinavlm_abc_artifact_paths import local_artifact
from retinavlm_olives_report import parse_report_claims, report_claim_counts
from train_retinavlm_adapter_olives import QUESTIONS, SPLIT_SHA, load_rows, sha256


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty result table")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_arm(root, arm, official, lock, lock_sha, map_set=None,
             remote_root=None, local_root=None):
    root = local_artifact(root, remote_root, local_root)
    receipts = sorted(root.glob("test_*.receipt.json"))
    if not receipts:
        raise FileNotFoundError(f"No locked test segments for {arm}")
    selected = lock["arms"][arm]
    by_index = {}
    provenance = []
    for receipt_path in receipts:
        receipt = json.loads(receipt_path.read_text())
        rows_file = local_artifact(receipt["rows_file"], remote_root, local_root)
        expected_rows_file = root / (f"test_{receipt['start']:05d}_"
                                     f"{receipt['stop']:05d}.jsonl")
        if (receipt.get("status") != "complete_abc_inference_segment"
                or rows_file != expected_rows_file
                or receipt.get("arm") != arm or receipt.get("split") != "test"
                or receipt.get("checkpoint_sha256") != selected["checkpoint_sha256"]
                or receipt.get("checkpoint_step") != selected["step"]
                or receipt.get("selection_lock_sha256") != lock_sha
                or receipt.get("split_sha256") != SPLIT_SHA
                or receipt.get("map_set_sha256") != lock["map_set_sha256"]
                or receipt.get("rows_sha256") != sha256(rows_file)
                or receipt.get("rows") != receipt["stop"] - receipt["start"]):
            raise ValueError("Test segment provenance mismatch")
        with rows_file.open(encoding="utf-8") as file:
            segment = [json.loads(line) for line in file]
        if len(segment) != receipt["rows"]:
            raise ValueError("Test segment row count mismatch")
        for offset, row in enumerate(segment):
            index = receipt["start"] + offset
            if (row.get("index") != index or index in by_index or index >= len(official)
                    or row.get("arm") != arm or row.get("split") != "test"
                    or row.get("source_image_path") != official[index]["source_image_path"]
                    or row.get("patient_key") != official[index]["patient_key"]
                    or row.get("checkpoint_sha256") != selected["checkpoint_sha256"]
                    or row.get("map_set_sha256") != lock["map_set_sha256"]
                    or set(row.get("finding_predictions", {})) != set(QUESTIONS)
                    or row.get("parsed_report_claims") != parse_report_claims(
                        row.get("generated_report", ""))):
                raise ValueError("Test row identity, model, or report parser mismatch")
            for finding in QUESTIONS:
                expected = official[index][finding]
                label = int(expected) if expected in {"0", "1"} else None
                prediction = row["finding_predictions"][finding]
                if prediction["expert_label"] != label:
                    raise ValueError("Test finding label mismatch")
                if map_set is not None:
                    definition = map_set["findings"].get(finding)
                    layers = prediction["map_layers"]
                    if definition is None and layers:
                        raise ValueError("Unvalidated finding has map-layer scores")
                    if definition is not None:
                        if [item["layer"] for item in layers] != definition["layers"]:
                            raise ValueError("Map-layer sequence changed on test")
                        for item in layers:
                            if (item["map_sha256"] != definition["map_sha256"]
                                    or not math.isfinite(float(item["raw_score"]))
                                    or not math.isfinite(float(item["calibrated_logit"]))
                                    or item["diagnostic_present"] != (item["calibrated_logit"] >= 0)):
                                raise ValueError("Test map score or version invalid")
            by_index[index] = row
        provenance.append({"receipt": str(receipt_path),
                           "receipt_sha256": sha256(receipt_path),
                           "rows_sha256": receipt["rows_sha256"]})
    if set(by_index) != set(range(len(official))):
        raise ValueError(f"{arm} did not score every test scan exactly once")
    return [by_index[index] for index in range(len(official))], provenance


def finding_rows(arm, rows, official, thresholds):
    output = []
    for finding in QUESTIONS:
        selected = [(row["finding_predictions"][finding], row["patient_key"])
                    for row in rows if row["finding_predictions"][finding]["expert_label"] is not None]
        labels = [item[0]["expert_label"] for item in selected]
        margins = [float(item[0]["answer_margin"]) for item in selected]
        threshold = float(thresholds[finding])
        score = f1_counts(labels, [margin >= threshold for margin in margins])
        zero = f1_counts(labels, [margin >= 0 for margin in margins])
        always_present = f1_counts(labels, [True] * len(labels))
        output.append({"arm": arm, "finding": finding,
                       "test_patients": len({patient for _, patient in selected}),
                       "labeled_scans": len(labels), "positive_scans": sum(labels),
                       "positive_prevalence": sum(labels) / len(labels) if labels else None,
                       "always_present_F1": always_present["F1"],
                       "threshold_margin": threshold, **score,
                       "zero_margin_F1": zero["F1"],
                       "zero_margin_recall": zero["recall"]})
    return output


def report_rows(arm, rows, official):
    tally = {finding: defaultdict(int) for finding in QUESTIONS}
    patients = {finding: set() for finding in QUESTIONS}
    for row, source in zip(rows, official, strict=True):
        counted = report_claim_counts(source, row["parsed_report_claims"])
        for finding, block in counted.items():
            patients[finding].add(row["patient_key"])
            for key, value in block.items():
                tally[finding][key] += value
    output = []
    for finding in QUESTIONS:
        counts = tally[finding]
        positive = counts["positive_label"]
        negative = counts["negative_label"]
        output.append({"arm": arm, "finding": finding,
                       "test_patients": len(patients[finding]),
                       "positive_scans": positive, "negative_scans": negative,
                       "supported_present": counts["supported_present"],
                       "unsupported_present": counts["unsupported_present"],
                       "miss": counts["miss"],
                       "ambiguous_or_missing": counts["ambiguous_or_missing"],
                       "unsupported_rate_per_negative": counts["unsupported_present"] / negative
                           if negative else None,
                       "miss_rate_per_positive": counts["miss"] / positive if positive else None})
    return output


def detail_rows(rows_by_arm, lock):
    """Keep each scored scan, report claim, and validated map layer in the report data."""
    predictions, claims, map_layers, reports = [], [], [], []
    for arm, rows in rows_by_arm.items():
        selected = lock["arms"][arm]
        for row in rows:
            common = {"arm": arm, "checkpoint_step": selected["step"],
                      "checkpoint_sha256": selected["checkpoint_sha256"],
                      "scan_index": row["index"], "patient_key": row["patient_key"],
                      "source_image_path": row["source_image_path"]}
            reports.append({**common, "generated_report": row["generated_report"],
                            "parsed_report_claims_json": json.dumps(
                                row["parsed_report_claims"], sort_keys=True)})
            for finding in QUESTIONS:
                finding_row = row["finding_predictions"][finding]
                label = finding_row["expert_label"]
                margin = float(finding_row["answer_margin"])
                threshold = float(selected["thresholds_margin"][finding])
                prediction = margin >= threshold
                predictions.append({**common, "finding": finding,
                                    "expert_label": label, "answer_margin": margin,
                                    "threshold_margin": threshold,
                                    "thresholded_answer": "present" if prediction else "absent",
                                    "zero_margin_answer": finding_row["zero_margin_answer"],
                                    "correct_at_locked_threshold": int(prediction == label)
                                    if label is not None else None})
                claim = row["parsed_report_claims"][finding]
                claims.append({**common, "finding": finding, "expert_label": label,
                               "parsed_claim": claim,
                               "unsupported_present": int(label == 0 and claim == 1),
                               "miss": int(label == 1 and claim != 1),
                               "ambiguous_or_missing": int(claim is None)})
                for score in finding_row["map_layers"]:
                    map_layers.append({**common, "finding": finding,
                                       "expert_label": label,
                                       "final_answer": "present" if prediction else "absent",
                                       "layer": score["layer"],
                                       "raw_score": score["raw_score"],
                                       "calibrated_logit": score["calibrated_logit"],
                                       "diagnostic_present": int(score["diagnostic_present"]),
                                       "map_sha256": score["map_sha256"]})
    return {"scan_predictions": predictions, "scan_report_claims": claims,
            "scan_map_layers": map_layers, "scan_reports": reports}


def rank_auc(labels, scores):
    """Pooled AUROC with average ranks for tied map scores."""
    positives = sum(label == 1 for label in labels)
    negatives = sum(label == 0 for label in labels)
    if not positives or not negatives or len(labels) != len(scores):
        return None
    ordered = sorted(zip(scores, labels, strict=True))
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2
        rank_sum += average_rank * sum(label == 1 for _, label in ordered[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def map_separation_rows(arm, rows):
    output = []
    for finding in QUESTIONS:
        relevant = [row for row in rows
                    if row["finding_predictions"][finding]["map_layers"]
                    and row["finding_predictions"][finding]["expert_label"] is not None]
        if not relevant:
            continue
        layers = [entry["layer"] for entry in
                  relevant[0]["finding_predictions"][finding]["map_layers"]]
        for position, layer in enumerate(layers):
            labels = [row["finding_predictions"][finding]["expert_label"] for row in relevant]
            scores = [float(row["finding_predictions"][finding]["map_layers"]
                            [position]["calibrated_logit"]) for row in relevant]
            if any(not math.isfinite(score) for score in scores):
                raise ValueError("Nonfinite map score in test table")
            positive_scores = [score for score, label in zip(scores, labels) if label == 1]
            negative_scores = [score for score, label in zip(scores, labels) if label == 0]
            output.append({"arm": arm, "finding": finding, "layer": layer,
                           "positive_scans": len(positive_scores),
                           "negative_scans": len(negative_scores),
                           "pooled_AUROC": rank_auc(labels, scores),
                           "positive_median_logit": statistics.median(positive_scores)
                               if positive_scores else None,
                           "negative_median_logit": statistics.median(negative_scores)
                               if negative_scores else None,
                           "positive_above_zero_fraction":
                               sum(score >= 0 for score in positive_scores) / len(positive_scores)
                               if positive_scores else None,
                           "negative_above_zero_fraction":
                               sum(score >= 0 for score in negative_scores) / len(negative_scores)
                               if negative_scores else None,
                           "claim_boundary": "pooled internal diagnostic, not final-answer F1"})
    return output


def paired_bootstrap(rows_by_arm, official, lock, seed, replicates):
    by_patient = defaultdict(list)
    for index, source in enumerate(official):
        by_patient[source["patient_key"]].append(index)
    patients = sorted(by_patient)
    rng = random.Random(seed)
    differences = {"C_minus_A_macro_F1": [], "C_minus_B_macro_F1": [],
                   "C_minus_A_macro_recall": [], "C_minus_B_macro_recall": []}
    for _ in range(replicates):
        sampled = [rng.choice(patients) for _ in patients]
        indices = [index for patient in sampled for index in by_patient[patient]]
        per_arm = {}
        for arm in ARMS:
            f1s, recalls = [], []
            for finding in QUESTIONS:
                pairs = [(rows_by_arm[arm][i]["finding_predictions"][finding], official[i][finding])
                         for i in indices if official[i][finding] in {"0", "1"}]
                labels = [int(label) for _, label in pairs]
                threshold = float(lock["arms"][arm]["thresholds_margin"][finding])
                predicted = [float(pred["answer_margin"]) >= threshold for pred, _ in pairs]
                metrics = f1_counts(labels, predicted)
                f1s.append(metrics["F1"])
                recalls.append(metrics["recall"])
            per_arm[arm] = (sum(f1s) / len(f1s), sum(recalls) / len(recalls))
        for comparator in ("A", "B"):
            differences[f"C_minus_{comparator}_macro_F1"].append(
                per_arm["C"][0] - per_arm[comparator][0])
            differences[f"C_minus_{comparator}_macro_recall"].append(
                per_arm["C"][1] - per_arm[comparator][1])
    return {name: [sorted(values)[int((len(values) - 1) * 0.025)],
                   sorted(values)[int((len(values) - 1) * 0.975)]]
            for name, values in differences.items()}


def map_rows(arm, rows, lock, seed=4106, bootstrap_replicates=2000):
    output = []
    for finding in QUESTIONS:
        relevant = [row for row in rows
                    if row["finding_predictions"][finding]["map_layers"]
                    and row["finding_predictions"][finding]["expert_label"] is not None]
        if not relevant:
            continue
        layers = [item["layer"] for item in relevant[0]["finding_predictions"][finding]["map_layers"]]
        if any([item["layer"] for item in row["finding_predictions"][finding]["map_layers"]] != layers
               for row in relevant):
            raise ValueError("Map layers differ across test scans")
        positives = [row for row in relevant if row["finding_predictions"][finding]["expert_label"] == 1]
        negatives = [row for row in relevant if row["finding_predictions"][finding]["expert_label"] == 0]
        def trajectory(row):
            return [bool(item["diagnostic_present"])
                    for item in row["finding_predictions"][finding]["map_layers"]]
        def first_readable_pair_end(row):
            trace = trajectory(row)
            return next((j + 1 for j in range(len(layers) - 1)
                         if trace[j] and trace[j + 1]), None)
        readable = [row for row in positives if first_readable_pair_end(row) is not None]
        late_retention_eligible = [row for row in readable
                                   if first_readable_pair_end(row) < len(layers) - 1]
        false_emergence = sum(any(trajectory(row)[j] and trajectory(row)[j + 1]
                                  for j in range(len(layers) - 1)) for row in negatives)
        threshold = float(lock["arms"][arm]["thresholds_margin"][finding])
        tp_cases = [row for row in positives if
                    float(row["finding_predictions"][finding]["answer_margin"]) >= threshold]
        fn_cases = [row for row in positives if
                    float(row["finding_predictions"][finding]["answer_margin"]) < threshold]
        fp_cases = [row for row in negatives if
                    float(row["finding_predictions"][finding]["answer_margin"]) >= threshold]
        tn_cases = [row for row in negatives if
                    float(row["finding_predictions"][finding]["answer_margin"]) < threshold]
        if len(layers) >= 3:
            retained_late = sum(trajectory(row)[-1] for row in late_retention_eligible)
            temporary_loss_recovery = sum(
                any(not trajectory(row)[j] for j in range(first_readable_pair_end(row) + 1,
                                                           len(layers) - 1))
                and trajectory(row)[-1] for row in late_retention_eligible)
            lost_middle_late = sum(trajectory(row)[-2] and not trajectory(row)[-1]
                                   for row in positives)
            false_middle_late = sum(not trajectory(row)[-2] and trajectory(row)[-1]
                                    for row in negatives)
            error_transitions = {
                "FN_middle_to_late_loss": sum(trajectory(row)[-2] and not trajectory(row)[-1]
                                              for row in fn_cases),
                "TP_middle_to_late_loss": sum(trajectory(row)[-2] and not trajectory(row)[-1]
                                              for row in tp_cases),
                "FP_middle_to_late_false_signal": sum(not trajectory(row)[-2] and trajectory(row)[-1]
                                                      for row in fp_cases),
                "TN_middle_to_late_false_signal": sum(not trajectory(row)[-2] and trajectory(row)[-1]
                                                      for row in tn_cases),
            }
        else:
            retained_late = lost_middle_late = false_middle_late = None
            temporary_loss_recovery = None
            error_transitions = {key: None for key in
                                 ("FN_middle_to_late_loss", "TP_middle_to_late_loss",
                                  "FP_middle_to_late_false_signal", "TN_middle_to_late_false_signal")}
        intervals = {}
        if len(layers) >= 3:
            by_patient = defaultdict(list)
            for row in relevant:
                label = row["finding_predictions"][finding]["expert_label"]
                predicted = float(row["finding_predictions"][finding]["answer_margin"]) >= threshold
                trace = trajectory(row)
                category = ("TP" if predicted else "FN") if label else ("FP" if predicted else "TN")
                transition = (trace[-2] and not trace[-1]) if label else (not trace[-2] and trace[-1])
                by_patient[row["patient_key"]].append((category, transition))
            patient_keys = sorted(by_patient)
            rng = random.Random(seed + sum(map(ord, arm + finding)))
            sampled_rates = {key: [] for key in ("FN", "TP", "FP", "TN")}
            sampled_differences = {key: [] for key in
                                   ("FN_minus_TP_loss_rate",
                                    "FP_minus_TN_false_signal_rate")}
            for _ in range(bootstrap_replicates):
                counts = {key: [0, 0] for key in sampled_rates}
                for _ in patient_keys:
                    for category, transition in by_patient[rng.choice(patient_keys)]:
                        counts[category][0] += int(transition)
                        counts[category][1] += 1
                for category, (numerator, denominator) in counts.items():
                    if denominator:
                        sampled_rates[category].append(numerator / denominator)
                if counts["FN"][1] and counts["TP"][1]:
                    sampled_differences["FN_minus_TP_loss_rate"].append(
                        counts["FN"][0] / counts["FN"][1]
                        - counts["TP"][0] / counts["TP"][1])
                if counts["FP"][1] and counts["TN"][1]:
                    sampled_differences["FP_minus_TN_false_signal_rate"].append(
                        counts["FP"][0] / counts["FP"][1]
                        - counts["TN"][0] / counts["TN"][1])
            for category, samples in sampled_rates.items():
                samples.sort()
                intervals[f"{category}_transition_ci95_low"] = (
                    samples[int((len(samples) - 1) * 0.025)] if samples else None)
                intervals[f"{category}_transition_ci95_high"] = (
                    samples[int((len(samples) - 1) * 0.975)] if samples else None)
            for name, samples in sampled_differences.items():
                samples.sort()
                intervals[f"{name}_ci95_low"] = (
                    samples[int((len(samples) - 1) * 0.025)] if samples else None)
                intervals[f"{name}_ci95_high"] = (
                    samples[int((len(samples) - 1) * 0.975)] if samples else None)
        else:
            for category in ("FN", "TP", "FP", "TN"):
                intervals[f"{category}_transition_ci95_low"] = None
                intervals[f"{category}_transition_ci95_high"] = None
            for name in ("FN_minus_TP_loss_rate", "FP_minus_TN_false_signal_rate"):
                intervals[f"{name}_ci95_low"] = None
                intervals[f"{name}_ci95_high"] = None
        output.append({"arm": arm, "finding": finding,
                       "selected_layers": ",".join(map(str, layers)),
                       "positive_scans": len(positives), "negative_scans": len(negatives),
                       "emerged_positive_scans": len(readable),
                       "emergence_coverage": len(readable) / len(positives) if positives else None,
                       "late_retention_eligible_scans": len(late_retention_eligible),
                       "retained_late_scans": retained_late,
                       "conditional_late_retention": retained_late / len(late_retention_eligible)
                           if retained_late is not None and late_retention_eligible else None,
                       "temporary_loss_then_recovery_scans": temporary_loss_recovery,
                       "false_emergence_scans": false_emergence,
                       "false_emergence_rate": false_emergence / len(negatives) if negatives else None,
                       "negative_late_false_signal_scans": sum(trajectory(row)[-1]
                                                               for row in negatives),
                       "positive_middle_to_late_loss": lost_middle_late,
                       "negative_middle_to_late_false_signal": false_middle_late,
                       "FN_scans": len(fn_cases), "TP_scans": len(tp_cases),
                       "FP_scans": len(fp_cases), "TN_scans": len(tn_cases),
                       **error_transitions,
                       "FN_loss_rate": error_transitions["FN_middle_to_late_loss"] / len(fn_cases)
                           if len(layers) >= 3 and fn_cases else None,
                       "TP_loss_rate": error_transitions["TP_middle_to_late_loss"] / len(tp_cases)
                           if len(layers) >= 3 and tp_cases else None,
                       "FP_false_signal_rate": error_transitions["FP_middle_to_late_false_signal"] / len(fp_cases)
                           if len(layers) >= 3 and fp_cases else None,
                       "TN_false_signal_rate": error_transitions["TN_middle_to_late_false_signal"] / len(tn_cases)
                           if len(layers) >= 3 and tn_cases else None,
                       "FN_minus_TP_loss_rate": (
                           error_transitions["FN_middle_to_late_loss"] / len(fn_cases)
                           - error_transitions["TP_middle_to_late_loss"] / len(tp_cases))
                           if len(layers) >= 3 and fn_cases and tp_cases else None,
                       "FP_minus_TN_false_signal_rate": (
                           error_transitions["FP_middle_to_late_false_signal"] / len(fp_cases)
                           - error_transitions["TN_middle_to_late_false_signal"] / len(tn_cases))
                           if len(layers) >= 3 and fp_cases and tn_cases else None,
                       **intervals,
                       "limitation": "two layers cannot estimate post-emergence retention"
                           if len(layers) < 3 else
                           "late retention excludes scans first readable at the final pair; fewer than four layers cannot capture a later loss and recovery" if len(layers) < 4 else
                           "late retention excludes scans first readable at the final pair; internal diagnostic, not lesion localization"})
    return output


def run(args):
    if args.bootstrap_replicates < 1:
        raise ValueError("Patient bootstrap must have at least one replicate")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Test report directory must be empty")
    lock = json.loads(args.selection_lock.read_text())
    map_set = json.loads(args.map_set.read_text())
    if (lock.get("status") != "locked_abc_validation_selection"
            or lock.get("split_sha256") != SPLIT_SHA
            or lock.get("test_accessed") is not False
            or set(lock.get("arms", {})) != set(ARMS)
            or map_set.get("status") != "validated_retinavlm_olives_map_set"
            or sha256(args.map_set) != lock.get("map_set_sha256")):
        raise ValueError("No intact validation-only A/B/C selection lock")
    manifest = json.loads(args.test_manifest.read_text())
    if (manifest.get("status") != "locked_abc_test_segments"
            or manifest.get("selection_lock_sha256") != sha256(args.selection_lock)
            or set(manifest.get("arms", {})) != set(ARMS)):
        raise ValueError("Test segment manifest does not match lock")
    official = load_rows(args.split_csv)["test"]
    rows_by_arm = {}
    receipts = {}
    for arm in ARMS:
        rows_by_arm[arm], receipts[arm] = read_arm(
            Path(manifest["arms"][arm]), arm, official, lock,
            sha256(args.selection_lock), map_set,
            getattr(args, "remote_root", None), getattr(args, "local_root", None))
    tables = {"finding_f1": [], "report_claims": [], "map_resilience": [],
              "map_layer_separation": []}
    for arm in ARMS:
        tables["finding_f1"].extend(finding_rows(
            arm, rows_by_arm[arm], official, lock["arms"][arm]["thresholds_margin"]))
        tables["report_claims"].extend(report_rows(arm, rows_by_arm[arm], official))
        tables["map_resilience"].extend(map_rows(
            arm, rows_by_arm[arm], lock, args.seed, args.bootstrap_replicates))
        tables["map_layer_separation"].extend(map_separation_rows(arm, rows_by_arm[arm]))
    tables.update(detail_rows(rows_by_arm, lock))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, rows in tables.items():
        path = args.output_dir / (name + ".csv")
        write_csv(path, rows)
        hashes[name] = sha256(path)
    summary = {
        "status": "locked_reused_test_abc_report_complete",
        "split_sha256": SPLIT_SHA,
        "selection_lock_sha256": sha256(args.selection_lock),
        "test_manifest_sha256": sha256(args.test_manifest),
        "test_patients": len({row["patient_key"] for row in official}),
        "test_scans": len(official),
        "seven_finding_macro_F1": {arm: sum(row["F1"] for row in tables["finding_f1"]
                                             if row["arm"] == arm) / len(QUESTIONS)
                                   for arm in ARMS},
        "paired_patient_bootstrap_ci95": paired_bootstrap(
            rows_by_arm, official, lock, args.seed, args.bootstrap_replicates),
        "source_segment_receipts": receipts,
        "table_sha256": hashes,
        "claim_boundary": "patient-disjoint reused OLIVES test; report parser is literal and extraction errors require manual audit",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": summary["status"], "output": str(args.output_dir)}), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--map-set", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=4106)
    parser.add_argument("--remote-root", type=Path)
    parser.add_argument("--local-root", type=Path)
    run(parser.parse_args())
