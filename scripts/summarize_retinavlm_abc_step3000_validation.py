"""Compare fixed-step A/B/C validation before deciding whether to train to 6000.

This is an interim validation report, not a checkpoint selection or test result.
The primary answer metric uses the prespecified zero answer margin; thresholds
optimized on these same validation scans are labeled apparent and secondary.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from lock_retinavlm_olives_abc_validation import ARMS, read_candidate
from retinavlm_abc_artifact_paths import local_artifact
from retinavlm_olives_report import parse_report_claims
from summarize_retinavlm_olives_abc_test import (
    detail_rows,
    finding_rows,
    map_rows,
    map_separation_rows,
    paired_bootstrap,
    report_rows,
    write_csv,
)
from train_retinavlm_adapter_olives import QUESTIONS, SPLIT_SHA, load_rows, sha256


def verified_rows(candidate, arm, map_set, remote_root, local_root):
    by_index = {}
    for source in candidate["segment_receipts"]:
        receipt = json.loads(Path(source["path"]).read_text())
        path = local_artifact(receipt["rows_file"], remote_root, local_root)
        with path.open(encoding="utf-8") as file:
            for line in file:
                row = json.loads(line)
                index = row["index"]
                if index in by_index or row["arm"] != arm:
                    raise ValueError("Duplicate or wrong-arm validation row")
                if row.get("parsed_report_claims") != parse_report_claims(
                        row.get("generated_report", "")):
                    raise ValueError("Generated report parser mismatch")
                for finding, prediction in row["finding_predictions"].items():
                    definition = map_set["findings"].get(finding)
                    layers = prediction["map_layers"]
                    if definition is None and layers:
                        raise ValueError("Unvalidated finding has map scores")
                    if definition is not None:
                        if [item["layer"] for item in layers] != definition["layers"]:
                            raise ValueError("Validation map-layer sequence changed")
                        for item in layers:
                            if (item["map_sha256"] != definition["map_sha256"]
                                    or not math.isfinite(float(item["raw_score"]))
                                    or not math.isfinite(float(item["calibrated_logit"]))
                                    or item["diagnostic_present"]
                                    != (item["calibrated_logit"] >= 0)):
                                raise ValueError("Validation map score or version invalid")
                by_index[index] = row
    if set(by_index) != set(range(candidate["validation_scans"])):
        raise ValueError("Missing validation rows")
    return [by_index[index] for index in range(candidate["validation_scans"])]


def rename_denominator(rows):
    for row in rows:
        if "test_patients" in row:
            row["validation_patients"] = row.pop("test_patients")
    return rows


def run(args):
    if args.bootstrap_replicates < 1 or (
            args.output_dir.exists() and any(args.output_dir.iterdir())):
        raise ValueError("Invalid bootstrap count or nonempty output directory")
    official = load_rows(args.split_csv)["validation"]
    if len(official) != 1470 or len({row["patient_key"] for row in official}) != 13:
        raise ValueError("Frozen validation cohort changed")
    manifest = json.loads(args.candidates.read_text())
    map_set = json.loads(args.map_set.read_text())
    map_sha = sha256(args.map_set)
    if (manifest.get("status") != "prespecified_abc_validation_candidates"
            or set(manifest.get("arms", {})) != set(ARMS)
            or manifest.get("map_set_sha256") != map_sha
            or map_set.get("status") != "validated_retinavlm_olives_map_set"):
        raise ValueError("Frozen candidate or map-set provenance changed")

    candidates, rows_by_arm = {}, {}
    for arm in ARMS:
        entries = [entry for entry in manifest["arms"][arm] if entry["step"] == 3000]
        if len(entries) != 1:
            raise ValueError(f"Expected one prespecified step-3000 candidate for {arm}")
        candidates[arm] = read_candidate(entries[0], arm, official, map_sha,
                                         args.remote_root, args.local_root)
        rows_by_arm[arm] = verified_rows(candidates[arm], arm, map_set,
                                         args.remote_root, args.local_root)

    zero = {finding: 0.0 for finding in QUESTIONS}
    comparison_lock = {"arms": {arm: {
        "step": 3000, "checkpoint_sha256": candidates[arm]["checkpoint_sha256"],
        "thresholds_margin": zero,
    } for arm in ARMS}}
    tables = {"finding_f1_zero_margin": [], "report_claims": [],
              "map_resilience_zero_margin": [], "map_layer_separation": [],
              "apparent_same_validation_threshold_f1": [],
              "scan_predictions": [], "scan_report_claims": [],
              "scan_map_layers": [], "scan_reports": []}
    for arm in ARMS:
        rows = rows_by_arm[arm]
        tables["finding_f1_zero_margin"].extend(rename_denominator(
            finding_rows(arm, rows, official, zero)))
        tables["report_claims"].extend(rename_denominator(
            report_rows(arm, rows, official)))
        tables["map_resilience_zero_margin"].extend(map_rows(
            arm, rows, comparison_lock, args.seed, args.bootstrap_replicates))
        tables["map_layer_separation"].extend(map_separation_rows(arm, rows))
        details = detail_rows({arm: rows}, comparison_lock)
        for name, detail in details.items():
            tables[name].extend(detail)
        for finding, block in candidates[arm]["finding_metrics"].items():
            tables["apparent_same_validation_threshold_f1"].append({
                "arm": arm, "step": 3000, "finding": finding,
                "labeled_scans": block["labeled_scans"],
                "threshold_margin": block["threshold_margin"],
                "apparent_validation_F1": block["validation"]["F1"],
                "apparent_validation_recall": block["validation"]["recall"],
                "zero_margin_F1": block["zero_margin"]["F1"],
                "limitation": "threshold fitted and scored on the same validation cohort",
            })

    candidate_sha = sha256(args.candidates)
    for row in tables["scan_predictions"]:
        row["correct_at_zero_margin"] = row.pop("correct_at_locked_threshold")
    for rows in tables.values():
        for row in rows:
            arm = row["arm"]
            row.update({"split": "validation", "checkpoint_step": 3000,
                        "checkpoint_sha256": candidates[arm]["checkpoint_sha256"],
                        "map_set_sha256": map_sha,
                        "candidate_manifest_sha256": candidate_sha,
                        "cohort_scans": len(official), "cohort_patients": 13})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, rows in tables.items():
        path = args.output_dir / f"{name}.csv"
        write_csv(path, rows)
        hashes[name] = sha256(path)
    macro_f1 = {arm: sum(row["F1"] for row in tables["finding_f1_zero_margin"]
                             if row["arm"] == arm) / len(QUESTIONS) for arm in ARMS}
    macro_recall = {arm: sum(row["recall"] for row in tables["finding_f1_zero_margin"]
                                 if row["arm"] == arm) / len(QUESTIONS) for arm in ARMS}
    bootstrap = paired_bootstrap(rows_by_arm, official, comparison_lock,
                                 args.seed, args.bootstrap_replicates)
    summary = {
        "status": "interim_abc_step3000_validation_complete",
        "analysis_script_sha256": sha256(Path(__file__)),
        "split_sha256": SPLIT_SHA, "candidate_manifest_sha256": candidate_sha,
        "map_set_sha256": map_sha, "step": 3000,
        "validation_scans": len(official), "validation_patients": 13,
        "primary_threshold": "prespecified zero answer margin",
        "seven_finding_macro_F1_zero_margin": macro_f1,
        "seven_finding_macro_recall_zero_margin": macro_recall,
        "C_minus_A_macro_F1_zero_margin": macro_f1["C"] - macro_f1["A"],
        "C_minus_B_macro_F1_zero_margin": macro_f1["C"] - macro_f1["B"],
        "paired_patient_bootstrap_ci95_zero_margin": bootstrap,
        "source_segment_receipts": {arm: candidates[arm]["segment_receipts"]
                                    for arm in ARMS},
        "checkpoint_sha256": {arm: candidates[arm]["checkpoint_sha256"]
                              for arm in ARMS},
        "table_sha256": hashes, "test_accessed": False,
        "scan_level_tables": ["scan_predictions", "scan_report_claims",
                              "scan_map_layers", "scan_reports"],
        "six_thousand_step_training_resumed": False,
        "claim_boundary": "interim validation comparison; same-cohort optimized thresholds are descriptive only; no test result",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": summary["status"],
                      "output": str(args.output_dir)}), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--map-set", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=4106)
    parser.add_argument("--remote-root", type=Path, required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    run(parser.parse_args())
