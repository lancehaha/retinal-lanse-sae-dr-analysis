"""Report the user-limited first 800 validation scans for matched A/B/C models.

This is an exploratory validation subset, not the full 1,470-scan validation
cohort, a checkpoint-selection result, or a held-out test result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lock_retinavlm_olives_abc_validation import ARMS
from retinavlm_abc_artifact_paths import local_artifact
from summarize_retinavlm_abc_step3000_validation import rename_denominator, verified_rows
from summarize_retinavlm_olives_abc_test import (
    detail_rows, finding_rows, map_rows, map_separation_rows,
    paired_bootstrap, report_rows, write_csv,
)
from train_retinavlm_adapter_olives import QUESTIONS, SPLIT_SHA, load_rows, sha256


BOUNDARIES = ((0, 400), (400, 800))


def read_subset_arm(arm, entry, official, map_set, map_sha, args):
    checkpoint = local_artifact(entry["checkpoint"], args.remote_root, args.local_root)
    root = local_artifact(entry["evaluation_dir"], args.remote_root, args.local_root)
    checkpoint_sha = sha256(checkpoint)
    sources = []
    for start, stop in BOUNDARIES:
        stem = f"validation_{start:05d}_{stop:05d}"
        rows_file = root / f"{stem}.jsonl"
        receipt_file = root / f"{stem}.receipt.json"
        audit_file = args.stage_dir / (f"abc_validation_{arm}_step{args.step}_"
                                       f"{start:05d}_{stop:05d}_local_audit.json")
        if not all(path.is_file() for path in (rows_file, receipt_file, audit_file)):
            raise FileNotFoundError(f"Completed, audited segment missing: {arm} {stem}")
        receipt = json.loads(receipt_file.read_text())
        audit = json.loads(audit_file.read_text())
        if (receipt.get("status") != "complete_abc_inference_segment"
                or receipt.get("arm") != arm
                or receipt.get("split") != "validation"
                or receipt.get("checkpoint_step") != args.step
                or receipt.get("checkpoint_sha256") != checkpoint_sha
                or receipt.get("map_set_sha256") != map_sha
                or receipt.get("split_sha256") != SPLIT_SHA
                or receipt.get("start") != start or receipt.get("stop") != stop
                or receipt.get("rows") != stop - start
                or receipt.get("rows_sha256") != sha256(rows_file)
                or local_artifact(receipt["rows_file"], args.remote_root,
                                  args.local_root) != rows_file
                or audit.get("status") != "abc_validation_segment_local_audit_passed"
                or audit.get("arm") != arm
                or audit.get("start") != start or audit.get("stop") != stop
                or audit.get("rows_sha256") != receipt["rows_sha256"]
                or audit.get("segment_receipt_sha256") != sha256(receipt_file)):
            raise ValueError(f"Segment provenance mismatch: {arm} {stem}")
        sources.append({"path": str(receipt_file),
                        "sha256": sha256(receipt_file),
                        "rows_sha256": receipt["rows_sha256"],
                        "local_audit": str(audit_file),
                        "local_audit_sha256": sha256(audit_file)})
    candidate = {"segment_receipts": sources, "validation_scans": len(official),
                 "checkpoint_sha256": checkpoint_sha}
    rows = verified_rows(candidate, arm, map_set, args.remote_root, args.local_root)
    for index, row in enumerate(rows):
        source = official[index]
        if (row.get("index") != index
                or row.get("split") != "validation"
                or row.get("checkpoint_step") != args.step
                or row.get("checkpoint_sha256") != checkpoint_sha
                or row.get("map_set_sha256") != map_sha
                or row.get("source_image_path") != source["source_image_path"]
                or row.get("patient_key") != source["patient_key"]
                or set(row.get("finding_predictions", {})) != set(QUESTIONS)):
            raise ValueError(f"Scan identity/provenance mismatch: {arm} {index}")
        for finding in QUESTIONS:
            label = source[finding]
            expected = int(label) if label in {"0", "1"} else None
            if row["finding_predictions"][finding]["expert_label"] != expected:
                raise ValueError(f"Expert label mismatch: {arm} {index} {finding}")
    return candidate, rows


def run(args):
    if args.bootstrap_replicates < 1 or args.output_dir.exists():
        raise ValueError("Invalid bootstrap count or output directory already exists")
    all_validation = load_rows(args.split_csv)["validation"]
    if len(all_validation) != 1470 or len({r["patient_key"] for r in all_validation}) != 13:
        raise ValueError("Frozen full validation cohort changed")
    official = all_validation[:800]
    manifest = json.loads(args.candidates.read_text())
    map_set = json.loads(args.map_set.read_text())
    map_sha = sha256(args.map_set)
    if (manifest.get("status") != "prespecified_abc_validation_candidates"
            or set(manifest.get("arms", {})) != set(ARMS)
            or manifest.get("map_set_sha256") != map_sha
            or map_set.get("status") != "validated_retinavlm_olives_map_set"
            or map_set.get("split_sha256") != SPLIT_SHA):
        raise ValueError("Frozen candidate or map-set provenance changed")

    candidates, rows_by_arm = {}, {}
    for arm in ARMS:
        entries = [e for e in manifest["arms"][arm] if e["step"] == args.step]
        if len(entries) != 1:
            raise ValueError(f"Expected one step-{args.step} checkpoint for {arm}")
        candidates[arm], rows_by_arm[arm] = read_subset_arm(
            arm, entries[0], official, map_set, map_sha, args)

    zero = {finding: 0.0 for finding in QUESTIONS}
    comparison_lock = {"arms": {arm: {
        "step": args.step, "checkpoint_sha256": candidates[arm]["checkpoint_sha256"],
        "thresholds_margin": zero,
    } for arm in ARMS}}
    tables = {"finding_f1_zero_margin": [], "report_claims": [],
              "map_resilience_zero_margin": [], "map_layer_separation": [],
              "scan_predictions": [], "scan_report_claims": [],
              "scan_map_layers": [], "scan_reports": []}
    patient_count = len({row["patient_key"] for row in official})
    candidate_sha = sha256(args.candidates)
    for arm in ARMS:
        rows = rows_by_arm[arm]
        tables["finding_f1_zero_margin"].extend(rename_denominator(
            finding_rows(arm, rows, official, zero)))
        tables["report_claims"].extend(rename_denominator(
            report_rows(arm, rows, official)))
        tables["map_resilience_zero_margin"].extend(map_rows(
            arm, rows, comparison_lock, args.seed, args.bootstrap_replicates))
        tables["map_layer_separation"].extend(map_separation_rows(arm, rows))
        for name, detail in detail_rows({arm: rows}, comparison_lock).items():
            tables[name].extend(detail)
    for row in tables["scan_predictions"]:
        row["correct_at_zero_margin"] = row.pop("correct_at_locked_threshold")
    for rows in tables.values():
        for row in rows:
            arm = row["arm"]
            row.update({"split": "validation_first800", "checkpoint_step": args.step,
                        "checkpoint_sha256": candidates[arm]["checkpoint_sha256"],
                        "map_set_sha256": map_sha,
                        "candidate_manifest_sha256": candidate_sha,
                        "cohort_scans": 800, "cohort_patients": patient_count})

    args.output_dir.mkdir(parents=True)
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
        "status": f"exploratory_abc_step{args.step}_first800_validation_complete",
        "analysis_script_sha256": sha256(Path(__file__)),
        "split_sha256": SPLIT_SHA,
        "candidate_manifest_sha256": candidate_sha,
        "map_set_sha256": map_sha, "step": args.step,
        "validation_scans_scored_per_arm": 800,
        "total_validation_scans": 1470,
        "validation_patients_in_subset": patient_count,
        "selection": "first 800 frozen validation scan indices 0 through 799",
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
        "table_sha256": hashes,
        "test_accessed_by_this_analysis": False,
        "claim_boundary": "User-limited first-800 validation subset; no full validation or held-out test claim. Maps were selected using validation patients.",
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
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, default=3000)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=4106)
    parser.add_argument("--remote-root", type=Path, required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    run(parser.parse_args())
