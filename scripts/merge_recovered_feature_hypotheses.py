from __future__ import annotations

"""Merge and audit the sharded Experiment 5B unparsed-card recovery run.

This utility is intentionally label-agnostic: it only joins per-card records,
checks coverage/uniqueness, and writes the consolidated audit artifact to the
external experiment volume.
"""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sample_paths(record: dict) -> list[dict]:
    out = []
    for key in ("discovery_high", "generation_low"):
        out.extend(record.get(key) or [])
    for validation in record.get("validation", []):
        samples = validation.get("samples", {})
        out.extend(samples.get("high") or [])
        out.extend(samples.get("low") or [])
    return out


def group_patients(items: list[dict]) -> set[str]:
    return {str(x.get("patient_id")) for x in items if x.get("patient_id") is not None}


def audit_record(record: dict) -> list[str]:
    errors: list[str] = []
    fid = record.get("feature_id")
    if record.get("labels_hidden") is not True:
        errors.append("labels_hidden_not_true")
    if len(record.get("discovery_high") or []) != 32:
        errors.append("discovery_high_not_32")
    if len(record.get("generation_low") or []) != 32:
        errors.append("generation_low_not_32")
    groups = [("discovery", record.get("discovery_high") or []),
              ("generation_low", record.get("generation_low") or [])]
    for i, validation in enumerate(record.get("validation", [])):
        samples = validation.get("samples", {})
        groups.extend([(f"validation_{i}_high", samples.get("high") or []),
                       (f"validation_{i}_low", samples.get("low") or [])])
        for side in ("high", "low"):
            if len(samples.get(side) or []) != 32:
                errors.append(f"{validation.get('phase', i)}_{side}_not_32")
    seen_paths: set[str] = set()
    for item in sample_paths(record):
        path = str(item.get("relative_path", ""))
        if not path:
            errors.append("sample_missing_path")
        elif path in seen_paths:
            errors.append(f"duplicate_sample_path:{path}")
        seen_paths.add(path)
    # Sampling is patient-disjoint where feasible; flag violations for audit,
    # but do not silently discard a record when the finite cohort makes this
    # impossible.
    for i, (name_a, group_a) in enumerate(groups):
        for name_b, group_b in groups[i + 1:]:
            overlap = group_patients(group_a) & group_patients(group_b)
            if overlap:
                errors.append(f"patient_overlap:{name_a}:{name_b}:{len(overlap)}")
    return errors


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--original-jsonl", required=True)
    p.add_argument("--original-summary", required=True)
    p.add_argument("--prior-hypotheses", required=True)
    p.add_argument("--recovery-jsonl", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--combined-jsonl", default=None)
    args = p.parse_args()
    original = Path(args.original_jsonl)
    original_summary = Path(args.original_summary)
    prior = Path(args.prior_hypotheses)
    shards = [Path(x) for x in args.recovery_jsonl]
    out = Path(args.output)
    if not str(out).startswith("/Volumes/Elements/Retina-LanSE/"):
        raise ValueError("consolidated output must be on /Volumes/Elements/Retina-LanSE")
    old_records = load_jsonl(original)
    shard_records = [r for path in shards for r in load_jsonl(path)]
    prior_rows = load_jsonl(prior)
    expected_recovery = {int(r["feature_id"]) for r in prior_rows if r.get("status") == "unparsed"}
    actual_ids = [int(r["feature_id"]) for r in shard_records]
    duplicate_ids = sorted({x for x in actual_ids if actual_ids.count(x) > 1})
    actual_set = set(actual_ids)
    missing_ids = sorted(expected_recovery - actual_set)
    unexpected_ids = sorted(actual_set - expected_recovery)
    merged = old_records + shard_records
    merged_ids = [int(r["feature_id"]) for r in merged]
    merged_duplicate_ids = sorted({x for x in merged_ids if merged_ids.count(x) > 1})
    record_errors = {str(r.get("feature_id")): audit_record(r) for r in shard_records}
    record_errors = {k: v for k, v in record_errors.items() if v}
    final_counts = {
        "passed_initial": sum(r.get("final_status") == "passed_initial" for r in shard_records),
        "revised_and_passed": sum(r.get("final_status") == "revised_and_passed" for r in shard_records),
        "failed_or_unavailable": sum(r.get("final_status") not in {"passed_initial", "revised_and_passed"} for r in shard_records),
        "hypothesis_unavailable": sum(r.get("final_status") == "hypothesis_unavailable" for r in shard_records),
    }
    result = {
        "schema_version": "1.0",
        "status": "complete" if not missing_ids and not duplicate_ids and not unexpected_ids else "incomplete",
        "original_features": len(old_records),
        "recovery_features_expected": len(expected_recovery),
        "recovery_features_observed": len(shard_records),
        "total_features_expected": len(old_records) + len(expected_recovery),
        "total_features_observed": len(merged),
        "original_summary": json.loads(original_summary.read_text(encoding="utf-8")),
        "recovery_status_counts": final_counts,
        "coverage": {"missing_feature_ids": missing_ids, "unexpected_feature_ids": unexpected_ids,
                     "duplicate_feature_ids": duplicate_ids, "merged_duplicate_feature_ids": merged_duplicate_ids},
        "record_audit_error_count": len(record_errors),
        "record_audit_errors": record_errors,
        "input_hashes": {str(path): sha256(path) for path in [original, original_summary, prior, *shards]},
        "recovery_jsonl": [str(x) for x in shards],
        "original_jsonl": str(original),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.combined_jsonl:
        combined = Path(args.combined_jsonl)
        if not str(combined).startswith("/Volumes/Elements/Retina-LanSE/"):
            raise ValueError("combined output must be on /Volumes/Elements/Retina-LanSE")
        combined.parent.mkdir(parents=True, exist_ok=True)
        with combined.open("w", encoding="utf-8") as f:
            for record in merged:
                f.write(json.dumps(record, sort_keys=True) + "\n")
        result["combined_jsonl"] = str(combined)
        # Rewrite the audit after adding the convenience index path.
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
