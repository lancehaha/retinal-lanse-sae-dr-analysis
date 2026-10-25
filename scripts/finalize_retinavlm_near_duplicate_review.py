"""Close the OLIVES cross-split difference-hash review gate.

Candidate hashes are only a screen. Every reported pair must be reviewed as
distinct before the official image audit can be marked ready for model use.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from retinavlm_verified_images import ARCHIVE_MD5, EXPERT_IMAGES, sha256


def pair_key(left: str, right: str) -> tuple[str, str]:
    if not left or not right or left == right:
        raise ValueError("Invalid candidate pair")
    return tuple(sorted((left, right)))


def finalize(image_audit: Path, candidate_report: Path,
             review_csv: Path | None) -> dict:
    audit = json.loads(image_audit.read_text(encoding="utf-8"))
    report = json.loads(candidate_report.read_text(encoding="utf-8"))
    if (audit.get("status") != "official_images_matched_and_exact_duplicate_audited"
            or audit.get("source_archive_md5") != ARCHIVE_MD5
            or audit.get("matched_images") != EXPERT_IMAGES
            or audit.get("near_duplicate_gate") != "not_run"
            or report.get("status") != "candidate_screen_only_review_required"
            or report.get("source_manifest_sha256") != audit.get("manifest_sha256")
            or report.get("source_image_audit_sha256") != sha256(image_audit)
            or report.get("images") != EXPERT_IMAGES
            or report.get("max_difference_hash_distance") != 4
            or report.get("truncated") is not False):
        raise ValueError("Candidate report is incomplete or not bound to this official audit")
    candidates = report.get("candidates")
    if (not isinstance(candidates, list)
            or report.get("candidate_pair_count") != len(candidates)
            or report.get("reported_pair_count") != len(candidates)):
        raise ValueError("Every candidate pair must be present in the report")
    expected = {pair_key(item["left_source"], item["right_source"]) for item in candidates}
    if len(expected) != len(candidates):
        raise ValueError("Duplicate candidate pairs in report")

    decisions = {}
    if expected:
        if review_csv is None:
            raise ValueError("Review CSV required for flagged candidate pairs")
        with review_csv.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not {"left_source", "right_source", "decision"}.issubset(reader.fieldnames or []):
                raise ValueError("Review CSV lacks required columns")
            for row in reader:
                key = pair_key(row["left_source"], row["right_source"])
                if key in decisions or row["decision"] not in {"distinct", "duplicate", "uncertain"}:
                    raise ValueError("Repeated or invalid review decision")
                decisions[key] = row["decision"]
        if set(decisions) != expected:
            raise ValueError("Review decisions do not cover exactly the candidate pairs")
        if any(decision != "distinct" for decision in decisions.values()):
            raise ValueError("Cross-split duplicate or unresolved candidate remains")
    elif review_csv is not None:
        raise ValueError("No candidate pairs require a review CSV")

    audit["near_duplicate_gate"] = "reviewed_pass"
    audit["near_duplicate_screen"] = {
        "method": "64-bit grayscale difference hash",
        "max_distance": 4,
        "candidate_pairs_reviewed": len(expected),
        "candidate_report_sha256": sha256(candidate_report),
        "review_csv_sha256": sha256(review_csv) if review_csv is not None else None,
        "pre_review_image_audit_sha256": report["source_image_audit_sha256"],
    }
    temporary = image_audit.with_suffix(image_audit.suffix + ".writing")
    temporary.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, image_audit)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("official_image_audit", type=Path)
    parser.add_argument("candidate_report", type=Path)
    parser.add_argument("--review-csv", type=Path)
    args = parser.parse_args()
    print(json.dumps(finalize(args.official_image_audit, args.candidate_report,
                              args.review_csv), indent=2))
