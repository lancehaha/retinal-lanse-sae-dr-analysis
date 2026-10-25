from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_global_sae_search import require_external


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a provenance-preserving clean IDRiD testing manifest")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit", required=True)
    args = parser.parse_args()
    input_path = require_external(Path(args.input), "IDRiD raw manifest")
    output_path = require_external(Path(args.output), "IDRiD clean manifest")
    audit_path = require_external(Path(args.audit), "IDRiD duplicate audit")
    records = read_jsonl(input_path)
    if not records:
        raise RuntimeError("raw IDRiD manifest is empty")
    training_hashes = defaultdict(list)
    for record in records:
        if record.get("official_split") == "training":
            training_hashes[record["sha256"]].append(record["relative_path"])
    testing = [record for record in records if record.get("official_split") == "testing"]
    if not testing:
        raise RuntimeError("raw manifest has no official testing records")
    kept: list[dict] = []
    excluded: list[dict] = []
    for record in testing:
        duplicate_training = training_hashes.get(record["sha256"], [])
        if duplicate_training:
            excluded.append({
                "relative_path": record["relative_path"],
                "sha256": record["sha256"],
                "grade": record.get("grade"),
                "referable_dr": record.get("referable_dr"),
                "reason": "exact_sha256_duplicate_of_official_training_record",
                "training_matches": duplicate_training,
            })
        else:
            kept.append(record)
    if not kept:
        raise RuntimeError("duplicate filtering removed every IDRiD testing record")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_suffix(output_path.suffix + ".partial")
    with partial.open("w", encoding="utf-8") as handle:
        for record in kept:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    partial.replace(output_path)
    audit = {
        "schema_version": "1.0",
        "status": "complete_idrid_clean_testing_manifest",
        "input_manifest": str(input_path),
        "raw_testing_images": len(testing),
        "kept_testing_images": len(kept),
        "excluded_testing_images": len(excluded),
        "excluded": excluded,
        "policy": "exclude_exact_testing_image_duplicates_of_official_training_images; preserve_raw_manifest_and_archives",
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
