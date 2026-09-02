from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit EyePACS manifest counts, pairing, and cross-split duplicates.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest = Path(args.manifest).resolve()
    output = Path(args.output).resolve()
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]

    patients: dict[str, list[dict]] = defaultdict(list)
    exact: dict[str, list[dict]] = defaultdict(list)
    perceptual: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        patients[record["patient_id"]].append(record)
        exact[record["sha256"]].append(record)
        perceptual[record["perceptual_hash"]].append(record)

    exact_groups = [group for group in exact.values() if len(group) > 1]
    exact_cross_split = [group for group in exact_groups if len({item["split"] for item in group}) > 1]
    perceptual_groups = [group for group in perceptual.values() if len(group) > 1]
    perceptual_cross_split = [
        group for group in perceptual_groups if len({item["split"] for item in group}) > 1
    ]
    inconsistent_patient_splits = {
        patient_id: sorted({item["split"] for item in group})
        for patient_id, group in patients.items()
        if len({item["split"] for item in group}) > 1
    }
    non_bilateral_patients = {
        patient_id: sorted(item["laterality"] for item in group)
        for patient_id, group in patients.items()
        if sorted(item["laterality"] for item in group) != ["left", "right"]
    }

    def examples(groups: list[list[dict]], limit: int = 20) -> list[dict]:
        return [
            {
                "hash": group[0]["sha256"],
                "paths": [item["relative_path"] for item in group],
                "patients": [item["patient_id"] for item in group],
                "splits": [item["split"] for item in group],
            }
            for group in groups[:limit]
        ]

    report = {
        "schema_version": "1.0",
        "manifest": str(manifest),
        "records": len(records),
        "patients": len(patients),
        "split_images": dict(sorted(Counter(item["split"] for item in records).items())),
        "source_partition_images": dict(
            sorted(Counter(item["source_partition"] for item in records).items())
        ),
        "inconsistent_patient_split_count": len(inconsistent_patient_splits),
        "non_bilateral_patient_count": len(non_bilateral_patients),
        "exact_duplicate_groups": len(exact_groups),
        "exact_duplicate_images": sum(len(group) for group in exact_groups),
        "exact_cross_split_groups": len(exact_cross_split),
        "exact_cross_split_examples": examples(exact_cross_split),
        "perceptual_hash_collision_groups": len(perceptual_groups),
        "perceptual_hash_cross_split_groups": len(perceptual_cross_split),
        "perceptual_hash_cross_split_examples": [
            {
                "perceptual_hash": group[0]["perceptual_hash"],
                "paths": [item["relative_path"] for item in group],
                "patients": [item["patient_id"] for item in group],
                "splits": [item["split"] for item in group],
            }
            for group in perceptual_cross_split[:20]
        ],
    }
    report["patient_split_gate"] = len(inconsistent_patient_splits) == 0
    report["exact_duplicate_leakage_gate"] = len(exact_cross_split) == 0
    report["pass"] = report["patient_split_gate"] and report["exact_duplicate_leakage_gate"]

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
