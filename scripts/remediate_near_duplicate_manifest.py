from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retina_lanse.data.manifest import sha256_path
from scripts.run_global_sae_search import require_external


SPLIT_PRIORITY = ("train", "validation", "internal_test")


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def remediate_records(records: list[dict], audit: dict) -> tuple[list[dict], dict]:
    confirmed = audit.get("confirmed", [])
    if (
        audit.get("status") != "fail"
        or audit.get("pass") is not False
        or not confirmed
        or audit.get("confirmed_cross_split_near_duplicates") != len(confirmed)
    ):
        raise RuntimeError("source audit must be a failed audit with confirmed duplicate pairs")

    patient_splits: dict[str, str] = {}
    patient_rows: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        patient = str(record["patient_id"])
        split = str(record["split"])
        prior = patient_splits.setdefault(patient, split)
        if prior != split:
            raise RuntimeError(f"patient already crosses splits: {patient}")
        patient_rows[patient].append(record)

    union_find = UnionFind()
    for pair in confirmed:
        left, right = str(pair["left_patient"]), str(pair["right_patient"])
        if left not in patient_splits or right not in patient_splits:
            raise RuntimeError("confirmed duplicate patient is absent from source manifest")
        if patient_splits[left] != pair["left_split"] or patient_splits[right] != pair["right_split"]:
            raise RuntimeError("confirmed duplicate split disagrees with source manifest")
        if patient_splits[left] == patient_splits[right]:
            raise RuntimeError("remediation input unexpectedly contains a within-split pair")
        union_find.union(left, right)

    members_by_root: dict[str, set[str]] = defaultdict(set)
    for patient in union_find.parent:
        members_by_root[union_find.find(patient)].add(patient)

    excluded_patients: set[str] = set()
    components = []
    for component_index, members in enumerate(
        sorted(members_by_root.values(), key=lambda values: sorted(values)), start=1
    ):
        splits = {patient_splits[patient] for patient in members}
        anchor_split = next((split for split in SPLIT_PRIORITY if split in splits), None)
        if anchor_split is None:
            raise RuntimeError(f"unsupported split in duplicate component: {sorted(splits)}")
        kept = sorted(patient for patient in members if patient_splits[patient] == anchor_split)
        excluded = sorted(patient for patient in members if patient_splits[patient] != anchor_split)
        if not kept or not excluded:
            raise RuntimeError("duplicate component did not cross a supported split boundary")
        excluded_patients.update(excluded)
        components.append(
            {
                "component_id": component_index,
                "anchor_split": anchor_split,
                "members": [
                    {
                        "patient_id": patient,
                        "split": patient_splits[patient],
                        "images": len(patient_rows[patient]),
                    }
                    for patient in sorted(members)
                ],
                "kept_patients": kept,
                "excluded_patients": excluded,
            }
        )

    if any(patient_splits[patient] == "train" for patient in excluded_patients):
        raise RuntimeError("fixed priority policy unexpectedly excluded a training patient")

    retained = [record for record in records if str(record["patient_id"]) not in excluded_patients]
    retained_patients = {str(record["patient_id"]) for record in retained}
    for pair in confirmed:
        if str(pair["left_patient"]) in retained_patients and str(pair["right_patient"]) in retained_patients:
            raise RuntimeError("a confirmed cross-split duplicate pair survived remediation")

    image_counts = Counter(str(record["split"]) for record in retained)
    patient_sets: dict[str, set[str]] = defaultdict(set)
    for record in retained:
        patient_sets[str(record["split"])].add(str(record["patient_id"]))
    excluded_rows = [
        {
            "patient_id": patient,
            "split": patient_splits[patient],
            "images": len(patient_rows[patient]),
            "relative_paths": sorted(str(row["relative_path"]) for row in patient_rows[patient]),
        }
        for patient in sorted(excluded_patients)
    ]
    summary = {
        "policy": "connected_components_keep_highest_priority_split_exclude_all_other_patients",
        "split_priority": list(SPLIT_PRIORITY),
        "clinical_labels_used": False,
        "patients_reassigned": 0,
        "source_records": len(records),
        "retained_records": len(retained),
        "excluded_images": len(records) - len(retained),
        "excluded_patients": len(excluded_patients),
        "excluded_patient_records": excluded_rows,
        "components": components,
        "retained_split_images": dict(sorted(image_counts.items())),
        "retained_split_patients": {
            split: len(patients) for split, patients in sorted(patient_sets.items())
        },
        "training_records_unchanged": not any(
            patient_splits[patient] == "train" for patient in excluded_patients
        ),
    }
    return retained, summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a label-free, patient-level decontaminated publication manifest."
    )
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--failed-audit", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    source = require_external(Path(args.source_manifest), "source manifest")
    audit_path = require_external(Path(args.failed_audit), "failed near-duplicate audit")
    output = require_external(Path(args.output_manifest), "publication manifest")
    report_path = require_external(Path(args.report), "remediation report")
    if source.resolve() == output.resolve():
        raise RuntimeError("source and output manifests must be different files")
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("manifest_sha256") != sha256_path(source):
        raise RuntimeError("failed audit does not belong to the source manifest")
    retained, summary = remediate_records(records, audit)

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    with partial.open("w", encoding="utf-8") as handle:
        for record in retained:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    partial.replace(output)

    report = {
        "schema_version": "1.0",
        "status": "complete",
        "scope": "label_free_patient_level_near_duplicate_remediation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(source),
        "source_manifest_sha256": sha256_path(source),
        "source_failed_audit": str(audit_path),
        "source_failed_audit_sha256": sha256_path(audit_path),
        "output_manifest": str(output),
        "output_manifest_sha256": sha256_path(output),
        **summary,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
