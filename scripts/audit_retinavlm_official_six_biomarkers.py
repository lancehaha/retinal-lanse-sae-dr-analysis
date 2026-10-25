"""Audit six OLIVES biomarkers on the verified official expert image cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


PATH_COLUMN = "Path (Trial/Arm/Folder/Visit/Eye/Image Name)"
COLUMNS = {
    "B1": "IR HRF",
    "B2": "Partially attached vitreous face",
    "B3": "Fully attached vitreous face",
    "B4": "Fluid (IRF)",
    "B5": "DRT/ME",
    "B6": "Vitreous debris",
}
SPLITS = ("train", "validation", "test")


def normalized_path(value: str) -> str:
    return "/" + value.strip().strip("/")


def label_status(value: str) -> str:
    value = value.strip()
    if value in {"0", "0.0"}:
        return "negative"
    if value in {"1", "1.0"}:
        return "positive"
    if not value:
        return "missing"
    return "invalid"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run(labels_zip: Path, image_manifest: Path, out: Path) -> dict:
    with zipfile.ZipFile(labels_zip) as archive:
        names = [name for name in archive.namelist()
                 if name.endswith("/ml_centric_labels/Biomarker_Clinical_Data_Images.csv")]
        if len(names) != 1:
            raise ValueError(f"Expected one official expert CSV, found {names}")
        with archive.open(names[0]) as binary:
            reader = csv.DictReader(io.TextIOWrapper(binary, encoding="utf-8-sig"))
            required = {PATH_COLUMN, "Patient_ID", "Eye_ID", *COLUMNS.values()}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"Missing columns: {required - set(reader.fieldnames or [])}")
            labels = list(reader)

    by_path = {}
    for row in labels:
        path = normalized_path(row[PATH_COLUMN])
        if path in by_path:
            raise ValueError(f"Duplicate official label path: {path}")
        by_path[path] = row

    with image_manifest.open(newline="", encoding="utf-8") as f:
        images = list(csv.DictReader(f))
    if len({normalized_path(r["source_image_path"]) for r in images}) != len(images):
        raise ValueError("Duplicate verified image path")
    if set(by_path) != {normalized_path(r["source_image_path"]) for r in images}:
        raise ValueError("Official labels and verified images have different path sets")

    patient_splits = defaultdict(set)
    patient_eyes = defaultdict(set)
    grouped = {code: defaultdict(lambda: defaultdict(Counter)) for code in COLUMNS}
    all_counts = {code: Counter() for code in COLUMNS}
    for image in images:
        path = normalized_path(image["source_image_path"])
        source = by_path[path]
        split = image["split"]
        if split not in SPLITS:
            raise ValueError(f"Unexpected split: {split}")
        trial = path.strip("/").split("/")[0]
        patient = (trial, source["Patient_ID"].strip())
        if not patient[1] or image["Patient_ID"] != patient[1]:
            raise ValueError(f"Patient ID mismatch: {path}")
        if image["Eye_ID"] != source["Eye_ID"].strip():
            raise ValueError(f"Eye ID mismatch: {path}")
        patient_splits[patient].add(split)
        patient_eyes[patient].add(image["Eye_ID"])
        for code, column in COLUMNS.items():
            status = label_status(source[column])
            all_counts[code][status] += 1
            grouped[code][split][patient][status] += 1
    leaks = {str(patient): sorted(splits) for patient, splits in patient_splits.items()
             if len(splits) != 1}
    if leaks:
        raise ValueError(f"Patient split leakage: {leaks}")

    findings = {}
    for code, column in COLUMNS.items():
        per_split = {}
        for split in SPLITS:
            patients = grouped[code][split]
            counts = Counter()
            for statuses in patients.values():
                counts.update(statuses)
            per_split[split] = {
                "images": sum(counts.values()),
                "labels": {kind: counts[kind] for kind in
                           ("positive", "negative", "missing", "invalid")},
                "patients": len(patients),
                "patients_with_positive": sum(p["positive"] > 0 for p in patients.values()),
                "patients_with_negative": sum(p["negative"] > 0 for p in patients.values()),
                "patients_with_only_verified_negative": sum(
                    p["negative"] > 0 and p["positive"] == p["missing"] == p["invalid"] == 0
                    for p in patients.values()),
            }
        findings[code] = {
            "official_column": column,
            "labels": {kind: all_counts[code][kind] for kind in
                       ("positive", "negative", "missing", "invalid")},
            "by_existing_fluid_pilot_split": per_split,
        }

    report = {
        "status": "official_six_biomarker_label_and_patient_audit_complete",
        "scope": "official OLIVES expert-labeled scans with verified image paths; not HF challenge test",
        "labels_zip_sha256": digest(labels_zip),
        "verified_image_manifest_sha256": digest(image_manifest),
        "official_label_rows": len(labels),
        "verified_images": len(images),
        "patients": len(patient_splits),
        "patients_with_multiple_eyes": sum(len(eyes) > 1 for eyes in patient_eyes.values()),
        "patient_split_overlap": len(leaks),
        "split_note": "Existing split belongs to earlier fluid pilot; this audit does not freeze it for six-biomarker training.",
        "findings": findings,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels_zip", type=Path)
    parser.add_argument("image_manifest", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    report = run(args.labels_zip, args.image_manifest, args.out)
    print(json.dumps({key: report[key] for key in
                      ("status", "verified_images", "patients", "patient_split_overlap")}, indent=2))
