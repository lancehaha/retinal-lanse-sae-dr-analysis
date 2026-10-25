"""Freeze the audited patient-disjoint official OLIVES split with six labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from audit_retinavlm_official_six_biomarkers import COLUMNS, PATH_COLUMN, label_status, normalized_path


EXPECTED_IMAGE_MANIFEST_SHA256 = "bd55283dd468ced0a81e4751a5e669962c56495fc03abda7efc2a69d0a2779ce"
EXPECTED_LABELS_ZIP_SHA256 = "8c63ef84541b747c91de3c754801012e0742e8672243a219bc49f809c78b9362"
SPLITS = ("train", "validation", "test")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_labels(path: Path) -> dict[str, dict[str, str]]:
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist()
                 if name.endswith("/ml_centric_labels/Biomarker_Clinical_Data_Images.csv")]
        if len(names) != 1:
            raise ValueError(f"Expected exactly one expert label CSV, found {len(names)}")
        with archive.open(names[0]) as binary:
            rows = list(csv.DictReader(io.TextIOWrapper(binary, encoding="utf-8-sig")))
    by_path = {}
    for row in rows:
        key = normalized_path(row[PATH_COLUMN])
        if key in by_path:
            raise ValueError(f"Duplicate label path: {key}")
        by_path[key] = row
    return by_path


def run(labels_zip: Path, image_manifest: Path, image_audit: Path,
        out_csv: Path, out_audit: Path) -> dict:
    if sha256(labels_zip) != EXPECTED_LABELS_ZIP_SHA256:
        raise ValueError("Official label ZIP differs from audited source")
    if sha256(image_manifest) != EXPECTED_IMAGE_MANIFEST_SHA256:
        raise ValueError("Verified image manifest differs from audited source")
    source_audit = json.loads(image_audit.read_text(encoding="utf-8"))
    if source_audit["manifest_sha256"] != EXPECTED_IMAGE_MANIFEST_SHA256:
        raise ValueError("Image audit does not match the verified image manifest")
    if source_audit["cross_split_decoded_pixel_duplicate_hashes"] != 0:
        raise ValueError("Image audit reports cross-split exact duplicates")
    if source_audit["near_duplicate_gate"] != "reviewed_pass":
        raise ValueError("Near-duplicate review gate has not passed")

    labels = load_labels(labels_zip)
    with image_manifest.open(newline="", encoding="utf-8") as handle:
        images = list(csv.DictReader(handle))
    if len(images) != 9408 or len(labels) != len(images):
        raise ValueError("Unexpected cohort size")
    if {normalized_path(r["source_image_path"]) for r in images} != set(labels):
        raise ValueError("Image and label path sets differ")

    fields = ("source_image_path", "trial", "Patient_ID", "Eye_ID", "patient_key",
              "split", "IRF", "SRF", "fluid", *COLUMNS,
              "sha256", "pixel_sha256", "width", "height", "mode")
    result_rows = []
    patient_splits: dict[str, set[str]] = defaultdict(set)
    pixel_splits: dict[str, set[str]] = defaultdict(set)
    summary = {split: {code: Counter() for code in COLUMNS} for split in SPLITS}
    patient_sets = {split: set() for split in SPLITS}
    trial_counts = {split: Counter() for split in SPLITS}
    for image in sorted(images, key=lambda r: normalized_path(r["source_image_path"])):
        path = normalized_path(image["source_image_path"])
        official = labels[path]
        split = image["split"]
        if split not in SPLITS:
            raise ValueError(f"Unknown split: {split}")
        trial = path.strip("/").split("/")[0]
        if trial != image["trial"] or image["Patient_ID"] != official["Patient_ID"].strip():
            raise ValueError(f"Patient or trial mismatch: {path}")
        if image["Eye_ID"] != official["Eye_ID"].strip():
            raise ValueError(f"Eye mismatch: {path}")
        patient_key = f'{trial}/{image["Patient_ID"]}'
        patient_splits[patient_key].add(split)
        pixel_splits[image["pixel_sha256"]].add(split)
        patient_sets[split].add(patient_key)
        trial_counts[split][trial] += 1
        row = {key: image[key] for key in fields if key in image}
        row["patient_key"] = patient_key
        for code, column in COLUMNS.items():
            status = label_status(official[column])
            if status == "invalid":
                raise ValueError(f"Invalid {code} label: {path}")
            row[code] = "" if status == "missing" else str(int(status == "positive"))
            summary[split][code][status] += 1
        if row["B4"] != image["IRF"]:
            raise ValueError(f"Official B4/IRF mismatch: {path}")
        result_rows.append(row)

    overlap = {key: sorted(value) for key, value in patient_splits.items() if len(value) > 1}
    duplicate_overlap = {key: sorted(value) for key, value in pixel_splits.items() if len(value) > 1}
    if overlap or duplicate_overlap:
        raise ValueError(f"Split leakage: patients={overlap}, pixels={duplicate_overlap}")
    if tuple(len(patient_sets[s]) for s in SPLITS) != (61, 13, 13):
        raise ValueError("Unexpected patient allocation")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result_rows)
    audit = {
        "status": "frozen_official_olives_six_biomarker_split_v1",
        "cohort": "official expert-labeled OLIVES scans, not Hugging Face challenge test",
        "assignment": "preserved audited patient-disjoint fluid-pilot assignments; no reassignment",
        "labels_zip_sha256": EXPECTED_LABELS_ZIP_SHA256,
        "source_image_manifest_sha256": EXPECTED_IMAGE_MANIFEST_SHA256,
        "source_image_audit_sha256": sha256(image_audit),
        "split_manifest_sha256": sha256(out_csv),
        "images": len(result_rows),
        "patients": len(patient_splits),
        "patient_overlap": len(overlap),
        "cross_split_exact_pixel_duplicates": len(duplicate_overlap),
        "near_duplicate_review": source_audit["near_duplicate_gate"],
        "splits": {
            split: {
                "images": sum(trial_counts[split].values()),
                "patients": len(patient_sets[split]),
                "trial_images": dict(sorted(trial_counts[split].items())),
                "findings": {
                    code: {status: summary[split][code][status]
                           for status in ("positive", "negative", "missing")}
                    for code in COLUMNS
                },
            }
            for split in SPLITS
        },
        "missing_label_rule": "empty field means missing; never treat as negative",
        "training_sampling": "sample patients evenly, then images within patient; do not move images across splits",
        "test_policy": "do not use test predictions for prompt, layer, threshold, or hyperparameter selection",
    }
    out_audit.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels_zip", type=Path)
    parser.add_argument("image_manifest", type=Path)
    parser.add_argument("image_audit", type=Path)
    parser.add_argument("out_csv", type=Path)
    parser.add_argument("out_audit", type=Path)
    args = parser.parse_args()
    result = run(args.labels_zip, args.image_manifest, args.image_audit,
                 args.out_csv, args.out_audit)
    print(json.dumps({key: result[key] for key in
                      ("status", "images", "patients", "patient_overlap", "splits")}, indent=2))
