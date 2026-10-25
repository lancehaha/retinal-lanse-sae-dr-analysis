"""Audit OLIVES expert biomarker labels and make patient-disjoint pilot splits.

This script reads only the small official labels archive. It does not assert that
the corresponding OCT images have been obtained or that a model has been run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


CSV_IN_ZIP = (
    "OLIVES_Dataset_Labels/ml_centric_labels/"
    "Biomarker_Clinical_Data_Images.csv"
)
PATH_COL = "Path (Trial/Arm/Folder/Visit/Eye/Image Name)"
REQUIRED = {PATH_COL, "Fluid (IRF)", "Fluid (SRF)", "Eye_ID", "Patient_ID"}


def label(value: str) -> int | None:
    value = value.strip()
    if value in {"0", "0.0"}:
        return 0
    if value in {"1", "1.0"}:
        return 1
    return None


def run(source: Path, out_dir: Path, seed: int) -> dict:
    with zipfile.ZipFile(source) as archive:
        with archive.open(CSV_IN_ZIP) as binary:
            reader = csv.DictReader(io.TextIOWrapper(binary, encoding="utf-8-sig"))
            if not REQUIRED.issubset(reader.fieldnames or []):
                raise ValueError(f"Missing columns: {sorted(REQUIRED - set(reader.fieldnames or []))}")
            rows = list(reader)

    seen_paths: set[str] = set()
    valid = []
    unknown = Counter()
    patient_trials: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        path = row[PATH_COL].strip()
        patient = row["Patient_ID"].strip()
        eye = row["Eye_ID"].strip()
        irf, srf = label(row["Fluid (IRF)"]), label(row["Fluid (SRF)"])
        if not path or not patient or not eye or irf is None or srf is None:
            unknown["missing_or_nonbinary_required_field"] += 1
            continue
        if path in seen_paths:
            unknown["duplicate_path"] += 1
            continue
        seen_paths.add(path)
        trial = path.strip("/").split("/")[0]
        patient_trials[patient].add(trial)
        valid.append({
            "source_image_path": path,
            "trial": trial,
            "Patient_ID": patient,
            "Eye_ID": eye,
            "IRF": irf,
            "SRF": srf,
            "fluid": int(bool(irf or srf)),
        })

    # Trial-stratify patients, while keeping every eye/visit/scan for a patient
    # in exactly one partition. This split is a pilot candidate, not an image
    # compatibility or near-duplicate clearance.
    patients_by_trial: dict[str, list[str]] = defaultdict(list)
    for patient, trials in patient_trials.items():
        if len(trials) != 1:
            raise ValueError(f"Patient {patient} appears in multiple trials")
        patients_by_trial[next(iter(trials))].append(patient)
    partitions: dict[str, str] = {}
    rng = random.Random(seed)
    for trial in sorted(patients_by_trial):
        patients = sorted(patients_by_trial[trial])
        rng.shuffle(patients)
        n_test = round(len(patients) * 0.15)
        n_val = round(len(patients) * 0.15)
        for i, patient in enumerate(patients):
            partitions[patient] = (
                "test" if i < n_test else
                "validation" if i < n_test + n_val else "train"
            )
    for row in valid:
        row["split"] = partitions[row["Patient_ID"]]

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "olives_expert_fluid_manifest.csv"
    fields = ["source_image_path", "trial", "Patient_ID", "Eye_ID", "IRF", "SRF", "fluid", "split"]
    with manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(valid)

    summary = {
        "status": "labels_audited_images_unverified",
        "source": "https://zenodo.org/records/7105232",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_rows": len(rows),
        "valid_rows": len(valid),
        "excluded_rows": dict(unknown),
        "patients": len(partitions),
        "trials": dict(Counter(row["trial"] for row in valid)),
        "fluid_labels": dict(Counter(str(row["fluid"]) for row in valid)),
        "splits": {
            split: {
                "patients": len({row["Patient_ID"] for row in valid if row["split"] == split}),
                "scans": sum(row["split"] == split for row in valid),
                "fluid_positive": sum(row["split"] == split and row["fluid"] == 1 for row in valid),
            }
            for split in ("train", "validation", "test")
        },
        "split_seed": seed,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "gates": {
            "image_path_match": "not_run",
            "image_format_and_crop": "not_run",
            "image_near_duplicates": "not_run",
            "tokenizer_single_token": "not_run",
            "original_model_baseline": "not_run",
            "j_lens_image_signal": "not_run",
        },
    }
    (out_dir / "olives_label_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Official OLIVES_Dataset_Labels.zip")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--seed", type=int, default=4106)
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.out_dir, args.seed), indent=2))
