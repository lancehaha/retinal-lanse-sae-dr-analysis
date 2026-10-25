"""Package fixed within-patient IRF validation pairs for a cluster baseline gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import tarfile
from collections import defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run(split_csv: Path, split_audit: Path, image_manifest: Path,
        image_root: Path, out_tar: Path) -> dict:
    audit = json.loads(split_audit.read_text(encoding="utf-8"))
    if audit["status"] != "frozen_official_olives_six_biomarker_split_v1":
        raise ValueError("Split is not frozen v1")
    if sha256(split_csv) != audit["split_manifest_sha256"]:
        raise ValueError("Split manifest hash mismatch")
    if sha256(image_manifest) != audit["source_image_manifest_sha256"]:
        raise ValueError("Source image manifest hash mismatch")
    sources = {row["source_image_path"]: row for row in read_csv(image_manifest)}
    pairs = defaultdict(lambda: defaultdict(list))
    for row in read_csv(split_csv):
        if row["split"] == "validation" and row["B4"] in {"0", "1"} and row["mode"] == "L":
            pairs[row["patient_key"]][row["B4"]].append(row)
    if len(pairs) != 13 or any(set(classes) != {"0", "1"} for classes in pairs.values()):
        raise ValueError("Expected 13 validation patients, each with both IRF classes")

    selected = []
    for patient in sorted(pairs):
        for label in ("0", "1"):
            # Content-hash order fixes selection without looking at model outputs.
            row = min(pairs[patient][label], key=lambda r: (r["sha256"], r["source_image_path"]))
            source = sources[row["source_image_path"]]
            path = image_root / row["source_image_path"].lstrip("/")
            if path.is_symlink() or not path.is_file() or sha256(path) != row["sha256"]:
                raise ValueError(f"Missing or changed verified image: {path}")
            if source["sha256"] != row["sha256"]:
                raise ValueError("Source manifest hash disagrees with split")
            selected.append({
                "source_image_path": row["source_image_path"],
                "patient_key": patient,
                "trial": row["trial"],
                "IRF": label,
                "sha256": row["sha256"],
                "image_archive_path": "images/" + row["source_image_path"].lstrip("/"),
            })

    fields = list(selected[0])
    out_tar.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_tar, "w:gz") as archive:
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
        data = buffer.getvalue().encode("utf-8")
        member = tarfile.TarInfo("retinavlm_irf_gate/selected_manifest.csv")
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))
        receipt = {
            "status": "fixed_irf_validation_gate_13_patient_pairs",
            "split_manifest_sha256": sha256(split_csv),
            "images": len(selected),
            "patients": len(pairs),
            "selection": "lexicographically smallest file SHA256 per patient and binary IRF label",
            "split": "validation",
            "test_images_used": 0,
        }
        data = (json.dumps(receipt, indent=2) + "\n").encode("utf-8")
        member = tarfile.TarInfo("retinavlm_irf_gate/selection_receipt.json")
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))
        for row in selected:
            path = image_root / row["source_image_path"].lstrip("/")
            archive.add(path, arcname="retinavlm_irf_gate/" + row["image_archive_path"], recursive=False)
    return {"archive": str(out_tar), "archive_sha256": sha256(out_tar), **receipt}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split_csv", type=Path)
    parser.add_argument("split_audit", type=Path)
    parser.add_argument("image_manifest", type=Path)
    parser.add_argument("image_root", type=Path)
    parser.add_argument("out_tar", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.split_csv, args.split_audit, args.image_manifest,
                         args.image_root, args.out_tar), indent=2))
