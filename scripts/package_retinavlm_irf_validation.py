"""Package the frozen OLIVES validation scans for an encoder IRF comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import tarfile
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_bytes(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(data)
    member.mode = 0o644
    archive.addfile(member, io.BytesIO(data))


def run(split_csv: Path, split_audit: Path, image_root: Path, output: Path) -> dict:
    audit = json.loads(split_audit.read_text(encoding="utf-8"))
    split_hash = sha256(split_csv)
    if audit["status"] != "frozen_official_olives_six_biomarker_split_v1" or split_hash != audit["split_manifest_sha256"]:
        raise ValueError("Frozen split or its hash did not match the audit")
    with split_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if row["split"] == "validation"]
    patients = {row["patient_key"] for row in selected}
    if len(rows) != 9408 or len(selected) != 1470 or len(patients) != 13:
        raise ValueError("Unexpected validation scan/patient count")
    if any(row["IRF"] not in {"0", "1"} for row in selected):
        raise ValueError("Missing IRF validation label")
    if len({row["source_image_path"] for row in selected}) != len(selected):
        raise ValueError("Repeated image path")
    for index, row in enumerate(selected, start=1):
        relative = row["source_image_path"].lstrip("/")
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("Unsafe image path")
        image = image_root / relative
        if image.is_symlink() or not image.is_file() or sha256(image) != row["sha256"]:
            raise ValueError(f"Image missing or changed: {image}")
        if index % 500 == 0:
            print(f"Verified {index}/{len(selected)} validation images", flush=True)
    fields = ("source_image_path", "patient_key", "IRF", "sha256", "width", "height", "mode")
    slim = [{key: row[key] for key in fields} for row in selected]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(slim)
    manifest_bytes = buffer.getvalue().encode("utf-8")
    receipt = {
        "status": "frozen_olives_validation_only_irf_package",
        "split_manifest_sha256": split_hash,
        "validation_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "images": len(selected), "patients": len(patients),
        "IRF_counts": dict(Counter(row["IRF"] for row in selected)),
        "train_images": 0, "test_images": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz", compresslevel=1) as archive:
        add_bytes(archive, "retinavlm_irf_validation/validation_manifest.csv", manifest_bytes)
        add_bytes(archive, "retinavlm_irf_validation/receipt.json", (json.dumps(receipt, indent=2) + "\n").encode())
        for index, row in enumerate(selected, start=1):
            relative = row["source_image_path"].lstrip("/")
            archive.add(image_root / relative,
                        arcname="retinavlm_irf_validation/images/" + relative,
                        recursive=False)
            if index % 500 == 0:
                print(f"Packed {index}/{len(selected)} validation images", flush=True)
    receipt["archive_sha256"] = sha256(output)
    receipt["archive_bytes"] = output.stat().st_size
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--split-audit", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.split_csv, args.split_audit, args.image_root, args.output), indent=2))
