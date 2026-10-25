"""Verify the frozen OLIVES validation images before A/B/C inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


EXPECTED_SPLIT_SHA256 = "28690d3ced5cdd3349100448b8f471cb56bb1690b0502c28148d055a266068fe"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(split_csv: Path, image_root: Path, out: Path) -> dict:
    if out.exists():
        raise FileExistsError(f"Audit receipt already exists: {out}")
    split_hash = sha256(split_csv)
    if split_hash != EXPECTED_SPLIT_SHA256:
        raise ValueError("Frozen OLIVES split manifest hash changed")
    with split_csv.open(newline="", encoding="utf-8") as source:
        rows = [row for row in csv.DictReader(source) if row["split"] == "validation"]
    if len(rows) != 1470 or len({row["patient_key"] for row in rows}) != 13:
        raise ValueError("Unexpected validation scan or patient count")
    failures = []
    bytes_checked = 0
    for row in rows:
        path = image_root / row["source_image_path"].lstrip("/")
        if not path.is_file() or path.is_symlink():
            failures.append({"source_image_path": row["source_image_path"], "reason": "missing_or_symlink"})
            continue
        bytes_checked += path.stat().st_size
        if sha256(path) != row["sha256"]:
            failures.append({"source_image_path": row["source_image_path"], "reason": "sha256_mismatch"})
    receipt = {
        "status": "validation_images_verified" if not failures else "validation_image_audit_failed",
        "split_sha256": split_hash,
        "validation_scans": len(rows),
        "validation_patients": len({row["patient_key"] for row in rows}),
        "bytes_checked": bytes_checked,
        "failures": failures,
        "test_images_accessed": False,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": receipt["status"], "failures": len(failures),
                      "validation_scans": len(rows)}), flush=True)
    if failures:
        raise ValueError("Validation image audit failed; see receipt")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit(args.split_csv, args.image_root, args.out)
