"""Read-only full OCTAVE training-image shape/type inventory from verified ZIP."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import tifffile


PREFIX = "OCTAVE/nnUNet_raw/Dataset001_OCTAVE/imagesTr/"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(archive: Path, acquisition: Path, inventory_path: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("Output already exists")
    source = json.loads(acquisition.read_text())
    inventory = json.loads(inventory_path.read_text())
    archive_sha = sha256(archive)
    if (source.get("archive_sha256") != archive_sha
            or inventory.get("archive_sha256") != archive_sha
            or inventory.get("training_volumes") != 198):
        raise ValueError("Source provenance mismatch")
    records = []
    with zipfile.ZipFile(archive) as bundle:
        for row in sorted(inventory["volumes"], key=lambda row: row["name"]):
            stem = row["name"]
            payload = bundle.read(PREFIX + stem + "_0000.tif")
            values = tifffile.imread(io.BytesIO(payload))
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite source image: {stem}")
            records.append({"name": stem, "member_sha256": hashlib.sha256(payload).hexdigest(),
                            "shape": list(values.shape), "dtype": str(values.dtype),
                            "minimum": float(values.min()), "maximum": float(values.max())})
    result = {"status": "external_source_image_geometry_audited_no_olives_use",
              "archive_sha256": archive_sha,
              "acquisition_receipt_sha256": sha256(acquisition),
              "label_inventory_sha256": sha256(inventory_path),
              "training_volumes": len(records),
              "shape_dtype_counts": dict(Counter(f"{r['shape']} {r['dtype']}" for r in records)),
              "records": records, "bulk_extraction": False, "olives_accessed": False,
              "claim_boundary": "Source geometry/type inventory only, no OLIVES map gate or model training"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--acquisition-receipt", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.archive, args.acquisition_receipt, args.inventory, args.output)
    print(json.dumps({"status": result["status"],
                      "training_volumes": result["training_volumes"],
                      "shape_dtype_counts": result["shape_dtype_counts"]}), flush=True)
