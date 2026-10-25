"""Cache verified OCTAVE TIFF volumes as uint8 NumPy arrays on cluster scratch."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import tifffile


PREFIX = "OCTAVE/nnUNet_raw/Dataset001_OCTAVE/"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(archive: Path, acquisition: Path, inventory_path: Path,
        split_path: Path, output_root: Path) -> dict:
    if output_root.exists():
        raise ValueError("Output root already exists; preserve any prior attempt")
    source = json.loads(acquisition.read_text())
    inventory = json.loads(inventory_path.read_text())
    split = json.loads(split_path.read_text())
    archive_sha = sha256(archive)
    if (source.get("archive_sha256") != archive_sha
            or inventory.get("archive_sha256") != archive_sha
            or split.get("archive_sha256") != archive_sha
            or split.get("source_inventory_sha256") != sha256(inventory_path)
            or len(split.get("fit_volumes", [])) != 158
            or len(split.get("evaluation_volumes", [])) != 40):
        raise ValueError("Source provenance or frozen split mismatch")
    names = sorted(set(split["fit_volumes"] + split["evaluation_volumes"]))
    expected = {row["name"]: row for row in inventory["volumes"]}
    if len(names) != 198 or set(names) != set(expected):
        raise ValueError("Frozen split does not cover exact external inventory")
    output_root.mkdir(parents=True)
    records = []
    with zipfile.ZipFile(archive) as bundle:
        for stem in names:
            image_bytes = bundle.read(PREFIX + f"imagesTr/{stem}_0000.tif")
            label_bytes = bundle.read(PREFIX + f"labelsTr/{stem}.tif")
            image = tifffile.imread(io.BytesIO(image_bytes))
            label = tifffile.imread(io.BytesIO(label_bytes))
            if (image.ndim != 3 or image.shape[:2] != (19, 496)
                    or image.shape[2] not in (1024, 1536) or label.shape != image.shape
                    or image.dtype != np.uint8 or label.dtype.kind not in "ui"
                    or int(label.min()) < 0 or int(label.max()) > 13):
                raise ValueError(f"Invalid image/label values: {stem}")
            for key, value in (("HYA", 9), ("RHS", 10), ("VIT", 8)):
                slices = int(np.count_nonzero(np.any(label == value, axis=(1, 2))))
                if slices != expected[stem]["class_presence"][key]:
                    raise ValueError(f"Inventory mismatch: {stem} {key}")
            image_path = output_root / f"{stem}_image.npy"
            label_path = output_root / f"{stem}_label.npy"
            np.save(image_path, image, allow_pickle=False)
            np.save(label_path, label.astype(np.uint8), allow_pickle=False)
            records.append({"name": stem,
                            "group": "fit" if stem in split["fit_volumes"] else "evaluation",
                            "image_member_sha256": hashlib.sha256(image_bytes).hexdigest(),
                            "label_member_sha256": hashlib.sha256(label_bytes).hexdigest(),
                            "image_cache_sha256": sha256(image_path),
                            "label_cache_sha256": sha256(label_path),
                            "shape": list(image.shape),
                            "image_cache_bytes": image_path.stat().st_size,
                            "label_cache_bytes": label_path.stat().st_size})
    receipt = {"status": "external_anatomy_cache_verified_no_olives_use",
               "archive_sha256": archive_sha,
               "acquisition_receipt_sha256": sha256(acquisition),
               "inventory_sha256": sha256(inventory_path),
               "frozen_split_sha256": sha256(split_path),
               "volume_count": len(records), "records": records,
               "local_bulk_extraction": False,
               "cluster_derived_cache_only": True,
               "olives_accessed": False,
               "claim_boundary": "External anatomy feasibility cache only; no OLIVES map gate or model training"}
    (output_root / "cache_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--acquisition-receipt", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.archive, args.acquisition_receipt, args.inventory,
                 args.split, args.output_root)
    print(json.dumps({"status": result["status"], "volume_count": result["volume_count"]}),
          flush=True)
