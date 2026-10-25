"""Read-only OCTAVE label-volume inventory from a verified external archive."""

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


PREFIX = "OCTAVE/nnUNet_raw/Dataset001_OCTAVE/"
CLASSES = {"VIT": 8, "HYA": 9, "RHS": 10}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(archive: Path, acquisition_receipt: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("Output already exists")
    source = json.loads(acquisition_receipt.read_text())
    if (source.get("status") != "archive_verified_not_extracted"
            or source.get("archive_sha256") != sha256(archive)):
        raise ValueError("Archive provenance mismatch")
    totals = {name: {"volumes": 0, "slices": 0, "pixels": 0}
              for name in CLASSES}
    volumes = []
    with zipfile.ZipFile(archive) as bundle:
        config_bytes = bundle.read(PREFIX + "dataset.json")
        config = json.loads(config_bytes)
        if (config.get("numTraining") != 198 or config.get("labels", {}) != {
                "background": 0, "SRM": 1, "HRM": 2, "FLU": 3,
                "HTD": 4, "RPE": 5, "RET": 6, "CHO": 7, "VIT": 8,
                "HYA": 9, "RHS": 10, "ART": 11, "ERM": 12, "SES": 13}):
            raise ValueError("Unexpected OCTAVE class map")
        labels = sorted(name for name in bundle.namelist()
                        if name.startswith(PREFIX + "labelsTr/")
                        and name.endswith(".tif"))
        images = {Path(name).name.removesuffix("_0000.tif")
                  for name in bundle.namelist()
                  if name.startswith(PREFIX + "imagesTr/")
                  and name.endswith("_0000.tif")}
        if len(labels) != 198 or len(images) != 198:
            raise ValueError("Incomplete training image or label inventory")
        for label_path in labels:
            stem = Path(label_path).stem
            if stem not in images:
                raise ValueError(f"Image missing for {stem}")
            payload = bundle.read(label_path)
            local = {name: {"slices": 0, "pixels": 0} for name in CLASSES}
            values = tifffile.imread(io.BytesIO(payload))
            if values.ndim == 2:
                values = values[None, ...]
            if values.ndim != 3 or values.dtype.kind not in "ui":
                raise ValueError(f"Unexpected label volume for {stem}: {values.shape} {values.dtype}")
            slice_count = values.shape[0]
            for name, value in CLASSES.items():
                matches = values == value
                local[name]["pixels"] = int(np.count_nonzero(matches))
                local[name]["slices"] = int(np.count_nonzero(np.any(matches, axis=(1, 2))))
            for name in CLASSES:
                totals[name]["volumes"] += local[name]["slices"] > 0
                totals[name]["slices"] += local[name]["slices"]
                totals[name]["pixels"] += local[name]["pixels"]
            volumes.append({"name": stem, "slices": slice_count,
                            "class_presence": {name: local[name]["slices"]
                                               for name in CLASSES}})
    receipt = {
        "status": "external_anatomy_inventory_audited_no_olives_use",
        "archive_sha256": source["archive_sha256"],
        "acquisition_receipt_sha256": sha256(acquisition_receipt),
        "dataset_json_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "training_volumes": len(volumes),
        "training_slices": sum(item["slices"] for item in volumes),
        "classes": totals,
        "volume_slices_distribution": dict(Counter(str(item["slices"]) for item in volumes)),
        "volumes": volumes,
        "bulk_extraction": False,
        "olives_validation_or_test_accessed": False,
        "claim_boundary": "External HYA/RHS/VIT segmentation, not OLIVES B2/B3 binary labels or a map gate",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--acquisition-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.archive, args.acquisition_receipt, args.output)
    print(json.dumps({key: result[key] for key in
                      ("status", "training_volumes", "training_slices", "classes")}),
          flush=True)
