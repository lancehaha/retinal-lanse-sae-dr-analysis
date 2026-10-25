"""Read-only, stratified OCTAVE image/mask geometry audit from the verified ZIP."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image


PREFIX = "OCTAVE/nnUNet_raw/Dataset001_OCTAVE/"
GROUPS = ("CNV", "DME", "DRU", "ERM", "GA", "NORMAL")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select(inventory: dict) -> list[dict]:
    chosen = []
    for group in GROUPS:
        rows = sorted((row for row in inventory["volumes"]
                       if row["name"].split("_")[1] == group),
                      key=lambda row: row["name"])
        for condition, predicate in (
            ("hya_rhs_present", lambda row: row["class_presence"]["HYA"] > 0
             and row["class_presence"]["RHS"] > 0),
            ("both_absent", lambda row: row["class_presence"]["HYA"] == 0
             and row["class_presence"]["RHS"] == 0),
        ):
            found = next((row for row in rows if predicate(row)), None)
            if found is not None:
                chosen.append({"group": group, "condition": condition, "name": found["name"]})
    return chosen


def run(archive: Path, acquisition: Path, inventory_path: Path, output_root: Path) -> dict:
    output = output_root / "image_mask_alignment.json"
    if output.exists():
        raise ValueError("Output already exists")
    source = json.loads(acquisition.read_text())
    inventory = json.loads(inventory_path.read_text())
    archive_sha = sha256(archive)
    if (source.get("archive_sha256") != archive_sha
            or inventory.get("archive_sha256") != archive_sha
            or inventory.get("acquisition_receipt_sha256") != sha256(acquisition)
            or inventory.get("training_volumes") != 198):
        raise ValueError("Archive/inventory provenance mismatch")
    selected = select(inventory)
    if len(selected) < 10:
        raise ValueError("Insufficient stratified sample")
    output_root.mkdir(parents=True, exist_ok=True)
    audited = []
    with zipfile.ZipFile(archive) as bundle:
        for item in selected:
            stem = item["name"]
            image_bytes = bundle.read(PREFIX + "imagesTr/" + stem + "_0000.tif")
            label_bytes = bundle.read(PREFIX + "labelsTr/" + stem + ".tif")
            image = tifffile.imread(io.BytesIO(image_bytes))
            label = tifffile.imread(io.BytesIO(label_bytes))
            if (image.shape != label.shape or image.ndim != 3 or image.shape[0] != 19
                    or label.dtype.kind not in "ui" or not np.isfinite(image).all()):
                raise ValueError(f"Image/mask geometry or value mismatch for {stem}")
            ids, counts = np.unique(label, return_counts=True)
            if any(int(value) > 13 or int(value) < 0 for value in ids):
                raise ValueError(f"Unexpected label IDs for {stem}")
            anatomy_count = np.count_nonzero((label == 9) | (label == 10), axis=(1, 2))
            slice_index = int(np.argmax(anatomy_count)) if anatomy_count.any() else 9
            plane = image[slice_index].astype(np.float32)
            low, high = np.percentile(plane, [1, 99])
            if high <= low:
                raise ValueError(f"Flat image plane for {stem}")
            gray = np.uint8(np.clip((plane - low) / (high - low) * 255, 0, 255))
            rgb = np.stack([gray] * 3, axis=-1)
            hya = label[slice_index] == 9
            rhs = label[slice_index] == 10
            rgb[hya] = np.uint8(0.5 * rgb[hya] + 0.5 * np.array([255, 40, 40]))
            rgb[rhs] = np.uint8(0.5 * rgb[rhs] + 0.5 * np.array([20, 220, 255]))
            overlay = output_root / f"{stem}_slice{slice_index:02d}_overlay.png"
            Image.fromarray(rgb).save(overlay)
            audited.append({**item, "image_member_sha256": hashlib.sha256(image_bytes).hexdigest(),
                            "label_member_sha256": hashlib.sha256(label_bytes).hexdigest(),
                            "image_shape": list(image.shape), "label_shape": list(label.shape),
                            "image_dtype": str(image.dtype), "label_dtype": str(label.dtype),
                            "slice_index": slice_index,
                            "image_slice_p01_p99": [float(low), float(high)],
                            "image_slice_mean": float(plane.mean()),
                            "image_slice_std": float(plane.std()),
                            "image_slice_fraction_below_16": float(np.mean(plane < 16)),
                            "horizontal_resize_factor_to_192": image.shape[2] / 192,
                            "vertical_resize_factor_to_192": image.shape[1] / 192,
                            "label_id_pixels": {str(int(i)): int(n) for i, n in zip(ids, counts)},
                            "overlay": overlay.name, "overlay_sha256": sha256(overlay)})
    result = {"status": "external_read_only_sample_alignment_audit",
              "archive_sha256": archive_sha, "acquisition_receipt_sha256": sha256(acquisition),
              "inventory_sha256": sha256(inventory_path),
              "selection_rule": "lexicographically first HYA+RHS-present and both-absent volume per disease group, where available",
              "sample_count": len(audited), "samples": audited,
              "bulk_extraction": False, "olives_accessed": False,
              "claim_boundary": "Shape and sampled overlay check only; no clinical label equivalence, registration proof, or OLIVES map gate"}
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--acquisition-receipt", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({key: value for key, value in run(
        args.archive, args.acquisition_receipt, args.inventory, args.output_root).items()
        if key in ("status", "sample_count")}), flush=True)
