"""Audit fixed top-crop feasibility on SHA-verified training OCT images.

The row-mean thresholds are image-content heuristics, not retinal segmentation
or clinical labels. This audit does not change the model input or map gates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with args.split.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "train"]
    if not rows:
        raise ValueError("No original-training scans")
    thresholds = (15, 25, 40)
    crop_fractions = (0.80, 0.85, 0.90)
    counts = {str(t): Counter() for t in thresholds}
    for row in rows:
        image_path = args.images / row["source_image_path"].lstrip("/")
        if not image_path.is_file() or image_path.is_symlink() or sha256(image_path) != row["sha256"]:
            raise ValueError(f"Image missing or SHA mismatch: {row['source_image_path']}")
        with Image.open(image_path) as image:
            gray = np.asarray(image.convert("L"), dtype=np.uint8)
        mean_by_row = gray.mean(axis=1)
        for threshold in thresholds:
            bright = np.flatnonzero(mean_by_row > threshold)
            if not len(bright):
                counts[str(threshold)]["no_row_above_threshold"] += 1
                continue
            lower_fraction = int(bright[-1]) / gray.shape[0]
            for crop in crop_fractions:
                if lower_fraction >= crop:
                    counts[str(threshold)][f"content_at_or_below_top_{crop:.2f}_crop"] += 1
    output = {
        "status": "training_image_geometry_diagnostic_only_no_model_input_change",
        "split_sha256": sha256(args.split),
        "training_scans_sha_verified": len(rows),
        "row_mean_thresholds": list(thresholds),
        "candidate_top_crop_fractions": list(crop_fractions),
        "counts": {threshold: dict(counter) for threshold, counter in counts.items()},
        "limitation": "Row mean brightness is a heuristic, not retinal/vitreous segmentation; a crop overlapping a bright row is risky, while zero such rows does not prove anatomy is preserved.",
        "original_validation_accessed": False,
        "test_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"status": output["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
