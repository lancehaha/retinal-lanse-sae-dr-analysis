"""Read-only native-image geometry and intensity audit on original OLIVES training patients."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(split: Path, image_root: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("Output already exists")
    with split.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_patient = defaultdict(list)
    for row in rows:
        if row["split"] == "train":
            by_patient[row["patient_key"]].append(row)
    if len(by_patient) != 61:
        raise ValueError(f"Expected 61 original training patients, got {len(by_patient)}")
    samples = []
    for patient, candidates in sorted(by_patient.items()):
        candidates.sort(key=lambda row: row["source_image_path"])
        row = candidates[len(candidates) // 2]
        path = image_root / row["source_image_path"].lstrip("/")
        if not path.is_file() or sha256(path) != row["sha256"]:
            raise ValueError(f"Source image missing or SHA mismatch: {patient}")
        with Image.open(path) as image:
            width, height = image.size
            mode = image.mode
            if (str(width), str(height), mode) != (row["width"], row["height"], row["mode"]):
                raise ValueError(f"Official geometry mismatch: {patient}")
            gray = np.asarray(image.convert("L"), dtype=np.uint8)
        samples.append({"patient_key": patient, "source_image_path": row["source_image_path"],
                        "sha256": row["sha256"], "shape": [height, width], "mode": mode,
                        "mean": float(gray.mean()), "std": float(gray.std()),
                        "p01": float(np.percentile(gray, 1)),
                        "p50": float(np.percentile(gray, 50)),
                        "p99": float(np.percentile(gray, 99)),
                        "fraction_below_16": float(np.mean(gray < 16)),
                        "horizontal_resize_factor_to_192": width / 192,
                        "vertical_resize_factor_to_192": height / 192})
    result = {"status": "original_train_only_domain_audit",
              "split_sha256": sha256(split), "selection": "lexicographic middle source path per original training patient",
              "patient_count": len(samples),
              "native_shapes": dict(Counter(f"{s['shape'][1]}x{s['shape'][0]}" for s in samples)),
              "modes": dict(Counter(s["mode"] for s in samples)),
              "median_mean": float(np.median([s["mean"] for s in samples])),
              "median_std": float(np.median([s["std"] for s in samples])),
              "median_fraction_below_16": float(np.median([s["fraction_below_16"] for s in samples])),
              "samples": samples,
              "validation_or_test_accessed": False,
              "claim_boundary": "Native image domain description only; no label gate or model performance"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.split, args.image_root, args.output)
    print(json.dumps({k: result[k] for k in ("status", "patient_count", "native_shapes", "modes",
                                               "median_mean", "median_std", "median_fraction_below_16")}))
