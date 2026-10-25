#!/usr/bin/env python3
"""Read-only preflight for the isolated OCTAVE segmentation comparison."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt_path = args.cache_root / "cache_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    split = json.loads(args.split.read_text())
    assert receipt["frozen_split_sha256"] == sha256(args.split)
    fit = set(split["fit_volumes"])
    evaluation = set(split["evaluation_volumes"])
    assert len(fit) == 158 and len(evaluation) == 40 and not fit & evaluation
    records = {r["name"]: r for r in receipt["records"]}
    assert set(records) == fit | evaluation
    positives = Counter()
    image_pixels = 0
    positive_volumes = Counter()
    positive_slices = Counter()
    tile_count = 0
    for name in sorted(fit):
        record = records[name]
        image = np.load(args.cache_root / f"{name}_image.npy", mmap_mode="r")
        label = np.load(args.cache_root / f"{name}_label.npy", mmap_mode="r")
        assert tuple(image.shape) == tuple(label.shape) == tuple(record["shape"])
        assert image.dtype == np.uint8 and label.dtype == np.uint8
        assert image.shape[:2] == (19, 496) and image.shape[2] in (1024, 1536)
        starts = list(range(0, image.shape[2] - 511, 256))
        assert starts[-1] == image.shape[2] - 512
        tile_count += len(starts)
        image_pixels += label.size
        for key, value in (("HYA", 9), ("RHS", 10)):
            mask = label == value
            count = int(np.count_nonzero(mask))
            positives[key] += count
            positive_volumes[key] += count > 0
            positive_slices[key] += int(np.count_nonzero(np.any(mask, axis=(1, 2))))
    weights = {key: min(20.0, (image_pixels - count) / max(count, 1)) for key, count in positives.items()}
    assert positive_volumes["HYA"] == 48 and positive_volumes["RHS"] == 46
    assert positives["HYA"] > 0 and positives["RHS"] > 0
    result = {
        "status": "external_anatomy_training_preflight_pass_no_training",
        "cache_receipt_sha256": sha256(receipt_path),
        "split_sha256": sha256(args.split),
        "fit_volumes": len(fit),
        "evaluation_volumes_unread": len(evaluation),
        "fit_native_pixels": image_pixels,
        "fit_positive_pixels": dict(positives),
        "fit_positive_volumes": dict(positive_volumes),
        "fit_positive_slices": dict(positive_slices),
        "fit_only_positive_weights_cap20": weights,
        "all_tile_positions_per_fit_selected_slice": tile_count,
        "olives_accessed": False,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
