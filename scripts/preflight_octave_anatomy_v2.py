#!/usr/bin/env python3
"""Count OCTAVE v2 subtrain pixels without opening either evaluation group."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Refusing to overwrite preflight"
    receipt_path = args.cache_root / "cache_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    split = json.loads(args.split.read_text())
    subtrain = set(split["subtrain_volumes"])
    development = set(split["development_volumes"])
    opened = set(split["opened_v1_evaluation_volumes_excluded"])
    assert len(subtrain) == 126 and len(development) == 32 and len(opened) == 40
    assert not (subtrain & development or subtrain & opened or development & opened)
    assert receipt["frozen_split_sha256"] == split["parent_split_sha256"]
    records = {row["name"]: row for row in receipt["records"]}
    assert set(records) == subtrain | development | opened
    positives = Counter()
    positive_volumes = Counter()
    positive_slices = Counter()
    pixels = 0
    tile_positions = 0
    for name in sorted(subtrain):
        record = records[name]
        image = np.load(args.cache_root / f"{name}_image.npy", mmap_mode="r")
        label = np.load(args.cache_root / f"{name}_label.npy", mmap_mode="r")
        assert tuple(image.shape) == tuple(label.shape) == tuple(record["shape"])
        assert image.dtype == label.dtype == np.uint8
        assert image.shape[:2] == (19, 496) and image.shape[2] in (1024, 1536)
        starts = list(range(0, image.shape[2] - 511, 256))
        assert starts[-1] == image.shape[2] - 512
        tile_positions += len(starts)
        pixels += label.size
        for finding, label_id in (("HYA", 9), ("RHS", 10)):
            present = label == label_id
            count = int(np.count_nonzero(present))
            positives[finding] += count
            positive_volumes[finding] += count > 0
            positive_slices[finding] += int(np.count_nonzero(np.any(present, axis=(1, 2))))
    assert positive_volumes["HYA"] == 39 and positive_volumes["RHS"] == 37
    assert all(positives[k] > 0 for k in ("HYA", "RHS"))
    weights = {k: min(200.0, (pixels - v) / v) for k, v in positives.items()}
    result = {
        "status": "v2_exploratory_subtrain_only_preflight_pass_no_training",
        "cache_receipt_sha256": sha256(receipt_path),
        "split_sha256": sha256(args.split),
        "protocol_sha256": sha256(args.protocol),
        "subtrain_volumes": 126,
        "development_volumes_unread": 32,
        "opened_v1_evaluation_volumes_unread": 40,
        "subtrain_native_pixels": pixels,
        "subtrain_positive_pixels": dict(positives),
        "subtrain_positive_volumes": dict(positive_volumes),
        "subtrain_positive_slices": dict(positive_slices),
        "subtrain_only_positive_weights_cap200": weights,
        "all_tile_positions_per_subtrain_selected_slice": tile_positions,
        "olives_accessed": False,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
