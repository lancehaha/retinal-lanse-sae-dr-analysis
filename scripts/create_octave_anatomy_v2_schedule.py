#!/usr/bin/env python3
"""Freeze matched v2 draws from OCTAVE subtrain volumes only."""

import argparse
import hashlib
import json
import random
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
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Refusing to overwrite schedule"
    split = json.loads(args.split.read_text())
    preflight = json.loads(args.preflight.read_text())
    subtrain = sorted(split["subtrain_volumes"])
    assert len(subtrain) == len(set(subtrain)) == 126
    assert preflight["split_sha256"] == sha256(args.split)
    assert preflight["protocol_sha256"] == sha256(args.protocol)
    assert preflight["cache_receipt_sha256"] == sha256(args.cache_root / "cache_receipt.json")
    positive, absent, positive_slices = [], [], {}
    for name in subtrain:
        mask = np.load(args.cache_root / f"{name}_label.npy", mmap_mode="r")
        slices = np.flatnonzero(np.any((mask == 9) | (mask == 10), axis=(1, 2))).tolist()
        if slices:
            positive.append(name)
            positive_slices[name] = slices
        else:
            absent.append(name)
    assert positive and absent
    rng = random.Random(42)
    epochs = []
    for _ in range(20):
        draws = []
        for _ in range(63):
            name = positive[rng.randrange(len(positive))]
            slices = positive_slices[name]
            draws.append({"volume": name, "slice": slices[rng.randrange(len(slices))], "source_stratum": "positive"})
            name = absent[rng.randrange(len(absent))]
            draws.append({"volume": name, "slice": rng.randrange(19), "source_stratum": "absent"})
        rng.shuffle(draws)
        epochs.append(draws)
    result = {
        "status": "v2_exploratory_subtrain_only_matched_arm_schedule_no_training",
        "seed": 42,
        "epochs": epochs,
        "draws_per_epoch": 126,
        "positive_draws_per_epoch": 63,
        "absent_draws_per_epoch": 63,
        "positive_subtrain_volumes": len(positive),
        "absent_subtrain_volumes": len(absent),
        "cache_receipt_sha256": sha256(args.cache_root / "cache_receipt.json"),
        "split_sha256": sha256(args.split),
        "preflight_sha256": sha256(args.preflight),
        "protocol_sha256": sha256(args.protocol),
        "development_volumes_read": False,
        "opened_v1_evaluation_volumes_read": False,
        "olives_accessed": False,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "epochs"}))


if __name__ == "__main__":
    main()
