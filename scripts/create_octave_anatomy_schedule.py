#!/usr/bin/env python3
"""Freeze matched, fit-only OCTAVE volume/slice draws before model training."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    split = json.loads(args.split.read_text())
    fit = sorted(split["fit_volumes"])
    assert len(fit) == 158 and len(set(fit)) == 158
    positive, absent = [], []
    positive_slices = {}
    for name in fit:
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
    for epoch in range(20):
        draws = []
        for _ in range(79):
            name = positive[rng.randrange(len(positive))]
            slices = positive_slices[name]
            draws.append({"volume": name, "slice": slices[rng.randrange(len(slices))], "source_stratum": "positive"})
            name = absent[rng.randrange(len(absent))]
            draws.append({"volume": name, "slice": rng.randrange(19), "source_stratum": "absent"})
        rng.shuffle(draws)
        epochs.append(draws)
    output = {
        "status": "frozen_fit_only_matched_arm_schedule_no_model_training",
        "seed": 42,
        "epochs": epochs,
        "draws_per_epoch": 158,
        "positive_draws_per_epoch": 79,
        "absent_draws_per_epoch": 79,
        "positive_fit_volumes": len(positive),
        "absent_fit_volumes": len(absent),
        "cache_receipt_sha256": sha256(args.cache_root / "cache_receipt.json"),
        "split_sha256": sha256(args.split),
        "evaluation_volumes_read": False,
        "olives_accessed": False,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: value for key, value in output.items() if key != "epochs"}))


if __name__ == "__main__":
    main()
