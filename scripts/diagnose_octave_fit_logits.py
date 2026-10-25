#!/usr/bin/env python3
"""Read-only fit-volume logit diagnostic for locked OCTAVE checkpoints."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from train_octave_anatomy_feasibility import UNet


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def predict_slice(model, frame, device):
    image = np.asarray(frame, dtype=np.float32) / 255.0
    width = image.shape[1]
    logits_sum = np.zeros((2, 496, width), dtype=np.float32)
    coverage = np.zeros(width, dtype=np.float32)
    with torch.inference_mode():
        for x in range(0, width - 511, 256):
            tile = np.pad(image[:, x:x + 512], ((0, 16), (0, 0)))
            logits = model(torch.from_numpy(tile[None, None]).to(device))[0, :, :496].cpu().numpy()
            logits_sum[:, :, x:x + 512] += logits
            coverage[x:x + 512] += 1
    assert np.all(coverage > 0)
    return 1 / (1 + np.exp(-np.clip(logits_sum / coverage[None, None, :], -30, 30)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--true-checkpoint", type=Path, required=True)
    parser.add_argument("--shifted-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    split = json.loads(args.split.read_text())
    fit = sorted(split["fit_volumes"])
    evaluation = set(split["evaluation_volumes"])
    assert len(fit) == 158 and len(evaluation) == 40 and not set(fit) & evaluation
    positive = {"HYA": [], "RHS": []}
    absent = []
    for name in fit:
        label = np.load(args.cache_root / f"{name}_label.npy", mmap_mode="r")
        has_hya = bool(np.any(label == 9))
        has_rhs = bool(np.any(label == 10))
        if has_hya:
            positive["HYA"].append(name)
        if has_rhs:
            positive["RHS"].append(name)
        if not has_hya and not has_rhs:
            absent.append(name)
    select = lambda names: sorted(names, key=lambda name: hashlib.sha256(name.encode()).hexdigest())[:8]
    selected = [(name, "HYA") for name in select(positive["HYA"])]
    selected += [(name, "RHS") for name in select(positive["RHS"])]
    selected += [(name, "ABSENT") for name in select(absent)]
    device = torch.device("cuda")
    assert torch.cuda.is_available()
    models = {}
    checkpoint_hashes = {}
    for arm, path in (("true", args.true_checkpoint), ("shifted", args.shifted_checkpoint)):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert payload["provenance"]["arm"] == arm and payload["provenance"]["epochs"] == 20
        assert payload["provenance"]["evaluation_volumes_read"] is False
        model = UNet().to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        models[arm] = model
        checkpoint_hashes[arm] = sha256(path)
    records = []
    for name, selection in selected:
        image = np.load(args.cache_root / f"{name}_image.npy", mmap_mode="r")
        label = np.load(args.cache_root / f"{name}_label.npy", mmap_mode="r")
        value = {"HYA": 9, "RHS": 10}.get(selection)
        slice_index = int(np.flatnonzero(np.any(label == value, axis=(1, 2)))[0]) if value else 0
        truth = np.asarray(label[slice_index])
        row = {"volume": name, "selection": selection, "slice": slice_index, "arms": {}}
        for arm, model in models.items():
            probability = predict_slice(model, image[slice_index], device)
            row["arms"][arm] = {}
            for channel, (finding, class_id) in enumerate((("HYA", 9), ("RHS", 10))):
                gt = truth == class_id
                bg = ~gt
                values = probability[channel]
                row["arms"][arm][finding] = {
                    "ground_truth_positive_pixels": int(np.count_nonzero(gt)),
                    "predicted_positive_pixels_at_frozen_0_5": int(np.count_nonzero(values >= 0.5)),
                    "max_probability": float(values.max()),
                    "positive_mean_probability": float(values[gt].mean()) if np.any(gt) else None,
                    "background_mean_probability": float(values[bg].mean()),
                    "background_p99_probability": float(np.quantile(values[bg], 0.99)),
                }
        records.append(row)
        print(json.dumps({"completed_fit_volume": name, "count": len(records)}), flush=True)
    result = {
        "status": "read_only_fit_volume_logit_diagnostic_no_eligibility",
        "cache_receipt_sha256": sha256(args.cache_root / "cache_receipt.json"),
        "split_sha256": sha256(args.split),
        "checkpoint_sha256": checkpoint_hashes,
        "selection_rule": "first 8 by SHA256(volume ID) per HYA-positive, RHS-positive and both-absent fit stratum",
        "records": records,
        "evaluation_volumes_read": False,
        "olives_accessed": False,
        "claim_boundary": "Training-volume diagnostic after failed held-out feasibility; no threshold retuning or gate eligibility",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
