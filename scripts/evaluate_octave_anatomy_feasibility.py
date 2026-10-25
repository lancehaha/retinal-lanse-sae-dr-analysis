#!/usr/bin/env python3
"""Paired volume-disjoint OCTAVE feasibility evaluation after both fits lock."""

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
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


def evaluate_one(model, image, label, device):
    counts = {name: {key: 0 for key in ("tp", "fp", "fn", "gt", "pixels")} for name in ("HYA", "RHS")}
    with torch.inference_mode():
        for slice_index in range(image.shape[0]):
            frame = np.asarray(image[slice_index], dtype=np.float32) / 255.0
            width = frame.shape[1]
            starts = list(range(0, width - 511, 256))
            logits_sum = np.zeros((2, 496, width), dtype=np.float32)
            coverage = np.zeros(width, dtype=np.float32)
            for x in starts:
                tile = np.pad(frame[:, x:x + 512], ((0, 16), (0, 0)))
                tensor = torch.from_numpy(tile[None, None]).to(device)
                logits = model(tensor)[0, :, :496].cpu().numpy()
                logits_sum[:, :, x:x + 512] += logits
                coverage[x:x + 512] += 1
            assert np.all(coverage > 0)
            prediction = logits_sum / coverage[None, None, :] >= 0
            truth = np.asarray(label[slice_index], dtype=np.uint8)
            for channel, (name, value) in enumerate((("HYA", 9), ("RHS", 10))):
                pred = prediction[channel]
                gt = truth == value
                row = counts[name]
                row["tp"] += int(np.count_nonzero(pred & gt))
                row["fp"] += int(np.count_nonzero(pred & ~gt))
                row["fn"] += int(np.count_nonzero(~pred & gt))
                row["gt"] += int(np.count_nonzero(gt))
                row["pixels"] += gt.size
    result = {}
    for name, row in counts.items():
        result[name] = dict(row)
        result[name]["dice"] = 2 * row["tp"] / (2 * row["tp"] + row["fp"] + row["fn"]) if row["gt"] else None
        result[name]["sensitivity"] = row["tp"] / row["gt"] if row["gt"] else None
        result[name]["absent_false_positive_fraction"] = row["fp"] / row["pixels"] if not row["gt"] else None
    return result


def mean(values):
    return sum(values) / len(values) if values else None


def summarize(records):
    output = {}
    for arm in ("true", "shifted"):
        output[arm] = {}
        for finding in ("HYA", "RHS"):
            rows = [row["metrics"][arm][finding] for row in records]
            output[arm][finding] = {
                "positive_volumes": sum(row["gt"] > 0 for row in rows),
                "absent_volumes": sum(row["gt"] == 0 for row in rows),
                "mean_positive_volume_dice": mean([row["dice"] for row in rows if row["dice"] is not None]),
                "mean_positive_volume_sensitivity": mean([row["sensitivity"] for row in rows if row["sensitivity"] is not None]),
                "mean_absent_volume_false_positive_fraction": mean([row["absent_false_positive_fraction"] for row in rows if row["absent_false_positive_fraction"] is not None]),
            }
    return output


def percentile_interval(values):
    finite = sorted(value for value in values if value is not None)
    if not finite:
        return None
    return [finite[int(0.025 * (len(finite) - 1))], finite[int(0.975 * (len(finite) - 1))]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--true-checkpoint", type=Path, required=True)
    parser.add_argument("--shifted-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output_root.exists(), "Refusing to overwrite evaluation output"
    split = json.loads(args.split.read_text())
    receipt = json.loads((args.cache_root / "cache_receipt.json").read_text())
    assert receipt["frozen_split_sha256"] == sha256(args.split)
    fit = set(split["fit_volumes"])
    evaluation = set(split["evaluation_volumes"])
    assert len(fit) == 158 and len(evaluation) == 40 and not fit & evaluation
    device = torch.device("cuda")
    assert torch.cuda.is_available()
    models = {}
    checkpoint_paths = {"true": args.true_checkpoint, "shifted": args.shifted_checkpoint}
    checkpoint_hashes = {}
    for arm, path in checkpoint_paths.items():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        provenance = payload["provenance"]
        assert provenance["arm"] == arm and provenance["epochs"] == 20
        assert provenance["status"] == "external_anatomy_fit_complete_no_evaluation"
        assert provenance["split_sha256"] == sha256(args.split)
        assert provenance["evaluation_volumes_read"] is False and provenance["olives_accessed"] is False
        model = UNet().to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        models[arm] = model
        checkpoint_hashes[arm] = sha256(path)
    records = []
    for name in sorted(evaluation):
        image = np.load(args.cache_root / f"{name}_image.npy", mmap_mode="r")
        label = np.load(args.cache_root / f"{name}_label.npy", mmap_mode="r")
        assert image.shape == label.shape and image.shape[:2] == (19, 496)
        group = name.removeprefix("OCTAVE_").rsplit("_", 1)[0]
        metrics = {arm: evaluate_one(model, image, label, device) for arm, model in models.items()}
        records.append({"volume": name, "disease_group": group, "metrics": metrics})
        print(json.dumps({"completed_volume": name, "count": len(records)}), flush=True)
    summary = summarize(records)
    by_disease = {group: summarize([row for row in records if row["disease_group"] == group]) for group in sorted({row["disease_group"] for row in records})}
    rng = random.Random(42)
    bootstrap = defaultdict(list)
    for _ in range(1000):
        sample = [records[rng.randrange(len(records))] for _ in records]
        sample_summary = summarize(sample)
        for finding in ("HYA", "RHS"):
            true_dice = sample_summary["true"][finding]["mean_positive_volume_dice"]
            shifted_dice = sample_summary["shifted"][finding]["mean_positive_volume_dice"]
            bootstrap[(finding, "true_dice")].append(true_dice)
            bootstrap[(finding, "shifted_dice")].append(shifted_dice)
            bootstrap[(finding, "paired_dice_delta")].append(true_dice - shifted_dice if true_dice is not None and shifted_dice is not None else None)
    intervals = {finding: {metric: percentile_interval(bootstrap[(finding, metric)]) for metric in ("true_dice", "shifted_dice", "paired_dice_delta")} for finding in ("HYA", "RHS")}
    args.output_root.mkdir(parents=True)
    with (args.output_root / "per_volume.jsonl").open("w") as stream:
        for row in records:
            stream.write(json.dumps(row) + "\n")
    summary_receipt = {
        "status": "external_anatomy_volume_disjoint_feasibility_only",
        "split_sha256": sha256(args.split),
        "cache_receipt_sha256": sha256(args.cache_root / "cache_receipt.json"),
        "checkpoint_sha256": checkpoint_hashes,
        "volumes": len(records),
        "patient_disjointness": "unverified_external_patient_ids",
        "summary": summary,
        "by_disease_group": by_disease,
        "volume_bootstrap_95pct": intervals,
        "per_volume_jsonl_sha256": sha256(args.output_root / "per_volume.jsonl"),
        "olives_accessed": False,
        "claim_boundary": "OCTAVE HYA/RHS segmentation feasibility, not OLIVES B2/B3 labels or map eligibility",
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary_receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
