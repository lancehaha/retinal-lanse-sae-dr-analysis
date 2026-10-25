#!/usr/bin/env python3
"""Isolated OCTAVE HYA/RHS feasibility fit; never reads evaluation volumes."""

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class OCTAVETiles(Dataset):
    def __init__(self, root, draws, shifted):
        self.root = root
        self.shifted = shifted
        self.items = []
        self.arrays = {}
        for draw in draws:
            name = draw["volume"]
            if name not in self.arrays:
                image = np.load(root / f"{name}_image.npy", mmap_mode="r")
                label = np.load(root / f"{name}_label.npy", mmap_mode="r")
                self.arrays[name] = (image, label)
            width = self.arrays[name][0].shape[2]
            assert width in (1024, 1536)
            for x in range(0, width - 511, 256):
                self.items.append((name, draw["slice"], x))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        name, slice_index, x = self.items[index]
        image, label = self.arrays[name]
        frame = np.asarray(image[slice_index], dtype=np.uint8)
        mask = np.asarray(label[slice_index], dtype=np.uint8)
        if self.shifted:
            mask = np.roll(mask, mask.shape[1] // 2, axis=1)
        image_tile = frame[:, x:x + 512].astype(np.float32) / 255.0
        mask_tile = mask[:, x:x + 512]
        image_tile = np.pad(image_tile, ((0, 16), (0, 0)))
        target = np.stack((mask_tile == 9, mask_tile == 10)).astype(np.float32)
        target = np.pad(target, ((0, 0), (0, 16), (0, 0)))
        return torch.from_numpy(image_tile[None]), torch.from_numpy(target)


class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([
            ConvBlock(1, 16), ConvBlock(16, 32),
            ConvBlock(32, 64), ConvBlock(64, 128),
        ])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(128, 256)
        self.decoders = nn.ModuleList([
            ConvBlock(256 + 128, 128), ConvBlock(128 + 64, 64),
            ConvBlock(64 + 32, 32), ConvBlock(32 + 16, 16),
        ])
        self.head = nn.Conv2d(16, 2, 1)

    def forward(self, x):
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat((x, skip), dim=1))
        return self.head(x)


def loss_fn(logits, target, positive_weights):
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, target, pos_weight=positive_weights[None, :, None, None]
    )
    probability = logits.sigmoid()
    intersection = (probability * target).sum(dim=(0, 2, 3))
    denominator = probability.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3))
    dice = 1 - ((2 * intersection + 1) / (denominator + 1)).mean()
    return bce + dice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--arm", choices=("true", "shifted"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    receipt_path = args.cache_root / "cache_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    split = json.loads(args.split.read_text())
    schedule = json.loads(args.schedule.read_text())
    preflight = json.loads(args.preflight.read_text())
    assert receipt["frozen_split_sha256"] == sha256(args.split)
    assert schedule["cache_receipt_sha256"] == sha256(receipt_path)
    assert schedule["split_sha256"] == sha256(args.split)
    assert preflight["cache_receipt_sha256"] == sha256(receipt_path)
    assert preflight["split_sha256"] == sha256(args.split)
    assert len(schedule["epochs"]) == 20
    fit = set(split["fit_volumes"])
    evaluation = set(split["evaluation_volumes"])
    assert len(fit) == 158 and len(evaluation) == 40 and not fit & evaluation
    assert all(draw["volume"] in fit for epoch in schedule["epochs"] for draw in epoch)
    assert schedule["evaluation_volumes_read"] is False
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    assert torch.cuda.is_available()
    model = UNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    weights = preflight["fit_only_positive_weights_cap20"]
    positive_weights = torch.tensor([weights["HYA"], weights["RHS"]], device=device)
    args.output_root.mkdir(parents=True, exist_ok=True)
    result_path = args.output_root / ("dry_run.json" if args.dry_run else "train_receipt.json")
    assert not result_path.exists(), f"Refusing to overwrite {result_path}"
    if not args.dry_run:
        assert not (args.output_root / "epoch_20.pt").exists()
    history = []
    epoch_limit = 1 if args.dry_run else 20
    for epoch_index in range(epoch_limit):
        dataset = OCTAVETiles(args.cache_root, schedule["epochs"][epoch_index], args.arm == "shifted")
        loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
        model.train()
        total, batches = 0.0, 0
        for image, target in loader:
            image, target = image.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(image), target, positive_weights)
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            batches += 1
            if args.dry_run:
                break
        history.append({"epoch": epoch_index + 1, "mean_loss": total / batches, "batches": batches, "tiles": len(dataset)})
        print(json.dumps(history[-1]), flush=True)
    provenance = {
        "status": "fit_only_dry_run_pass" if args.dry_run else "external_anatomy_fit_complete_no_evaluation",
        "arm": args.arm,
        "epochs": epoch_limit,
        "cache_receipt_sha256": sha256(receipt_path),
        "split_sha256": sha256(args.split),
        "schedule_sha256": sha256(args.schedule),
        "preflight_sha256": sha256(args.preflight),
        "protocol_sha256": sha256(args.protocol),
        "evaluation_volumes_read": False,
        "olives_accessed": False,
        "history": history,
    }
    if not args.dry_run:
        torch.save({"state_dict": model.cpu().state_dict(), "provenance": provenance}, args.output_root / "epoch_20.pt")
        provenance["checkpoint_sha256"] = sha256(args.output_root / "epoch_20.pt")
        with (args.output_root / "train_history.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("epoch", "mean_loss", "batches", "tiles"))
            writer.writeheader()
            writer.writerows(history)
        provenance["history_csv_sha256"] = sha256(args.output_root / "train_history.csv")
    result_path.write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
