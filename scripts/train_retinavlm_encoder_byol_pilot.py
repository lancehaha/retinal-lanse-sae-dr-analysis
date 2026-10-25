"""Short, patient-balanced BYOL continuation of the released RetinaVLM encoder.

This is an unlabeled timing/engineering pilot. It does not train the adapter or
Llama, and its checkpoint must not be used as a validated clinical model.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path


EXPECTED_INDEX_SHA256 = "65dfbceb982e991eaa66fa381aaeaf042e393dcfa1f16bad14da41ff2c1fe7a2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_encoder_weights(encoder, checkpoint_dir: Path) -> None:
    from safetensors import safe_open

    index_path = checkpoint_dir / "model.safetensors.index.json"
    if sha256(index_path) != EXPECTED_INDEX_SHA256:
        raise ValueError("Released checkpoint index differs from pinned model")
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    prefix = "model.visual_encoder.model."
    expected = set(encoder.state_dict())
    available = {key.removeprefix(prefix): shard for key, shard in weight_map.items() if key.startswith(prefix)}
    if set(available) != expected:
        raise ValueError(f"Encoder keys differ: missing={sorted(expected - set(available))[:8]}, extra={sorted(set(available) - expected)[:8]}")
    state = {}
    for shard in sorted(set(available.values())):
        with safe_open(checkpoint_dir / shard, framework="pt", device="cpu") as handle:
            for key, source_shard in available.items():
                if source_shard == shard:
                    state[key] = handle.get_tensor(prefix + key)
    encoder.load_state_dict(state, strict=True)
    print(f"Loaded {len(state)} encoder tensors from released Specialist checkpoint", flush=True)


class TrainImages:
    def __init__(self, root: Path):
        from PIL import Image  # noqa: F401

        receipt = json.loads((root / "receipt.json").read_text(encoding="utf-8"))
        manifest_path = root / "train_manifest.csv"
        if receipt["status"] != "frozen_olives_train_only_byol_package" or sha256(manifest_path) != receipt["train_manifest_sha256"]:
            raise ValueError("Training package receipt mismatch")
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            self.rows = list(csv.DictReader(handle))
        patients = Counter(row["patient_key"] for row in self.rows)
        if len(self.rows) != 6664 or len(patients) != 61 or len({r["source_image_path"] for r in self.rows}) != 6664:
            raise ValueError("Training package count mismatch")
        self.weights = [1.0 / patients[row["patient_key"]] for row in self.rows]
        self.root = root / "images"
        self.augment = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from PIL import Image
        from torchvision import transforms as T

        row = self.rows[index]
        relative = row["source_image_path"].lstrip("/")
        path = self.root / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing or symlinked train image: {path}")
        with Image.open(path) as image:
            if image.mode not in {"L", "RGB"}:
                raise ValueError(f"Unexpected image mode: {path}")
            image = image.convert("L")
        # Keep preprocessing aligned with the working full-frame 192px inference
        # path. This remains an OLIVES-to-Topcon geometry limitation.
        image = image.resize((192, 192), Image.Resampling.BILINEAR)
        if self.augment is None:
            self.augment = T.Compose([
                T.RandomResizedCrop(192, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
                T.RandomHorizontalFlip(),
                T.RandomApply([T.ColorJitter(brightness=0.15, contrast=0.15)], p=0.8),
                T.ToTensor(),
            ])
        return self.augment(image), self.augment(image)


def projector():
    import torch.nn as nn

    return nn.Sequential(nn.Linear(2048, 4096, bias=False), nn.BatchNorm1d(4096),
                         nn.ReLU(inplace=True), nn.Linear(4096, 256))


def byol_loss(prediction, target):
    import torch.nn.functional as F

    prediction = F.normalize(prediction.float(), dim=-1)
    target = F.normalize(target.float(), dim=-1)
    return 2 - 2 * (prediction * target).sum(dim=-1).mean()


def save_checkpoint(path: Path, encoder, step: int, metrics: dict) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "format": "retinavlm_olives_byol_encoder_pilot_v1",
        "encoder_state_dict": {k: v.detach().cpu() for k, v in encoder.state_dict().items()},
        "step": step,
        "metrics": metrics,
    }, temporary)
    os.replace(temporary, path)


def run(args):
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    if not torch.cuda.is_available():
        raise RuntimeError("An NVIDIA GPU is required")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    sys.path.insert(0, str(args.authors_code))
    from models.ssl_config import ssl_backbone

    dataset = TrainImages(args.data_dir)
    load_start = time.monotonic()
    online = ssl_backbone(channels_in=1)
    load_encoder_weights(online, args.checkpoint_dir)
    target = copy.deepcopy(online)
    online_projection = projector()
    target_projection = copy.deepcopy(online_projection)
    import torch.nn as nn
    predictor = nn.Sequential(nn.Linear(256, 4096, bias=False), nn.BatchNorm1d(4096),
                              nn.ReLU(inplace=True), nn.Linear(4096, 256))
    device = torch.device("cuda")
    online, target = online.to(device).train(), target.to(device).eval()
    online_projection, target_projection = online_projection.to(device).train(), target_projection.to(device).eval()
    predictor = predictor.to(device).train()
    for module in (target, target_projection):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(list(online.parameters()) + list(online_projection.parameters()) + list(predictor.parameters()),
                                  lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(dataset.weights, num_samples=args.steps * args.batch_size,
                                    replacement=True, generator=generator)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                        num_workers=args.workers, pin_memory=True, drop_last=True,
                        persistent_workers=args.workers > 0)
    print(json.dumps({"status": "pilot_start", "train_images": len(dataset.rows),
                      "train_patients": 61, "steps": args.steps, "batch_size": args.batch_size,
                      "gpu": torch.cuda.get_device_name(0), "preprocessing": "full_frame_resize_192"}), flush=True)
    torch.cuda.synchronize()
    train_start = time.monotonic()
    losses = []
    for step, (first, second) in enumerate(loader, start=1):
        first, second = first.to(device, non_blocking=True), second.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            q1 = predictor(online_projection(online(first)))
            q2 = predictor(online_projection(online(second)))
            with torch.no_grad():
                z1 = target_projection(target(first))
                z2 = target_projection(target(second))
            loss = (byol_loss(q1, z2) + byol_loss(q2, z1)) / 2
        if not torch.isfinite(loss):
            raise ValueError(f"Nonfinite BYOL loss at step {step}")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        with torch.no_grad():
            for online_param, target_param in zip(online.parameters(), target.parameters()):
                target_param.data.lerp_(online_param.data, 1 - args.ema)
            for online_buffer, target_buffer in zip(online.buffers(), target.buffers()):
                target_buffer.copy_(online_buffer)
            for online_param, target_param in zip(online_projection.parameters(), target_projection.parameters()):
                target_param.data.lerp_(online_param.data, 1 - args.ema)
            for online_buffer, target_buffer in zip(online_projection.buffers(), target_projection.buffers()):
                target_buffer.copy_(online_buffer)
        losses.append(float(loss.detach().item()))
        if step % args.log_every == 0 or step == args.steps:
            torch.cuda.synchronize()
            seconds = time.monotonic() - train_start
            print(json.dumps({"step": step, "steps": args.steps, "loss": losses[-1],
                              "mean_loss_last_window": sum(losses[-args.log_every:]) / len(losses[-args.log_every:]),
                              "seconds_per_step": seconds / step}), flush=True)
    torch.cuda.synchronize()
    seconds = time.monotonic() - train_start
    metrics = {
        "status": "byol_encoder_pilot_complete", "train_images": 6664, "train_patients": 61,
        "validation_images_used": 0, "test_images_used": 0,
        "initialization": "released_RetinaVLM_Specialist_encoder",
        "split_manifest_sha256": json.loads((args.data_dir / "receipt.json").read_text())["split_manifest_sha256"],
        "steps": args.steps, "batch_size": args.batch_size, "lr": args.lr, "ema": args.ema,
        "seed": args.seed, "preprocessing": "full_frame_resize_192",
        "augmentation": "mild_crop_flip_brightness_contrast",
        "gpu": torch.cuda.get_device_name(0),
        "initial_loss": losses[0], "final_loss": losses[-1],
        "mean_loss_last_20": sum(losses[-20:]) / len(losses[-20:]),
        "training_seconds": seconds, "seconds_per_step": seconds / args.steps,
        "startup_seconds": train_start - load_start,
        "limitation": "BYOL training loss is not evidence of IRF accuracy; adapter remains unchanged",
    }
    save_checkpoint(args.out_checkpoint, online, args.steps, metrics)
    metrics["encoder_checkpoint"] = str(args.out_checkpoint)
    metrics["encoder_checkpoint_sha256"] = sha256(args.out_checkpoint)
    args.out_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.out_metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--authors-code", type=Path, required=True)
    parser.add_argument("--out-checkpoint", type=Path, required=True)
    parser.add_argument("--out-metrics", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--ema", type=float, default=0.996)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.steps <= 0 or args.batch_size <= 1 or args.workers < 0:
        parser.error("steps > 0, batch-size > 1, and workers >= 0 are required")
    run(args)
