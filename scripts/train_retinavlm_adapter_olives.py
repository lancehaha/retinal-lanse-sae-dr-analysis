"""Resumable, patient-balanced OLIVES answer alignment of RetinaVLM's visual adapter.

Two independently scheduled variants use the same released VLM/adapter start,
training examples, optimizer, and seed. Only the image encoder differs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

from retinavlm_binary_answer import answer_token_ids, answer_loss
from retinavlm_pre_answer import pre_answer_forward
from retinavlm_released_model import load_released_specialist
from retinavlm_matched_arm_losses import adapter_parameters


SPLIT_SHA = "28690d3ced5cdd3349100448b8f471cb56bb1690b0502c28148d055a266068fe"
ADAPTED_SHA = "c2e433c7d64d747b2b3f3a14f28cbd9c2d153cfb1b5270093b07b8b32c9ac50c"
QUESTIONS = {
    "B1": "Are intraretinal hyperreflective foci present in this OCT scan? Answer present or absent.",
    "B2": "Is a partially attached vitreous face present in this OCT scan? Answer present or absent.",
    "B3": "Is a fully attached vitreous face present in this OCT scan? Answer present or absent.",
    "B4": "Is intraretinal fluid present in this OCT scan? Answer present or absent.",
    "B5": "Is diffuse retinal thickening or macular edema present in this OCT scan? Answer present or absent.",
    "B6": "Is vitreous debris present in this OCT scan? Answer present or absent.",
    "SRF": "Is subretinal fluid present in this OCT scan? Answer present or absent.",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path):
    if sha256(path) != SPLIT_SHA:
        raise ValueError("Frozen split SHA-256 changed")
    with Path(path).open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 9408:
        raise ValueError("Unexpected frozen cohort size")
    groups = {s: [r for r in rows if r["split"] == s] for s in ("train", "validation", "test")}
    if {s: len(rs) for s, rs in groups.items()} != {"train": 6664, "validation": 1470, "test": 1274}:
        raise ValueError("Frozen split sizes changed")
    patients = {s: {r["patient_key"] for r in rs} for s, rs in groups.items()}
    if any(patients[a] & patients[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Patient leakage")
    return groups


def install_encoder(model, variant, adapted_checkpoint):
    if variant == "released":
        return
    if sha256(adapted_checkpoint) != ADAPTED_SHA:
        raise ValueError("BYOL encoder checksum mismatch")
    checkpoint = torch.load(adapted_checkpoint, map_location="cpu")
    if checkpoint.get("format") != "retinavlm_olives_byol_encoder_full_v1" or checkpoint.get("step") != 2083:
        raise ValueError("BYOL encoder provenance mismatch")
    model.model.visual_encoder.model.load_state_dict(checkpoint["encoder_state_dict"], strict=True)
    del checkpoint


def load_image(model, image_root, row):
    from PIL import Image
    path = image_root / row["source_image_path"].lstrip("/")
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Missing or linked OCT image: {path}")
    with Image.open(path) as image:
        if image.mode not in {"L", "RGB"}:
            raise ValueError(f"Unexpected OCT mode: {path}: {image.mode}")
        # Match the prior frozen-encoder F1 probe: one RGB source is converted
        # to grayscale before the authors' full-frame 192x192 resize.
        return model.convert_any_image_to_normalized_tensor(image.convert("L")).to(model.model.device)


def save_checkpoint(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".writing")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def run(args):
    if args.steps < 1 or args.save_every < 1 or args.variant not in {"released", "adapted_byol"}:
        raise ValueError("Invalid training arguments")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    groups = load_rows(args.split_csv)
    train = groups["train"]
    # Verify every training image exists before touching the model. Val/test are not read here.
    for row in train:
        path = args.train_images / row["source_image_path"].lstrip("/")
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Training OCT missing: {path}")
    by_patient = defaultdict(list)
    for row in train:
        by_patient[row["patient_key"]].append(row)
    patients = sorted(by_patient)
    if len(patients) != 61:
        raise ValueError("Expected 61 training patients")
    by_finding_label = {finding: {label: defaultdict(list) for label in (0, 1)}
                        for finding in QUESTIONS}
    for row in train:
        for finding in QUESTIONS:
            if row[finding] not in {"0", "1"}:
                continue  # An unknown label is neither a positive nor a negative target.
            by_finding_label[finding][int(row[finding])][row["patient_key"]].append(row)
    if any(not by_finding_label[f][label] for f in QUESTIONS for label in (0, 1)):
        raise ValueError("Training split lacks a class for a finding")
    eligible_patients = {f: {label: sorted(by_finding_label[f][label]) for label in (0, 1)}
                         for f in QUESTIONS}

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    model, _ = load_released_specialist(args.model_dir, args.authors_code)
    install_encoder(model, args.variant, args.adapted_checkpoint)
    parameters = adapter_parameters(model.model)
    model.eval()
    model.model.llama_proj.train()
    optimizer = torch.optim.AdamW(parameters, lr=args.lr)
    absent_id, present_id = answer_token_ids(model.model.get_tokenizer(), QUESTIONS["B5"])
    latest = args.out_dir / "latest.pt"
    start = 0
    if latest.exists():
        old = torch.load(latest, map_location="cpu")
        if any((old.get("variant") != args.variant, old.get("split_sha256") != SPLIT_SHA,
                old.get("adapted_encoder_sha256") != (ADAPTED_SHA if args.variant == "adapted_byol" else None),
                old.get("seed") != args.seed, old.get("lr") != args.lr)):
            raise ValueError("Resume checkpoint provenance mismatch")
        model.model.llama_proj.load_state_dict(old["adapter"], strict=True)
        optimizer.load_state_dict(old["optimizer"])
        random.setstate(old["python_random_state"])
        torch.set_rng_state(old["torch_random_state"])
        torch.cuda.set_rng_state_all(old["cuda_random_states"])
        start = old["step"]
        print(json.dumps({"resume_step": start, "variant": args.variant}), flush=True)
    labels = tuple(QUESTIONS)
    start_time = time.monotonic()
    window_loss = 0.0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for step in range(start + 1, args.steps + 1):
        finding = labels[(step - 1) % len(labels)]
        wanted = ((step - 1) // len(labels)) % 2
        patient = random.choice(eligible_patients[finding][wanted])
        row = random.choice(by_finding_label[finding][wanted][patient])
        image = load_image(model, args.train_images, row).unsqueeze(0)
        samples = {"Image": image, "Question": [QUESTIONS[finding]], "Answer": [""]}
        optimizer.zero_grad(set_to_none=True)
        logits, _ = pre_answer_forward(model.model, samples)
        target = torch.tensor([int(row[finding])], dtype=torch.long, device=logits.device)
        loss = answer_loss(logits, target, absent_id, present_id)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        window_loss += float(loss.detach())
        if step % args.save_every == 0 or step == args.steps:
            payload = {
                "format": "retinavlm_olives_adapter_alignment_v1", "variant": args.variant,
                "step": step, "seed": args.seed, "lr": args.lr, "split_sha256": SPLIT_SHA,
                "adapted_encoder_sha256": ADAPTED_SHA if args.variant == "adapted_byol" else None,
                "source_model_dir": str(args.model_dir), "train_images": len(train),
                "train_patients": len(patients), "trainable_parameters": sum(p.numel() for p in parameters),
                "adapter": {k: v.detach().cpu() for k, v in model.model.llama_proj.state_dict().items()},
                "optimizer": optimizer.state_dict(), "python_random_state": random.getstate(),
                "torch_random_state": torch.get_rng_state(), "cuda_random_states": torch.cuda.get_rng_state_all(),
            }
            save_checkpoint(latest, payload)
            print(json.dumps({"step": step, "variant": args.variant,
                              "mean_loss_window": window_loss / (step - start if step < args.save_every else args.save_every),
                              "elapsed_seconds": time.monotonic() - start_time, "checkpoint": str(latest)}), flush=True)
            window_loss = 0.0
    final = args.out_dir / "adapter_final.pt"
    save_checkpoint(final, {k: v for k, v in payload.items() if k not in {
        "optimizer", "python_random_state", "torch_random_state", "cuda_random_states"}})
    receipt = {"status": "adapter_alignment_complete", "variant": args.variant,
               "step": args.steps, "seed": args.seed, "lr": args.lr,
               "train_images": len(train), "train_patients": len(patients),
               "sampling": "round-robin finding, alternating label, uniform patient within finding and label",
               "frozen_encoder": True, "frozen_llama": True, "trainable_adapter_parameters": payload["trainable_parameters"],
               "image_preprocessing": "L/RGB input, convert to L, authors full-frame 192x192 resize and 0-1 scaling",
               "split_sha256": SPLIT_SHA, "adapter_checkpoint": str(final), "adapter_sha256": sha256(final),
               "gpu": torch.cuda.get_device_name(0), "elapsed_seconds": time.monotonic() - start_time}
    (args.out_dir / "training_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", required=True, choices=["released", "adapted_byol"])
    p.add_argument("--split-csv", type=Path, required=True)
    p.add_argument("--train-images", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--authors-code", type=Path, required=True)
    p.add_argument("--adapted-checkpoint", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-5)
    run(p.parse_args())
