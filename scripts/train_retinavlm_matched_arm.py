"""Train one RetinaVLM B/C/D arm on all verified OLIVES training scans.

Requires the released-model feasibility gates and a CUDA host. This runner
never reads the test partition. It saves an adapter-only checkpoint and full
validation predictions after each epoch; later selection is a separate step.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from pathlib import Path

import torch

from evaluate_retinavlm_fluid_arms import scores
from retinavlm_binary_answer import answer_token_ids, post_image_segment, predict_answer
from retinavlm_matched_arm_losses import FixedLensHead, ProbeHead, adapter_parameters, arm_loss
from retinavlm_pre_answer import pre_answer_forward
from retinavlm_released_model import SPECIALIST_REVISION, load_released_specialist
from retinavlm_verified_images import load_verified_manifest, open_verified_image, select_split, sha256


QUESTION = "Is intraretinal or subretinal fluid present in this OCT scan? Answer present or absent."


def load_fixed_lens(path: Path, manifest_digest: str, image_audit_digest: str,
                    layer_index: int) -> FixedLensHead:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (payload.get("status") != "real_model_lens_validated"
            or payload.get("manifest_sha256") != manifest_digest
            or payload.get("image_audit_sha256") != image_audit_digest
            or payload.get("checkpoint_revision") != SPECIALIST_REVISION
            or payload.get("layer_index") != layer_index
            or payload.get("readout_mode") != "answer"
            or payload.get("question") != QUESTION
            or payload.get("image_signal_gate") is not True
            or payload.get("jacobian_submatrix_gate") is not True
            or payload.get("adapter_gradient_gate") is not True):
        raise ValueError("Fixed J-Lens receipt has not passed real-model gates")
    direction = payload.get("direction")
    if not isinstance(direction, list) or not direction:
        raise ValueError("Fixed J-Lens direction is missing")
    return FixedLensHead(torch.tensor(direction, dtype=torch.float32),
                         float(payload["scale"]), float(payload["intercept"]))


def tensor_for_row(model, row: dict[str, str], image_root: Path) -> torch.Tensor:
    image = open_verified_image(row, image_root)
    if image.mode != "L":
        raise ValueError(f"OCT image mode {image.mode!r} needs a documented preprocessing decision")
    return model.convert_any_image_to_normalized_tensor(image).unsqueeze(0).to(model.model.device)


def write_csv_atomic(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["source_image_path", "image_sha256", "arm", "seed", "answer"])
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def run(*, arm: str, seed: int, epochs: int, learning_rate: float,
        auxiliary_weight: float, layer_index: int | None, lens_receipt: Path | None,
        model_dir: Path, authors_code: Path, manifest: Path, image_audit: Path,
        image_root: Path, output_dir: Path) -> dict:
    if arm not in {"B", "C", "D"} or seed < 0 or epochs < 1:
        raise ValueError("Choose B/C/D, nonnegative seed, and at least one epoch")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Learning rate must be positive and finite")
    if not math.isfinite(auxiliary_weight) or auxiliary_weight < 0:
        raise ValueError("Auxiliary loss weight must be nonnegative and finite")
    if (arm == "B" and (layer_index is not None or lens_receipt is not None)
            or arm == "C" and (layer_index is None or lens_receipt is not None)
            or arm == "D" and (layer_index is None or lens_receipt is None)):
        raise ValueError("B has no auxiliary layer; C needs a layer; D needs a layer and fixed lens")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Output directory must be empty to avoid mixing runs")

    rows = load_verified_manifest(manifest, image_audit)
    train_rows = sorted(select_split(rows, "train"), key=lambda row: row["source_image_path"])
    validation_rows = sorted(select_split(rows, "validation"), key=lambda row: row["source_image_path"])
    manifest_digest = sha256(manifest)
    image_audit_digest = sha256(image_audit)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # A rejected D receipt must stop before the costly model is loaded.
    lens = (load_fixed_lens(lens_receipt, manifest_digest, image_audit_digest, layer_index)
            if arm == "D" else None)
    model, _ = load_released_specialist(model_dir, authors_code)
    adapter = adapter_parameters(model.model)
    width = model.model.llama_model.config.hidden_size
    probe = ProbeHead(width).to(model.model.device) if arm == "C" else None
    if lens is not None:
        if lens.direction.numel() != width:
            raise ValueError("J-Lens direction width differs from decoder width")
        lens = lens.to(model.model.device)
    parameters = adapter + (list(probe.parameters()) if probe is not None else [])
    trainable_count = sum(parameter.numel() for parameter in parameters)
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    model.eval()
    model.model.llama_proj.train()

    absent_id = present_id = None
    history = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        ordered = train_rows.copy()
        random.Random(seed + epoch).shuffle(ordered)
        epoch_loss = 0.0
        for row in ordered:
            tensor = tensor_for_row(model, row, image_root)
            samples = {"Image": tensor, "Question": [QUESTION], "Answer": [""]}
            logits, hidden = pre_answer_forward(model.model, samples, layer_index)
            if absent_id is None:
                absent_id, present_id = answer_token_ids(
                    model.model.get_tokenizer(), post_image_segment(samples["Input"][0])
                )
            labels = torch.tensor([int(row["fluid"])], dtype=torch.long, device=logits.device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = arm_loss(arm, logits, labels, absent_id, present_id,
                               pre_answer_hidden=hidden, probe=probe, lens=lens,
                               auxiliary_weight=auxiliary_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite {arm} loss at {row['source_image_path']}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
            optimizer.step()
            epoch_loss += float(loss.detach())

        predictions = []
        validation_pairs = []
        with torch.inference_mode():
            for row in validation_rows:
                tensor = tensor_for_row(model, row, image_root)
                samples = {"Image": tensor, "Question": [QUESTION], "Answer": [""]}
                logits, _ = pre_answer_forward(model.model, samples)
                answer, _ = predict_answer(logits, absent_id, present_id)
                prediction = int(answer.item())
                predictions.append({"source_image_path": row["source_image_path"],
                                    "image_sha256": row["sha256"], "arm": arm,
                                    "seed": str(seed), "answer": "present" if prediction else "absent"})
                validation_pairs.append((int(row["fluid"]), prediction))
        epoch_dir = output_dir / f"epoch_{epoch:03d}"
        epoch_dir.mkdir(parents=True)
        write_csv_atomic(epoch_dir / "validation_predictions.csv", predictions)
        checkpoint = {
            "arm": arm, "seed": seed, "epoch": epoch,
            "manifest_sha256": manifest_digest,
            "image_audit_sha256": image_audit_digest,
            "specialist_revision": SPECIALIST_REVISION,
            "answer_token_ids": {"absent": absent_id, "present": present_id},
            "adapter": model.model.llama_proj.state_dict(),
            "probe": probe.state_dict() if probe is not None else None,
            "lens_receipt_sha256": sha256(lens_receipt) if lens_receipt is not None else None,
            "optimizer": optimizer.state_dict(),
        }
        temporary = epoch_dir / "checkpoint.writing.pt"
        torch.save(checkpoint, temporary)
        os.replace(temporary, epoch_dir / "checkpoint.pt")
        epoch_record = {"epoch": epoch, "train_scans": len(train_rows),
                        "mean_training_loss": epoch_loss / len(train_rows),
                        "validation_scans": len(validation_rows),
                        "validation_metrics": scores(validation_pairs)}
        history.append(epoch_record)
        print(json.dumps(epoch_record), flush=True)

    result = {"status": "completed_matched_arm_training_and_validation",
              "arm": arm, "seed": seed, "epochs": epochs,
              "model_revision": SPECIALIST_REVISION,
              "manifest_sha256": manifest_digest, "image_audit_sha256": image_audit_digest,
              "train_scans": len(train_rows), "validation_scans": len(validation_rows),
              "trainable_parameters": trainable_count,
              "adapter_parameters": sum(parameter.numel() for parameter in adapter),
              "probe_parameters": sum(parameter.numel() for parameter in probe.parameters()) if probe else 0,
              "learning_rate": learning_rate, "auxiliary_weight": auxiliary_weight,
              "layer_index": layer_index,
              "lens_receipt_sha256": sha256(lens_receipt) if lens_receipt is not None else None,
              "answer_token_ids": {"absent": absent_id, "present": present_id},
              "question": QUESTION,
              "preprocessing": "authors_convert_any_image_to_normalized_tensor; requires grayscale L",
              "gpu_name": torch.cuda.get_device_name(0),
              "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
              "history": history, "test_split_accessed": False}
    (output_dir / "training_receipt.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["B", "C", "D"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--auxiliary-weight", type=float, default=1.0)
    parser.add_argument("--layer-index", type=int)
    parser.add_argument("--lens-receipt", type=Path)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--authors-code", type=Path, required=True)
    parser.add_argument("--official-image-manifest", type=Path, required=True)
    parser.add_argument("--image-audit", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(
        arm=args.arm, seed=args.seed, epochs=args.epochs,
        learning_rate=args.learning_rate, auxiliary_weight=args.auxiliary_weight,
        layer_index=args.layer_index, lens_receipt=args.lens_receipt,
        model_dir=args.model_dir, authors_code=args.authors_code,
        manifest=args.official_image_manifest, image_audit=args.image_audit,
        image_root=args.image_root, output_dir=args.output_dir,
    ), indent=2))
