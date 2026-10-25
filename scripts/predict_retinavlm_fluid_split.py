"""Predict the fixed fluid answer for every verified validation/test scan.

Test inference requires a manifest-bound validation-selection lock, including
the selected checkpoint hash for B/C/D. The released A arm uses the pinned
Specialist checkpoint directly. Each output includes image hashes for the
matched-arm evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import torch

from retinavlm_binary_answer import answer_token_ids, post_image_segment, predict_answer
from retinavlm_pre_answer import pre_answer_forward
from retinavlm_released_model import SPECIALIST_REVISION, load_released_specialist
from retinavlm_verified_images import load_verified_manifest, open_verified_image, select_split, sha256


QUESTION = "Is intraretinal or subretinal fluid present in this OCT scan? Answer present or absent."


def verify_selection(arm: str, seed: int, checkpoint: Path | None,
                     selection_lock: Path | None, manifest_digest: str,
                     image_audit_digest: str, split: str) -> str | None:
    if arm not in {"A", "B", "C", "D"} or seed < 0:
        raise ValueError("Expected A/B/C/D and a nonnegative seed")
    if (arm == "A") != (checkpoint is None):
        raise ValueError("A uses the released model; B/C/D require a trained checkpoint")
    checkpoint_digest = sha256(checkpoint) if checkpoint is not None else None
    if split != "test":
        return checkpoint_digest
    if selection_lock is None:
        raise ValueError("Test predictions require a validation-selection lock")
    lock = json.loads(selection_lock.read_text(encoding="utf-8"))
    selected = lock.get("selected_configuration")
    if (lock.get("status") != "validation_selection_frozen"
            or lock.get("manifest_sha256") != manifest_digest
            or lock.get("image_audit_sha256") != image_audit_digest
            or not isinstance(selected, dict)
            or selected.get("released_revision") != SPECIALIST_REVISION):
        raise ValueError("Test lock does not match this manifest and released model")
    if arm != "A":
        checkpoints = selected.get("checkpoints")
        if not isinstance(checkpoints, dict) or checkpoints.get(f"{arm}/{seed}") != checkpoint_digest:
            raise ValueError(f"{arm}/{seed} checkpoint was not frozen on validation")
    return checkpoint_digest


def load_adapter_checkpoint(checkpoint: Path, arm: str, seed: int,
                            manifest_digest: str, image_audit_digest: str,
                            model) -> dict[str, int]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (payload.get("arm") != arm or payload.get("seed") != seed
            or payload.get("manifest_sha256") != manifest_digest
            or payload.get("image_audit_sha256") != image_audit_digest
            or payload.get("specialist_revision") != SPECIALIST_REVISION
            or not isinstance(payload.get("epoch"), int)
            or payload["epoch"] < 1):
        raise ValueError("Adapter checkpoint metadata does not match prediction request")
    token_ids = payload.get("answer_token_ids")
    if (not isinstance(token_ids, dict) or not all(isinstance(token_ids.get(k), int)
            for k in ("absent", "present"))):
        raise ValueError("Adapter checkpoint has no verified answer-token IDs")
    model.model.llama_proj.load_state_dict(payload["adapter"], strict=True)
    return token_ids


def run(*, arm: str, seed: int, split: str, checkpoint: Path | None,
        selection_lock: Path | None, model_dir: Path, authors_code: Path,
        manifest: Path, image_audit: Path, image_root: Path,
        output_dir: Path) -> dict:
    if split not in {"validation", "test"}:
        raise ValueError("Predictions must use validation or locked test")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Output directory must be empty to avoid mixing predictions")
    rows = load_verified_manifest(manifest, image_audit)
    manifest_digest = sha256(manifest)
    image_audit_digest = sha256(image_audit)
    checkpoint_digest = verify_selection(
        arm, seed, checkpoint, selection_lock, manifest_digest, image_audit_digest, split
    )
    selected = sorted(select_split(rows, split, manifest_digest=manifest_digest,
                                   selection_lock=selection_lock),
                      key=lambda row: row["source_image_path"])
    model, _ = load_released_specialist(model_dir, authors_code)
    expected_ids = (load_adapter_checkpoint(checkpoint, arm, seed, manifest_digest,
                                            image_audit_digest, model)
                    if checkpoint is not None else None)
    model.eval()
    predictions = []
    actual_ids = None
    with torch.inference_mode():
        for row in selected:
            image = open_verified_image(row, image_root)
            if image.mode != "L":
                raise ValueError(f"OCT image mode {image.mode!r} needs a documented preprocessing decision")
            tensor = model.convert_any_image_to_normalized_tensor(image).unsqueeze(0).to(model.model.device)
            samples = {"Image": tensor, "Question": [QUESTION], "Answer": [""]}
            logits, _ = pre_answer_forward(model.model, samples)
            if actual_ids is None:
                absent_id, present_id = answer_token_ids(
                    model.model.get_tokenizer(), post_image_segment(samples["Input"][0])
                )
                actual_ids = {"absent": absent_id, "present": present_id}
                if expected_ids is not None and actual_ids != expected_ids:
                    raise ValueError("Checkpoint and released tokenizer disagree on answer tokens")
            answer, _ = predict_answer(logits, actual_ids["absent"], actual_ids["present"])
            predictions.append({"source_image_path": row["source_image_path"],
                                "image_sha256": row["sha256"], "arm": arm,
                                "seed": str(seed),
                                "answer": "present" if answer.item() else "absent"})
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "predictions.csv"
    temporary = csv_path.with_suffix(".writing")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["source_image_path", "image_sha256", "arm", "seed", "answer"])
        writer.writeheader()
        writer.writerows(predictions)
    os.replace(temporary, csv_path)
    receipt = {"status": "complete_verified_split_predictions", "arm": arm,
               "seed": seed, "split": split, "scans": len(predictions),
               "manifest_sha256": manifest_digest, "image_audit_sha256": image_audit_digest,
               "checkpoint_sha256": checkpoint_digest,
               "selection_lock_sha256": sha256(selection_lock) if selection_lock else None,
               "released_revision": SPECIALIST_REVISION,
               "answer_token_ids": actual_ids, "question": QUESTION,
               "predictions_sha256": sha256(csv_path),
               "gpu_name": torch.cuda.get_device_name(0),
               "torch_version": torch.__version__, "cuda_version": torch.version.cuda}
    (output_dir / "prediction_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["A", "B", "C", "D"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", choices=["validation", "test"], required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--selection-lock", type=Path)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--authors-code", type=Path, required=True)
    parser.add_argument("--official-image-manifest", type=Path, required=True)
    parser.add_argument("--image-audit", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(
        arm=args.arm, seed=args.seed, split=args.split, checkpoint=args.checkpoint,
        selection_lock=args.selection_lock, model_dir=args.model_dir,
        authors_code=args.authors_code, manifest=args.official_image_manifest,
        image_audit=args.image_audit, image_root=args.image_root,
        output_dir=args.output_dir,
    ), indent=2))
