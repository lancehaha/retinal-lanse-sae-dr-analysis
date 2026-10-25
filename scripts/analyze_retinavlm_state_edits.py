"""Compare fluid-direction and orthogonal decoder edits on verified OCT cases.

Uses a deterministic sample of correct positives, false positives, false
negatives, and correct negatives from one completed prediction run. Test use
requires the frozen validation lock. This is an intervention on decoder state,
not evidence of lesion location or a complete reasoning trace.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import torch

from fit_retinavlm_fluid_lens_candidate import QUESTION, author_prompt, tensor_for_row
from predict_retinavlm_fluid_split import load_adapter_checkpoint, verify_selection
from retinavlm_binary_answer import answer_token_ids, post_image_segment
from retinavlm_released_model import SPECIALIST_REVISION, load_released_specialist
from retinavlm_state_edits import compare_state_edits
from retinavlm_verified_images import load_verified_manifest, select_split, sha256


CATEGORIES = ("correct_positive", "false_positive", "false_negative", "correct_negative")


def categorized_predictions(prediction_csv: Path, expected: dict[str, dict[str, str]],
                            arm: str, seed: int) -> dict[str, list[tuple[dict[str, str], str]]]:
    with prediction_csv.open(newline="", encoding="utf-8") as file:
        predictions = list(csv.DictReader(file))
    if len(predictions) != len(expected):
        raise ValueError("Mechanism input does not contain the complete split")
    categories = {name: [] for name in CATEGORIES}
    seen = set()
    for prediction in predictions:
        path = prediction["source_image_path"]
        if (path in seen or path not in expected
                or prediction["image_sha256"] != expected[path]["sha256"]
                or prediction["arm"] != arm or prediction["seed"] != str(seed)
                or prediction["answer"] not in {"present", "absent"}):
            raise ValueError(f"Invalid or repeated mechanism prediction: {path}")
        seen.add(path)
        row = expected[path]
        label = int(row["fluid"])
        answer = prediction["answer"]
        if label and answer == "present":
            category = "correct_positive"
        elif not label and answer == "present":
            category = "false_positive"
        elif label and answer == "absent":
            category = "false_negative"
        else:
            category = "correct_negative"
        categories[category].append((row, answer))
    if seen != set(expected):
        raise ValueError("Mechanism input has missing split predictions")
    for group in categories.values():
        group.sort(key=lambda item: item[0]["source_image_path"])
    return categories


def run(*, model_dir: Path, authors_code: Path, manifest: Path, image_audit: Path,
        image_root: Path, lens_receipt: Path, prediction_run_dir: Path,
        arm: str, seed: int, split: str, checkpoint: Path | None,
        selection_lock: Path | None, output: Path,
        max_per_category: int = 8, control_draws: int = 4,
        strength_fraction: float = 0.05) -> dict:
    if output.exists():
        raise FileExistsError("State-edit analysis receipt already exists")
    if arm not in {"A", "D"} or split not in {"validation", "test"}:
        raise ValueError("State edits compare released A or J-Lens D on validation/locked test")
    if max_per_category < 1 or control_draws < 1 or not math.isfinite(strength_fraction):
        raise ValueError("State-edit sample count and strength are invalid")
    all_rows = load_verified_manifest(manifest, image_audit)
    manifest_digest = sha256(manifest)
    image_audit_digest = sha256(image_audit)
    checkpoint_digest = verify_selection(arm, seed, checkpoint, selection_lock,
                                         manifest_digest, image_audit_digest, split)
    split_rows = select_split(all_rows, split, manifest_digest=manifest_digest,
                              selection_lock=selection_lock)
    expected = {row["source_image_path"]: row for row in split_rows}
    lens = json.loads(lens_receipt.read_text(encoding="utf-8"))
    if (lens.get("status") != "real_model_lens_validated"
            or lens.get("readout_mode") != "answer"
            or lens.get("question") != QUESTION
            or lens.get("manifest_sha256") != manifest_digest
            or lens.get("image_audit_sha256") != image_audit_digest
            or lens.get("checkpoint_revision") != SPECIALIST_REVISION
            or lens.get("image_signal_gate") is not True
            or lens.get("jacobian_submatrix_gate") is not True
            or lens.get("adapter_gradient_gate") is not True):
        raise ValueError("State-edit direction has not passed real-model lens gates")
    layer = int(lens["layer_index"])
    direction = torch.tensor(lens["direction"], dtype=torch.float32)
    orientation = 1.0 if float(lens["scale"]) > 0 else -1.0
    prediction_csv = prediction_run_dir / "predictions.csv"
    prediction_receipt_path = prediction_run_dir / "prediction_receipt.json"
    receipt = json.loads(prediction_receipt_path.read_text(encoding="utf-8"))
    if (receipt.get("status") != "complete_verified_split_predictions"
            or receipt.get("arm") != arm or receipt.get("seed") != seed
            or receipt.get("split") != split
            or receipt.get("manifest_sha256") != manifest_digest
            or receipt.get("image_audit_sha256") != image_audit_digest
            or receipt.get("predictions_sha256") != sha256(prediction_csv)
            or receipt.get("checkpoint_sha256") != checkpoint_digest
            or (split == "test" and receipt.get("selection_lock_sha256") != sha256(selection_lock))):
        raise ValueError("Mechanism prediction receipt does not match selected model and split")
    categories = categorized_predictions(prediction_csv, expected, arm, seed)
    model, _ = load_released_specialist(model_dir, authors_code)
    if checkpoint is not None:
        checkpoint_ids = load_adapter_checkpoint(checkpoint, arm, seed, manifest_digest,
                                                 image_audit_digest, model)
    else:
        checkpoint_ids = None
    model.eval()
    token_ids = answer_token_ids(model.model.get_tokenizer(),
                                 post_image_segment(author_prompt(model.model, QUESTION)))
    if checkpoint_ids is not None and checkpoint_ids != {"absent": token_ids[0], "present": token_ids[1]}:
        raise ValueError("Mechanism checkpoint and released tokenizer disagree")
    if receipt.get("answer_token_ids") != {"absent": token_ids[0], "present": token_ids[1]}:
        raise ValueError("Mechanism prediction receipt used different answer-token IDs")

    results = []
    for category in CATEGORIES:
        for position, (row, answer) in enumerate(categories[category][:max_per_category]):
            tensor = tensor_for_row(model, row, image_root)
            def make_samples():
                return {"Image": tensor, "Question": [QUESTION], "Answer": [""]}
            control_seeds = [seed * 100000 + CATEGORIES.index(category) * 10000
                             + position * 100 + draw for draw in range(control_draws)]
            effects = compare_state_edits(
                model.model, make_samples, layer, direction, token_ids[0], token_ids[1],
                strength_fraction=strength_fraction, control_seeds=control_seeds,
                orientation=orientation,
            )
            reproduced = "present" if effects["base_present_minus_absent"] >= 0 else "absent"
            if reproduced != answer:
                raise ValueError(f"Base prediction changed during state-edit run: {row['source_image_path']}")
            results.append({"source_image_path": row["source_image_path"],
                            "image_sha256": row["sha256"],
                            "Patient_ID": row["Patient_ID"], "trial": row["trial"],
                            "expert_fluid": int(row["fluid"]), "answer": answer,
                            "category": category, "control_seeds": control_seeds,
                            **effects})
    summary = {}
    for category in CATEGORIES:
        group = [row for row in results if row["category"] == category]
        summary[category] = {
            "available": len(categories[category]), "analyzed": len(group),
            "median_target_margin_change": (statistics.median(row["target_margin_change"] for row in group)
                                            if group else None),
            "median_control_absolute_margin_change": (
                statistics.median(abs(change) for row in group for change in row["control_margin_changes"])
                if group else None),
        }
    result = {"status": "completed_reversible_decoder_state_edit_analysis",
              "arm": arm, "seed": seed, "split": split,
              "manifest_sha256": manifest_digest,
              "image_audit_sha256": image_audit_digest,
              "lens_receipt_sha256": sha256(lens_receipt),
              "prediction_receipt_sha256": sha256(prediction_receipt_path),
              "checkpoint_sha256": checkpoint_digest,
              "selection_lock_sha256": sha256(selection_lock) if selection_lock else None,
              "layer_index": layer, "strength_fraction": strength_fraction,
              "control_draws": control_draws,
              "max_per_category": max_per_category,
              "summary": summary, "cases": results,
              "claim_boundary": "state edit effect on answer only; no lesion location or complete reasoning reconstruction",
              "test_split_accessed": split == "test",
              "gpu_name": torch.cuda.get_device_name(0),
              "torch_version": torch.__version__, "cuda_version": torch.version.cuda}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--authors-code", type=Path, required=True)
    parser.add_argument("--official-image-manifest", type=Path, required=True)
    parser.add_argument("--image-audit", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--lens-receipt", type=Path, required=True)
    parser.add_argument("--prediction-run-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=["A", "D"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", choices=["validation", "test"], required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--selection-lock", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-per-category", type=int, default=8)
    parser.add_argument("--control-draws", type=int, default=4)
    parser.add_argument("--strength-fraction", type=float, default=0.05)
    args = parser.parse_args()
    print(json.dumps(run(
        model_dir=args.model_dir, authors_code=args.authors_code,
        manifest=args.official_image_manifest, image_audit=args.image_audit,
        image_root=args.image_root, lens_receipt=args.lens_receipt,
        prediction_run_dir=args.prediction_run_dir, arm=args.arm, seed=args.seed,
        split=args.split, checkpoint=args.checkpoint, selection_lock=args.selection_lock,
        output=args.output, max_per_category=args.max_per_category,
        control_draws=args.control_draws, strength_fraction=args.strength_fraction,
    ), indent=2))
