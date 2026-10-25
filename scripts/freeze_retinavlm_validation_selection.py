"""Freeze validation-selected checkpoints before RetinaVLM test inference.

The selection input names one B, C, and D checkpoint per seed and the real
validated D lens receipt. This script verifies their run receipts, selected
validation predictions, and recall floor against the combined validation
evaluation. It never reads test images or predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch

from fit_retinavlm_fluid_lens_candidate import QUESTION
from retinavlm_released_model import SPECIALIST_REVISION
from retinavlm_verified_images import load_verified_manifest, sha256


def prediction_map(path: Path) -> dict[tuple[str, str, str], tuple[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    required = {"source_image_path", "image_sha256", "arm", "seed", "answer"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Incomplete validation predictions: {path}")
    result = {}
    for row in rows:
        key = (row["arm"], row["seed"], row["source_image_path"])
        if key in result or row["answer"] not in {"present", "absent"}:
            raise ValueError(f"Duplicate or invalid validation prediction: {key}")
        result[key] = (row["image_sha256"], row["answer"])
    return result


def run(*, manifest: Path, image_audit: Path, validation_evaluation: Path,
        selection_input: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("Validation selection lock already exists")
    rows = load_verified_manifest(manifest, image_audit)
    manifest_digest = sha256(manifest)
    image_audit_digest = sha256(image_audit)
    expected_train = sum(row["split"] == "train" for row in rows)
    expected_validation = sum(row["split"] == "validation" for row in rows)
    evaluation = json.loads(validation_evaluation.read_text(encoding="utf-8"))
    prediction_file = Path(evaluation.get("predictions_file", ""))
    if (evaluation.get("split") != "validation"
            or evaluation.get("manifest_sha256") != manifest_digest
            or evaluation.get("image_audit_sha256") != image_audit_digest
            or evaluation.get("scans") != expected_validation
            or not prediction_file.is_file()
            or evaluation.get("predictions_sha256") != sha256(prediction_file)):
        raise ValueError("Validation evaluation is not bound to these complete predictions")
    combined_predictions = prediction_map(prediction_file)
    selection = json.loads(selection_input.read_text(encoding="utf-8"))
    tolerance = selection.get("max_recall_drop")
    if (not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool)
            or not math.isfinite(tolerance) or not 0 <= tolerance <= 1):
        raise ValueError("Selection must predeclare a finite maximum recall drop")
    chosen = selection.get("checkpoints")
    lens_paths = selection.get("lens_receipts")
    if not isinstance(chosen, dict) or not isinstance(lens_paths, dict) or not chosen:
        raise ValueError("Selection must name checkpoints and D lens receipts")
    keys = set(chosen)
    seeds = {key.split("/", 1)[1] for key in keys if key.startswith("B/")}
    if not seeds or any(not seed.isdigit() for seed in seeds):
        raise ValueError("Selection has no valid matched seeds")
    expected_keys = {f"{arm}/{seed}" for seed in seeds for arm in ("B", "C", "D")}
    if keys != expected_keys or set(lens_paths) != {f"D/{seed}" for seed in seeds}:
        raise ValueError("Each selected seed needs exactly B/C/D and one D lens receipt")
    metrics = evaluation.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("Validation metrics are missing")
    for seed in seeds:
        if f"A/{seed}" not in metrics:
            raise ValueError(f"Released A/{seed} validation predictions are missing")
        d_recall = metrics[f"D/{seed}"]["recall"]
        if d_recall is None:
            raise ValueError(f"D/{seed} has undefined validation recall")
        for baseline in ("B", "C"):
            control_recall = metrics[f"{baseline}/{seed}"]["recall"]
            if control_recall is None or d_recall < control_recall - tolerance:
                raise ValueError(f"D/{seed} violates the validation recall floor versus {baseline}")

    selected_hashes = {}
    layers = {}
    weights = {}
    for key in sorted(expected_keys):
        arm, seed_text = key.split("/")
        path = Path(chosen[key])
        if not path.is_file():
            raise FileNotFoundError(f"Selected checkpoint is missing: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        receipt_path = path.parent.parent / "training_receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (checkpoint.get("arm") != arm or checkpoint.get("seed") != int(seed_text)
                or checkpoint.get("manifest_sha256") != manifest_digest
                or checkpoint.get("image_audit_sha256") != image_audit_digest
                or checkpoint.get("specialist_revision") != SPECIALIST_REVISION
                or receipt.get("status") != "completed_matched_arm_training_and_validation"
                or receipt.get("arm") != arm or receipt.get("seed") != int(seed_text)
                or receipt.get("manifest_sha256") != manifest_digest
                or receipt.get("image_audit_sha256") != image_audit_digest
                or receipt.get("train_scans") != expected_train
                or receipt.get("validation_scans") != expected_validation
                or checkpoint.get("epoch") not in [record["epoch"] for record in receipt.get("history", [])]):
            raise ValueError(f"Selected {key} checkpoint or training receipt does not match")
        if (arm == "C") != (checkpoint.get("probe") is not None):
            raise ValueError(f"Selected {key} probe state does not match arm")
        selected_predictions = prediction_map(path.parent / "validation_predictions.csv")
        subset = {entry: value for entry, value in combined_predictions.items()
                  if entry[0] == arm and entry[1] == seed_text}
        if selected_predictions != subset or len(subset) != expected_validation:
            raise ValueError(f"Selected {key} epoch predictions differ from validation evaluation")
        if arm == "D":
            lens_path = Path(lens_paths[key])
            lens = json.loads(lens_path.read_text(encoding="utf-8"))
            if (lens.get("status") != "real_model_lens_validated"
                    or lens.get("readout_mode") != "answer"
                    or lens.get("question") != QUESTION
                    or lens.get("manifest_sha256") != manifest_digest
                    or lens.get("image_audit_sha256") != image_audit_digest
                    or lens.get("checkpoint_revision") != SPECIALIST_REVISION
                    or lens.get("image_signal_gate") is not True
                    or lens.get("jacobian_submatrix_gate") is not True
                    or lens.get("adapter_gradient_gate") is not True
                    or checkpoint.get("lens_receipt_sha256") != sha256(lens_path)
                    or lens.get("layer_index") != receipt.get("layer_index")):
                raise ValueError(f"Selected {key} has no matching validated lens")
        selected_hashes[key] = sha256(path)
        layers[key] = receipt.get("layer_index")
        weights[key] = receipt.get("auxiliary_weight")

    lock = {
        "status": "validation_selection_frozen",
        "manifest_sha256": manifest_digest,
        "image_audit_sha256": image_audit_digest,
        "validation_evaluation_sha256": sha256(validation_evaluation),
        "validation_predictions_sha256": sha256(prediction_file),
        "selection_input_sha256": sha256(selection_input),
        "selected_configuration": {
            "released_revision": SPECIALIST_REVISION,
            "max_recall_drop": float(tolerance),
            "checkpoints": selected_hashes,
            "layer_indices": layers,
            "auxiliary_weights": weights,
            "selection_rationale": selection.get("selection_rationale", ""),
        },
        "matched_seeds": sorted(int(seed) for seed in seeds),
        "test_split_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return lock


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-image-manifest", type=Path, required=True)
    parser.add_argument("--image-audit", type=Path, required=True)
    parser.add_argument("--validation-evaluation", type=Path, required=True)
    parser.add_argument("--selection-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(manifest=args.official_image_manifest,
                         image_audit=args.image_audit,
                         validation_evaluation=args.validation_evaluation,
                         selection_input=args.selection_input,
                         output=args.output), indent=2))
