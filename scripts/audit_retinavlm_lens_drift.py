"""Measure fixed answer-prompt lens drift after D adapter training.

The decoder and J-Lens direction stay frozen in this phase. This audit checks
whether adapter updates move validation states or change the calibrated lens
signal. It does not select a checkpoint or read test patients.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch

from fit_retinavlm_fluid_lens_candidate import (
    CANDIDATE_STATUSES, QUESTION, score_one_image, tensor_for_row,
)
from predict_retinavlm_fluid_split import load_adapter_checkpoint
from retinavlm_released_model import SPECIALIST_REVISION, load_released_specialist
from retinavlm_verified_images import load_verified_manifest, select_split, sha256
from validate_retinavlm_fluid_lens import auc, percentile


def probability(logit: float) -> float:
    if logit >= 0:
        return 1 / (1 + math.exp(-logit))
    exponential = math.exp(logit)
    return exponential / (1 + exponential)


def paired_patient_auc_delta(rows: list[dict], replicates: int, seed: int) -> dict:
    if replicates < 1:
        raise ValueError("At least one patient bootstrap replicate is required")
    by_patient = defaultdict(list)
    trial_patients = defaultdict(set)
    for row in rows:
        key = (row["trial"], row["Patient_ID"])
        by_patient[key].append(row)
        trial_patients[row["trial"]].add(key)
    rng = random.Random(seed)
    values = []
    for _ in range(replicates):
        sampled = []
        for trial in sorted(trial_patients):
            patients = sorted(trial_patients[trial])
            for _ in patients:
                sampled.extend(by_patient[rng.choice(patients)])
        labels = {row["fluid"] for row in sampled}
        if len(labels) != 2:
            continue
        before = auc([(row["fluid"], row["before_logit"]) for row in sampled])
        after = auc([(row["fluid"], row["after_logit"]) for row in sampled])
        values.append(after - before)
    return {
        "ci95": [percentile(values, 0.025), percentile(values, 0.975)] if values else None,
        "valid_replicates": len(values), "requested_replicates": replicates,
        "trial_patient_counts": {trial: len(group) for trial, group in sorted(trial_patients.items())},
    }


def verify_inputs(manifest: Path, image_audit: Path, candidate_path: Path,
                  lens_path: Path, checkpoint_path: Path) -> tuple[list[dict[str, str]], dict, dict]:
    rows = load_verified_manifest(manifest, image_audit)
    validation_rows = sorted(select_split(rows, "validation"),
                             key=lambda row: row["source_image_path"])
    manifest_digest, audit_digest = sha256(manifest), sha256(image_audit)
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    lens = json.loads(lens_path.read_text(encoding="utf-8"))
    layer = lens.get("layer_index")
    key = str(layer)
    if (lens.get("status") != "real_model_lens_validated"
            or lens.get("readout_mode") != "answer" or lens.get("question") != QUESTION
            or lens.get("checkpoint_revision") != SPECIALIST_REVISION
            or lens.get("manifest_sha256") != manifest_digest
            or lens.get("image_audit_sha256") != audit_digest
            or lens.get("candidate_sha256") != sha256(candidate_path)
            or lens.get("image_signal_gate") is not True
            or lens.get("jacobian_submatrix_gate") is not True
            or lens.get("adapter_gradient_gate") is not True
            or not isinstance(layer, int) or layer < 0
            or candidate.get("status") != CANDIDATE_STATUSES["answer"]
            or candidate.get("readout_mode") != "answer"
            or candidate.get("question") != QUESTION
            or candidate.get("checkpoint_revision") != SPECIALIST_REVISION
            or candidate.get("manifest_sha256") != manifest_digest
            or candidate.get("image_audit_sha256") != audit_digest
            or candidate.get("test_split_accessed") is not False
            or key not in candidate.get("candidates", {})):
        raise ValueError("Lens and candidate are not bound to this official validation cohort")
    fitted = candidate["candidates"][key]
    if (fitted.get("direction") != lens.get("direction")
            or fitted.get("scale") != lens.get("scale")
            or fitted.get("intercept") != lens.get("intercept")):
        raise ValueError("Validated lens differs from fitted candidate")
    original_scores = candidate.get("validation_score_rows")
    expected = {row["source_image_path"]: row for row in validation_rows}
    if (not isinstance(original_scores, list) or len(original_scores) != len(expected)
            or {row["source_image_path"] for row in original_scores} != set(expected)):
        raise ValueError("Candidate lacks complete original validation scores")
    for row in original_scores:
        gold = expected[row["source_image_path"]]
        if (row["Patient_ID"] != gold["Patient_ID"] or row["trial"] != gold["trial"]
                or row["fluid"] != int(gold["fluid"])
                or key not in row["scores"]
                or not math.isfinite(float(row["scores"][key]))):
            raise ValueError("Original lens scores differ from expert validation rows")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    training_path = checkpoint_path.parent.parent / "training_receipt.json"
    training = json.loads(training_path.read_text(encoding="utf-8"))
    if (checkpoint.get("arm") != "D"
            or checkpoint.get("manifest_sha256") != manifest_digest
            or checkpoint.get("image_audit_sha256") != audit_digest
            or checkpoint.get("specialist_revision") != SPECIALIST_REVISION
            or checkpoint.get("lens_receipt_sha256") != sha256(lens_path)
            or not isinstance(checkpoint.get("seed"), int)
            or not isinstance(checkpoint.get("epoch"), int)
            or training.get("status") != "completed_matched_arm_training_and_validation"
            or training.get("arm") != "D"
            or training.get("seed") != checkpoint.get("seed")
            or training.get("manifest_sha256") != manifest_digest
            or training.get("image_audit_sha256") != audit_digest
            or training.get("lens_receipt_sha256") != sha256(lens_path)
            or training.get("layer_index") != layer
            or checkpoint.get("epoch") not in [item["epoch"] for item in training.get("history", [])]):
        raise ValueError("D checkpoint or training receipt differs from the validated lens")
    return validation_rows, candidate, lens


def run(*, model_dir: Path, authors_code: Path, manifest: Path, image_audit: Path,
        image_root: Path, candidate_path: Path, lens_path: Path,
        checkpoint_path: Path, output: Path, bootstrap_replicates: int = 2000) -> dict:
    if output.exists():
        raise FileExistsError("Lens drift audit already exists")
    validation_rows, candidate, lens = verify_inputs(
        manifest, image_audit, candidate_path, lens_path, checkpoint_path
    )
    layer = lens["layer_index"]
    key = str(layer)
    direction = torch.tensor(lens["direction"], dtype=torch.float32)
    scale, intercept = float(lens["scale"]), float(lens["intercept"])
    original = {row["source_image_path"]: row for row in candidate["validation_score_rows"]}
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model, _ = load_released_specialist(model_dir, authors_code)
    load_adapter_checkpoint(checkpoint_path, "D", checkpoint["seed"],
                            sha256(manifest), sha256(image_audit), model)
    model.eval()
    score_rows = []
    with torch.inference_mode():
        for row in validation_rows:
            source = row["source_image_path"]
            before = scale * float(original[source]["scores"][key]) + intercept
            after_raw = score_one_image(model.model, tensor_for_row(model, row, image_root),
                                        {layer: direction}, QUESTION)[layer]
            after = scale * after_raw + intercept
            if not all(map(math.isfinite, (before, after))):
                raise FloatingPointError("D lens score drift is nonfinite")
            score_rows.append({"source_image_path": source, "Patient_ID": row["Patient_ID"],
                               "trial": row["trial"], "fluid": int(row["fluid"]),
                               "before_logit": before, "after_logit": after})
    before_auc = auc([(row["fluid"], row["before_logit"]) for row in score_rows])
    after_auc = auc([(row["fluid"], row["after_logit"]) for row in score_rows])
    if abs(before_auc - float(lens["validation_auc"])) > 1e-6:
        raise ValueError("Pretraining validation AUC differs from validated lens receipt")
    result = {
        "status": "adapter_training_lens_distribution_drift_assessed",
        "manifest_sha256": sha256(manifest),
        "image_audit_sha256": sha256(image_audit),
        "candidate_sha256": sha256(candidate_path),
        "lens_receipt_sha256": sha256(lens_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "arm": "D", "seed": checkpoint["seed"], "epoch": checkpoint["epoch"],
        "question": QUESTION, "layer_index": layer,
        "validation_scans": len(score_rows),
        "before_auc": before_auc, "after_auc": after_auc,
        "after_minus_before_auc": after_auc - before_auc,
        "paired_patient_bootstrap_auc_delta": paired_patient_auc_delta(
            score_rows, bootstrap_replicates, checkpoint["seed"]
        ),
        "mean_absolute_logit_change": sum(abs(row["after_logit"] - row["before_logit"])
                                          for row in score_rows) / len(score_rows),
        "before_brier": sum((probability(row["before_logit"]) - row["fluid"]) ** 2
                            for row in score_rows) / len(score_rows),
        "after_brier": sum((probability(row["after_logit"]) - row["fluid"]) ** 2
                           for row in score_rows) / len(score_rows),
        "score_rows": score_rows,
        "test_split_accessed": False,
        "interpretation_boundary": "Adapter-state distribution and fixed-calibration diagnostic only",
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
    }
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
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--lens-receipt", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    result = run(model_dir=args.model_dir, authors_code=args.authors_code,
                 manifest=args.official_image_manifest, image_audit=args.image_audit,
                 image_root=args.image_root, candidate_path=args.candidate,
                 lens_path=args.lens_receipt, checkpoint_path=args.checkpoint,
                 output=args.output, bootstrap_replicates=args.bootstrap_replicates)
    print(json.dumps({key: value for key, value in result.items() if key != "score_rows"}, indent=2))
