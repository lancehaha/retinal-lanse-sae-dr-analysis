"""Read-only epsilon-ladder diagnostic for failed v3 map gradient checks.

This does not replace or revise any frozen map gate decision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from evaluate_retinavlm_adapter_olives import image_batch
from fit_retinavlm_olives_jlens_map import finite_difference_check
from retinavlm_binary_answer import answer_token_ids
from retinavlm_released_model import load_released_specialist
from train_retinavlm_adapter_olives import (
    ADAPTED_SHA, QUESTIONS, SPLIT_SHA, install_encoder, load_rows, sha256,
)


EPSILONS = (0.0625, 0.125, 0.25, 0.5, 1.0, 2.0)


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    groups = load_rows(args.split_csv)
    train = {row["source_image_path"]: row for row in groups["train"]}
    model, _ = load_released_specialist(args.model_dir, args.authors_code)
    install_encoder(model, "adapted_byol", args.adapted_checkpoint)
    checkpoint = torch.load(args.adapter_checkpoint, map_location="cpu")
    if (checkpoint.get("split_sha256") != SPLIT_SHA
            or checkpoint.get("adapted_encoder_sha256") != ADAPTED_SHA
            or checkpoint.get("format") != "retinavlm_olives_adapter_alignment_v1"
            or checkpoint.get("variant") != "adapted_byol"
            or checkpoint.get("step") != 10000):
        raise ValueError("Adapter checkpoint provenance mismatch")
    model.model.llama_proj.load_state_dict(checkpoint["adapter"], strict=True)
    del checkpoint
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    results = {}
    for finding in ("B2", "B6"):
        plan_path = args.plan_root / f"{finding}_selection_plan.json"
        plan = json.loads(plan_path.read_text())
        summary_path = args.map_root / finding / "summary.json"
        summary = json.loads(summary_path.read_text())
        if (plan.get("protocol") != "retinavlm_jlens_train_gate_v3"
                or plan.get("finding") != finding
                or plan.get("split_sha256") != SPLIT_SHA
                or summary.get("finding") != finding
                or summary.get("split_sha256") != SPLIT_SHA
                or summary.get("selection_plan_sha256") != sha256(plan_path)
                or summary.get("adapter_sha256") != sha256(args.adapter_checkpoint)
                or summary.get("map_eval_cohort") != "held-out original training patients"):
            raise ValueError("Map/plan provenance mismatch")
        first = plan["direction_paths"][0]
        if first not in train:
            raise ValueError("Direction scan is outside original training split")
        image = image_batch(model, [train[first]], args.train_images)
        absent_id, present_id = answer_token_ids(
            model.model.get_tokenizer(), QUESTIONS[finding]
        )
        if summary["answer_token_ids"] != {"absent": absent_id, "present": present_id}:
            raise ValueError("Answer tokens changed")
        layers = {}
        for layer in (20, 28):
            original = summary["actual_model_gradient_checks"][str(layer)]
            coordinate = int(original["coordinate"])
            analytic = float(original["analytic"])
            samples = []
            for epsilon in EPSILONS:
                numeric = finite_difference_check(
                    model, image, finding, layer, coordinate,
                    absent_id, present_id, epsilon=epsilon,
                )
                samples.append({"epsilon": epsilon, "finite_difference": numeric,
                                "absolute_error": abs(analytic - numeric),
                                "within_original_tolerance":
                                    abs(analytic - numeric) <= max(0.10, 0.50 * abs(analytic))})
            layers[str(layer)] = {"coordinate": coordinate, "analytic": analytic,
                                  "original_epsilon_0_25": original, "epsilon_ladder": samples}
            print(json.dumps({"finding": finding, "layer": layer}), flush=True)
        results[finding] = {"summary_sha256": sha256(summary_path),
                            "selection_plan_sha256": sha256(plan_path),
                            "first_direction_scan": first, "layers": layers}
    receipt = {"status": "diagnostic_only_no_gate_revision",
               "split_sha256": SPLIT_SHA, "adapter_sha256": sha256(args.adapter_checkpoint),
               "test_accessed": False, "original_validation_accessed": False,
               "findings": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": receipt["status"], "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("split-csv", "train-images", "model-dir", "authors-code",
                 "adapted-checkpoint", "adapter-checkpoint", "plan-root", "map-root",
                 "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
