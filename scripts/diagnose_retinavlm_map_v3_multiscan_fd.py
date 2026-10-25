"""Read-only multi-scan finite-difference diagnostic for failed v3 maps.

Uses direction-fit scans only. No original validation/test access or gate revision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from evaluate_retinavlm_adapter_olives import image_batch
from fit_retinavlm_olives_jlens_map import finite_difference_check
from retinavlm_binary_answer import answer_token_ids
from retinavlm_pre_answer import decoder_blocks, pre_answer_forward
from retinavlm_projected_jacobian import answer_margin_directions_for_batch
from retinavlm_released_model import load_released_specialist
from train_retinavlm_adapter_olives import (
    ADAPTED_SHA, QUESTIONS, SPLIT_SHA, install_encoder, load_rows, sha256,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("split-csv", "train-images", "model-dir", "authors-code",
                 "adapted-checkpoint", "adapter-checkpoint", "plan-root", "map-root",
                 "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    train = {row["source_image_path"]: row
             for row in load_rows(args.split_csv)["train"]}
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
    blocks = decoder_blocks(model.model)
    result = {
        "status": "diagnostic_only_no_gate_revision",
        "split_sha256": SPLIT_SHA,
        "adapter_sha256": sha256(args.adapter_checkpoint),
        "original_validation_accessed": False,
        "test_accessed": False,
        "findings": {},
    }
    for finding, layer in (("B2", 20), ("B6", 28)):
        plan_path = args.plan_root / f"{finding}_selection_plan.json"
        summary_path = args.map_root / finding / "summary.json"
        plan = json.loads(plan_path.read_text())
        summary = json.loads(summary_path.read_text())
        if (plan.get("protocol") != "retinavlm_jlens_train_gate_v3"
                or plan.get("finding") != finding
                or plan.get("split_sha256") != SPLIT_SHA
                or summary.get("finding") != finding
                or summary.get("selection_plan_sha256") != sha256(plan_path)
                or summary.get("adapter_sha256") != sha256(args.adapter_checkpoint)):
            raise ValueError("Selection plan or candidate provenance mismatch")
        direction_paths = plan["direction_paths"]
        first = direction_paths[0]
        if first not in train:
            raise ValueError("Direction image missing from original training split")
        chosen = [first]
        for label in ("0", "1"):
            chosen += [path for path in direction_paths
                       if path != first and train[path][finding] == label][:3]
        chosen = list(dict.fromkeys(chosen))
        absent_id, present_id = answer_token_ids(
            model.model.get_tokenizer(), QUESTIONS[finding]
        )
        if summary["answer_token_ids"] != {"absent": absent_id, "present": present_id}:
            raise ValueError("Answer tokens changed")
        checks = []
        for path in chosen:
            image = image_batch(model, [train[path]], args.train_images)

            def forward():
                logits, _ = pre_answer_forward(
                    model.model,
                    {"Image": image, "Question": [QUESTIONS[finding]], "Answer": [""]},
                )
                return logits

            vector = answer_margin_directions_for_batch(
                forward, blocks, [layer], absent_id, present_id,
            )[layer][0]
            coordinate = int(vector.abs().argmax())
            analytic = float(vector[coordinate])
            values = []
            for epsilon in (0.25, 1.0):
                numeric = finite_difference_check(
                    model, image, finding, layer, coordinate,
                    absent_id, present_id, epsilon=epsilon,
                )
                tolerance = max(0.10, 0.50 * abs(analytic))
                values.append({"epsilon": epsilon, "numeric": numeric,
                               "absolute_error": abs(analytic - numeric),
                               "original_tolerance": tolerance,
                               "within_original_tolerance":
                                   abs(analytic - numeric) <= tolerance})
            checks.append({"source_image_path": path, "label": train[path][finding],
                           "coordinate": coordinate, "analytic": analytic,
                           "epsilon_checks": values})
            print(json.dumps({"finding": finding, "checked": len(checks),
                              "total": len(chosen)}), flush=True)
        result["findings"][finding] = {
            "layer": layer,
            "plan_sha256": sha256(plan_path),
            "candidate_summary_sha256": sha256(summary_path),
            "first_scan_matches_original_gate": chosen[0] == direction_paths[0],
            "checks": checks,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
