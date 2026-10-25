"""Read-only B2 finite-difference hidden-edit precision diagnostic.

Uses only the frozen first direction-fit image. Never revises map eligibility.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from evaluate_retinavlm_adapter_olives import image_batch
from retinavlm_binary_answer import answer_token_ids
from retinavlm_pre_answer import decoder_blocks, pre_answer_forward
from retinavlm_released_model import load_released_specialist
from train_retinavlm_adapter_olives import ADAPTED_SHA, QUESTIONS, SPLIT_SHA, install_encoder, load_rows, sha256


def main():
    p = argparse.ArgumentParser()
    for key in ("split-csv", "train-images", "model-dir", "authors-code",
                "adapted-checkpoint", "adapter-checkpoint", "plan-root", "map-root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    plan_path = a.plan_root / "B2_selection_plan.json"
    summary_path = a.map_root / "B2" / "summary.json"
    plan, summary = json.loads(plan_path.read_text()), json.loads(summary_path.read_text())
    if (plan["split_sha256"] != SPLIT_SHA or plan["protocol"] != "retinavlm_jlens_train_gate_v3"
            or summary["selection_plan_sha256"] != sha256(plan_path)
            or summary["adapter_sha256"] != sha256(a.adapter_checkpoint)):
        raise ValueError("Frozen candidate provenance mismatch")
    row_map = {r["source_image_path"]: r for r in load_rows(a.split_csv)["train"]}
    path = plan["direction_paths"][0]
    if path not in row_map:
        raise ValueError("First direction image is outside original training split")
    model, _ = load_released_specialist(a.model_dir, a.authors_code)
    install_encoder(model, "adapted_byol", a.adapted_checkpoint)
    ckpt = torch.load(a.adapter_checkpoint, map_location="cpu")
    if (ckpt.get("split_sha256") != SPLIT_SHA or ckpt.get("adapted_encoder_sha256") != ADAPTED_SHA
            or ckpt.get("format") != "retinavlm_olives_adapter_alignment_v1"
            or ckpt.get("variant") != "adapted_byol" or ckpt.get("step") != 10000):
        raise ValueError("Adapter checkpoint mismatch")
    model.model.llama_proj.load_state_dict(ckpt["adapter"], strict=True)
    del ckpt
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    absent_id, present_id = answer_token_ids(model.model.get_tokenizer(), QUESTIONS["B2"])
    if summary["answer_token_ids"] != {"absent": absent_id, "present": present_id}:
        raise ValueError("Answer token mismatch")
    layer = 16
    original = summary["actual_model_gradient_checks"][str(layer)]
    coordinate = int(original["coordinate"])
    block = decoder_blocks(model.model)[layer]
    image = image_batch(model, [row_map[path]], a.train_images)
    observations = []
    for epsilon in (0.25, 1.0):
        for sign in (-1, 1):
            state = {}

            def edit(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                before = hidden[:, -1, coordinate].clone()
                updated = hidden.clone()
                updated[:, -1, coordinate] += sign * epsilon
                after = updated[:, -1, coordinate]
                state.update({"hidden_dtype": str(hidden.dtype), "before": float(before.item()),
                              "after": float(after.item()), "effective_delta": float((after - before).item())})
                return (updated, *output[1:]) if isinstance(output, tuple) else updated

            handle = block.register_forward_hook(edit)
            try:
                with torch.inference_mode():
                    logits, _ = pre_answer_forward(
                        model.model, {"Image": image, "Question": [QUESTIONS["B2"]], "Answer": [""]}
                    )
                    margin = float((logits[0, present_id] - logits[0, absent_id]).float())
            finally:
                handle.remove()
            observations.append({"requested_delta": sign * epsilon, "margin": margin, **state})
    pairs = []
    for i, epsilon in enumerate((0.25, 1.0)):
        neg, pos = observations[2 * i:2 * i + 2]
        numerator = pos["margin"] - neg["margin"]
        actual_span = pos["effective_delta"] - neg["effective_delta"]
        pairs.append({"epsilon": epsilon, "nominal_numeric_gradient": numerator / (2 * epsilon),
                      "actual_edit_span": actual_span,
                      "effective_numeric_gradient": numerator / actual_span if actual_span else None})
    result = {"status": "read_only_precision_diagnostic_no_gate_revision", "finding": "B2",
              "layer": layer, "coordinate": coordinate, "first_direction_path": path,
              "original_gradient_check": original, "observations": observations, "pairs": pairs,
              "plan_sha256": sha256(plan_path), "candidate_summary_sha256": sha256(summary_path),
              "adapter_sha256": sha256(a.adapter_checkpoint), "split_sha256": SPLIT_SHA,
              "original_validation_accessed": False, "test_accessed": False}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "output": str(a.output)}), flush=True)


if __name__ == "__main__":
    main()
