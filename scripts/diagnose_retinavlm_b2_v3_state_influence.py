"""Exploratory B2 answer-state intervention on failed v3 statistical gate.

This tests output influence of the selected decoder directions with the same
image and question, using equal-norm orthogonal controls. It does not test
generated report claims or authorize J-Lens training on its own.
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

from audit_retinavlm_olives_map_candidate import sha256
from evaluate_retinavlm_adapter_olives import image_batch
from retinavlm_binary_answer import answer_token_ids
from retinavlm_released_model import load_released_specialist
from retinavlm_state_edits import compare_state_edits
from train_retinavlm_adapter_olives import (ADAPTED_SHA, QUESTIONS, SPLIT_SHA,
                                            install_encoder, load_rows)


CATEGORIES = ("true_positive", "false_positive", "false_negative", "true_negative")
DIAGNOSTIC_LAYERS = (16, 30)


def choose_cases(rows, finding, per_category):
    """Use distinct patients where available, in fixed path order."""
    pools = {category: defaultdict(list) for category in CATEGORIES}
    for row in rows:
        label = int(row["label"])
        predicted = float(row["final_answer_margin"]) >= 0
        category = CATEGORIES[(0 if predicted else 2) + (0 if label else 1)]
        pools[category][row["patient_key"]].append(row)
    selected = {}
    for category, by_patient in pools.items():
        for group in by_patient.values():
            group.sort(key=lambda item: item["source_image_path"])
        patients = sorted(by_patient)
        chosen = []
        while len(chosen) < per_category and any(by_patient.values()):
            for patient in patients:
                if by_patient[patient] and len(chosen) < per_category:
                    chosen.append(by_patient[patient].pop(0))
        selected[category] = chosen
    return selected, {key: sum(map(len, groups.values())) + len(selected[key])
                      for key, groups in pools.items()}


def run(args):
    if (args.output.exists() or args.per_category < 2 or args.controls < 3
            or args.reference_batch_size < 1):
        raise ValueError("Output exists or intervention settings are invalid")
    gate_path = args.map_dir / "statistical_gate.json"
    summary_path = args.map_dir / "summary.json"
    map_path = args.map_dir / "map_and_probe.pt"
    score_path = args.map_dir / "validation_map_scores.csv"
    gate = json.loads(gate_path.read_text())
    summary = json.loads(summary_path.read_text())
    plan = json.loads(args.selection_plan.read_text())
    if ("inference_batch_size" in summary
            and summary["inference_batch_size"] != args.reference_batch_size):
        raise ValueError("Intervention reference batch size differs from map fitting")
    if (gate.get("status") != "candidate_statistical_gate_failed"
            or gate.get("finding") != summary.get("finding")
            or gate.get("map_sha256") != sha256(map_path)
            or gate.get("summary_sha256") != sha256(summary_path)
            or gate.get("validation_scores_sha256") != sha256(score_path)
            or summary.get("split_sha256") != SPLIT_SHA
            or summary.get("adapted_encoder_sha256") != ADAPTED_SHA
            or gate.get("selection_plan_sha256") != sha256(args.selection_plan)
            or summary.get("selection_plan_sha256") != sha256(args.selection_plan)
            or gate.get("map_eval_cohort") != "held-out original training patients"
            or summary.get("map_eval_cohort") != "held-out original training patients"
            or plan.get("protocol") != "retinavlm_jlens_train_gate_v3"
            or plan.get("finding") != summary.get("finding")
            or plan.get("split_sha256") != SPLIT_SHA
            or not gate.get("selected_layers")):
        raise ValueError("Failed-gate diagnostic provenance mismatch")
    if (summary["finding"] != "B2" or gate.get("selected_layers") != [30]
            or not all(value for name, value in gate["layers"]["16"]["checks"].items()
                       if name != "finite_difference_error_within_tolerance")
            or gate["layers"]["16"]["checks"]["finite_difference_error_within_tolerance"]
            or not gate["layers"]["30"]["statistical_gate_passed"]):
        raise ValueError("Unexpected B2 failed-gate diagnostic state")
    finding = summary["finding"]
    groups = load_rows(args.split_csv)
    official = {row["source_image_path"]: row for row in groups["train"]}
    with score_path.open(newline="", encoding="utf-8") as file:
        table = list(csv.DictReader(file))
    layer = int(gate["selected_layers"][0])
    score_rows = [row for row in table if int(row["layer"]) == layer]
    if len(score_rows) != summary["validation_scans"]:
        raise ValueError("Incomplete validation score table")
    for row in score_rows:
        source = row["source_image_path"]
        if (source not in official or row["patient_key"] != official[source]["patient_key"]
                or row["label"] != official[source][finding]):
            raise ValueError("Validation label or patient mismatch")
    validation_rows = [official[path] for path in plan["gate_paths"]]
    if ([row["source_image_path"] for row in score_rows]
            != [row["source_image_path"] for row in validation_rows]):
        raise ValueError("Reference validation batch order differs from map fitting")
    source_index = {row["source_image_path"]: index
                    for index, row in enumerate(validation_rows)}
    selected, available = choose_cases(score_rows, finding, args.per_category)
    checkpoint = torch.load(args.adapter_checkpoint, map_location="cpu")
    if (checkpoint.get("format") != "retinavlm_olives_adapter_alignment_v1"
            or checkpoint.get("variant") != "adapted_byol"
            or checkpoint.get("step") != 10000
            or checkpoint.get("split_sha256") != SPLIT_SHA
            or checkpoint.get("adapted_encoder_sha256") != ADAPTED_SHA
            or sha256(args.adapter_checkpoint) != summary.get("adapter_sha256")):
        raise ValueError("Adapter checkpoint differs from fitted map")
    model, _ = load_released_specialist(args.model_dir, args.authors_code)
    install_encoder(model, "adapted_byol", args.adapted_checkpoint)
    model.model.llama_proj.load_state_dict(checkpoint["adapter"], strict=True)
    del checkpoint
    model.eval()
    absent_id, present_id = answer_token_ids(model.model.get_tokenizer(), QUESTIONS[finding])
    if summary["answer_token_ids"] != {"absent": absent_id, "present": present_id}:
        raise ValueError("Answer-token provenance mismatch")
    maps = torch.load(map_path, map_location="cpu")
    directions = maps["directions"]
    results = []
    reference_images = {}
    for layer in DIAGNOSTIC_LAYERS:
        layer = int(layer)
        direction = directions[layer]
        scale = float(summary["calibration"][str(layer)]["scale"])
        if not math.isfinite(scale) or scale == 0:
            raise ValueError("Degenerate map calibration")
        # Remove a fixed 5% of the state norm along the finding-positive axis.
        # A negative answer-margin change is the prespecified expected effect.
        orientation = -1.0 if scale > 0 else 1.0
        for category_index, category in enumerate(CATEGORIES):
            for index, score_row in enumerate(selected[category]):
                source = score_row["source_image_path"]
                position = source_index[source]
                batch_start = (position // args.reference_batch_size) * args.reference_batch_size
                batch_rows = validation_rows[batch_start:batch_start + args.reference_batch_size]
                if batch_start not in reference_images:
                    reference_images[batch_start] = image_batch(
                        model, batch_rows, args.train_images)
                image = reference_images[batch_start]
                target_index = position - batch_start
                def samples():
                    return {"Image": image,
                            "Question": [QUESTIONS[finding]] * len(batch_rows),
                            "Answer": [""] * len(batch_rows)}
                seeds = [41060000 + layer * 100000 + category_index * 1000
                         + index * 10 + draw for draw in range(args.controls)]
                effects = compare_state_edits(
                    model.model, samples, layer, direction, absent_id, present_id,
                    strength_fraction=0.05, control_seeds=seeds, orientation=orientation,
                    target_index=target_index)
                reference_margin = float(score_row["final_answer_margin"])
                observed_margin = effects["base_present_minus_absent"]
                if abs(observed_margin - reference_margin) > 0.05:
                    raise ValueError(
                        f"Baseline margin did not reproduce for {source}: "
                        f"saved={reference_margin:.6f}, single_image={observed_margin:.6f}, "
                        f"absolute_difference={abs(observed_margin-reference_margin):.6f}"
                    )
                results.append({"source_image_path": source,
                                "patient_key": score_row["patient_key"],
                                "finding": finding, "expert_label": int(score_row["label"]),
                                "layer": layer, "category": category,
                                "reference_batch_start": batch_start,
                                "reference_batch_index": target_index,
                                "control_seeds": seeds, **effects})
                print(json.dumps({"layer": layer, "category": category,
                                  "case": index + 1}), flush=True)
    decisions = {}
    for layer in DIAGNOSTIC_LAYERS:
        subset = [row for row in results if row["layer"] == layer]
        positives = [row for row in subset if row["category"] == "true_positive"]
        controls = [abs(value) for row in subset for value in row["control_margin_changes"]]
        target = [row["target_margin_change"] for row in subset]
        control_median = statistics.median(controls) if controls else float("inf")
        decisions[str(layer)] = {
            "true_positive_cases": len(positives),
            "true_positive_patients": len({row["patient_key"] for row in positives}),
            "target_negative_fraction": sum(value < 0 for value in target) / len(target)
                if target else 0.0,
            "median_target_change": statistics.median(target) if target else None,
            "median_absolute_control_change": control_median,
            "answer_influence_gate": (
                len(positives) >= 2
                and len({row["patient_key"] for row in positives}) >= 2
                and sum(value < 0 for value in target) / len(target) >= 0.75
                and statistics.median(target) <= -2 * control_median),
        }
    receipt = {
        "status": "exploratory_answer_state_intervention_no_eligibility",
        "original_statistical_gate_status": gate["status"],
        "finding": finding, "selected_layers": list(DIAGNOSTIC_LAYERS),
        "available_categories": available, "per_category_limit": args.per_category,
        "reference_batch_size": args.reference_batch_size,
        "controls_per_case": args.controls, "strength_fraction": 0.05,
        "statistical_gate_sha256": sha256(gate_path),
        "selection_plan_sha256": sha256(args.selection_plan),
        "map_eval_cohort": "held-out original training patients",
        "map_sha256": sha256(map_path), "adapter_sha256": sha256(args.adapter_checkpoint),
        "test_accessed": False, "original_validation_accessed": False,
        "report_claims_checked": False,
        "decisions": decisions, "cases": results,
        "claim_boundary": "Exploratory output influence on reused training gate; statistical gate remains failed, no map eligibility or downstream claim",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": receipt["status"], "output": str(args.output)}), flush=True)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("map-dir", "selection-plan", "split-csv", "train-images", "model-dir", "authors-code",
                 "adapted-checkpoint", "adapter-checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--per-category", type=int, default=4)
    parser.add_argument("--controls", type=int, default=4)
    parser.add_argument("--reference-batch-size", type=int, default=8,
                        help="Use the map fitter's validation batch size to reproduce saved margins")
    run(parser.parse_args())
