"""Validation-only decoder state edits for the selected A/B/C checkpoints.

The fixed B4/B5 directions are compared with equal-norm orthogonal controls
on the same image and question. The test split is never accessed.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
import statistics

import torch

from evaluate_retinavlm_adapter_olives import image_batch
from evaluate_retinavlm_olives_abc_segment import restore_lora, verify_checkpoint
from profile_retinavlm_olives_lora import load_aligned_model
from retinavlm_binary_answer import answer_token_ids
from retinavlm_state_edits import compare_state_edits
from train_retinavlm_adapter_olives import QUESTIONS, SPLIT_SHA, load_rows, sha256
from train_retinavlm_olives_abc import load_map_set


CATEGORIES = ("true_positive", "false_positive", "false_negative", "true_negative")


def choose_cases(rows, finding, threshold, per_category):
    pools = {category: defaultdict(list) for category in CATEGORIES}
    for row in rows:
        prediction = row["finding_predictions"][finding]
        label = prediction["expert_label"]
        if label is None:
            continue
        present = float(prediction["answer_margin"]) >= threshold
        category = ("true_positive" if label == 1 and present else
                    "false_negative" if label == 1 else
                    "false_positive" if present else "true_negative")
        pools[category][row["patient_key"]].append(row)
    selected, available = {}, {}
    for category, by_patient in pools.items():
        available[category] = sum(len(group) for group in by_patient.values())
        for group in by_patient.values():
            group.sort(key=lambda item: item["source_image_path"])
        patients = sorted(by_patient)
        chosen = []
        while len(chosen) < per_category and any(by_patient.values()):
            for patient in patients:
                if by_patient[patient] and len(chosen) < per_category:
                    chosen.append(by_patient[patient].pop(0))
        selected[category] = chosen
    return selected, available


def read_locked_validation(directory, arm, checkpoint_sha, map_sha, lock_candidate,
                           official):
    expected = {Path(entry["path"]).name: entry for entry in
                lock_candidate["segment_receipts"]}
    by_index = {}
    for name, frozen in expected.items():
        receipt_path = directory / name
        if sha256(receipt_path) != frozen["sha256"]:
            raise ValueError("Validation receipt changed after selection lock")
        receipt = json.loads(receipt_path.read_text())
        rows_file = Path(receipt["rows_file"])
        if (receipt.get("status") != "complete_abc_inference_segment"
                or receipt.get("arm") != arm or receipt.get("split") != "validation"
                or receipt.get("checkpoint_sha256") != checkpoint_sha
                or receipt.get("map_set_sha256") != map_sha
                or receipt.get("split_sha256") != SPLIT_SHA
                or receipt.get("rows_sha256") != frozen["rows_sha256"]
                or receipt.get("rows_sha256") != sha256(rows_file)
                or rows_file != directory /
                    f"validation_{receipt['start']:05d}_{receipt['stop']:05d}.jsonl"):
            raise ValueError("Validation source rows differ from frozen candidate")
        for line in rows_file.open(encoding="utf-8"):
            row = json.loads(line)
            index = row["index"]
            if (index in by_index or index >= len(official)
                    or row["source_image_path"] != official[index]["source_image_path"]
                    or row["patient_key"] != official[index]["patient_key"]
                    or row["checkpoint_sha256"] != checkpoint_sha):
                raise ValueError("Validation row identity mismatch")
            by_index[index] = row
    if set(by_index) != set(range(len(official))):
        raise ValueError("Intervention requires complete validation scan coverage")
    return [by_index[index] for index in range(len(official))]


def run(args):
    if (args.output.exists() or args.arm not in {"A", "B", "C"}
            or args.per_category < 2 or args.controls < 3
            or not torch.cuda.is_available()):
        raise ValueError("Invalid or unavailable A/B/C state-edit run")
    maps, map_sha = load_map_set(args.map_set, sha256(args.adapter_checkpoint))
    if set(maps) != {"B4", "B5"}:
        raise ValueError("Expected frozen B4/B5 singleton maps")
    checkpoint, config = verify_checkpoint(args, map_sha)
    checkpoint_sha = sha256(args.checkpoint)
    lock = json.loads(args.selection_lock.read_text())
    selected = lock.get("arms", {}).get(args.arm, {})
    if (lock.get("status") != "locked_abc_validation_selection"
            or lock.get("split_sha256") != SPLIT_SHA
            or lock.get("map_set_sha256") != map_sha
            or lock.get("test_accessed") is not False
            or selected.get("checkpoint_sha256") != checkpoint_sha
            or selected.get("step") != checkpoint["step"]):
        raise ValueError("Checkpoint is not validation selected")
    candidates = [entry for entry in lock["all_validation_candidates"][args.arm]
                  if entry["step"] == selected["step"]]
    if len(candidates) != 1:
        raise ValueError("Missing frozen validation candidate")
    official = load_rows(args.split_csv)["validation"]
    validation = read_locked_validation(args.validation_dir, args.arm,
                                        checkpoint_sha, map_sha, candidates[0], official)
    model = load_aligned_model(args)
    mini = model.model
    restore_lora(mini, checkpoint, config)
    del checkpoint
    results, available = [], {}
    with torch.inference_mode():
        for finding, definition in maps.items():
            absent_id, present_id = answer_token_ids(mini.get_tokenizer(), QUESTIONS[finding])
            cases, counts = choose_cases(validation, finding,
                                         float(selected["thresholds_margin"][finding]),
                                         args.per_category)
            available[finding] = counts
            for layer in definition["layers"]:
                calibration = definition["calibration"][layer]
                scale = float(calibration["scale"])
                if not math.isfinite(scale) or scale == 0:
                    raise ValueError("Degenerate fixed map calibration")
                orientation = -1.0 if scale > 0 else 1.0
                for category_index, category in enumerate(CATEGORIES):
                    for case_index, row in enumerate(cases[category]):
                        image = image_batch(model, [official[row["index"]]], args.image_root)
                        def samples():
                            return {"Image": image, "Question": [QUESTIONS[finding]],
                                    "Answer": [""]}
                        seeds = [41060000 + layer * 100000 + category_index * 1000
                                 + case_index * 10 + draw for draw in range(args.controls)]
                        effects = compare_state_edits(
                            mini, samples, layer, definition["directions"][layer],
                            absent_id, present_id, strength_fraction=0.05,
                            control_seeds=seeds, orientation=orientation)
                        saved_margin = float(row["finding_predictions"][finding]
                                             ["answer_margin"])
                        if abs(effects["base_present_minus_absent"] - saved_margin) > 0.05:
                            raise ValueError("Singleton answer margin did not reproduce")
                        results.append({"arm": args.arm, "finding": finding,
                                        "layer": layer, "category": category,
                                        "scan_index": row["index"],
                                        "patient_key": row["patient_key"],
                                        "source_image_path": row["source_image_path"],
                                        "expert_label": row["finding_predictions"][finding]
                                                           ["expert_label"],
                                        "reference_answer_margin": saved_margin,
                                        "control_seeds": seeds, **effects})
                        print(json.dumps({"arm": args.arm, "finding": finding,
                                          "layer": layer, "category": category,
                                          "case": case_index + 1}), flush=True)
    decisions = {}
    for finding, definition in maps.items():
        decisions[finding] = {}
        for layer in definition["layers"]:
            subset = [row for row in results
                      if row["finding"] == finding and row["layer"] == layer]
            target = [row["target_margin_change"] for row in subset]
            controls = [abs(value) for row in subset
                        for value in row["control_margin_changes"]]
            positives = [row for row in subset if row["category"] == "true_positive"]
            control_median = statistics.median(controls) if controls else float("inf")
            decisions[finding][str(layer)] = {
                "true_positive_cases": len(positives),
                "true_positive_patients": len({row["patient_key"] for row in positives}),
                "target_negative_fraction": sum(value < 0 for value in target) / len(target)
                    if target else 0.0,
                "median_target_change": statistics.median(target) if target else None,
                "median_absolute_control_change": control_median,
                "answer_influence_gate": bool(
                    len(positives) >= 2
                    and len({row["patient_key"] for row in positives}) >= 2
                    and target
                    and sum(value < 0 for value in target) / len(target) >= 0.75
                    and statistics.median(target) <= -2 * control_median),
            }
    result = {"status": "abc_posttraining_state_intervention_complete",
              "arm": args.arm, "checkpoint_step": selected["step"],
              "checkpoint_sha256": checkpoint_sha,
              "selection_lock_sha256": sha256(args.selection_lock),
              "map_set_sha256": map_sha, "split_sha256": SPLIT_SHA,
              "per_category_limit": args.per_category,
              "controls_per_case": args.controls, "strength_fraction": 0.05,
              "available_categories": available, "decisions": decisions,
              "cases": results, "test_accessed": False,
              "report_claims_checked": False,
              "claim_boundary": "same-image answer-state influence, not lesion localization or report-claim effect"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "output": str(args.output)}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("split-csv", "validation-dir", "image-root", "model-dir",
                 "authors-code", "adapted-checkpoint", "adapter-checkpoint",
                 "map-set", "checkpoint", "selection-lock", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--arm", choices=("A", "B", "C"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-category", type=int, default=4)
    parser.add_argument("--controls", type=int, default=4)
    args = parser.parse_args()
    args.split = "validation"
    run(args)
