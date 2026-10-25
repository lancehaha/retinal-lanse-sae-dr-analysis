"""Check fixed B4/B5 maps against paired and blank validation OCT images.

One opposite-label pair per eligible validation patient is scored under the
same finding question. This is a map/image-dependence diagnostic, not a new
threshold-fitting pass and never touches the test split.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path

import torch

from evaluate_retinavlm_adapter_olives import image_batch
from evaluate_retinavlm_olives_abc_segment import restore_lora, verify_checkpoint
from profile_retinavlm_olives_lora import load_aligned_model
from retinavlm_binary_answer import answer_token_ids
from retinavlm_pre_answer import pre_answer_forward_layers
from retinavlm_projected_jacobian import projected_score
from train_retinavlm_adapter_olives import QUESTIONS, SPLIT_SHA, load_rows, sha256
from train_retinavlm_olives_abc import load_map_set


def select_pairs(rows, findings, max_patients):
    if max_patients < 1:
        raise ValueError("Choose at least one validation patient")
    result = []
    for finding in sorted(findings):
        by_patient = defaultdict(lambda: {0: [], 1: []})
        for row in rows:
            if row[finding] in {"0", "1"}:
                by_patient[row["patient_key"]][int(row[finding])].append(row)
        eligible = [patient for patient in sorted(by_patient)
                    if by_patient[patient][0] and by_patient[patient][1]]
        if not eligible:
            raise ValueError(f"No within-patient opposite-label pairs for {finding}")
        for patient in eligible[:max_patients]:
            negative = min(by_patient[patient][0], key=lambda row: row["source_image_path"])
            positive = min(by_patient[patient][1], key=lambda row: row["source_image_path"])
            result.append({"finding": finding, "patient_key": patient,
                           "negative": negative, "positive": positive})
    return result


def score(mini, image, question, selected, absent_id, present_id):
    samples = {"Image": image, "Question": [question], "Answer": [""]}
    logits, states = pre_answer_forward_layers(mini, samples, selected["layers"])
    margin = float((logits[0, present_id] - logits[0, absent_id]).float())
    if not math.isfinite(margin):
        raise FloatingPointError("Nonfinite map-control answer margin")
    layers = []
    for layer in selected["layers"]:
        raw = float(projected_score(states[layer], selected["directions"][layer])[0])
        calibration = selected["calibration"][layer]
        calibrated = float(calibration["scale"] * raw + calibration["intercept"])
        if not math.isfinite(raw) or not math.isfinite(calibrated):
            raise FloatingPointError("Nonfinite map-control score")
        layers.append({"layer": layer, "raw_score": raw,
                       "calibrated_logit": calibrated,
                       "diagnostic_present": calibrated >= 0,
                       "map_sha256": selected["version"]})
    return {"answer_margin": margin, "map_layers": layers}


def run(args):
    if args.arm not in {"A", "B", "C"} or not torch.cuda.is_available():
        raise ValueError("A/B/C control run requires CUDA")
    validation = load_rows(args.split_csv)["validation"]
    maps, map_set_sha = load_map_set(args.map_set, sha256(args.adapter_checkpoint))
    if not maps or set(maps) != {"B4", "B5"}:
        raise ValueError("Expected the frozen B4/B5 singleton map set")
    checkpoint, config = verify_checkpoint(args, map_set_sha)
    checkpoint_sha = sha256(args.checkpoint)
    lock = json.loads(args.selection_lock.read_text())
    selected = lock.get("arms", {}).get(args.arm, {})
    if (lock.get("status") != "locked_abc_validation_selection"
            or lock.get("split_sha256") != SPLIT_SHA
            or lock.get("map_set_sha256") != map_set_sha
            or lock.get("test_accessed") is not False
            or selected.get("checkpoint_sha256") != checkpoint_sha
            or selected.get("step") != checkpoint["step"]):
        raise ValueError("Map control checkpoint is not validation selected")
    pairs = select_pairs(validation, maps, args.max_patients)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    part = args.out_dir / "map_controls.jsonl"
    receipt = args.out_dir / "map_controls.receipt.json"
    if receipt.exists():
        old = json.loads(receipt.read_text())
        if (old.get("status") == "complete_abc_posttraining_map_controls"
                and old.get("checkpoint_sha256") == checkpoint_sha
                and old.get("rows_sha256") == sha256(part)
                and old.get("pairs") == len(pairs)):
            print(json.dumps({"status": "verified_existing_map_controls",
                              "receipt": str(receipt)}), flush=True)
            return old
        raise ValueError("Existing map-control receipt does not match")
    existing = []
    if part.exists():
        raw = part.read_bytes()
        if raw and not raw.endswith(b"\n"):
            raise ValueError("Incomplete map-control row needs review")
        existing = [json.loads(line) for line in raw.splitlines()]
        if len(existing) > len(pairs):
            raise ValueError("Too many existing map-control rows")
        for old, pair in zip(existing, pairs):
            if (old.get("arm") != args.arm or old.get("finding") != pair["finding"]
                    or old.get("patient_key") != pair["patient_key"]
                    or old.get("checkpoint_sha256") != checkpoint_sha):
                raise ValueError("Map-control resume identity mismatch")
    model = load_aligned_model(args)
    mini = model.model
    restore_lora(mini, checkpoint, config)
    del checkpoint
    answer_ids = {finding: answer_token_ids(mini.get_tokenizer(), QUESTIONS[finding])
                  for finding in maps}
    blank_scores = {}
    with part.open("a", encoding="utf-8") as file, torch.inference_mode():
        for pair in pairs[len(existing):]:
            finding = pair["finding"]
            definition = maps[finding]
            absent_id, present_id = answer_ids[finding]
            negative_image = image_batch(model, [pair["negative"]], args.image_root)
            positive_image = image_batch(model, [pair["positive"]], args.image_root)
            if finding not in blank_scores:
                blank_scores[finding] = score(mini, torch.zeros_like(positive_image),
                                               QUESTIONS[finding], definition,
                                               absent_id, present_id)
            negative_score = score(mini, negative_image, QUESTIONS[finding], definition,
                                   absent_id, present_id)
            positive_score = score(mini, positive_image, QUESTIONS[finding], definition,
                                   absent_id, present_id)
            row = {"arm": args.arm, "checkpoint_sha256": checkpoint_sha,
                   "checkpoint_step": selected["step"],
                   "map_set_sha256": map_set_sha, "finding": finding,
                   "patient_key": pair["patient_key"],
                   "negative_source_image_path": pair["negative"]["source_image_path"],
                   "positive_source_image_path": pair["positive"]["source_image_path"],
                   "negative_real": negative_score, "positive_real": positive_score,
                   "blank": blank_scores[finding],
                   "swap_definition": "same question with within-patient opposite-label image",
                   "test_accessed": False}
            file.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            file.flush()
            os.fsync(file.fileno())
            print(json.dumps({"arm": args.arm, "finding": finding,
                              "pairs_scored": len(existing) + 1}), flush=True)
            existing.append(row)
    result = {"status": "complete_abc_posttraining_map_controls",
              "arm": args.arm, "checkpoint_sha256": checkpoint_sha,
              "checkpoint_step": selected["step"], "map_set_sha256": map_set_sha,
              "selection_lock_sha256": sha256(args.selection_lock),
              "split_sha256": SPLIT_SHA, "pairs": len(pairs),
              "pairs_by_finding": {finding: sum(p["finding"] == finding for p in pairs)
                                   for finding in maps},
              "rows_file": str(part), "rows_sha256": sha256(part),
              "test_accessed": False, "test_targets_used": False,
              "claim_boundary": "paired and blank image dependence diagnostic, not lesion localization",
              "gpu": torch.cuda.get_device_name(0)}
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "receipt": str(receipt)}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("split-csv", "image-root", "model-dir", "authors-code",
                 "adapted-checkpoint", "adapter-checkpoint", "map-set",
                 "checkpoint", "selection-lock", "out-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--arm", choices=("A", "B", "C"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-patients", type=int, default=13)
    args = parser.parse_args()
    args.split = "validation"
    run(args)
