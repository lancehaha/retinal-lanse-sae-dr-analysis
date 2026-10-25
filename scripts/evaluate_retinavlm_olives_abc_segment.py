"""Write one resumable A/B/C validation or locked-test inference segment.

Rows contain final forced-choice answers, generated report claims, and all
selected map-layer scores. No threshold or checkpoint is selected here.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch

from evaluate_retinavlm_adapter_olives import image_batch
from profile_retinavlm_olives_lora import add_lora, load_aligned_model
from retinavlm_binary_answer import answer_token_ids
from retinavlm_olives_report import REPORT_QUESTION, parse_report_claims
from retinavlm_pre_answer import pre_answer_forward, pre_answer_forward_layers
from retinavlm_projected_jacobian import projected_score
from train_retinavlm_adapter_olives import QUESTIONS, SPLIT_SHA, load_rows, sha256
from train_retinavlm_olives_abc import load_map_set


def verify_checkpoint(args, map_set_sha):
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint.get("run_config")
    if (not isinstance(config, dict) or config.get("format") != "retinavlm_olives_abc_lora_v1"
            or config.get("arm") != args.arm or config.get("seed") != args.seed
            or config.get("map_set_sha256") != map_set_sha
            or config.get("split_sha256") != SPLIT_SHA
            or config.get("adapter_sha256") != sha256(args.adapter_checkpoint)
            or not isinstance(checkpoint.get("step"), int)
            or checkpoint["step"] < 1):
        raise ValueError("A/B/C checkpoint provenance mismatch")
    if args.split == "test":
        if args.selection_lock is None:
            raise ValueError("Test requires a validation-selected checkpoint lock")
        lock = json.loads(args.selection_lock.read_text())
        selected = lock.get("arms", {}).get(args.arm, {})
        if (lock.get("status") != "locked_abc_validation_selection"
                or lock.get("split_sha256") != SPLIT_SHA
                or lock.get("map_set_sha256") != map_set_sha
                or selected.get("checkpoint_sha256") != sha256(args.checkpoint)
                or selected.get("step") != checkpoint["step"]):
            raise ValueError("Test checkpoint is not the validation-locked selection")
    return checkpoint, config


def restore_lora(mini, checkpoint, config):
    add_lora(mini, config["rank"], config["alpha"])
    named = dict(mini.llama_model.named_parameters())
    trained = checkpoint.get("lora_parameters")
    if (not isinstance(trained, dict) or not trained
            or set(trained) != {name for name, parameter in named.items()
                               if parameter.requires_grad}):
        raise ValueError("LoRA checkpoint parameter set mismatch")
    for name, value in trained.items():
        if named[name].shape != value.shape:
            raise ValueError(f"LoRA tensor shape mismatch: {name}")
        named[name].data.copy_(value.to(named[name].device, dtype=named[name].dtype))
    mini.eval()
    for parameter in mini.parameters():
        parameter.requires_grad_(False)


def existing_count(path, start, stop, arm, split, checkpoint_sha, official_rows=None):
    if not path.exists():
        return 0
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError("Incomplete final JSONL line; preserve and inspect before resuming")
    rows = raw.splitlines()
    if len(rows) > stop - start:
        raise ValueError("Segment has more rows than requested")
    for position, line in enumerate(rows):
        row = json.loads(line)
        source = official_rows[start + position] if official_rows is not None else None
        if (row.get("index") != start + position or row.get("arm") != arm
                or row.get("split") != split
                or row.get("checkpoint_sha256") != checkpoint_sha
                or (source is not None and (
                    row.get("source_image_path") != source["source_image_path"]
                    or row.get("patient_key") != source["patient_key"]))):
            raise ValueError("Existing segment rows do not match this run")
    return len(rows)


def run(args):
    if (args.split not in {"validation", "test"} or args.arm not in {"A", "B", "C"}
            or args.start < 0 or args.stop <= args.start or args.report_tokens < 64):
        raise ValueError("Invalid evaluation segment")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    groups = load_rows(args.split_csv)
    rows = groups[args.split]
    if args.stop > len(rows):
        raise ValueError("Segment extends past split")
    maps, map_set_sha = load_map_set(args.map_set, sha256(args.adapter_checkpoint))
    checkpoint, config = verify_checkpoint(args, map_set_sha)
    checkpoint_sha = sha256(args.checkpoint)
    part = args.out_dir / f"{args.split}_{args.start:05d}_{args.stop:05d}.jsonl"
    receipt = part.with_suffix(".receipt.json")
    if receipt.exists():
        old = json.loads(receipt.read_text())
        if (old.get("status") == "complete_abc_inference_segment"
                and old.get("rows_sha256") == sha256(part)
                and old.get("checkpoint_sha256") == checkpoint_sha
                and old.get("rows") == args.stop - args.start):
            print(json.dumps({"status": "verified_existing_segment", "receipt": str(receipt)}), flush=True)
            return old
        raise ValueError("Existing segment receipt does not match")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    done = existing_count(part, args.start, args.stop, args.arm, args.split,
                          checkpoint_sha, rows)
    model = load_aligned_model(args)
    mini = model.model
    restore_lora(mini, checkpoint, config)
    checkpoint_step = checkpoint["step"]
    del checkpoint
    absent_id, present_id = answer_token_ids(mini.get_tokenizer(), QUESTIONS["B5"])
    with part.open("a", encoding="utf-8") as file, torch.inference_mode():
        for index in range(args.start + done, args.stop):
            source = rows[index]
            image = image_batch(model, [source], args.image_root)
            findings = {}
            for finding, question in QUESTIONS.items():
                samples = {"Image": image, "Question": [question], "Answer": [""]}
                if finding in maps:
                    selected = maps[finding]
                    logits, states = pre_answer_forward_layers(mini, samples, selected["layers"])
                    map_scores = []
                    for layer in selected["layers"]:
                        raw = float(projected_score(
                            states[layer], selected["directions"][layer])[0])
                        calibration = selected["calibration"][layer]
                        calibrated = float(calibration["scale"] * raw + calibration["intercept"])
                        map_scores.append({"layer": layer, "raw_score": raw,
                                           "calibrated_logit": calibrated,
                                           "diagnostic_present": calibrated >= 0,
                                           "map_sha256": selected["version"]})
                else:
                    logits, _ = pre_answer_forward(mini, samples)
                    map_scores = []
                margin = float((logits[0, present_id] - logits[0, absent_id]).float())
                if not math.isfinite(margin):
                    raise FloatingPointError("Nonfinite VLM answer margin")
                findings[finding] = {
                    "expert_label": int(source[finding]) if source[finding] in {"0", "1"} else None,
                    "answer_margin": margin,
                    "zero_margin_answer": "present" if margin >= 0 else "absent",
                    "map_layers": map_scores,
                }
            generated = mini.query(
                image, [REPORT_QUESTION], max_new_tokens=args.report_tokens,
                output_only=True)[0]
            claims = parse_report_claims(generated)
            record = {
                "index": index, "arm": args.arm, "seed": args.seed,
                "split": args.split, "source_image_path": source["source_image_path"],
                "patient_key": source["patient_key"],
                "checkpoint_sha256": checkpoint_sha, "checkpoint_step": checkpoint_step,
                "map_set_sha256": map_set_sha, "finding_predictions": findings,
                "generated_report": generated, "parsed_report_claims": claims,
            }
            file.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
            file.flush()
            os.fsync(file.fileno())
            if (index - args.start + 1) % 25 == 0 or index + 1 == args.stop:
                print(json.dumps({"arm": args.arm, "split": args.split,
                                  "scored": index + 1 - args.start,
                                  "segment_size": args.stop - args.start}), flush=True)
    result = {
        "status": "complete_abc_inference_segment", "arm": args.arm,
        "seed": args.seed, "split": args.split, "start": args.start, "stop": args.stop,
        "rows": args.stop - args.start, "rows_file": str(part),
        "rows_sha256": sha256(part), "checkpoint_sha256": checkpoint_sha,
        "checkpoint_step": checkpoint_step,
        "map_set_sha256": map_set_sha, "split_sha256": SPLIT_SHA,
        "report_tokens": args.report_tokens,
        "selection_lock_sha256": sha256(args.selection_lock) if args.selection_lock else None,
        "reused_patient_disjoint_test": args.split == "test",
        "gpu": torch.cuda.get_device_name(0),
    }
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "receipt": str(receipt)}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("split-csv", "image-root", "model-dir", "authors-code",
                 "adapted-checkpoint", "adapter-checkpoint", "map-set",
                 "checkpoint", "out-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--arm", choices=("A", "B", "C"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    parser.add_argument("--report-tokens", type=int, default=160)
    parser.add_argument("--selection-lock", type=Path)
    run(parser.parse_args())
