"""Audit all matched A/B/C batch records and export a reproducible training curve."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import deque
from pathlib import Path

from train_retinavlm_adapter_olives import SPLIT_SHA, sha256


ARMS = ("A", "B", "C")
LOSSES = ("total_loss", "answer_loss", "report_loss", "auxiliary_loss")


def read_arm(root: Path, arm: str, seed: int, map_sha: str, stop_step: int | None):
    directory = root / f"{arm}_seed{seed}"
    config_path = directory / "run_config.json"
    config = json.loads(config_path.read_text())
    total_steps = int(config["total_steps"])
    steps = total_steps if stop_step is None else stop_step
    if steps < 1 or steps > total_steps:
        raise ValueError("Invalid training report stop step")
    collected_segment = directory / "segments" / f"step_{steps:06d}"
    segment_directory = collected_segment if collected_segment.is_dir() else directory
    batch_path = segment_directory / "train_batches.jsonl"
    receipt_path = segment_directory / f"segment_{steps:06d}_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    latest = segment_directory / "latest.pt"
    if (config.get("format") != "retinavlm_olives_abc_lora_v1"
            or config.get("arm") != arm or config.get("seed") != seed
            or config.get("split_sha256") != SPLIT_SHA
            or config.get("map_set_sha256") != map_sha
            or receipt.get("status") != ("matched_abc_training_complete"
                                         if steps == total_steps else
                                         "matched_abc_segment_complete")
            or receipt.get("arm") != arm or receipt.get("seed") != seed
            or receipt.get("step") != steps or receipt.get("total_steps") != total_steps
            or receipt.get("run_config_sha256") != sha256(config_path)
            or receipt.get("batch_records_sha256") != sha256(batch_path)
            or receipt.get("checkpoint_sha256") != sha256(latest)):
        raise ValueError(f"{arm} training provenance or completion mismatch")
    rows = []
    with batch_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            row = json.loads(line)
            scores = row.get("sample_layer_scores", [])
            if (row.get("arm") != arm or row.get("seed") != seed
                    or row.get("global_step") != line_number
                    or row.get("map_version") != map_sha
                    or row.get("verified_positive_targets", -1)
                       + row.get("verified_negative_targets", -1)
                       + row.get("unknown_targets", -1) != 7
                    or not scores or any(item.get("map_version") is None for item in scores)
                    or len({(item["sample_key"], item["finding"], item["layer"])
                            for item in scores}) != len(scores)):
                raise ValueError(f"{arm} invalid batch record at step {line_number}")
            for key in LOSSES[:3]:
                if not math.isfinite(float(row[key])):
                    raise ValueError(f"{arm} nonfinite {key} at step {line_number}")
            if arm == "A" and row["auxiliary_loss"] is not None:
                raise ValueError("A batch unexpectedly has an auxiliary loss")
            if arm != "A" and not math.isfinite(float(row["auxiliary_loss"])):
                raise ValueError(f"{arm} nonfinite auxiliary loss")
            rows.append(row)
    if len(rows) != steps:
        raise ValueError(f"{arm} has {len(rows)} batch rows for {steps} steps")
    return config, receipt, rows, receipt_path, batch_path


def sample_signature(row):
    return tuple(sorted((item["sample_key"], item["patient_key"],
                         item["trained_finding"], item["trained_label"],
                         item["finding"], item["expert_label"], item["layer"])
                        for item in row["sample_layer_scores"]))


def run(args):
    if args.window < 1:
        raise ValueError("Window must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Training report directory must be empty")
    map_set = json.loads(args.map_set.read_text())
    if (map_set.get("status") != "validated_retinavlm_olives_map_set"
            or map_set.get("split_sha256") != SPLIT_SHA):
        raise ValueError("Validated map set required")
    map_sha = sha256(args.map_set)
    loaded = {arm: read_arm(args.run_root, arm, args.seed, map_sha,
                            args.stop_step) for arm in ARMS}
    configs = {arm: loaded[arm][0] for arm in ARMS}
    common = {key: value for key, value in configs["A"].items() if key != "arm"}
    if any({key: value for key, value in configs[arm].items() if key != "arm"} != common
           for arm in ARMS[1:]):
        raise ValueError("A/B/C training configurations are not matched")
    steps = common["total_steps"] if args.stop_step is None else args.stop_step
    for index in range(steps):
        signature = sample_signature(loaded["A"][2][index])
        if any(sample_signature(loaded[arm][2][index]) != signature for arm in ARMS[1:]):
            raise ValueError(f"A/B/C sampled cases differ at step {index + 1}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_path = args.output_dir / "training_curve.csv"
    with table_path.open("w", newline="", encoding="utf-8") as file:
        fields = ("arm", "seed", "step", "window_steps", "total_loss", "answer_loss",
                  "report_loss", "auxiliary_loss", "mean_total_loss_window",
                  "mean_answer_loss_window", "mean_report_loss_window",
                  "mean_auxiliary_loss_window", "learning_rate", "positive_targets",
                  "negative_targets", "unknown_targets", "sampled_patient_count",
                  "sampled_image_count", "mapped_finding", "mapped_layers",
                  "mean_calibrated_map_logit", "elapsed_seconds")
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for arm in ARMS:
            windows = {name: deque(maxlen=args.window) for name in LOSSES}
            for row in loaded[arm][2]:
                for name in LOSSES:
                    if row[name] is not None:
                        windows[name].append(float(row[name]))
                scores = row["sample_layer_scores"]
                writer.writerow({
                    "arm": arm, "seed": args.seed, "step": row["global_step"],
                    "window_steps": len(windows["total_loss"]),
                    **{name: row[name] for name in LOSSES},
                    **{f"mean_{name}_window": sum(windows[name]) / len(windows[name])
                       if windows[name] else "" for name in LOSSES},
                    "learning_rate": row["learning_rate"],
                    "positive_targets": row["verified_positive_targets"],
                    "negative_targets": row["verified_negative_targets"],
                    "unknown_targets": row["unknown_targets"],
                    "sampled_patient_count": row["patient_count"],
                    "sampled_image_count": row["image_count"],
                    "mapped_finding": ",".join(sorted({item["finding"] for item in scores})),
                    "mapped_layers": ",".join(map(str, sorted({item["layer"] for item in scores}))),
                    "mean_calibrated_map_logit": sum(float(item["calibrated_map_logit"])
                                                       for item in scores) / len(scores),
                    "elapsed_seconds": row["elapsed_seconds"],
                })
    summary = {
        "status": "matched_abc_training_curve_audited",
        "split_sha256": SPLIT_SHA, "map_set_sha256": map_sha,
        "seed": args.seed, "steps_per_arm": steps, "arms": list(ARMS),
        "matched_sample_schedule": True,
        "config_sha256": {arm: sha256(args.run_root / f"{arm}_seed{args.seed}" /
                                      "run_config.json") for arm in ARMS},
        "completion_receipt_sha256": {arm: sha256(loaded[arm][3]) for arm in ARMS},
        "batch_records_sha256": {arm: sha256(loaded[arm][4]) for arm in ARMS},
        "training_curve_sha256": sha256(table_path),
        "rows": steps * len(ARMS),
        "scope": "training diagnostics, not held-out finding accuracy",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": summary["status"], "rows": summary["rows"]}), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--map-set", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stop-step", type=int,
                        help="Audit a completed interim segment, such as step 3000")
    parser.add_argument("--window", type=int, default=100)
    run(parser.parse_args())
