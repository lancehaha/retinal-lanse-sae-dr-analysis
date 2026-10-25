"""Require hash-bound A/B/C one-step receipts before matched training."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from train_retinavlm_adapter_olives import SPLIT_SHA, sha256


def verify(args):
    map_set = json.loads(args.map_set.read_text())
    if (map_set.get("status") != "validated_retinavlm_olives_map_set"
            or map_set.get("split_sha256") != SPLIT_SHA
            or map_set.get("adapter_sha256") != sha256(args.adapter_checkpoint)):
        raise ValueError("Validated map-set provenance mismatch")
    map_sha = sha256(args.map_set)
    root = args.preflight_root
    a = json.loads((root / "one_step.json").read_text())
    if (a.get("status") != "matched_lora_one_step_preflight_complete"
            or a.get("split_sha256") != SPLIT_SHA
            or a.get("adapter_sha256") != map_set["adapter_sha256"]
            or a.get("rank") != args.rank or a.get("alpha") != args.alpha
            or a.get("lr") != args.lr or a.get("report_weight") != args.report_weight
            or not math.isfinite(float(a.get("gradient_norm", float("nan"))))
            or float(a["gradient_norm"]) <= 0):
        raise ValueError("Ordinary LoRA preflight does not match proposed training")
    summaries = {"A_profile": a}
    for arm in ("B", "C"):
        directory = root / arm
        receipt_path = directory / "segment_000001_receipt.json"
        config_path = directory / "run_config.json"
        checkpoint_path = directory / "checkpoint_step_000001.pt"
        batch_path = directory / "train_batches.jsonl"
        receipt = json.loads(receipt_path.read_text())
        config = json.loads(config_path.read_text())
        if (receipt.get("status") != "matched_abc_training_complete"
                or receipt.get("arm") != arm or receipt.get("seed") != args.seed
                or receipt.get("step") != 1 or receipt.get("total_steps") != 1
                or receipt.get("run_config_sha256") != sha256(config_path)
                or receipt.get("checkpoint_sha256") != sha256(checkpoint_path)
                or receipt.get("batch_records_sha256") != sha256(batch_path)
                or config.get("arm") != arm or config.get("seed") != args.seed
                or config.get("split_sha256") != SPLIT_SHA
                or config.get("adapter_sha256") != map_set["adapter_sha256"]
                or config.get("map_set_sha256") != map_sha
                or config.get("rank") != args.rank or config.get("alpha") != args.alpha
                or config.get("learning_rate") != args.lr
                or config.get("report_weight") != args.report_weight
                or config.get("auxiliary_weight") != args.auxiliary_weight
                or receipt.get("trainable_lora_parameters") != a["trainable_lora_parameters"]):
            raise ValueError(f"{arm} auxiliary preflight provenance mismatch")
        batch_lines = batch_path.read_text().splitlines()
        if len(batch_lines) != 1:
            raise ValueError(f"{arm} one-step preflight lacks its batch record")
        batch = json.loads(batch_lines[0])
        if (batch.get("arm") != arm or batch.get("seed") != args.seed
                or batch.get("global_step") != 1
                or batch.get("map_version") != map_sha
                or not batch.get("sample_layer_scores")):
            raise ValueError(f"{arm} one-step batch audit mismatch")
        if arm == "B" and (receipt.get("extra_probe_parameters", 0) <= 0
                           or receipt.get("probe_gradient_norm_step1", 0) <= 0):
            raise ValueError("B probe did not receive a gradient")
        if arm == "C" and (receipt.get("extra_probe_parameters") != 0
                           or receipt.get("auxiliary_gradient_norm_step1", 0) <= 0):
            raise ValueError("C J-Lens loss did not reach LoRA")
        summaries[arm] = receipt
    peaks = {name: float(item["max_gpu_memory_gib"]) for name, item in summaries.items()}
    if any(not math.isfinite(value) or value > args.memory_ceiling_gib
           for value in peaks.values()):
        raise ValueError("A/B/C one-step memory exceeds the prespecified ceiling")
    result = {"status": "matched_abc_preflight_verified",
              "map_set_sha256": map_sha,
              "adapter_sha256": map_set["adapter_sha256"],
              "split_sha256": SPLIT_SHA, "rank": args.rank,
              "alpha": args.alpha, "lr": args.lr,
              "report_weight": args.report_weight,
              "auxiliary_weight": args.auxiliary_weight,
              "seed": args.seed, "memory_ceiling_gib": args.memory_ceiling_gib,
              "one_step_peak_memory_gib": peaks,
              "preflight_receipt_sha256": {
                  "A": sha256(root / "one_step.json"),
                  "B": sha256(root / "B/segment_000001_receipt.json"),
                  "C": sha256(root / "C/segment_000001_receipt.json")}}
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError("Do not overwrite a prior preflight decision")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-root", type=Path, required=True)
    parser.add_argument("--map-set", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--alpha", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--report-weight", type=float, required=True)
    parser.add_argument("--auxiliary-weight", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--memory-ceiling-gib", type=float, default=74.0)
    parser.add_argument("--output", type=Path)
    verify(parser.parse_args())
