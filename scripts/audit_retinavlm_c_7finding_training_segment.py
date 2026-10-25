"""Audit a retrieved seven-finding C segment and its cumulative batch log."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from train_retinavlm_adapter_olives import SPLIT_SHA, sha256


def run(args: argparse.Namespace) -> dict:
    root = args.training_root / "C_seed42"
    segment = root / "segments" / f"step_{args.step:06d}"
    manifest = json.loads(args.manifest.read_text())
    transfer = json.loads(args.transfer_receipt.read_text())
    if (manifest.get("arm") != "C" or manifest.get("seed") != 42
            or manifest.get("stop_step") != args.step
            or transfer.get("status") != "cluster_artifacts_retrieved_and_verified"
            or transfer.get("manifest_sha256") != sha256(args.manifest)):
        raise ValueError("Transfer identity or manifest hash mismatch")
    declared = {(entry["remote"], entry["local"]) for entry in manifest["artifacts"]}
    actual = {(entry["remote"], entry["local"]) for entry in transfer["artifacts"]}
    if declared != actual or len(declared) != len(manifest["artifacts"]):
        raise ValueError("Transferred artifact set differs from manifest")
    for item in transfer["artifacts"]:
        path = Path(item["local"])
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise ValueError(f"Corrupted retrieved artifact: {path}")

    config_path = root / "run_config.json"
    config = json.loads(config_path.read_text())
    map_set = json.loads(args.map_set.read_text())
    expected_findings = set(map_set["findings"])
    if (config.get("arm") != "C" or config.get("seed") != 42
            or config.get("total_steps") != 3000 or config.get("rank") != 8
            or config.get("alpha") != 16 or config.get("learning_rate") != 1e-4
            or config.get("report_weight") != 1.0
            or config.get("auxiliary_weight") != 0.1
            or config.get("split_sha256") != SPLIT_SHA
            or config.get("map_set_sha256") != sha256(args.map_set)
            or set(config.get("mapped_findings", [])) != expected_findings
            or map_set.get("status") != "validated_retinavlm_olives_map_set"):
        raise ValueError("C configuration or validated map provenance mismatch")

    receipt_path = segment / f"segment_{args.step:06d}_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    batch_path = segment / "train_batches.jsonl"
    checkpoint_path = segment / "latest.pt"
    snapshot_path = root / "checkpoints" / f"checkpoint_step_{args.step:06d}.pt"
    expected_status = ("matched_abc_training_complete" if args.step == 3000
                       else "matched_abc_segment_complete")
    if (receipt.get("status") != expected_status
            or receipt.get("arm") != "C" or receipt.get("seed") != 42
            or receipt.get("step") != args.step
            or receipt.get("total_steps") != 3000
            or receipt.get("run_config_sha256") != sha256(config_path)
            or receipt.get("checkpoint_sha256") != sha256(checkpoint_path)
            or receipt.get("checkpoint_sha256") != sha256(snapshot_path)
            or receipt.get("batch_records_sha256") != sha256(batch_path)
            or receipt.get("validation_images_accessed") is not False
            or receipt.get("test_images_accessed") is not False
            or receipt.get("validation_targets_used") is not False
            or receipt.get("test_targets_used") is not False):
        raise ValueError("C receipt, checkpoint, or cumulative log mismatch")

    count = 0
    with batch_path.open(encoding="utf-8") as file:
        for count, line in enumerate(file, 1):
            row = json.loads(line)
            if (row.get("arm") != "C" or row.get("seed") != 42
                    or row.get("global_step") != count
                    or row.get("map_version") != config["map_set_sha256"]
                    or row.get("patient_count") != 1 or row.get("image_count") != 1
                    or sum(row[key] for key in ("verified_positive_targets",
                                                "verified_negative_targets",
                                                "unknown_targets")) != 7
                    or any(not math.isfinite(float(row[key]))
                           for key in ("total_loss", "answer_loss", "report_loss",
                                       "auxiliary_loss"))):
                raise ValueError(f"Invalid C batch at step {count}")
            scores = row.get("sample_layer_scores", [])
            keys = {(item["sample_key"], item["finding"], item["layer"])
                    for item in scores}
            if not scores or len(keys) != len(scores):
                raise ValueError(f"Missing or duplicated map scores at step {count}")
            for item in scores:
                definition = map_set["findings"].get(item["finding"])
                if (definition is None or item["map_version"] != definition["map_sha256"]
                        or item["layer"] not in definition["layers"]
                        or item["expert_label"] not in (0, 1)
                        or not math.isfinite(float(item["calibrated_map_logit"]))):
                    raise ValueError(f"Invalid mapped score at step {count}")
    if count != args.step:
        raise ValueError(f"Expected {args.step} exact batch rows, found {count}")
    prefix_same = None
    if args.previous_step:
        prior = root / "segments" / f"step_{args.previous_step:06d}" / "train_batches.jsonl"
        prefix_same = batch_path.read_bytes().startswith(prior.read_bytes())
        if not prefix_same:
            raise ValueError("Prior immutable batch prefix changed")
    result = {
        "status": "seven_finding_C_training_segment_audit_passed",
        "step": args.step, "prior_step": args.previous_step,
        "prior_batch_prefix_identical": prefix_same,
        "batch_rows": count, "job_id": manifest["job_id"],
        "map_set_sha256": sha256(args.map_set),
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "batch_records_sha256": receipt["batch_records_sha256"],
        "transfer_receipt_sha256": sha256(args.transfer_receipt),
        "segment_receipt_sha256": sha256(receipt_path),
        "scope": "Training provenance only; no validation or test outcome",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".writing")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"status": result["status"], "step": args.step}))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--map-set", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--transfer-receipt", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--previous-step", type=int)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
