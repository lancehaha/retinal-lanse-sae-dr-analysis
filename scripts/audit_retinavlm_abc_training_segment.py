"""Audit one locally collected A/B/C training segment and its batch continuity."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict:
    if args.arm not in "ABC" or args.step not in (1000, 3000, 3500, 4000, 6000):
        raise ValueError("Unexpected arm or segment step")
    root = args.local_root / "abc_training" / f"{args.arm}_seed{args.seed}"
    segment = root / "segments" / f"step_{args.step:06d}"
    manifest_path = (args.local_root / "cluster_stage" /
                     f"abc_{args.arm}_step{args.step}_transfer_manifest.json")
    transfer_path = (args.local_root / "cluster_stage" /
                     f"abc_{args.arm}_step{args.step}_transfer_receipt.json")
    manifest = json.loads(manifest_path.read_text())
    transfer = json.loads(transfer_path.read_text())
    if (manifest["arm"] != args.arm or manifest["seed"] != args.seed
            or manifest["stop_step"] != args.step
            or transfer["status"] != "cluster_artifacts_retrieved_and_verified"
            or transfer["manifest_sha256"] != sha256(manifest_path)):
        raise ValueError("Transfer identity or manifest hash mismatch")
    declared = {(entry["remote"], entry["local"]) for entry in manifest["artifacts"]}
    actual = {(entry["remote"], entry["local"]) for entry in transfer["artifacts"]}
    if declared != actual or len(declared) != len(manifest["artifacts"]):
        raise ValueError("Transferred artifact set differs from manifest")
    for item in transfer["artifacts"]:
        local = Path(item["local"])
        if not local.is_file() or sha256(local) != item["sha256"]:
            raise ValueError(f"Missing or corrupted local artifact: {local}")

    config_path = root / "run_config.json"
    config = json.loads(config_path.read_text())
    map_set_path = args.local_root / "jlens_maps_singleton" / "validated_map_set_combined.json"
    map_set = json.loads(map_set_path.read_text())
    if (sha256(map_set_path) != config["map_set_sha256"]
            or map_set["status"] != "validated_retinavlm_olives_map_set"):
        raise ValueError("Validated map set does not match run configuration")
    receipt_path = segment / f"segment_{args.step:06d}_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    batch_path = segment / "train_batches.jsonl"
    checkpoint_path = segment / "latest.pt"
    snapshot_path = root / "checkpoints" / f"checkpoint_step_{args.step:06d}.pt"
    expected_status = ("matched_abc_training_complete" if args.step == 6000
                       else "matched_abc_segment_complete")
    if (config["arm"] != args.arm or config["seed"] != args.seed
            or config["total_steps"] != 6000
            or receipt["status"] != expected_status
            or receipt["arm"] != args.arm or receipt["seed"] != args.seed
            or receipt["step"] != args.step or receipt["total_steps"] != 6000
            or sha256(config_path) != receipt["run_config_sha256"]
            or sha256(checkpoint_path) != receipt["checkpoint_sha256"]
            or sha256(snapshot_path) != receipt["checkpoint_sha256"]
            or sha256(batch_path) != receipt["batch_records_sha256"]):
        raise ValueError("Training config, receipt, checkpoint, or batch hash mismatch")

    losses = []
    count = 0
    with batch_path.open(encoding="utf-8") as file:
        for count, line in enumerate(file, 1):
            row = json.loads(line)
            if (row["arm"] != args.arm or row["seed"] != args.seed
                    or row["global_step"] != count
                    or row["map_version"] != config["map_set_sha256"]
                    or row["verified_positive_targets"]
                    + row["verified_negative_targets"]
                    + row["unknown_targets"] != 7):
                raise ValueError(f"Invalid batch identity/targets at step {count}")
            for key in ("total_loss", "answer_loss", "report_loss"):
                if not math.isfinite(float(row[key])):
                    raise ValueError(f"Nonfinite {key} at step {count}")
            if args.arm == "A":
                if row["auxiliary_loss"] is not None:
                    raise ValueError(f"A has auxiliary loss at step {count}")
            elif not math.isfinite(float(row["auxiliary_loss"])):
                raise ValueError(f"Nonfinite auxiliary loss at step {count}")
            scores = row["sample_layer_scores"]
            keys = {(item["sample_key"], item["finding"], item["layer"])
                    for item in scores}
            if not scores or len(keys) != len(scores):
                raise ValueError(f"Missing/duplicated sample map scores at step {count}")
            for item in scores:
                finding_map = map_set["findings"].get(item["finding"])
                if (finding_map is None
                        or item["map_version"] != finding_map["map_sha256"]
                        or item["layer"] not in finding_map["layers"]
                        or item["expert_label"] not in (0, 1)
                        or not math.isfinite(float(item["calibrated_map_logit"]))):
                    raise ValueError(f"Invalid sample map score at step {count}")
            losses.append(float(row["total_loss"]))
    if count != args.step:
        raise ValueError(f"Expected {args.step} batch rows, found {count}")

    prior_step = {1000: None, 3000: 1000, 3500: 3000,
                  4000: 3500 if (root / "segments/step_003500/train_batches.jsonl").is_file()
                  else 3000,
                  6000: 4000 if (root / "segments/step_004000/train_batches.jsonl").is_file()
                  else 3500 if (root / "segments/step_003500/train_batches.jsonl").is_file()
                  else 3000}[args.step]
    prefix_identical = None
    if prior_step is not None:
        prior = root / "segments" / f"step_{prior_step:06d}" / "train_batches.jsonl"
        prefix_identical = batch_path.read_bytes().startswith(prior.read_bytes())
        if not prefix_identical:
            raise ValueError(f"Batch history changed since step {prior_step}")
    result = {
        "status": "abc_training_segment_local_audit_passed",
        "arm": args.arm, "seed": args.seed, "step": args.step,
        "slurm_job_id": manifest["job_id"],
        "batch_rows": count, "prior_step": prior_step,
        "prior_batch_prefix_identical": prefix_identical,
        "finite_losses_and_valid_sample_scores": True,
        "transferred_artifacts": len(transfer["artifacts"]),
        "transfer_receipt_sha256": sha256(transfer_path),
        "segment_receipt_sha256": sha256(receipt_path),
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "batch_records_sha256": receipt["batch_records_sha256"],
        "last100_mean_total_loss": statistics.mean(losses[-100:]),
        "scope": "training provenance only; no held-out F1",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".writing")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"status": result["status"], "arm": args.arm,
                      "step": args.step, "rows": count, "output": str(args.output)}))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("A", "B", "C"))
    parser.add_argument("--step", required=True, type=int,
                        choices=(1000, 3000, 3500, 4000, 6000))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-root", type=Path,
                        default=Path("/Volumes/mac外接盘/RetinaVLM_JLens_OLIVES"))
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args())
