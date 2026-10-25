"""Declare a complete A/B/C training segment for hash-verified retrieval.

Cumulative logs and the advancing latest checkpoint are copied to immutable
segment-specific local paths. Historical checkpoint snapshots keep stable paths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build(arm: str, seed: int, stop_step: int, save_every: int, job_id: int,
          output: Path, remote_root: Path, local_root: Path) -> dict:
    if (arm not in {"A", "B", "C"} or seed < 0 or stop_step < 1
            or save_every < 1 or job_id < 1 or stop_step % save_every):
        raise ValueError("Invalid matched-arm segment identity")
    if output.exists():
        raise FileExistsError(output)
    name = f"{arm}_seed{seed}"
    remote_run = remote_root / "abc_training" / name
    local_run = local_root / "abc_training" / name
    local_segment = local_run / "segments" / f"step_{stop_step:06d}"
    artifacts = []

    def add(remote: Path, local: Path):
        artifacts.append({"remote": str(remote), "local": str(local)})

    add(remote_run / "run_config.json", local_run / "run_config.json")
    for step in range(save_every, stop_step + 1, save_every):
        filename = f"checkpoint_step_{step:06d}.pt"
        add(remote_run / filename, local_run / "checkpoints" / filename)
    for filename in ("latest.pt", "train_batches.jsonl", f"segment_{stop_step:06d}_receipt.json"):
        add(remote_run / filename, local_segment / filename)
    log = f"abc-train-{job_id}.out"
    add(remote_root / log, local_root / "logs" / log)
    receipt = {"status": "expected_after_abc_training_segment",
               "arm": arm, "seed": seed, "stop_step": stop_step,
               "save_every": save_every, "job_id": job_id,
               "artifacts": artifacts}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("A", "B", "C"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stop-step", type=int, required=True)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--remote-root", type=Path,
                        default=Path("/home/h/heshi/retinavlm-scratch"))
    parser.add_argument("--local-root", type=Path,
                        default=Path("/Volumes/mac外接盘/RetinaVLM_JLens_OLIVES"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.arm, args.seed, args.stop_step,
                           args.save_every, args.job_id, args.output,
                           args.remote_root, args.local_root), indent=2))
