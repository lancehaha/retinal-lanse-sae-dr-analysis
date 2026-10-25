"""Declare one A/B/C inference segment for hash-verified cluster retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build(split: str, arm: str, step: int, start: int, stop: int, job_id: int,
          output: Path, remote_root: Path, local_root: Path) -> dict:
    if (split not in {"validation", "test"} or arm not in {"A", "B", "C"}
            or step < 1 or start < 0 or stop <= start or job_id < 1 or output.exists()):
        raise ValueError("Invalid inference segment identity or manifest already exists")
    directory = "abc_validation_singleton" if split == "validation" else "abc_test_singleton"
    folder = Path(directory) / f"step_{step:06d}" / arm
    stem = f"{split}_{start:05d}_{stop:05d}"
    entries = [
        {"remote": str(remote_root / folder / (stem + ".jsonl")),
         "local": str(local_root / folder / (stem + ".jsonl"))},
        {"remote": str(remote_root / folder / (stem + ".receipt.json")),
         "local": str(local_root / folder / (stem + ".receipt.json"))},
        {"remote": str(remote_root / f"abc-eval-{job_id}.out"),
         "local": str(local_root / "logs" / f"abc-eval-{job_id}.out")},
    ]
    result = {"status": "expected_after_abc_inference_segment",
              "split": split, "arm": arm, "checkpoint_step": step,
              "start": start, "stop": stop, "job_id": job_id,
              "artifacts": entries}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--arm", choices=("A", "B", "C"), required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--remote-root", type=Path,
                        default=Path("/home/h/heshi/retinavlm-scratch"))
    parser.add_argument("--local-root", type=Path,
                        default=Path("/Volumes/mac外接盘/RetinaVLM_JLens_OLIVES"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.split, args.arm, args.step, args.start,
                           args.stop, args.job_id, args.output,
                           args.remote_root, args.local_root), indent=2))
