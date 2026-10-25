"""Declare one validation-selected A/B/C state-edit job for hash retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build(arm: str, job_id: int, output: Path, remote_root: Path,
          local_root: Path) -> dict:
    if arm not in {"A", "B", "C"} or job_id < 1 or output.exists():
        raise ValueError("Invalid or duplicate intervention transfer manifest")
    log = f"abc-intervention-{job_id}.out"
    result = {
        "status": "expected_after_abc_posttraining_intervention",
        "arm": arm, "job_id": job_id,
        "artifacts": [
            {"remote": str(remote_root / "abc_posttraining_interventions" / f"{arm}.json"),
             "local": str(local_root / "abc_posttraining_interventions" / f"{arm}.json")},
            {"remote": str(remote_root / log),
             "local": str(local_root / "logs" / log)},
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("A", "B", "C"), required=True)
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--remote-root", type=Path,
                        default=Path("/home/h/heshi/retinavlm-scratch"))
    parser.add_argument("--local-root", type=Path,
                        default=Path("/Volumes/mac外接盘/RetinaVLM_JLens_OLIVES"))
    args = parser.parse_args()
    print(json.dumps(build(args.arm, args.job_id, args.output,
                           args.remote_root, args.local_root), indent=2))
