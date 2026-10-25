"""Declare combined singleton maps, exact-map B/C preflight, and frozen candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build(combine_job: int, auxiliary_job: int, candidates_job: int,
          output: Path, remote_root: Path, local_root: Path) -> dict:
    if any(job < 1 for job in (combine_job, auxiliary_job, candidates_job)) or output.exists():
        raise ValueError("Invalid job ID or manifest already exists")
    artifacts = []

    def add(relative: str):
        artifacts.append({"remote": str(remote_root / relative),
                          "local": str(local_root / relative)})

    add("jlens_maps_singleton/validated_map_set_combined.json")
    add(f"jlens_maps_singleton/final_map_gate_inventory_{combine_job}.json")
    add(f"jlens-map-combine-{combine_job}.out")
    add("abc_preflight_singleton/one_step.json")
    for arm in ("B", "C"):
        for name in ("run_config.json", "checkpoint_step_000001.pt", "latest.pt",
                     "train_batches.jsonl", "segment_000001_receipt.json"):
            add(f"abc_preflight_singleton/{arm}/{name}")
    add(f"abc-aux-preflight-{auxiliary_job}.out")
    add("abc_validation_singleton/prespecified_candidates.json")
    add(f"abc-validation-candidates-{candidates_job}.out")
    result = {"status": "expected_after_singleton_map_preflight_and_candidate_freeze",
              "combine_job": combine_job, "auxiliary_job": auxiliary_job,
              "candidates_job": candidates_job, "artifacts": artifacts}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combine-job", type=int, required=True)
    parser.add_argument("--auxiliary-job", type=int, required=True)
    parser.add_argument("--candidates-job", type=int, required=True)
    parser.add_argument("--remote-root", type=Path,
                        default=Path("/home/h/heshi/retinavlm-scratch"))
    parser.add_argument("--local-root", type=Path,
                        default=Path("/Volumes/mac外接盘/RetinaVLM_JLens_OLIVES"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.combine_job, args.auxiliary_job,
                           args.candidates_job, args.output,
                           args.remote_root, args.local_root), indent=2))
