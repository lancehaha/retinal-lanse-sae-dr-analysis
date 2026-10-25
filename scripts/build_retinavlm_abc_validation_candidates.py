"""Freeze matched validation checkpoint candidates before any A/B/C test run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ARMS = ("A", "B", "C")
STEPS = (1000, 3000, 6000)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def build(map_set: Path, training_root: Path, evaluation_root: Path,
          output: Path, seed: int = 42) -> dict:
    if seed < 0 or output.exists():
        raise ValueError("Invalid seed or candidate manifest already exists")
    map_data = json.loads(map_set.read_text())
    if (map_data.get("status") != "validated_retinavlm_olives_map_set"
            or map_data.get("test_accessed") is not False
            or not map_data.get("findings")):
        raise ValueError("Candidate schedule requires a validated map set")
    arms = {}
    for arm in ARMS:
        arms[arm] = [
            {"step": step,
             "checkpoint": str(training_root / f"{arm}_seed{seed}"
                               / f"checkpoint_step_{step:06d}.pt"),
             "evaluation_dir": str(evaluation_root / f"step_{step:06d}" / arm)}
            for step in STEPS]
    result = {
        "status": "prespecified_abc_validation_candidates",
        "map_set_sha256": digest(map_set),
        "seed": seed, "steps": list(STEPS), "arms": arms,
        "selection_rule": "highest validation seven-finding macro-F1 per arm; tie earliest step",
        "test_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-set", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(build(args.map_set, args.training_root,
                           args.evaluation_root, args.output, args.seed), indent=2))
