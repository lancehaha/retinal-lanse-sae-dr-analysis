"""Freeze one A/B/C test-pass directory layout from the validation lock."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ARMS = ("A", "B", "C")


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def build(lock_path: Path, test_root: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    lock = json.loads(lock_path.read_text())
    if (lock.get("status") != "locked_abc_validation_selection"
            or lock.get("test_accessed") is not False
            or set(lock.get("arms", {})) != set(ARMS)):
        raise ValueError("Test manifest requires a complete validation-only lock")
    arms = {}
    for arm in ARMS:
        selected = lock["arms"][arm]
        step = selected.get("step")
        if (not isinstance(step, int) or step < 1
                or len(selected.get("checkpoint_sha256", "")) != 64
                or set(selected.get("thresholds_margin", {}))
                   != {"B1", "B2", "B3", "B4", "B5", "B6", "SRF"}):
            raise ValueError(f"Incomplete validation selection for {arm}")
        arms[arm] = str(test_root / f"step_{step:06d}" / arm)
    result = {
        "status": "locked_abc_test_segments",
        "selection_lock_sha256": digest(lock_path),
        "split_sha256": lock["split_sha256"],
        "arms": arms,
        "prespecified_segments": [[0, 637], [637, 1274]],
        "test_accessed": False,
        "test_cohort_status": "patient-disjoint, reused from Stage 3",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.selection_lock, args.test_root, args.output), indent=2))
