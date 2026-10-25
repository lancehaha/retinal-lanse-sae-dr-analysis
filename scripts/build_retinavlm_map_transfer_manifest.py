"""Declare map artifacts for hash-verified cluster retrieval.

Run after a map finalizer; --passed must reflect validation_gate.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .resolve_retinavlm_map_finding import resolve
except ImportError:  # Direct execution as a script.
    from resolve_retinavlm_map_finding import resolve


MAP_FILES = (
    "summary.json", "map_and_probe.pt", "validation_map_scores.csv",
    "image_controls.csv", "statistical_gate.json", "state_intervention.json",
    "validation_gate.json",
)


def build(finding: str, job_ids: list[str], passed: bool,
          remote_root: Path, local_root: Path, output: Path,
          map_directory: str = "jlens_maps",
          array_task: int | None = None,
          selection: Path | None = None) -> dict:
    if finding not in {"B1", "B2", "B3", "B4", "B5", "B6", "SRF"}:
        raise ValueError("Invalid finding")
    if len(job_ids) != 4 or any(not job_id.isdecimal() for job_id in job_ids):
        raise ValueError("Expected four numeric map/audit/intervention/finalizer job IDs")
    if output.exists():
        raise FileExistsError(output)
    if map_directory not in {"jlens_maps", "jlens_maps_singleton"}:
        raise ValueError("Unknown map protocol directory")
    if array_task is not None:
        if selection is None or resolve(selection, array_task) != finding:
            raise ValueError("Array finding does not match frozen pilot selection")
    elif selection is not None:
        raise ValueError("--selection requires --array-task")
    remote_map = remote_root / map_directory / finding
    local_map = local_root / map_directory / finding
    artifacts = []

    def add(remote: Path, local: Path):
        artifacts.append({"remote": str(remote), "local": str(local)})

    if array_task is None:
        add(remote_root / map_directory / "validation_pilot_selection.json",
            local_root / map_directory / "validation_pilot_selection.json")
    for name in MAP_FILES:
        add(remote_map / name, local_map / name)
    if passed:
        if array_task is None:
            add(remote_root / map_directory / "validated_map_set.json",
                local_root / map_directory / "validated_map_set.json")
        else:
            add(remote_map / "validated_map_set.json",
                local_map / "validated_map_set.json")
    for prefix, job_id in zip(("jlens-map", "jlens-map-audit", "jlens-map-edit", "jlens-map-final"), job_ids):
        filename = f"{prefix}-{job_id}.out"
        add(remote_root / filename, local_root / "logs" / filename)
    receipt = {"status": "expected_after_pilot_map_finalizer" if array_task is None
               else "expected_after_array_map_finalizer",
               "finding": finding, "validation_gate_passed": passed,
               "job_ids": job_ids, "array_task": array_task, "artifacts": artifacts}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finding", required=True)
    parser.add_argument("--job-ids", nargs=4, required=True,
                        metavar=("MAP", "AUDIT", "INTERVENTION", "FINALIZER"))
    parser.add_argument("--passed", action="store_true")
    parser.add_argument("--map-directory", choices=("jlens_maps", "jlens_maps_singleton"),
                        default="jlens_maps")
    parser.add_argument("--array-task", type=int,
                        help="Remaining-finding Slurm array index (0 to 5)")
    parser.add_argument("--selection", type=Path,
                        help="Retrieved validation_pilot_selection.json; required with --array-task")
    parser.add_argument("--remote-root", type=Path,
                        default=Path("/home/h/heshi/retinavlm-scratch"))
    parser.add_argument("--local-root", type=Path,
                        default=Path("/Volumes/mac外接盘/RetinaVLM_JLens_OLIVES"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.finding, args.job_ids, args.passed,
                           args.remote_root, args.local_root, args.output,
                           args.map_directory, args.array_task,
                           args.selection), indent=2))
