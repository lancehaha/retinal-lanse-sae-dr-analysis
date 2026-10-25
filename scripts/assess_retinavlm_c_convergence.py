"""Apply the frozen training-only convergence rule to a C batch log."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


LOSSES = ("total_loss", "answer_loss", "report_loss", "auxiliary_loss")


def assess(path: Path, step: int) -> dict:
    if step < 2000 or step > 3000 or (step > 2000 and step % 100):
        raise ValueError("Decision step must be 2000 or a 100-step increment through 3000")
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if len(records) != step or any(row.get("global_step") != i or row.get("arm") != "C"
                                   for i, row in enumerate(records, 1)):
        raise ValueError("C batch log must be contiguous and end at the decision step")
    versions = {row.get("map_version") for row in records}
    if len(versions) != 1 or None in versions:
        raise ValueError("Map version changed within training")
    previous = records[-400:-200]
    recent = records[-200:]
    means = {}
    for key in LOSSES:
        values = [row.get(key) for row in previous + recent]
        if any(not isinstance(value, (int, float)) or not math.isfinite(value)
               for value in values):
            raise ValueError(f"Missing or nonfinite {key}")
        means[key] = {
            "previous_200": sum(row[key] for row in previous) / 200,
            "recent_200": sum(row[key] for row in recent) / 200,
        }
    old = means["total_loss"]["previous_200"]
    new = means["total_loss"]["recent_200"]
    if old <= 0:
        raise ValueError("Nonpositive total loss mean")
    relative_decrease = (old - new) / old
    component_increases = {}
    for key in LOSSES[1:]:
        prior = means[key]["previous_200"]
        latest = means[key]["recent_200"]
        component_increases[key] = ((latest - prior) / prior if prior > 0
                                    else (0.0 if latest <= prior else math.inf))
    plateau = 0 <= relative_decrease <= 0.01 and all(
        value <= 0.02 for value in component_increases.values())
    return {
        "status": "c_training_convergence_decision",
        "step": step,
        "map_set_sha256": versions.pop(),
        "window_steps": {"previous": [step - 399, step - 200],
                         "recent": [step - 199, step]},
        "mean_losses": means,
        "relative_total_loss_decrease": relative_decrease,
        "relative_component_increases": component_increases,
        "converged": plateau,
        "action": "stop" if plateau or step == 3000 else "continue_100_steps",
        "reason": "plateau" if plateau else ("hard_cap" if step == 3000 else "not_converged"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-log", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = assess(args.batch_log, args.step)
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
