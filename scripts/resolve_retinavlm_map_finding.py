"""Resolve one of the six non-pilot findings from a frozen validation selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FINDINGS = ("B1", "B2", "B3", "B4", "B5", "B6", "SRF")


def resolve(selection: Path, index: int) -> str:
    record = json.loads(selection.read_text())
    chosen = record.get("chosen_finding")
    if (record.get("status") != "validation_pilot_finding_selected"
            or chosen not in FINDINGS
            or record.get("test_predictions_accessed") is not False):
        raise ValueError("No frozen validation-only pilot finding")
    remaining = tuple(finding for finding in FINDINGS if finding != chosen)
    if index < 0 or index >= len(remaining):
        raise ValueError("Remaining-finding array index must be 0 through 5")
    return remaining[index]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args()
    print(resolve(args.selection, args.index))
