"""Inventory all singleton map gates and combine only independently validated maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from combine_retinavlm_validated_maps import combine, sha256


FINDINGS = ("B1", "B2", "B3", "B4", "B5", "B6", "SRF")


def finalize(root: Path, inventory_path: Path, output_path: Path) -> dict:
    if inventory_path.exists() or output_path.exists():
        raise FileExistsError("Map inventory or combined set already exists")
    selected = json.loads((root / "validation_pilot_selection.json").read_text())["chosen_finding"]
    inventory = {"status": "singleton_map_inventory", "pilot_finding": selected,
                 "findings": {}, "complete": True, "validated_findings": [],
                 "test_accessed": False}
    inputs = []
    for finding in FINDINGS:
        gate_path = root / finding / "validation_gate.json"
        if not gate_path.is_file():
            inventory["findings"][finding] = {"status": "missing_final_gate"}
            inventory["complete"] = False
            continue
        gate = json.loads(gate_path.read_text())
        if gate.get("finding") != finding or gate.get("test_accessed") is not False:
            raise ValueError(f"Invalid gate provenance: {finding}")
        entry = {"status": gate.get("status"), "gate_sha256": sha256(gate_path),
                 "selected_layers": gate.get("selected_layers", [])}
        inventory["findings"][finding] = entry
        if gate.get("status") == "validated_image_conditioned_map":
            source = root / finding / "validated_map_set.json"
            if finding == selected:
                source = root / "validated_map_set.json"
            if not source.is_file():
                raise FileNotFoundError(f"Validated map set missing for {finding}: {source}")
            entry["source_map_set"] = str(source)
            entry["source_map_set_sha256"] = sha256(source)
            inputs.append(source)
            inventory["validated_findings"].append(finding)
        elif gate.get("status") != "image_conditioned_map_gate_failed":
            raise ValueError(f"Unknown final gate status: {finding}")
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(json.dumps(inventory, indent=2) + "\n")
    if not inventory["complete"] or not inputs:
        raise ValueError("Cannot combine: missing final gates or no validated finding")
    combined = combine(inputs, output_path)
    if set(combined["findings"]) != set(inventory["validated_findings"]):
        raise ValueError("Combined set differs from frozen final-gate inventory")
    return {"status": "singleton_map_set_combined", "inventory": str(inventory_path),
            "inventory_sha256": sha256(inventory_path), "output": str(output_path),
            "output_sha256": sha256(output_path),
            "validated_findings": inventory["validated_findings"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(finalize(args.map_root, args.inventory, args.output)), flush=True)
