"""Combine independently validated finding maps without changing their readouts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def combine(inputs: list[Path], output: Path) -> dict:
    if not inputs or output.exists():
        raise ValueError("Provide at least one source and a new output path")
    findings = {}
    sources = []
    common = None
    for path in inputs:
        source = json.loads(path.read_text())
        if (source.get("status") != "validated_retinavlm_olives_map_set"
                or source.get("test_accessed") is not False
                or not isinstance(source.get("findings"), dict)
                or len(source["findings"]) != 1):
            raise ValueError(f"Invalid single-finding validated map set: {path}")
        provenance = (source.get("split_sha256"), source.get("adapter_sha256"))
        if any(not isinstance(value, str) or len(value) != 64 for value in provenance):
            raise ValueError(f"Missing split or adapter hash: {path}")
        if common is None:
            common = provenance
        elif provenance != common:
            raise ValueError("Map sets use different split or adapter checkpoints")
        finding, entry = next(iter(source["findings"].items()))
        if finding in findings or finding not in {"B1", "B2", "B3", "B4", "B5", "B6", "SRF"}:
            raise ValueError(f"Duplicate or invalid finding: {finding}")
        directory = Path(entry["directory"])
        gate_path = directory / "validation_gate.json"
        gate = json.loads(gate_path.read_text())
        summary = json.loads((directory / "summary.json").read_text())
        intervention = json.loads((directory / "state_intervention.json").read_text())
        if (gate.get("status") != "validated_image_conditioned_map"
                or summary.get("inference_batch_size") != 1
                or intervention.get("reference_batch_size") != 1
                or gate.get("finding") != finding
                or gate.get("split_sha256") != common[0]
                or gate.get("adapter_sha256") != common[1]
                or gate.get("test_accessed") is not False
                or gate.get("selected_layers") != entry.get("layers")
                or sha256(gate_path) != entry.get("gate_sha256")
                or sha256(directory / "map_and_probe.pt") != entry.get("map_sha256")
                or sha256(directory / "summary.json") != entry.get("summary_sha256")):
            raise ValueError(f"Source finding bundle failed hash/gate check: {finding}")
        findings[finding] = entry
        sources.append({"path": str(path.resolve()), "sha256": sha256(path), "finding": finding})
    result = {"status": "validated_retinavlm_olives_map_set",
              "split_sha256": common[0], "adapter_sha256": common[1],
              "findings": {key: findings[key] for key in ("B1", "B2", "B3", "B4", "B5", "B6", "SRF") if key in findings},
              "selection": "union of independently validated finding maps; no refitting",
              "test_accessed": False, "source_map_sets": sources}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-set", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = combine(args.map_set, args.output)
    print(json.dumps({"status": result["status"],
                      "findings": list(result["findings"]),
                      "output": str(args.output)}))
