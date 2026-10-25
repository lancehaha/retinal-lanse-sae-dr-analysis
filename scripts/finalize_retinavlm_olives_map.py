"""Bind statistical and answer-intervention evidence into one map gate.

Only validation patients are used. Generated-report effects are a downstream
diagnostic after matched report training, not a condition of this answer map.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from audit_retinavlm_olives_map_candidate import SPLIT_SHA, sha256


def finalize(map_dir: Path, map_set: Path):
    names = ("summary.json", "map_and_probe.pt", "validation_map_scores.csv",
             "image_controls.csv", "statistical_gate.json", "state_intervention.json")
    paths = {name: map_dir / name for name in names}
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("Incomplete candidate map and intervention evidence")
    summary = json.loads(paths["summary.json"].read_text())
    statistical = json.loads(paths["statistical_gate.json"].read_text())
    intervention = json.loads(paths["state_intervention.json"].read_text())
    if (summary.get("split_sha256") != SPLIT_SHA
            or summary.get("test_accessed") is not False
            or statistical.get("finding") != summary.get("finding")
            or statistical.get("map_sha256") != sha256(paths["map_and_probe.pt"])
            or statistical.get("summary_sha256") != sha256(paths["summary.json"])
            or statistical.get("validation_scores_sha256") != sha256(paths["validation_map_scores.csv"])
            or statistical.get("image_controls_sha256") != sha256(paths["image_controls.csv"])):
        raise ValueError("Map/statistical provenance mismatch")
    passed_statistical = statistical.get("status") == "candidate_statistical_gate_passed_needs_intervention"
    selected = statistical.get("selected_layers", [])
    validated_layers = []
    if passed_statistical:
        if intervention.get("status") not in {
                "candidate_answer_intervention_passed_needs_final_audit",
                "candidate_answer_intervention_failed"}:
            raise ValueError("Missing complete intervention result for statistical candidate")
        if (intervention.get("finding") != summary.get("finding")
                or intervention.get("selected_layers") != selected
                or intervention.get("statistical_gate_sha256") != sha256(paths["statistical_gate.json"])
                or intervention.get("map_sha256") != sha256(paths["map_and_probe.pt"])
                or intervention.get("adapter_sha256") != summary.get("adapter_sha256")
                or intervention.get("test_accessed") is not False):
            raise ValueError("Intervention provenance mismatch")
        with paths["validation_map_scores.csv"].open(newline="", encoding="utf-8") as file:
            scores = list(csv.DictReader(file))
        lookup = {(int(row["layer"]), row["source_image_path"]): row for row in scores}
        cases = intervention["cases"]
        seen = set()
        for case in cases:
            key = (int(case["layer"]), case["source_image_path"])
            if (key in seen or key not in lookup or key[0] not in selected
                    or case["finding"] != summary["finding"]
                    or case["patient_key"] != lookup[key]["patient_key"]
                    or int(case["expert_label"]) != int(lookup[key]["label"])
                    or len(case["control_margin_changes"]) != intervention["controls_per_case"]
                    or abs(float(case["base_present_minus_absent"])
                           - float(lookup[key]["final_answer_margin"])) > 0.05
                    or not math.isfinite(float(case["target_margin_change"]))):
                raise ValueError("State-edit case does not match validation map row")
            seen.add(key)
        for layer in selected:
            layer_cases = [case for case in cases if int(case["layer"]) == layer]
            if not statistical["layers"][str(layer)]["statistical_gate_passed"]:
                raise ValueError("Selected layer lacks statistical evidence")
            decision = intervention["decisions"][str(layer)]
            if len(layer_cases) < 4:
                raise ValueError("Intervention lacks per-layer cases")
            if decision["answer_influence_gate"]:
                validated_layers.append(layer)
        if ((intervention["status"] == "candidate_answer_intervention_passed_needs_final_audit")
                != (len(validated_layers) == len(selected))):
            raise ValueError("Intervention status disagrees with per-layer decisions")
    else:
        if intervention.get("status") != "state_intervention_skipped_statistical_gate_failed":
            raise ValueError("Failed statistical map unexpectedly ran intervention")
    passed = passed_statistical and len(validated_layers) >= 2
    gate = {
        "status": "validated_image_conditioned_map" if passed else "image_conditioned_map_gate_failed",
        "finding": summary["finding"], "selected_layers": validated_layers if passed else [],
        "statistically_selected_layers": selected,
        "intervention_rejected_layers": [layer for layer in selected if layer not in validated_layers],
        "map_sha256": sha256(paths["map_and_probe.pt"]),
        "summary_sha256": sha256(paths["summary.json"]),
        "statistical_gate_sha256": sha256(paths["statistical_gate.json"]),
        "state_intervention_sha256": sha256(paths["state_intervention.json"]),
        "split_sha256": SPLIT_SHA, "adapter_sha256": summary["adapter_sha256"],
        "validation_patients": summary["validation_patients"],
        "test_accessed": False, "report_claims_checked": False,
        "scope": "image-conditioned pre-answer finding readout and answer-state influence",
    }
    gate_path = map_dir / "validation_gate.json"
    if gate_path.exists() or (passed and map_set.exists()):
        raise FileExistsError("Do not overwrite a completed map decision")
    gate_path.write_text(json.dumps(gate, indent=2) + "\n")
    if passed:
        weights = [1 / len(validated_layers)] * len(validated_layers)
        receipt = {
            "status": "validated_retinavlm_olives_map_set",
            "split_sha256": SPLIT_SHA, "adapter_sha256": summary["adapter_sha256"],
            "findings": {summary["finding"]: {
                "directory": str(map_dir.resolve()),
                "map_sha256": gate["map_sha256"],
                "summary_sha256": gate["summary_sha256"],
                "gate_sha256": sha256(gate_path),
                "layers": validated_layers, "weights": weights,
            }},
            "selection": "validation-only statistical gate plus same-image answer-state interventions",
            "test_accessed": False,
        }
        map_set.parent.mkdir(parents=True, exist_ok=True)
        map_set.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": gate["status"], "finding": gate["finding"],
                      "selected_layers": gate["selected_layers"],
                      "map_set": str(map_set) if passed else None}), flush=True)
    return gate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-dir", type=Path, required=True)
    parser.add_argument("--map-set", type=Path, required=True)
    args = parser.parse_args()
    finalize(args.map_dir, args.map_set)
