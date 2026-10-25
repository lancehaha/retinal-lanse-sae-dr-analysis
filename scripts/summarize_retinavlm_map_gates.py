"""Export an auditable validation map-gate table, including pending and failed findings.

This table describes map eligibility and validation diagnostics. It does not
measure held-out test F1 or a trained A/B/C model's clinical performance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


FINDINGS = ("B1", "B2", "B3", "B4", "B5", "B6", "SRF")
LAYERS = (8, 16, 24, 30)
FIELDS = (
    "finding", "layer", "map_status", "statistical_status", "intervention_status",
    "validation_gate_status", "validated_layer", "validation_patients",
    "validation_scans", "validation_auc", "patient_bootstrap_auc_lower",
    "patient_bootstrap_auc_upper", "linear_probe_validation_auc",
    "within_patient_opposite_label_pairs", "positive_higher_pair_fraction",
    "median_absolute_blank_change", "calibrated_median_blank_change",
    "statistical_gate_passed", "check_auc_at_least_0_60",
    "check_auc_ci_lower_above_0_50", "check_opposite_label_patients",
    "check_positive_higher_pair_fraction", "check_blank_change",
    "check_jacobian_finite_difference", "answer_influence_gate", "target_negative_fraction",
    "median_target_change", "median_absolute_control_change", "split_sha256",
    "adapter_sha256", "map_sha256", "summary_sha256", "statistical_gate_sha256",
    "state_intervention_sha256", "validation_gate_sha256",
)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def summarize(map_root: Path, output_csv: Path, output_json: Path) -> dict:
    if output_csv.exists() or output_json.exists():
        raise FileExistsError("Choose new output paths; existing report evidence is immutable")
    rows = []
    reference = None
    finding_statuses = {}
    for finding in FINDINGS:
        folder = map_root / finding
        paths = {name: folder / name for name in (
            "summary.json", "map_and_probe.pt", "statistical_gate.json",
            "state_intervention.json", "validation_gate.json")}
        summary, stat, intervention, gate = (
            load(paths[name]) for name in (
                "summary.json", "statistical_gate.json", "state_intervention.json",
                "validation_gate.json"))
        for obj in (summary, stat, gate):
            if obj is not None and obj.get("finding") != finding:
                raise ValueError(f"Finding mismatch in {folder}")
        skipped_intervention = intervention is not None and (
            intervention.get("status") == "state_intervention_skipped_statistical_gate_failed")
        if intervention is not None and not skipped_intervention and intervention.get("finding") != finding:
            raise ValueError(f"Finding mismatch in {folder}")
        if summary is not None:
            pair = (summary["split_sha256"], summary["adapter_sha256"])
            if reference is None:
                reference = pair
            elif pair != reference:
                raise ValueError(f"Mixed split or adapter provenance in {folder}")
        if stat is not None:
            if summary is None or stat["summary_sha256"] != digest(paths["summary.json"]):
                raise ValueError(f"Statistical-to-summary hash mismatch in {folder}")
            if not paths["map_and_probe.pt"].is_file() or stat["map_sha256"] != digest(paths["map_and_probe.pt"]):
                raise ValueError(f"Statistical-to-map hash mismatch in {folder}")
        if intervention is not None:
            if skipped_intervention:
                if stat is None or stat.get("status") != "candidate_statistical_gate_failed" or (
                    intervention.get("statistical_gate_status") != stat["status"]
                ):
                    raise ValueError(f"Invalid skipped intervention in {folder}")
            elif stat is None or intervention["statistical_gate_sha256"] != digest(paths["statistical_gate.json"]):
                raise ValueError(f"Intervention-to-statistical hash mismatch in {folder}")
        if gate is not None:
            if intervention is None or gate["state_intervention_sha256"] != digest(paths["state_intervention.json"]):
                raise ValueError(f"Gate-to-intervention hash mismatch in {folder}")
            if gate["summary_sha256"] != digest(paths["summary.json"]):
                raise ValueError(f"Gate-to-summary hash mismatch in {folder}")
        finding_statuses[finding] = (gate or intervention or stat or summary or {}).get("status", "pending")
        for layer in LAYERS:
            s = (summary or {}).get("layers", {}).get(str(layer), {})
            t = (stat or {}).get("layers", {}).get(str(layer), {})
            checks = t.get("checks", {})
            decision = (intervention or {}).get("decisions", {}).get(str(layer), {})
            ci = s.get("patient_bootstrap_auc_ci95", [None, None])
            row = {name: None for name in FIELDS}
            row.update({
                "finding": finding, "layer": layer,
                "map_status": (summary or {}).get("status", "pending"),
                "statistical_status": (stat or {}).get("status", "pending"),
                "intervention_status": (intervention or {}).get("status", "pending"),
                "validation_gate_status": (gate or {}).get("status", "pending"),
                "validated_layer": layer in (gate or {}).get("selected_layers", []),
                "validation_patients": (summary or {}).get("validation_patients"),
                "validation_scans": (summary or {}).get("validation_scans"),
                "validation_auc": s.get("validation_auc"),
                "patient_bootstrap_auc_lower": ci[0],
                "patient_bootstrap_auc_upper": ci[1],
                "linear_probe_validation_auc": s.get("linear_probe_validation_auc"),
                "within_patient_opposite_label_pairs": s.get("within_patient_opposite_label_pairs"),
                "positive_higher_pair_fraction": s.get("positive_higher_pair_fraction"),
                "median_absolute_blank_change": s.get("median_absolute_blank_change"),
                "calibrated_median_blank_change": t.get("calibrated_median_blank_change"),
                "statistical_gate_passed": t.get("statistical_gate_passed"),
                "check_auc_at_least_0_60": checks.get("validation_auc_at_least_0_60"),
                "check_auc_ci_lower_above_0_50": checks.get("patient_bootstrap_lower_above_0_50"),
                "check_opposite_label_patients": checks.get("six_opposite_label_patients"),
                "check_positive_higher_pair_fraction": checks.get("positive_higher_pair_fraction_at_least_0_60"),
                "check_blank_change": checks.get("calibrated_blank_change_at_least_0_10"),
                "check_jacobian_finite_difference": checks.get("finite_difference_error_within_tolerance"),
                "answer_influence_gate": decision.get("answer_influence_gate"),
                "target_negative_fraction": decision.get("target_negative_fraction"),
                "median_target_change": decision.get("median_target_change"),
                "median_absolute_control_change": decision.get("median_absolute_control_change"),
                "split_sha256": (summary or {}).get("split_sha256"),
                "adapter_sha256": (summary or {}).get("adapter_sha256"),
                "map_sha256": digest(paths["map_and_probe.pt"]) if paths["map_and_probe.pt"].is_file() else None,
                "summary_sha256": digest(paths["summary.json"]) if summary else None,
                "statistical_gate_sha256": digest(paths["statistical_gate.json"]) if stat else None,
                "state_intervention_sha256": digest(paths["state_intervention.json"]) if intervention else None,
                "validation_gate_sha256": digest(paths["validation_gate.json"]) if gate else None,
            })
            rows.append(row)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as dest:
        writer = csv.DictWriter(dest, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    receipt = {
        "status": "map_gate_table_complete" if all(
            (map_root / finding / "validation_gate.json").is_file() for finding in FINDINGS)
            else "map_gate_table_partial",
        "finding_statuses": finding_statuses,
        "rows": len(rows), "split_sha256": reference[0] if reference else None,
        "adapter_sha256": reference[1] if reference else None,
        "table": str(output_csv), "table_sha256": digest(output_csv),
        "claim_boundary": "validation map diagnostics only; no held-out test or A/B/C performance",
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.map_root, args.output_csv, args.output_json), indent=2))
