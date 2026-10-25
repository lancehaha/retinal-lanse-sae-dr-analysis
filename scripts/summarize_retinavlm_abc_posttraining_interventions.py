"""Audit A/B/C state edits and export same-image intervention tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from train_retinavlm_adapter_olives import SPLIT_SHA, load_rows, sha256


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty intervention table")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    lock = json.loads(args.selection_lock.read_text())
    map_set = json.loads(args.map_set.read_text())
    if (lock.get("status") != "locked_abc_validation_selection"
            or lock.get("test_accessed") is not False
            or lock.get("split_sha256") != SPLIT_SHA
            or lock.get("map_set_sha256") != sha256(args.map_set)
            or set(lock.get("arms", {})) != {"A", "B", "C"}
            or set(map_set.get("findings", {})) != {"B4", "B5"}):
        raise ValueError("Incomplete validation-only intervention provenance")
    official = {row["source_image_path"]: row for row in
                load_rows(args.split_csv)["validation"]}
    aggregates, case_rows, sources = [], [], {}
    for arm in "ABC":
        path = args.intervention_root / f"{arm}.json"
        receipt = json.loads(path.read_text())
        if (receipt.get("status") != "abc_posttraining_state_intervention_complete"
                or receipt.get("arm") != arm
                or receipt.get("checkpoint_sha256")
                   != lock["arms"][arm]["checkpoint_sha256"]
                or receipt.get("checkpoint_step") != lock["arms"][arm]["step"]
                or receipt.get("selection_lock_sha256") != sha256(args.selection_lock)
                or receipt.get("map_set_sha256") != sha256(args.map_set)
                or receipt.get("split_sha256") != SPLIT_SHA
                or receipt.get("test_accessed") is not False
                or receipt.get("report_claims_checked") is not False
                or receipt.get("controls_per_case") < 3
                or receipt.get("strength_fraction") != 0.05):
            raise ValueError(f"Intervention receipt mismatch for {arm}")
        sources[arm] = sha256(path)
        cases = receipt["cases"]
        for case in cases:
            finding, layer = case["finding"], case["layer"]
            definition = map_set["findings"].get(finding)
            source = official.get(case["source_image_path"])
            if (definition is None or layer not in definition["layers"]
                    or source is None or case["patient_key"] != source["patient_key"]
                    or case["expert_label"] != int(source[finding])
                    or case["arm"] != arm
                    or not math.isfinite(float(case["target_margin_change"]))
                    or abs(case["base_present_minus_absent"]
                           - case["reference_answer_margin"]) > 0.05
                    or abs(case["target_margin_change"] -
                           (case["target_present_minus_absent"]
                            - case["base_present_minus_absent"])) > 1e-5
                    or len(case["control_margin_changes"])
                       != receipt["controls_per_case"]
                    or abs(case["target_control_max_absolute_cosine"]) > 1e-4):
                raise ValueError("Intervention case identity or effect invalid")
            threshold = float(lock["arms"][arm]["thresholds_margin"][finding])
            present = case["reference_answer_margin"] >= threshold
            label = case["expert_label"]
            category = ("true_positive" if label == 1 and present else
                        "false_negative" if label == 1 else
                        "false_positive" if present else "true_negative")
            if case["category"] != category:
                raise ValueError("Intervention category differs from locked answer")
            if any(not math.isfinite(float(value)) for value in
                   case["control_margin_changes"]):
                raise ValueError("Nonfinite orthogonal control change")
            case_rows.append({"arm": arm, "checkpoint_step": receipt["checkpoint_step"],
                              "finding": finding, "layer": layer,
                              "category": category, "patient_key": case["patient_key"],
                              "source_image_path": case["source_image_path"],
                              "expert_label": label,
                              "reference_answer_margin": case["reference_answer_margin"],
                              "target_margin_change": case["target_margin_change"],
                              "median_abs_orthogonal_change": statistics.median(
                                  abs(value) for value in case["control_margin_changes"]),
                              "hidden_norm": case["hidden_norm"],
                              "edit_norm": case["edit_norm"],
                              "max_abs_target_control_cosine":
                                  case["target_control_max_absolute_cosine"],
                              "control_margin_changes_json": json.dumps(
                                  case["control_margin_changes"])})
        for finding, definition in map_set["findings"].items():
            for layer in definition["layers"]:
                subset = [case for case in cases
                          if case["finding"] == finding and case["layer"] == layer]
                target = [case["target_margin_change"] for case in subset]
                controls = [abs(value) for case in subset
                            for value in case["control_margin_changes"]]
                positives = [case for case in subset
                             if case["category"] == "true_positive"]
                target_median = statistics.median(target) if target else None
                control_median = statistics.median(controls) if controls else None
                gate = bool(len(positives) >= 2
                            and len({case["patient_key"] for case in positives}) >= 2
                            and target
                            and sum(value < 0 for value in target) / len(target) >= 0.75
                            and target_median <= -2 * control_median)
                frozen = receipt["decisions"][finding][str(layer)]
                if (frozen["answer_influence_gate"] != gate
                        or frozen["true_positive_cases"] != len(positives)
                        or frozen["median_target_change"] != target_median
                        or frozen["median_absolute_control_change"] != control_median):
                    raise ValueError("Intervention gate differs from source cases")
                aggregates.append({"arm": arm, "checkpoint_step": receipt["checkpoint_step"],
                                   "finding": finding, "layer": layer,
                                   "cases": len(subset), "true_positive_cases": len(positives),
                                   "true_positive_patients": len({case["patient_key"]
                                                                  for case in positives}),
                                   "target_negative_fraction":
                                       sum(value < 0 for value in target) / len(target)
                                       if target else None,
                                   "median_target_change": target_median,
                                   "median_abs_orthogonal_change": control_median,
                                   "answer_influence_gate": int(gate),
                                   "claim_boundary": "answer-state effect; no lesion localization"})
    args.output_dir.mkdir(parents=True)
    aggregate_path = args.output_dir / "intervention_summary.csv"
    cases_path = args.output_dir / "intervention_cases.csv"
    write_csv(aggregate_path, aggregates)
    write_csv(cases_path, case_rows)
    result = {"status": "abc_posttraining_interventions_summarized",
              "selection_lock_sha256": sha256(args.selection_lock),
              "map_set_sha256": sha256(args.map_set),
              "source_receipt_sha256": sources,
              "rows": len(aggregates), "cases": len(case_rows),
              "table_sha256": {"aggregate": sha256(aggregate_path),
                               "cases": sha256(cases_path)},
              "test_accessed": False,
              "claim_boundary": "validation same-image output influence only"}
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "output": str(args.output_dir)}))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("split-csv", "selection-lock", "map-set", "intervention-root",
                 "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
