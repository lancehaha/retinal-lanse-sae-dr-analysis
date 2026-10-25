"""Verify A/B/C paired-image controls and export map-dependence tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import statistics

from evaluate_retinavlm_abc_posttraining_map_controls import select_pairs
from retinavlm_abc_artifact_paths import local_artifact
from train_retinavlm_adapter_olives import SPLIT_SHA, load_rows, sha256


def interval(values, replicates, seed):
    if not values:
        return (None, None)
    rng = random.Random(seed)
    samples = sorted(sum(rng.choice(values) for _ in values) / len(values)
                     for _ in range(replicates))
    return (samples[int((replicates - 1) * 0.025)],
            samples[int((replicates - 1) * 0.975)])


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty map-control table")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    if args.bootstrap_replicates < 1 or args.output_dir.exists():
        raise ValueError("Invalid output or bootstrap count")
    lock = json.loads(args.selection_lock.read_text())
    map_set = json.loads(args.map_set.read_text())
    if (lock.get("status") != "locked_abc_validation_selection"
            or lock.get("split_sha256") != SPLIT_SHA
            or lock.get("map_set_sha256") != sha256(args.map_set)
            or lock.get("test_accessed") is not False
            or set(lock.get("arms", {})) != {"A", "B", "C"}
            or set(map_set.get("findings", {})) != {"B4", "B5"}):
        raise ValueError("Incomplete validation-only map-control provenance")
    official = load_rows(args.split_csv)["validation"]
    expected = select_pairs(official, map_set["findings"], 13)
    expected_keys = [(pair["finding"], pair["patient_key"],
                      pair["negative"]["source_image_path"],
                      pair["positive"]["source_image_path"]) for pair in expected]
    by_arm = {}
    receipt_hashes = {}
    for arm in "ABC":
        directory = args.control_root / arm
        receipt_path = directory / "map_controls.receipt.json"
        receipt = json.loads(receipt_path.read_text())
        rows_path = local_artifact(receipt["rows_file"], args.remote_root, args.local_root)
        if (receipt.get("status") != "complete_abc_posttraining_map_controls"
                or receipt.get("arm") != arm
                or receipt.get("checkpoint_sha256")
                   != lock["arms"][arm]["checkpoint_sha256"]
                or receipt.get("checkpoint_step") != lock["arms"][arm]["step"]
                or receipt.get("map_set_sha256") != sha256(args.map_set)
                or receipt.get("selection_lock_sha256") != sha256(args.selection_lock)
                or receipt.get("split_sha256") != SPLIT_SHA
                or receipt.get("test_accessed") is not False
                or receipt.get("test_targets_used") is not False
                or rows_path != directory / "map_controls.jsonl"
                or receipt.get("rows_sha256") != sha256(rows_path)):
            raise ValueError(f"Map-control receipt mismatch for {arm}")
        rows = [json.loads(line) for line in rows_path.open(encoding="utf-8")]
        keys = [(row["finding"], row["patient_key"],
                 row["negative_source_image_path"],
                 row["positive_source_image_path"]) for row in rows]
        if (keys != expected_keys or len(rows) != receipt["pairs"]
                or receipt.get("pairs_by_finding") != {
                    finding: sum(row["finding"] == finding for row in rows)
                    for finding in map_set["findings"]}):
            raise ValueError(f"Map-control patient pairs differ for {arm}")
        for row in rows:
            if (row.get("arm") != arm
                    or row.get("checkpoint_sha256")
                       != lock["arms"][arm]["checkpoint_sha256"]
                    or row.get("map_set_sha256") != sha256(args.map_set)
                    or row.get("test_accessed") is not False):
                raise ValueError("Map-control row provenance mismatch")
            definition = map_set["findings"][row["finding"]]
            for state in ("negative_real", "positive_real", "blank"):
                block = row[state]
                if (not math.isfinite(float(block["answer_margin"]))
                        or [item["layer"] for item in block["map_layers"]]
                           != definition["layers"]):
                    raise ValueError("Map-control score shape or margin invalid")
                for item in block["map_layers"]:
                    if (item["map_sha256"] != definition["map_sha256"]
                            or not math.isfinite(float(item["calibrated_logit"]))
                            or item["diagnostic_present"]
                               != (item["calibrated_logit"] >= 0)):
                        raise ValueError("Map-control layer score invalid")
        by_arm[arm] = rows
        receipt_hashes[arm] = sha256(receipt_path)
    aggregate, per_pair = [], []
    for arm, rows in by_arm.items():
        for finding, definition in map_set["findings"].items():
            relevant = [row for row in rows if row["finding"] == finding]
            for position, layer in enumerate(definition["layers"]):
                positive = [float(row["positive_real"]["map_layers"][position]
                                  ["calibrated_logit"]) for row in relevant]
                negative = [float(row["negative_real"]["map_layers"][position]
                                  ["calibrated_logit"]) for row in relevant]
                blank = [float(row["blank"]["map_layers"][position]
                               ["calibrated_logit"]) for row in relevant]
                wins = [int(pos > neg) for pos, neg in zip(positive, negative, strict=True)]
                low, high = interval(wins, args.bootstrap_replicates,
                                     args.seed + 1000 * ord(arm) + 10 * layer)
                aggregate.append({"arm": arm, "checkpoint_step": lock["arms"][arm]["step"],
                                  "finding": finding, "layer": layer,
                                  "paired_patients": len(relevant),
                                  "positive_higher_pairs": sum(wins),
                                  "positive_higher_fraction": sum(wins) / len(wins),
                                  "positive_higher_ci95_low": low,
                                  "positive_higher_ci95_high": high,
                                  "median_positive_minus_negative_logit": statistics.median(
                                      pos - neg for pos, neg in zip(positive, negative)),
                                  "median_abs_positive_blank_change": statistics.median(
                                      abs(pos - zero) for pos, zero in zip(positive, blank)),
                                  "median_abs_negative_blank_change": statistics.median(
                                      abs(neg - zero) for neg, zero in zip(negative, blank)),
                                  "positive_above_zero_fraction": sum(x >= 0 for x in positive) / len(positive),
                                  "negative_above_zero_fraction": sum(x >= 0 for x in negative) / len(negative),
                                  "claim_boundary": "same-patient image controls; not lesion localization"})
                for row, pos, neg, zero in zip(relevant, positive, negative, blank):
                    per_pair.append({"arm": arm, "checkpoint_step": lock["arms"][arm]["step"],
                                     "finding": finding, "layer": layer,
                                     "patient_key": row["patient_key"],
                                     "negative_source_image_path": row["negative_source_image_path"],
                                     "positive_source_image_path": row["positive_source_image_path"],
                                     "positive_logit": pos, "negative_logit": neg,
                                     "blank_logit": zero,
                                     "positive_higher": int(pos > neg)})
    args.output_dir.mkdir(parents=True)
    aggregate_path = args.output_dir / "map_image_dependence.csv"
    pair_path = args.output_dir / "map_image_dependence_pairs.csv"
    write_csv(aggregate_path, aggregate)
    write_csv(pair_path, per_pair)
    result = {"status": "abc_posttraining_map_controls_summarized",
              "selection_lock_sha256": sha256(args.selection_lock),
              "map_set_sha256": sha256(args.map_set),
              "source_receipt_sha256": receipt_hashes,
              "rows": len(aggregate), "pair_layer_rows": len(per_pair),
              "table_sha256": {"aggregate": sha256(aggregate_path),
                               "per_pair": sha256(pair_path)},
              "test_accessed": False,
              "claim_boundary": "validation image-dependence diagnostics, not final-answer accuracy"}
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "output": str(args.output_dir)}))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("split-csv", "selection-lock", "map-set", "control-root",
                 "output-dir", "remote-root", "local-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=4106)
    run(parser.parse_args())
