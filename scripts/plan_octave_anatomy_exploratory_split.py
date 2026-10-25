"""Freeze a disease/annotation-stratified OCTAVE volume split for exploratory anatomy fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


SEED = "retinavlm-octave-anatomy-20261024-v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(inventory_path: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("Output already exists")
    inventory = json.loads(inventory_path.read_text())
    if inventory.get("status") != "external_anatomy_inventory_audited_no_olives_use" or inventory.get("training_volumes") != 198:
        raise ValueError("Unexpected inventory")
    strata = defaultdict(list)
    for row in inventory["volumes"]:
        group = row["name"].split("_")[1]
        positive = int(row["class_presence"]["HYA"] > 0 or row["class_presence"]["RHS"] > 0)
        strata[(group, positive)].append(row)
    fit, evaluation = [], []
    strata_counts = []
    for (group, positive), rows in sorted(strata.items()):
        ordered = sorted(rows, key=lambda row: hashlib.sha256((SEED + "|" + row["name"]).encode()).hexdigest())
        count = len(rows)
        eval_count = min(count - 1, max(1, round(0.2 * count))) if count >= 2 else 0
        evaluation.extend(row["name"] for row in ordered[:eval_count])
        fit.extend(row["name"] for row in ordered[eval_count:])
        strata_counts.append({"disease_group": group, "hya_or_rhs_volume_positive": bool(positive),
                              "fit_volumes": count - eval_count, "evaluation_volumes": eval_count})
    if len(fit) != 158 or len(evaluation) != 40 or set(fit) & set(evaluation):
        raise ValueError("Unexpected split sizes or overlap")
    result = {"status": "frozen_exploratory_external_volume_split_no_model_training",
              "source_inventory_sha256": sha256(inventory_path),
              "archive_sha256": inventory["archive_sha256"],
              "seed": SEED, "rule": "Within disease group and HYA-or-RHS presence: SHA-order; hold out rounded 20%, at least one and leaving one fit volume",
              "fit_volumes": sorted(fit), "evaluation_volumes": sorted(evaluation),
              "strata_counts": strata_counts,
              "olives_accessed": False,
              "independence_boundary": "Volume-disjoint only; no patient identifiers in OCTAVE archive, so patient-disjointness cannot be verified",
              "use_boundary": "Anatomy feasibility only; not an OLIVES B2/B3 map gate or independent clinical test"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.inventory, args.output)
    print(json.dumps({"status": result["status"], "fit_volumes": len(result["fit_volumes"]),
                      "evaluation_volumes": len(result["evaluation_volumes"])}))
