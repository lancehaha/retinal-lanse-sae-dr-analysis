"""Sample parsed A/B/C report claims for a blinded extraction-error review."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from lock_retinavlm_olives_abc_validation import ARMS
from summarize_retinavlm_olives_abc_test import read_arm
from train_retinavlm_adapter_olives import QUESTIONS, SPLIT_SHA, load_rows, sha256


def pseudonym(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


def category(label, claim):
    if label == 1:
        return "supported_present" if claim == 1 else "miss_or_unparsed"
    if claim == 1:
        return "unsupported_present"
    return "supported_absent" if claim == 0 else "negative_unparsed"


def choose_patient_diverse(rows, limit, rng):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["patient_key"]].append(row)
    patients = sorted(grouped)
    rng.shuffle(patients)
    for patient in patients:
        grouped[patient].sort(key=lambda row: row["source_image_path"])
        rng.shuffle(grouped[patient])
    selected = []
    while len(selected) < limit and any(grouped.values()):
        for patient in patients:
            if grouped[patient] and len(selected) < limit:
                selected.append(grouped[patient].pop())
    return selected


def run(args):
    if args.output.exists() or args.per_stratum < 1:
        raise ValueError("Audit output exists or sample count is invalid")
    lock = json.loads(args.selection_lock.read_text())
    map_set = json.loads(args.map_set.read_text())
    manifest = json.loads(args.test_manifest.read_text())
    if (lock.get("status") != "locked_abc_validation_selection"
            or lock.get("split_sha256") != SPLIT_SHA
            or lock.get("map_set_sha256") != sha256(args.map_set)
            or map_set.get("status") != "validated_retinavlm_olives_map_set"
            or manifest.get("status") != "locked_abc_test_segments"
            or manifest.get("selection_lock_sha256") != sha256(args.selection_lock)
            or set(manifest.get("arms", {})) != set(ARMS)):
        raise ValueError("Test and selection provenance mismatch")
    official = load_rows(args.split_csv)["test"]
    strata = defaultdict(list)
    for arm in ARMS:
        rows, _ = read_arm(Path(manifest["arms"][arm]), arm, official, lock,
                           sha256(args.selection_lock), map_set,
                           getattr(args, "remote_root", None),
                           getattr(args, "local_root", None))
        for row, source in zip(rows, official, strict=True):
            for finding in QUESTIONS:
                if source[finding] not in {"0", "1"}:
                    continue
                label = int(source[finding])
                claim = row["parsed_report_claims"][finding]
                group = category(label, claim)
                strata[(arm, finding, group)].append({
                    "arm": arm, "finding": finding, "category": group,
                    "expert_label": label, "parsed_claim": claim,
                    "patient_key": row["patient_key"],
                    "source_image_path": row["source_image_path"],
                    "generated_report": row["generated_report"],
                })
    rng = random.Random(args.seed)
    selected = []
    denominators = []
    for key in sorted(strata):
        group = strata[key]
        denominators.append({"arm": key[0], "finding": key[1],
                             "category": key[2], "available": len(group)})
        selected.extend(choose_patient_diverse(group, args.per_stratum, rng))
    if not selected:
        raise ValueError("No report cases available for audit")
    output = []
    for number, row in enumerate(selected, 1):
        output.append({"audit_case": number,
                       "arm": row["arm"], "finding": row["finding"],
                       "category": row["category"],
                       "expert_label": row["expert_label"],
                       "parsed_claim": row["parsed_claim"],
                       "patient_pseudonym": pseudonym(row["patient_key"]),
                       "image_pseudonym": pseudonym(row["source_image_path"]),
                       "source_image_path": row["source_image_path"],
                       "generated_report": row["generated_report"],
                       "reviewer_corrected_claim": "",
                       "reviewer_note": ""})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    receipt = {"status": "report_extraction_audit_sample_prepared",
               "selection_lock_sha256": sha256(args.selection_lock),
               "test_manifest_sha256": sha256(args.test_manifest),
               "audit_csv_sha256": sha256(args.output),
               "cases": len(output), "per_stratum_limit": args.per_stratum,
               "seed": args.seed, "available_denominators": denominators,
               "review_status": "unreviewed",
               "limitation": "sampled parser review; not an expert image relabeling"}
    receipt_path = args.output.with_suffix(".receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": receipt["status"], "cases": len(output),
                      "output": str(args.output)}), flush=True)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--map-set", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-stratum", type=int, default=3)
    parser.add_argument("--seed", type=int, default=4106)
    parser.add_argument("--remote-root", type=Path)
    parser.add_argument("--local-root", type=Path)
    run(parser.parse_args())
