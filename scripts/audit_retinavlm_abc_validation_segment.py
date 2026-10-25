"""Audit one collected A/B/C validation segment before aggregate reporting."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from collect_retinavlm_abc_validation_segments import receipt_verified
from retinavlm_abc_artifact_paths import local_artifact
from retinavlm_olives_report import parse_report_claims
from train_retinavlm_adapter_olives import QUESTIONS, SPLIT_SHA, load_rows, sha256


def run(args: argparse.Namespace) -> dict:
    manifest = json.loads(args.manifest.read_text())
    if not receipt_verified(args.manifest, args.transfer_receipt):
        raise ValueError("Cluster/local transfer receipt is not verified")
    arm, step = manifest["arm"], manifest["checkpoint_step"]
    start, stop = manifest["start"], manifest["stop"]
    if arm not in "ABC" or not 0 <= start < stop <= 1470:
        raise ValueError("Unexpected validation segment identity")
    artifacts = {Path(entry["remote"]).name: Path(entry["local"])
                 for entry in manifest["artifacts"]}
    base = f"validation_{start:05d}_{stop:05d}"
    rows_path = artifacts[base + ".jsonl"]
    result_path = artifacts[base + ".receipt.json"]
    segment = json.loads(result_path.read_text())
    candidate_set = json.loads(args.candidates.read_text())
    map_set = json.loads(args.map_set.read_text())
    map_sha = sha256(args.map_set)
    entries = [entry for entry in candidate_set["arms"][arm]
               if entry["step"] == step]
    if (len(entries) != 1
            or candidate_set["map_set_sha256"] != map_sha
            or map_set["status"] != "validated_retinavlm_olives_map_set"
            or map_set["split_sha256"] != SPLIT_SHA):
        raise ValueError("Frozen candidate/map mismatch")
    checkpoint = local_artifact(entries[0]["checkpoint"], args.remote_root,
                                args.local_root)
    checkpoint_sha = sha256(checkpoint)
    if (segment.get("status") != "complete_abc_inference_segment"
            or segment.get("split") != "validation"
            or segment.get("arm") != arm
            or segment.get("checkpoint_step") != step
            or segment.get("checkpoint_sha256") != checkpoint_sha
            or segment.get("map_set_sha256") != map_sha
            or segment.get("split_sha256") != SPLIT_SHA
            or segment.get("start") != start or segment.get("stop") != stop
            or segment.get("rows") != stop - start
            or segment.get("rows_sha256") != sha256(rows_path)):
        raise ValueError("Completed validation segment receipt mismatch")
    official = load_rows(args.split_csv)["validation"]
    if len(official) != 1470:
        raise ValueError("Validation cohort size changed")
    count = 0
    with rows_path.open(encoding="utf-8") as file:
        for count, line in enumerate(file, 1):
            row = json.loads(line)
            index = start + count - 1
            if index >= stop:
                raise ValueError("More rows than declared in segment")
            source = official[index]
            if (row.get("index") != index
                    or row.get("arm") != arm or row.get("split") != "validation"
                    or row.get("checkpoint_step") != step
                    or row.get("checkpoint_sha256") != checkpoint_sha
                    or row.get("map_set_sha256") != map_sha
                    or row.get("source_image_path") != source["source_image_path"]
                    or row.get("patient_key") != source["patient_key"]
                    or row.get("parsed_report_claims") != parse_report_claims(
                        row.get("generated_report", ""))
                    or set(row.get("finding_predictions", {})) != set(QUESTIONS)):
                raise ValueError(f"Invalid completed validation row {index}")
            for finding in QUESTIONS:
                prediction = row["finding_predictions"][finding]
                label = source[finding]
                expected_label = int(label) if label in {"0", "1"} else None
                definition = map_set["findings"].get(finding)
                if (prediction["expert_label"] != expected_label
                        or not math.isfinite(float(prediction["answer_margin"]))
                        or [item["layer"] for item in prediction["map_layers"]]
                        != (definition["layers"] if definition else [])):
                    raise ValueError(f"Finding mismatch at scan {index}: {finding}")
                for item in prediction["map_layers"]:
                    if (item["map_sha256"] != definition["map_sha256"]
                            or not math.isfinite(float(item["raw_score"]))
                            or not math.isfinite(float(item["calibrated_logit"]))
                            or item["diagnostic_present"]
                            != (item["calibrated_logit"] >= 0)):
                        raise ValueError(f"Map mismatch at scan {index}: {finding}")
    if count != stop - start:
        raise ValueError(f"Expected {stop - start} rows, found {count}")
    audit = {"status": "abc_validation_segment_local_audit_passed",
             "job_id": manifest["job_id"], "arm": arm, "checkpoint_step": step,
             "start": start, "stop": stop, "rows": count,
             "checkpoint_sha256": checkpoint_sha, "map_set_sha256": map_sha,
             "rows_sha256": sha256(rows_path),
             "segment_receipt_sha256": sha256(result_path),
             "transfer_receipt_sha256": sha256(args.transfer_receipt),
             "claim_boundary": "one completed validation segment; no full-cohort metric"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".writing")
    temporary.write_text(json.dumps(audit, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"status": audit["status"], "job_id": audit["job_id"],
                      "rows": count, "output": str(args.output)}), flush=True)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--transfer-receipt", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--map-set", type=Path, required=True)
    parser.add_argument("--remote-root", type=Path,
                        default=Path("/home/h/heshi/retinavlm-scratch"))
    parser.add_argument("--local-root", type=Path,
                        default=Path("/Volumes/mac外接盘/RetinaVLM_JLens_OLIVES"))
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
