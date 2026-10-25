"""Preserve and audit an immutable prefix of a running validation segment."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import subprocess

from retinavlm_abc_artifact_paths import local_artifact
from retinavlm_olives_report import parse_report_claims
from retrieve_hashed_cluster_artifacts import run as retrieve
from train_retinavlm_adapter_olives import QUESTIONS, SPLIT_SHA, load_rows, sha256


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict:
    schedule = json.loads(args.schedule.read_text())
    matches = [job for job in schedule["jobs"] if job["job_id"] == args.job_id]
    if len(matches) != 1:
        raise ValueError("Job ID not unique in frozen validation schedule")
    job = matches[0]
    start, stop = job["start"], job["stop"]
    if not 1 <= args.rows <= stop - start:
        raise ValueError("Invalid fixed-prefix length")
    full_manifest = json.loads(Path(job["manifest"]).read_text())
    expected = (args.job_id, job["arm"], job["step"], start, stop)
    actual = tuple(full_manifest[key] for key in
                   ("job_id", "arm", "checkpoint_step", "start", "stop"))
    if actual != expected:
        raise ValueError("Validation job/manifest mismatch")
    source = next((Path(item["remote"]) for item in full_manifest["artifacts"]
                   if item["remote"].endswith(f"validation_{start:05d}_{stop:05d}.jsonl")), None)
    if source is None:
        raise ValueError("No expected full validation JSONL")
    prefix_end = start + args.rows
    name = f"validation_{start:05d}_{prefix_end:05d}.progress.jsonl"
    remote = source.with_name(name)
    local = (args.local_root / "abc_validation_singleton" /
             f"step_{job['step']:06d}" / job["arm"] / "progress_snapshots" / name)
    remote_command = (
        "set -eu; "
        f"source={shlex.quote(str(source))}; snapshot={shlex.quote(str(remote))}; "
        f"test \"$(wc -l < \"$source\")\" -ge {args.rows}; "
        "if ! test -f \"$snapshot\"; then "
        f"head -n {args.rows} \"$source\" > \"$snapshot.writing\"; "
        f"test \"$(wc -l < \"$snapshot.writing\")\" -eq {args.rows}; "
        "mv \"$snapshot.writing\" \"$snapshot\"; fi; "
        "sha256sum -- \"$snapshot\""
    )
    result = subprocess.run(
        ["/bin/zsh", str(args.expect_wrapper), "ssh", "-F", str(args.ssh_config),
         "-o", "ControlMaster=no", "retinavlm-xlogin", remote_command],
        capture_output=True, text=True, timeout=90, check=False,
    )
    if result.returncode:
        raise RuntimeError(f"Cluster snapshot failed: {result.stdout[-600:]} {result.stderr[-300:]}")
    stem = f"abc_validation_{job['arm']}_step{job['step']}_{start:05d}_{prefix_end:05d}_progress"
    manifest_path = args.local_root / "cluster_stage" / f"{stem}_transfer_manifest.json"
    receipt_path = args.local_root / "cluster_stage" / f"{stem}_transfer_receipt.json"
    manifest = {"status": "expected_running_validation_progress_backup",
                "job_id": args.job_id, "arm": job["arm"], "step": job["step"],
                "start": start, "stop": prefix_end,
                "artifacts": [{"remote": str(remote), "local": str(local)}]}
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Existing progress manifest changed")
    else:
        save_json(manifest_path, manifest)
    retrieve(argparse.Namespace(manifest=manifest_path, receipt=receipt_path,
                                ssh_config=args.ssh_config,
                                expect_wrapper=args.expect_wrapper,
                                host="retinavlm-xlogin"))
    transfer = json.loads(receipt_path.read_text())
    if (transfer["status"] != "cluster_artifacts_retrieved_and_verified"
            or transfer["manifest_sha256"] != sha256(manifest_path)
            or len(transfer["artifacts"]) != 1
            or transfer["artifacts"][0]["sha256"] != sha256(local)):
        raise ValueError("Progress transfer receipt mismatch")
    official = load_rows(args.split_csv)["validation"]
    map_set = json.loads(args.map_set.read_text())
    map_sha = sha256(args.map_set)
    candidate_set = json.loads(args.candidates.read_text())
    entries = [item for item in candidate_set["arms"][job["arm"]]
               if item["step"] == job["step"]]
    if (len(entries) != 1 or map_set["status"] != "validated_retinavlm_olives_map_set"
            or map_set["split_sha256"] != SPLIT_SHA
            or candidate_set["map_set_sha256"] != map_sha):
        raise ValueError("Frozen candidate/map provenance changed")
    checkpoint = local_artifact(entries[0]["checkpoint"], args.remote_root, args.local_root)
    checkpoint_sha = sha256(checkpoint)
    count = 0
    with local.open(encoding="utf-8") as file:
        for count, line in enumerate(file, 1):
            row = json.loads(line)
            index = start + count - 1
            source_row = official[index]
            if (row.get("index") != index or row.get("arm") != job["arm"]
                    or row.get("split") != "validation"
                    or row.get("checkpoint_step") != job["step"]
                    or row.get("checkpoint_sha256") != checkpoint_sha
                    or row.get("map_set_sha256") != map_sha
                    or row.get("source_image_path") != source_row["source_image_path"]
                    or row.get("patient_key") != source_row["patient_key"]
                    or row.get("parsed_report_claims") != parse_report_claims(
                        row.get("generated_report", ""))
                    or set(row.get("finding_predictions", {})) != set(QUESTIONS)):
                raise ValueError(f"Invalid validation row {index}")
            for finding in QUESTIONS:
                prediction = row["finding_predictions"][finding]
                label = source_row[finding]
                expected_label = int(label) if label in {"0", "1"} else None
                definition = map_set["findings"].get(finding)
                if (prediction["expert_label"] != expected_label
                        or not math.isfinite(float(prediction["answer_margin"]))
                        or [item["layer"] for item in prediction["map_layers"]]
                        != (definition["layers"] if definition else [])):
                    raise ValueError(f"Finding row mismatch at {index}: {finding}")
                for item in prediction["map_layers"]:
                    if (item["map_sha256"] != definition["map_sha256"]
                            or not math.isfinite(float(item["calibrated_logit"]))):
                        raise ValueError(f"Map mismatch at {index}: {finding}")
    if count != args.rows:
        raise ValueError(f"Expected {args.rows} progress rows, found {count}")
    audit = {"status": "running_validation_progress_backup_verified",
             "job_id": args.job_id, "arm": job["arm"], "step": job["step"],
             "scan_indices": [start, prefix_end - 1], "rows": count,
             "rows_sha256": sha256(local), "checkpoint_sha256": checkpoint_sha,
             "map_set_sha256": map_sha,
             "transfer_receipt_sha256": sha256(receipt_path),
             "claim_boundary": "progress recovery only; no completed-segment metric"}
    audit_path = local.parent / f"validation_{start:05d}_{prefix_end:05d}.progress_audit.json"
    save_json(audit_path, audit)
    print(json.dumps({"status": audit["status"], "rows": count,
                      "audit": str(audit_path)}), flush=True)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--map-set", type=Path, required=True)
    parser.add_argument("--remote-root", type=Path, default=Path("/home/h/heshi/retinavlm-scratch"))
    parser.add_argument("--local-root", type=Path, default=Path("/Volumes/mac外接盘/RetinaVLM_JLens_OLIVES"))
    parser.add_argument("--ssh-config", type=Path, required=True)
    parser.add_argument("--expect-wrapper", type=Path, required=True)
    run(parser.parse_args())
