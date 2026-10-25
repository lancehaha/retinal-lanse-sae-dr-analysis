"""Collect completed A/B/C validation segments from a frozen Slurm schedule.

Jobs still pending/running are left alone. A completed job is copied using its
declared manifest, with cluster/local SHA-256 parity checked per artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess

from retrieve_hashed_cluster_artifacts import sha256


def slurm_states(wrapper: Path, config: Path, job_ids: list[int]) -> dict[int, tuple[str, str]]:
    command = f"sacct -X -n -P -j {','.join(map(str, job_ids))} --format=JobIDRaw,State,ExitCode"
    result = subprocess.run(
        ["/bin/zsh", str(wrapper), "ssh", "-F", str(config),
         "-o", "ControlMaster=no", "retinavlm-xlogin", command],
        capture_output=True, text=True, timeout=90, check=False,
    )
    if result.returncode:
        raise RuntimeError(f"Slurm query failed: {result.stderr[-800:]}")
    states = {}
    for line in result.stdout.replace("\r", "").splitlines():
        match = re.fullmatch(r"(\d+)\|([^|]+)\|(\d+:\d+)", line.strip())
        if match:
            states[int(match[1])] = (match[2].split()[0].rstrip("+"), match[3])
    missing = set(job_ids) - states.keys()
    if missing:
        raise RuntimeError(f"No Slurm accounting state for: {sorted(missing)}")
    return states


def receipt_verified(manifest_path: Path, receipt_path: Path) -> bool:
    if not receipt_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("status") != "cluster_artifacts_retrieved_and_verified"
            or receipt.get("manifest_sha256") != sha256(manifest_path)):
        return False
    declared = {(entry["remote"], entry["local"]) for entry in manifest["artifacts"]}
    actual = {(entry["remote"], entry["local"]) for entry in receipt.get("artifacts", [])}
    if declared != actual or len(declared) != len(manifest["artifacts"]):
        return False
    for entry in receipt["artifacts"]:
        path = Path(entry["local"])
        if not path.is_file() or sha256(path) != entry["sha256"]:
            return False
    return True


def audit_verified(manifest_path: Path, receipt_path: Path, audit_path: Path) -> bool:
    if not audit_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text())
    audit = json.loads(audit_path.read_text())
    rows_name = (f"validation_{manifest['start']:05d}_"
                 f"{manifest['stop']:05d}.jsonl")
    rows = [Path(item["local"]) for item in manifest["artifacts"]
            if Path(item["local"]).name == rows_name]
    return (len(rows) == 1
            and audit.get("status") == "abc_validation_segment_local_audit_passed"
            and audit.get("job_id") == manifest["job_id"]
            and audit.get("rows_sha256") == sha256(rows[0])
            and audit.get("transfer_receipt_sha256") == sha256(receipt_path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--ssh-config", type=Path, required=True)
    parser.add_argument("--expect-wrapper", type=Path, required=True)
    parser.add_argument("--collect", action="store_true", help="Copy completed jobs")
    parser.add_argument("--max-new", type=int, default=36)
    args = parser.parse_args()
    if args.max_new < 1:
        raise ValueError("--max-new must be positive")
    schedule = json.loads(args.schedule.read_text())
    jobs = schedule["jobs"]
    if len(jobs) == 6:
        expected = [(arm, 3500, start, stop)
                    for arm in "ABC" for start, stop in ((0, 400), (400, 800))]
        found = [(job["arm"], job["step"], job["start"], job["stop"])
                 for job in jobs]
        if found != expected:
            raise ValueError("Unexpected six-segment step-3500 validation schedule")
    elif len(jobs) != 36:
        raise ValueError("Expected six updated or 36 original validation segments")
    submitted = [job for job in jobs if job["job_id"] is not None]
    states = slurm_states(args.expect_wrapper, args.ssh_config,
                          [int(job["job_id"]) for job in submitted])
    summary = {"submitted": len(submitted), "unsent": len(jobs) - len(submitted),
               "verified": 0, "collected": 0, "audited": 0,
               "new_audits": 0, "waiting": 0, "failed": []}
    for job in submitted:
        job_id = int(job["job_id"])
        state, exit_code = states[job_id]
        if state != "COMPLETED" or exit_code != "0:0":
            if state in {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING"}:
                summary["waiting"] += 1
            else:
                summary["failed"].append({"job_id": job_id, "state": state,
                                          "exit_code": exit_code})
            continue
        manifest = Path(job["manifest"])
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        identity = json.loads(manifest.read_text())
        expected = (job_id, job["arm"], job["step"], job["start"], job["stop"])
        found = tuple(identity[key] for key in
                      ("job_id", "arm", "checkpoint_step", "start", "stop"))
        if found != expected:
            raise ValueError(f"Manifest identity mismatch for job {job_id}")
        receipt = manifest.with_name(manifest.name.replace("_transfer_manifest.json",
                                                            "_transfer_receipt.json"))
        if not receipt_verified(manifest, receipt):
            if not args.collect or summary["collected"] >= args.max_new:
                print(json.dumps({"ready_to_collect": job_id, "manifest": str(manifest)}),
                      flush=True)
                continue
            command = ["python3", str(Path(__file__).with_name("retrieve_hashed_cluster_artifacts.py")),
                       "--manifest", str(manifest), "--receipt", str(receipt),
                       "--ssh-config", str(args.ssh_config),
                       "--expect-wrapper", str(args.expect_wrapper)]
            subprocess.run(command, check=True)
            if not receipt_verified(manifest, receipt):
                raise RuntimeError(f"Retrieved receipt failed local audit: {receipt}")
            summary["collected"] += 1
        summary["verified"] += 1
        audit = manifest.with_name(manifest.name.replace("_transfer_manifest.json",
                                                        "_local_audit.json"))
        if args.collect and not audit_verified(manifest, receipt, audit):
            root = args.schedule.parent.parent
            command = ["python3", str(Path(__file__).with_name(
                           "audit_retinavlm_abc_validation_segment.py")),
                       "--manifest", str(manifest), "--transfer-receipt", str(receipt),
                       "--split-csv", str(root / "official_expert_data" /
                                          "official_six_biomarker_split_v1.csv"),
                       "--candidates", str(schedule.get("candidates") or
                                           root / "abc_validation_singleton" /
                                           "prespecified_candidates.json"),
                       "--map-set", str(root / "jlens_maps_singleton" /
                                        "validated_map_set_combined.json"),
                       "--output", str(audit)]
            subprocess.run(command, check=True)
            if not audit_verified(manifest, receipt, audit):
                raise RuntimeError(f"Completed validation segment audit failed: {audit}")
            summary["new_audits"] += 1
        if audit_verified(manifest, receipt, audit):
            summary["audited"] += 1
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
