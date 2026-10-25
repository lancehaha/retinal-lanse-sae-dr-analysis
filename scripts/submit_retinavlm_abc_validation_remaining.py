"""Submit unsent validation segments as Slurm submit slots become available.

The schedule is durable: after an ambiguous SSH result, reconcile the unique
Slurm job name before another submission. Never open the test split here.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess

from build_retinavlm_abc_eval_transfer_manifest import build
from retrieve_hashed_cluster_artifacts import sha256


def save(path: Path, schedule: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(schedule, indent=2) + "\n")
    temporary.replace(path)


def remote(wrapper: Path, config: Path, command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/bin/zsh", str(wrapper), "ssh", "-F", str(config), "-o", "ControlMaster=no",
         "retinavlm-xlogin", command],
        capture_output=True, text=True, timeout=90, check=False,
    )


def recover_job(wrapper: Path, config: Path, name: str) -> int | None:
    result = remote(wrapper, config,
                    f"sacct -X -n -P -u \"$USER\" --name {name} "
                    "--format=JobIDRaw,JobName,State")
    if result.returncode:
        raise RuntimeError(f"Cannot query Slurm accounting for {name}: {result.stderr}")
    found = set()
    for line in result.stdout.splitlines():
        fields = line.strip().split("|")
        if len(fields) >= 2 and fields[1] == name and fields[0].isdigit():
            found.add(int(fields[0]))
    if len(found) > 1:
        raise RuntimeError(f"Multiple jobs named {name}: {sorted(found)}")
    return next(iter(found), None)


def ensure_manifest(item: dict) -> None:
    job_id = item["job_id"]
    path = Path(item["manifest"])
    if path.exists():
        existing = json.loads(path.read_text())
        expected = (item["arm"], item["step"], item["start"], item["stop"], job_id)
        actual = (existing["arm"], existing["checkpoint_step"],
                  existing["start"], existing["stop"], existing["job_id"])
        if actual != expected:
            raise ValueError(f"Manifest identity mismatch: {path}")
        return
    build("validation", item["arm"], item["step"], item["start"],
          item["stop"], job_id, path,
          Path("/home/h/heshi/retinavlm-scratch"),
          Path("/Volumes/mac外接盘/RetinaVLM_JLens_OLIVES"))


def short_training_ready(schedule: dict) -> bool:
    """Only release step-3500 validation after all three local training audits."""
    root = Path("/Volumes/mac外接盘/RetinaVLM_JLens_OLIVES")
    summary_path = root / "abc_training/training_curves_step3500/summary.json"
    if not summary_path.is_file():
        return False
    summary = json.loads(summary_path.read_text())
    if (summary.get("status") != "matched_abc_training_curve_audited"
            or summary.get("steps_per_arm") != 3500
            or summary.get("rows") != 10500
            or summary.get("matched_sample_schedule") is not True):
        return False
    for arm in "ABC":
        audit_path = root / "cluster_stage" / f"abc_{arm}_step3500_local_audit.json"
        checkpoint = (root / "abc_training" / f"{arm}_seed42" / "checkpoints" /
                      "checkpoint_step_003500.pt")
        if not audit_path.is_file() or not checkpoint.is_file():
            return False
        audit = json.loads(audit_path.read_text())
        if (audit.get("status") != "abc_training_segment_local_audit_passed"
                or audit.get("arm") != arm or audit.get("step") != 3500
                or audit.get("batch_rows") != 3500
                or audit.get("prior_batch_prefix_identical") is not True
                or audit.get("checkpoint_sha256") != sha256(checkpoint)
                or audit.get("slurm_job_id") != schedule["training_jobs"][arm]):
            return False
    return True


def dependency_option(wrapper: Path, config: Path, prior: int | None) -> str:
    if prior is None:
        return ""
    result = remote(wrapper, config,
                    f"sacct -X -n -P -j {prior} --format=JobIDRaw,State")
    if result.returncode:
        raise RuntimeError(f"Cannot inspect prerequisite {prior}: {result.stderr}")
    states = [line.strip().split("|")[1].split()[0]
              for line in result.stdout.splitlines()
              if line.strip().startswith(f"{prior}|")]
    if len(states) != 1:
        raise RuntimeError(f"Ambiguous prerequisite state for {prior}")
    if states[0] == "COMPLETED":
        return ""
    if states[0] in {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING"}:
        return f"--dependency=afterok:{prior} "
    raise RuntimeError(f"Prerequisite {prior} ended in {states[0]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--ssh-config", type=Path, required=True)
    parser.add_argument("--expect-wrapper", type=Path, required=True)
    parser.add_argument("--submit", action="store_true",
                        help="Actually submit available jobs; default is a read-only preview")
    parser.add_argument("--max-new", type=int, default=10)
    args = parser.parse_args()
    if args.max_new < 1 or not args.expect_wrapper.is_file() or not args.ssh_config.is_file():
        raise ValueError("Invalid submission inputs")
    schedule = json.loads(args.schedule.read_text())
    jobs = schedule["jobs"]
    short = len(jobs) == 6
    if short:
        expected = [(arm, 3500, start, stop)
                    for arm in "ABC" for start, stop in ((0, 400), (400, 800))]
        found = [(j["arm"], j["step"], j["start"], j["stop"]) for j in jobs]
        if (found != expected or schedule.get("split") != "validation"
                or schedule.get("scans_per_arm") != 800):
            raise ValueError("Unexpected step-3500 first-800 schedule")
    elif len(jobs) != 36 or any(j["job_id"] is None for j in jobs[:26]):
        raise ValueError("Unexpected frozen validation schedule")
    seen_missing = False
    for item in jobs:
        if item["job_id"] is None:
            seen_missing = True
        elif seen_missing:
            raise ValueError("Submitted jobs must form one schedule prefix")
        else:
            ensure_manifest(item)
    first = next((i for i, item in enumerate(jobs) if item["job_id"] is None), None)
    if first is None:
        print(json.dumps({"status": "all_validation_segments_submitted", "jobs": len(jobs)}))
        return
    if short and not short_training_ready(schedule):
        print(json.dumps({"status": "waiting_for_all_step3500_training_audits",
                          "submitted": first, "remaining": len(jobs) - first}))
        return
    if not args.submit:
        print(json.dumps({"status": "preview", "next": jobs[first],
                          "dependency": jobs[first - 1]["job_id"] if first else None,
                          "remaining": len(jobs) - first}))
        return

    added = 0
    for i in range(first, len(jobs)):
        if added >= args.max_new:
            break
        item = jobs[i]
        prior = jobs[i - 1]["job_id"] if i else None
        if prior is None and i:
            raise ValueError("Missing preceding Slurm job")
        arm, step, start, stop = (item[k] for k in ("arm", "step", "start", "stop"))
        allowed_boundaries = (((0, 400), (400, 800)) if short else
                              ((0, 400), (400, 800), (800, 1200), (1200, 1470)))
        if (arm not in "ABC" or step not in ((3500,) if short else (1000, 3000, 6000))
                or (start, stop) not in allowed_boundaries):
            raise ValueError("Unexpected validation segment")
        name = f"abc-v-{arm}-{step}-{start}"
        recovered = recover_job(args.expect_wrapper, args.ssh_config, name)
        if recovered is None and item.get("submission_attempt_started"):
            raise RuntimeError(f"Ambiguous prior submission for {name}; inspect Slurm before retry")
        if recovered is None:
            count = remote(args.expect_wrapper, args.ssh_config,
                           'squeue -u "$USER" -h | wc -l')
            if count.returncode:
                raise RuntimeError(f"Cannot read submit count: {count.stderr}")
            numbers = re.findall(r"^\s*(\d+)\s*$", count.stdout, flags=re.MULTILINE)
            if len(numbers) != 1:
                raise RuntimeError(f"Unexpected submit count: {count.stdout}")
            if int(numbers[0]) >= 32:
                print(json.dumps({"status": "slurm_submit_limit", "added": added,
                                  "next": name}))
                break
            dep = dependency_option(args.expect_wrapper, args.ssh_config, prior)
            item["submission_attempt_started"] = datetime.now(timezone.utc).isoformat()
            save(args.schedule, schedule)
            command = (
                f"sbatch --parsable {dep}--job-name={name} "
                f"--export=ALL,ARM={arm},SPLIT=validation,START={start},STOP={stop},"
                f"SEED=42,CHECKPOINT=/home/h/heshi/retinavlm-scratch/abc_training/"
                f"{arm}_seed42/checkpoint_step_{step:06d}.pt,"
                "MAP_SET=/home/h/heshi/retinavlm-scratch/jlens_maps_singleton/"
                "validated_map_set_combined.json,"
                f"EVAL_ROOT=/home/h/heshi/retinavlm-scratch/abc_validation_singleton/"
                f"step_{step:06d} /home/h/heshi/retinavlm-scratch/abc_training_code/"
                "retinavlm_abc_eval_segment.sbatch"
            )
            submitted = remote(args.expect_wrapper, args.ssh_config, command)
            match = re.search(r"(?:^|\n)(\d{6,})\s*$", submitted.stdout)
            if submitted.returncode or not match:
                if "AssocMaxSubmitJobLimit" in submitted.stderr:
                    item.pop("submission_attempt_started", None)
                    save(args.schedule, schedule)
                    print(json.dumps({"status": "slurm_submit_limit", "added": added,
                                      "next": name}))
                    break
                raise RuntimeError(f"Submission result for {name} needs reconciliation: "
                                   f"{submitted.stdout} {submitted.stderr}")
            recovered = int(match.group(1))
        item["job_id"] = recovered
        item["manifest"] = str(args.schedule.parent /
                               f"abc_validation_{arm}_step{step}_{start:05d}_{stop:05d}"
                               "_transfer_manifest.json")
        ensure_manifest(item)
        item.pop("submission_attempt_started", None)
        schedule["submitted_count"] = i + 1
        schedule["remaining_count"] = len(jobs) - i - 1
        schedule["last_submitted_job_id"] = recovered
        schedule["remaining_dependency_anchor"] = recovered
        schedule["status"] = ("validation_all_segments_submitted" if i + 1 == len(jobs)
                              else "validation_submission_in_progress")
        save(args.schedule, schedule)
        added += 1
        print(json.dumps({"status": "submitted_or_recovered", "job_id": recovered,
                          "segment": name, "remaining": schedule["remaining_count"]}))


if __name__ == "__main__":
    main()
