"""Verify and stage an immutable A/B/C validation selection lock on the cluster."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import subprocess

from retinavlm_abc_artifact_paths import local_artifact


FINDINGS = {"B1", "B2", "B3", "B4", "B5", "B6", "SRF"}
REMOTE_ROOT = Path("/home/h/heshi/retinavlm-scratch")
LOCAL_ROOT = Path("/Volumes/mac外接盘/RetinaVLM_JLens_OLIVES")
REMOTE_LOCK = REMOTE_ROOT / "abc_validation_singleton" / "selection_lock.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(lock_path: Path, candidates: Path, map_set: Path,
             remote_root: Path, local_root: Path) -> dict:
    lock = json.loads(lock_path.read_text())
    if (lock.get("status") != "locked_abc_validation_selection"
            or lock.get("test_accessed") is not False
            or lock.get("test_targets_used") is not False
            or lock.get("prespecified_steps") != [1000, 3000, 6000]
            or set(lock.get("arms", {})) != {"A", "B", "C"}
            or lock.get("candidate_manifest_sha256") != sha256(candidates)
            or lock.get("map_set_sha256") != sha256(map_set)):
        raise ValueError("Incomplete or changed validation-only selection lock")
    candidate_data = json.loads(candidates.read_text())
    map_data = json.loads(map_set.read_text())
    if (candidate_data.get("status") != "prespecified_abc_validation_candidates"
            or map_data.get("status") != "validated_retinavlm_olives_map_set"
            or candidate_data.get("seed") != 42
            or candidate_data.get("steps") != [1000, 3000, 6000]
            or candidate_data.get("map_set_sha256") != lock["map_set_sha256"]
            or map_data.get("split_sha256") != lock["split_sha256"]):
        raise ValueError("Candidate schedule/map set does not match lock")
    for arm in "ABC":
        selected = lock["arms"][arm]
        checkpoint = local_artifact(selected["checkpoint"], remote_root, local_root)
        if (selected.get("step") not in (1000, 3000, 6000)
                or (selected["step"], selected["checkpoint"]) not in
                   {(entry["step"], entry["checkpoint"])
                    for entry in candidate_data["arms"][arm]}
                or set(selected.get("thresholds_margin", {})) != FINDINGS
                or any(not math.isfinite(float(value)) for value in
                       selected["thresholds_margin"].values())
                or not checkpoint.is_file()
                or sha256(checkpoint) != selected["checkpoint_sha256"]):
            raise ValueError(f"Invalid selected checkpoint or thresholds for {arm}")
    return {"status": "validation_lock_verified_locally",
            "lock_sha256": sha256(lock_path),
            "selected_steps": {arm: lock["arms"][arm]["step"] for arm in "ABC"}}


def command(args: argparse.Namespace, program: str, *parts: str) -> list[str]:
    return ["/bin/zsh", str(args.expect_wrapper), program, "-F",
            str(args.ssh_config), *parts]


def run_command(argv: list[str]) -> str:
    result = subprocess.run(argv, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=90, check=False)
    if result.returncode:
        raise RuntimeError(f"Cluster command failed: {result.stdout[-800:]}")
    return result.stdout.replace("\r", "")


def remote_sha(args: argparse.Namespace, path: Path) -> str | None:
    quoted = shlex.quote(str(path))
    output = run_command(command(args, "ssh", "-o", "ControlMaster=no",
                                 "retinavlm-xlogin",
                                 f"if test -f {quoted}; then sha256sum -- {quoted}; fi"))
    match = re.search(rf"(?m)^([0-9a-f]{{64}})\s+{re.escape(str(path))}$", output)
    return match.group(1) if match else None


def stage(args: argparse.Namespace, digest: str) -> str:
    if args.remote_lock != REMOTE_LOCK:
        raise ValueError(f"Remote lock must be {REMOTE_LOCK}")
    existing = remote_sha(args, args.remote_lock)
    if existing is not None:
        if existing != digest:
            raise ValueError("Cluster already has a different selection lock")
        return "verified_existing_cluster_lock"
    if not args.stage:
        return "ready_to_stage"
    temporary = Path(str(args.remote_lock) + ".partial." + digest[:12])
    run_command(command(args, "ssh", "-o", "ControlMaster=no",
                        "retinavlm-xlogin",
                        f"mkdir -p -- {shlex.quote(str(args.remote_lock.parent))}"))
    run_command(command(args, "scp", str(args.selection_lock),
                        f"retinavlm-xlogin:{temporary}"))
    if remote_sha(args, temporary) != digest:
        raise ValueError("Uploaded selection lock SHA-256 differs from local")
    run_command(command(args, "ssh", "-o", "ControlMaster=no",
                        "retinavlm-xlogin",
                        f"ln -- {shlex.quote(str(temporary))} "
                        f"{shlex.quote(str(args.remote_lock))} && "
                        f"chmod 600 -- {shlex.quote(str(args.remote_lock))} && "
                        f"rm -- {shlex.quote(str(temporary))}"))
    if remote_sha(args, args.remote_lock) != digest:
        raise ValueError("Cluster selection lock failed final SHA-256 check")
    return "selection_lock_staged_and_verified"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--map-set", type=Path, required=True)
    parser.add_argument("--remote-lock", type=Path, default=REMOTE_LOCK)
    parser.add_argument("--remote-root", type=Path, default=REMOTE_ROOT)
    parser.add_argument("--local-root", type=Path, default=LOCAL_ROOT)
    parser.add_argument("--ssh-config", type=Path, required=True)
    parser.add_argument("--expect-wrapper", type=Path, required=True)
    parser.add_argument("--stage", action="store_true")
    args = parser.parse_args()
    receipt = validate(args.selection_lock, args.candidates, args.map_set,
                       args.remote_root, args.local_root)
    receipt["cluster_status"] = stage(args, receipt["lock_sha256"])
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
