"""Copy declared Slurm artifacts and verify local bytes against cluster SHA-256."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path


HASH_LINE = re.compile(r"(?m)^([0-9a-f]{64})\s+(/.+)$")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def command(args, program, *parts):
    prefix = ["/bin/zsh", str(args.expect_wrapper)] if args.expect_wrapper else []
    return [*prefix, program, "-F", str(args.ssh_config), *parts]


def run_command(argv):
    result = subprocess.run(argv, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"Transfer command failed ({result.returncode}): {result.stdout[-1000:]}")
    return result.stdout


def remote_sha(args, remote):
    output = run_command(command(args, "ssh", "-o", "ControlMaster=no",
                                 args.host, "sha256sum -- " + shlex.quote(remote)))
    matches = HASH_LINE.findall(output.replace("\r", ""))
    matching = [digest for digest, path in matches if path == remote]
    if len(matching) != 1:
        raise ValueError(f"Cannot parse exact remote SHA-256 for {remote}: {output[-500:]}")
    return matching[0]


def run(args):
    manifest = json.loads(args.manifest.read_text())
    entries = manifest.get("artifacts")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Artifact manifest must contain nonempty artifacts list")
    seen = set()
    results = []
    for entry in entries:
        remote = entry["remote"]
        local = Path(entry["local"])
        if not remote.startswith("/home/h/heshi/") or not local.is_absolute():
            raise ValueError("Expected absolute authorized cluster/local artifact paths")
        if local in seen:
            raise ValueError(f"Duplicate destination: {local}")
        seen.add(local)
        expected = remote_sha(args, remote)
        local.parent.mkdir(parents=True, exist_ok=True)
        if local.is_file() and sha256(local) == expected:
            status = "verified_existing"
        else:
            partial = local.with_name(local.name + ".partial")
            if partial.exists():
                partial.unlink()
            run_command(command(args, "scp", f"{args.host}:{remote}", str(partial)))
            actual = sha256(partial)
            if actual != expected:
                raise ValueError(f"Cluster/local SHA-256 mismatch for {remote}")
            os.replace(partial, local)
            status = "copied_and_verified"
        results.append({"remote": remote, "local": str(local),
                        "sha256": expected, "size_bytes": local.stat().st_size,
                        "status": status})
        print(json.dumps({"local": str(local), "status": status,
                          "size_bytes": local.stat().st_size}), flush=True)
    receipt = {
        "status": "cluster_artifacts_retrieved_and_verified",
        "manifest_sha256": sha256(args.manifest),
        "host": args.host, "timestamp_unix": int(time.time()),
        "artifacts": results,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.receipt.with_name(args.receipt.name + ".writing")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    os.replace(temporary, args.receipt)
    print(json.dumps({"status": receipt["status"], "artifacts": len(results),
                      "receipt": str(args.receipt)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--ssh-config", type=Path, required=True)
    parser.add_argument("--expect-wrapper", type=Path)
    parser.add_argument("--host", default="retinavlm-xlogin")
    run(parser.parse_args())
