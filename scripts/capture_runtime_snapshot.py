from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from retina_lanse.data.manifest import sha256_path
from scripts.run_global_sae_search import require_external


def command(args: list[str], cwd: Path | None = None) -> str | None:
    result = subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture the immutable Experiment 1 runtime.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--upstream-repo", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    checkpoint = require_external(Path(args.checkpoint), "checkpoint")
    manifest = require_external(Path(args.dataset_manifest), "dataset manifest")
    upstream = require_external(Path(args.upstream_repo), "RETFound upstream repository")
    output = require_external(Path(args.output), "runtime snapshot")
    if not checkpoint.is_file() or not manifest.is_file() or not upstream.is_dir():
        raise RuntimeError("runtime snapshot input is absent")

    packages = {
        distribution.metadata["Name"]: distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    }
    git_commit = command(["git", "rev-parse", "HEAD"], upstream)
    git_status = command(["git", "status", "--porcelain"], upstream)
    if not git_commit:
        raise RuntimeError("RETFound upstream repository commit could not be resolved")
    snapshot = {
        "schema_version": "1.0",
        "status": "complete",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "cpu_brand": command(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "physical_memory_bytes": command(["sysctl", "-n", "hw.memsize"]),
        "torch": torch.__version__,
        "torchvision": packages.get("torchvision"),
        "numpy": packages.get("numpy"),
        "pillow": packages.get("Pillow"),
        "timm": packages.get("timm"),
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "torch_build_configuration": torch.__config__.show(),
        "installed_packages": dict(sorted(packages.items(), key=lambda item: item[0].lower())),
        "retfound_upstream_repository": str(upstream),
        "retfound_upstream_commit": git_commit,
        "retfound_upstream_dirty": bool(git_status),
        "retfound_upstream_status": git_status or "",
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": sha256_path(checkpoint),
        "dataset_manifest": str(manifest),
        "dataset_manifest_sha256": sha256_path(manifest),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": snapshot["status"],
                "output": str(output),
                "packages": len(snapshot["installed_packages"]),
                "retfound_upstream_commit": snapshot["retfound_upstream_commit"],
                "checkpoint_sha256": snapshot["checkpoint_sha256"],
                "dataset_manifest_sha256": snapshot["dataset_manifest_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
