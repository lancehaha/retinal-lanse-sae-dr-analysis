from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail closed unless publication artifacts target the external disk.")
    parser.add_argument("--config", default="configs/storage.yaml")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    root = Path(config["external_root"])
    problems: list[str] = []

    if not config.get("external_only"):
        problems.append("external_only must be true")
    if not str(root).startswith("/Volumes/"):
        problems.append(f"external_root is not mounted under /Volumes: {root}")
    if not root.exists():
        problems.append(f"external_root does not exist: {root}")

    for name, configured_path in config["paths"].items():
        path = Path(configured_path)
        try:
            path.relative_to(root)
        except ValueError:
            problems.append(f"{name} escapes external_root: {path}")
        if not path.exists():
            problems.append(f"missing storage directory {name}: {path}")

    if root.exists():
        usage = shutil.disk_usage(root)
        required = int(config["minimum_free_bytes_before_download"])
        if usage.free < required:
            problems.append(f"free space {usage.free} is below required {required}")
        print(json.dumps({
            "external_root": str(root),
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
            "minimum_free_bytes": required,
        }, indent=2))

    if problems:
        print(json.dumps({"storage_gate": "FAIL", "problems": problems}, indent=2))
        return 1
    print(json.dumps({"storage_gate": "PASS"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

