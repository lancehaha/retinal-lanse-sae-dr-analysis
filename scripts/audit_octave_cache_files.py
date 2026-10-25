#!/usr/bin/env python3
"""Read-only verification of an isolated OCTAVE derived cache."""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    receipt = json.loads((root / "cache_receipt.json").read_text())
    errors = []
    for record in receipt["records"]:
        for kind in ("image", "label"):
            path = root / f"{record['name']}_{kind}.npy"
            if not path.is_file() or path.stat().st_size != record[f"{kind}_cache_bytes"] or sha256(path) != record[f"{kind}_cache_sha256"]:
                errors.append(path.name)
    actual = {path.name for path in root.glob("*.npy")}
    expected = {f"{record['name']}_{kind}.npy" for record in receipt["records"] for kind in ("image", "label")}
    errors.extend(sorted(actual ^ expected))
    print(json.dumps({"status": "pass" if not errors else "fail", "records": len(receipt["records"]), "verified_files": len(expected), "errors": errors}))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
