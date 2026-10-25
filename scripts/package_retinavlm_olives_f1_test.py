"""Package only frozen OLIVES test images for transfer to the cluster."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import tarfile
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run(args):
    if sha256(args.split_csv) != "28690d3ced5cdd3349100448b8f471cb56bb1690b0502c28148d055a266068fe":
        raise ValueError("Frozen split changed")
    with args.split_csv.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "test"]
    if len(rows) != 1274 or len({r["patient_key"] for r in rows}) != 13:
        raise ValueError("Unexpected test cohort")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.out, "w:gz", compresslevel=1) as archive:
        for index, row in enumerate(rows, 1):
            relative = row["source_image_path"].lstrip("/")
            if ".." in Path(relative).parts:
                raise ValueError("Unsafe image path")
            image = args.images / relative
            if image.is_symlink() or sha256(image) != row["sha256"]:
                raise ValueError(f"Missing or changed image: {image}")
            archive.add(image, arcname="retinavlm_olives_f1_test/images/" + relative, recursive=False)
            if index % 250 == 0:
                print(f"Packed {index}/{len(rows)}", flush=True)
    return {"status": "test_only_package_complete", "images": len(rows),
            "archive": str(args.out), "sha256": sha256(args.out), "bytes": args.out.stat().st_size}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split-csv", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    print(json.dumps(run(p.parse_args()), indent=2))
