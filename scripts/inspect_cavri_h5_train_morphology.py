#!/usr/bin/env python3
"""Create a small deterministic, train-only CAVRI morphology review sheet."""

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import tifffile


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--geometry", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    receipt = json.loads(args.geometry.read_text())
    archive_sha = hashlib.sha256(args.archive.read_bytes()).hexdigest()
    assert archive_sha == receipt["archive_sha256"]
    rows = [r for r in receipt["records"] if r["group"] == "VMA-VMT"]
    assert len(rows) == 115
    rows = sorted(rows, key=lambda r: hashlib.sha256(r["archive_path"].encode()).digest())[:12]
    sheet = Image.new("RGB", (4 * 360, 3 * 270), "white")
    draw = ImageDraw.Draw(sheet)
    selected = []
    with zipfile.ZipFile(args.archive) as z:
        for i, r in enumerate(rows):
            name = r["archive_path"]
            assert "/train/VMA-VMT/" in name
            data = z.read(name)
            assert hashlib.sha256(data).hexdigest() == r["member_sha256"]
            a = tifffile.imread(io.BytesIO(data))
            assert a.ndim == 3 and a.shape[2] == 3 and a.dtype == np.uint8
            im = Image.fromarray(a)
            im.thumbnail((350, 235))
            x, y = i % 4 * 360 + 5, i // 4 * 270 + 5
            sheet.paste(im, (x, y))
            draw.text((x, y + 238), Path(name).name, fill="black")
            selected.append({"archive_path": name, "sha256": r["member_sha256"]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    result = {"status": "train_only_visual_morphology_review_no_label_adjudication", "archive_sha256": archive_sha,
              "geometry_receipt_sha256": hashlib.sha256(args.geometry.read_bytes()).hexdigest(),
              "selected": selected, "validation_test_read": False, "olives_accessed": False}
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
