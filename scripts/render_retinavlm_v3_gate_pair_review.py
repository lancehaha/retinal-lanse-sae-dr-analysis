"""Render frozen discordant training-gate OCT pairs for manual quality review."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.review_csv.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 28 or {r["finding"] for r in rows} != {"B2", "B6"}:
        raise ValueError("Unexpected frozen-pair review queue")
    output = []
    for finding in ("B2", "B6"):
        selected = [row for row in rows if row["finding"] == finding]
        for view in ("whole", "upper_55_percent"):
            columns = 4
            width, height, caption = 340, 340, 40
            canvas = Image.new("RGB", (columns * width,
                                       math.ceil(len(selected) / columns) * (height + caption)),
                               "#121212")
            draw = ImageDraw.Draw(canvas)
            for index, row in enumerate(selected):
                source = args.image_root / row["source_image_path"].lstrip("/")
                if not source.is_file() or source.is_symlink() or sha256(source) != row["sha256"]:
                    raise ValueError(f"Image missing or hash mismatch: {source}")
                with Image.open(source) as original:
                    gray = original.convert("L")
                    if view == "upper_55_percent":
                        gray = gray.crop((0, 0, gray.width, round(gray.height * 0.55)))
                    image = gray.resize((width, height))
                x, y = (index % columns) * width, (index // columns) * (height + caption)
                canvas.paste(image.convert("RGB"), (x, y + caption))
                label = row["expert_label"]
                color = "#7fdc9b" if label == "1" else "#ffbe7d"
                draw.rectangle((x, y + caption, x + width - 1, y + caption + height - 1),
                               outline=color, width=3)
                draw.text((x + 7, y + 10),
                          f"{finding}  {index + 1:02d}  label={label}  score={float(row['calibrated_logit']):+.2f}",
                          fill=color)
            path = args.output_dir / f"{finding}_{view}.png"
            if path.exists():
                raise FileExistsError(path)
            canvas.save(path)
            output.append({"path": str(path), "sha256": sha256(path),
                           "finding": finding, "view": view})
    receipt = {"status": "image_quality_review_only_no_relabeling",
               "review_csv_sha256": sha256(args.review_csv), "images_verified": len(rows),
               "rendered": output, "test_accessed": False,
               "original_validation_accessed": False}
    receipt_path = args.output_dir / "review_render_receipt.json"
    if receipt_path.exists():
        raise FileExistsError(receipt_path)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())
