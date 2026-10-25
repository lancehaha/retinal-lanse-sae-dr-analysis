"""Run the authors' open-ended demo question on all 28 released PNG examples."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from retinavlm_released_model import load_released_specialist


QUESTION = (
    "Write an extensive report describing the OCT image and listing any visible "
    "biomarkers or other observations. Do not provide disease stage or patient referral recommendations yet."
)


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def grayscale_pixels(path: Path) -> tuple[np.ndarray, str, tuple[int, int]]:
    with Image.open(path) as image:
        mode, size = image.mode, image.size
        if mode == "RGBA":
            r, g, b, a = image.split()
            if a.getextrema() != (255, 255) or r.tobytes() != g.tobytes() or r.tobytes() != b.tobytes():
                raise ValueError("RGBA channels are not identical opaque grayscale")
            return np.array(r), mode, size
        if mode == "P":
            r, g, b = image.convert("RGB").split()
            if r.tobytes() != g.tobytes() or r.tobytes() != b.tobytes():
                raise ValueError("Palette image is not grayscale")
            return np.array(r), mode, size
        raise ValueError(f"Unexpected demo image mode: {mode}")


def run(model_dir: Path, authors_code: Path, out_jsonl: Path) -> dict:
    image_dir = authors_code / "dataset/processed_images"
    paths = sorted(image_dir.glob("*.png"))
    if len(paths) != 28:
        raise ValueError(f"Expected 28 authors' demo PNGs, found {len(paths)}")
    model, _ = load_released_specialist(model_dir, authors_code)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    statuses = Counter()
    with out_jsonl.open("w", encoding="utf-8") as handle:
        for index, path in enumerate(paths, start=1):
            item = {
                "index": index,
                "image_name": path.name,
                "image_sha256": sha256(path),
                "source": "authors_repo_processed_images",
                "expert_label_available": False,
                "question": QUESTION,
            }
            try:
                pixels, mode, size = grayscale_pixels(path)
                item.update({"original_mode": mode, "original_size": size,
                             "model_input_channels": 1})
                item["generated_report"] = model.forward(pixels, [QUESTION], max_new_tokens=150)[0]
                item["status"] = "generated"
            except Exception as error:
                item["status"] = "error"
                item["error_type"] = type(error).__name__
                item["error_message"] = str(error)
            statuses[item["status"]] += 1
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            handle.flush()
            print(f'{index}/{len(paths)} {path.name}: {item["status"]}', flush=True)
    return {
        "status": "authors_demo_28_image_smoke_complete",
        "images": len(paths),
        "generated": statuses["generated"],
        "errors": statuses["error"],
        "per_image_results": str(out_jsonl),
        "expert_labels_available": False,
        "benchmark_result": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--authors-code", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.model_dir, args.authors_code, args.out_jsonl), indent=2))
