"""Compare authors' demo images with one OLIVES validation pair via the same API."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from retinavlm_released_model import load_released_specialist


QUESTIONS = {
    "binary": "Is intraretinal fluid present in this OCT scan? Answer present or absent.",
    "report": (
        "Write an extensive report describing the OCT image and listing any visible "
        "biomarkers or other observations. Do not provide disease stage or patient referral recommendations yet."
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def grayscale_input(path: Path, *, demo: bool) -> np.ndarray:
    with Image.open(path) as image:
        if demo:
            if image.mode != "RGBA" or image.size != (192, 192):
                raise ValueError(f"Unexpected demo image format: {path}")
            r, g, b, a = image.split()
            if a.getextrema() != (255, 255) or r.tobytes() != g.tobytes() or r.tobytes() != b.tobytes():
                raise ValueError(f"Demo image is not opaque grayscale RGBA: {path}")
            return np.array(r)
        if image.mode != "L":
            raise ValueError(f"Expected grayscale OLIVES image: {path}")
        return np.array(image)


def run(gate_dir: Path, model_dir: Path, authors_code: Path, out: Path) -> dict:
    demo_dir = authors_code / "dataset/processed_images"
    demos = sorted(path for path in demo_dir.glob("*.png")
                   if Image.open(path).mode == "RGBA")[:2]
    if len(demos) != 2:
        raise ValueError("Expected at least two authors' processed demo images")
    with (gate_dir / "selected_manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    first_patient = sorted({row["patient_key"] for row in rows})[0]
    olives = sorted((row for row in rows if row["patient_key"] == first_patient),
                    key=lambda row: row["IRF"])
    if len(olives) != 2 or {row["IRF"] for row in olives} != {"0", "1"}:
        raise ValueError("Expected one fixed positive/negative OLIVES pair")
    for row in olives:
        path = gate_dir / row["image_archive_path"]
        if not path.is_file() or sha256(path) != row["sha256"]:
            raise ValueError(f"Changed or missing OLIVES image: {path}")

    model, _ = load_released_specialist(model_dir, authors_code)
    samples = (
        [{"source": "authors_demo_unlabeled", "path": path, "expert_IRF": None} for path in demos]
        + [{"source": "OLIVES_validation", "path": gate_dir / row["image_archive_path"],
            "expert_IRF": int(row["IRF"])} for row in olives]
    )
    results = []
    for index, sample in enumerate(samples, start=1):
        pixels = grayscale_input(sample["path"], demo=sample["source"] == "authors_demo_unlabeled")
        outputs = {}
        for kind, question in QUESTIONS.items():
            outputs[kind] = model.forward(pixels, [question], max_new_tokens=120 if kind == "report" else 40)[0]
        results.append({
            "source": sample["source"],
            "image_path": str(sample["path"]),
            "image_sha256": sha256(sample["path"]),
            "expert_IRF": sample["expert_IRF"],
            "input_shape": list(pixels.shape),
            "generated": outputs,
        })
        print(f"Generated on {index}/{len(samples)} images", flush=True)
    result = {
        "status": "authors_demo_and_olives_same_wrapper_generation_complete",
        "benchmark_result": False,
        "note": "Authors' demo PNGs have no verified expert labels; outputs cannot be scored for accuracy.",
        "questions": QUESTIONS,
        "results": results,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--authors-code", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.gate_dir, args.model_dir, args.authors_code, args.out)
    print(json.dumps(result["results"], indent=2))
