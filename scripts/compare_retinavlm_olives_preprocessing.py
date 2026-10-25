"""Compare existing full-frame resize with paper-style half-scale central crop."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from PIL import Image

from retinavlm_binary_answer import answer_token_ids
from retinavlm_pre_answer import pre_answer_forward
from retinavlm_released_model import load_released_specialist


QUESTION = "Is intraretinal fluid present in this OCT scan? Answer present or absent."


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paper_style_crop(image: Image.Image) -> Image.Image:
    """Diagnostic approximation: half resolution, then centered 192x192 crop.

    The authors used this geometry on 416x512 Topcon slices. OLIVES has a
    different acquisition scale, so this is an ablation, not a claim of
    physically equivalent input pixels.
    """
    width, height = image.size
    half = image.resize((round(width / 2), round(height / 2)), Image.Resampling.BILINEAR)
    left = (half.width - 192) // 2
    top = (half.height - 192) // 2
    if left < 0 or top < 0:
        raise ValueError("Half-scale image is smaller than central crop")
    return half.crop((left, top, left + 192, top + 192))


def run(gate_dir: Path, model_dir: Path, authors_code: Path, out: Path) -> dict:
    import torch

    receipt = json.loads((gate_dir / "selection_receipt.json").read_text(encoding="utf-8"))
    if receipt["status"] != "fixed_irf_validation_gate_13_patient_pairs":
        raise ValueError("Unexpected selection receipt")
    with (gate_dir / "selected_manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 26 or len({r["patient_key"] for r in rows}) != 13:
        raise ValueError("Expected 26 scans from 13 fixed validation patients")
    for row in rows:
        path = gate_dir / row["image_archive_path"]
        if row["IRF"] not in {"0", "1"} or not path.is_file() or sha256(path) != row["sha256"]:
            raise ValueError("Invalid label, missing image, or changed image hash")

    model, _ = load_released_specialist(model_dir, authors_code)
    absent_id, present_id = answer_token_ids(model.model.get_tokenizer(), QUESTION)
    predictions = []
    for index, row in enumerate(rows, start=1):
        with Image.open(gate_dir / row["image_archive_path"]) as image:
            if image.mode != "L":
                raise ValueError("Expected grayscale OLIVES image")
            image = image.copy()
        variants = {"full_frame_resize": image, "half_scale_center_crop": paper_style_crop(image)}
        margins = {}
        for name, variant in variants.items():
            tensor = model.convert_any_image_to_normalized_tensor(variant).unsqueeze(0).to(model.model.device)
            samples = {"Image": tensor, "Question": [QUESTION], "Answer": [""]}
            with torch.inference_mode():
                logits, _ = pre_answer_forward(model.model, samples)
                margins[name] = float((logits[0, present_id] - logits[0, absent_id]).item())
        predictions.append({
            "source_image_path": row["source_image_path"],
            "patient_key": row["patient_key"],
            "expert_IRF": int(row["IRF"]),
            "margins": margins,
        })
        print(f"Scored {index}/{len(rows)} scans under both transforms", flush=True)

    summaries = {}
    for name in ("full_frame_resize", "half_scale_center_crop"):
        pairs = defaultdict(dict)
        labels = {0: [], 1: []}
        confusion = Counter()
        for item in predictions:
            margin = item["margins"][name]
            label = item["expert_IRF"]
            pairs[item["patient_key"]][label] = margin
            labels[label].append(margin)
            confusion[f"{label}{int(margin >= 0)}"] += 1
        if any(set(pair) != {0, 1} for pair in pairs.values()):
            raise ValueError("A patient pair is incomplete")
        summaries[name] = {
            "confusion_at_zero_margin": {
                "TN": confusion["00"], "FP": confusion["01"],
                "FN": confusion["10"], "TP": confusion["11"],
            },
            "positive_higher_within_patient": sum(pair[1] > pair[0] for pair in pairs.values()),
            "pairs": len(pairs),
            "negative_median_margin": median(labels[0]),
            "positive_median_margin": median(labels[1]),
        }
    result = {
        "status": "retinavlm_olives_preprocessing_ablation_complete",
        "benchmark_result": False,
        "source": "fixed 13-patient validation gate",
        "crop_note": "half-scale center crop imitates paper geometry, but OLIVES physical pixel scale is unverified",
        "question": QUESTION,
        "answer_token_ids": {"absent": absent_id, "present": present_id},
        "summaries": summaries,
        "predictions": predictions,
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
    print(json.dumps(result["summaries"], indent=2))
