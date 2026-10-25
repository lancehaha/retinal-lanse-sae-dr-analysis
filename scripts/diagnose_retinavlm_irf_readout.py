"""Compare IRF prompt variants and a blank-image control on validation pairs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

from retinavlm_binary_answer import ANSWER_FORMS, answer_token_ids
from retinavlm_pre_answer import pre_answer_forward
from retinavlm_released_model import load_released_specialist


PROMPTS = {
    "original": "Is intraretinal fluid present in this OCT scan? Answer present or absent.",
    "direct": "Does this OCT scan show intraretinal fluid (IRF)? Answer with exactly one word: present or absent.",
    "statement": "For intraretinal fluid in this OCT scan, state present or absent.",
}


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def run(gate_dir: Path, model_dir: Path, authors_code: Path, out: Path) -> dict:
    import torch
    from PIL import Image

    receipt = json.loads((gate_dir / "selection_receipt.json").read_text(encoding="utf-8"))
    if receipt["status"] != "fixed_irf_validation_gate_13_patient_pairs":
        raise ValueError("Unexpected selection receipt")
    with (gate_dir / "selected_manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 26 or len({r["patient_key"] for r in rows}) != 13:
        raise ValueError("Expected 26 fixed validation scans from 13 patients")
    for row in rows:
        path = gate_dir / row["image_archive_path"]
        if row["IRF"] not in {"0", "1"} or not path.is_file() or sha256(path) != row["sha256"]:
            raise ValueError("Invalid label, missing image, or changed image hash")

    model, _ = load_released_specialist(model_dir, authors_code)
    tokenizer = model.model.get_tokenizer()
    absent_id, present_id = answer_token_ids(tokenizer, PROMPTS["original"])

    def score(image: Image.Image, question: str) -> float:
        tensor = model.convert_any_image_to_normalized_tensor(image).unsqueeze(0).to(model.model.device)
        samples = {"Image": tensor, "Question": [question], "Answer": [""]}
        with torch.inference_mode():
            logits, _ = pre_answer_forward(model.model, samples)
            return float((logits[0, present_id] - logits[0, absent_id]).item())

    blank = Image.new("L", (504, 496), 0)
    blank_margins = {name: score(blank, question) for name, question in PROMPTS.items()}
    predictions = []
    for index, row in enumerate(rows, start=1):
        path = gate_dir / row["image_archive_path"]
        with Image.open(path) as image:
            if image.mode != "L":
                raise ValueError(f"Expected grayscale: {path}")
            image = image.copy()
        margins = {name: score(image, question) for name, question in PROMPTS.items()}
        predictions.append({
            "source_image_path": row["source_image_path"],
            "patient_key": row["patient_key"],
            "expert_IRF": int(row["IRF"]),
            "margins": margins,
        })
        print(f"Scored {index}/{len(rows)} scans for {len(PROMPTS)} prompts", flush=True)

    summaries = {}
    for name in PROMPTS:
        by_patient = defaultdict(dict)
        by_label = {0: [], 1: []}
        for item in predictions:
            margin = item["margins"][name]
            by_patient[item["patient_key"]][item["expert_IRF"]] = margin
            by_label[item["expert_IRF"]].append(margin)
        if any(set(pair) != {0, 1} for pair in by_patient.values()):
            raise ValueError("A patient pair is incomplete")
        summaries[name] = {
            "blank_margin": blank_margins[name],
            "negative_median_margin": median(by_label[0]),
            "positive_median_margin": median(by_label[1]),
            "positive_higher_within_patient": sum(pair[1] > pair[0] for pair in by_patient.values()),
            "pairs": len(by_patient),
            "scans_predicted_present_at_zero_margin": sum(
                item["margins"][name] >= 0 for item in predictions
            ),
            "real_image_margin_range": [
                min(item["margins"][name] for item in predictions),
                max(item["margins"][name] for item in predictions),
            ],
        }
    result = {
        "status": "retinavlm_irf_validation_readout_diagnostic_complete",
        "benchmark_result": False,
        "selection_receipt": receipt,
        "gpu_name": torch.cuda.get_device_name(0),
        "answer_token_forms": {"absent": ANSWER_FORMS[0], "present": ANSWER_FORMS[1]},
        "answer_token_ids": {"absent": absent_id, "present": present_id},
        "prompts": PROMPTS,
        "summary": summaries,
        "predictions": predictions,
        "blank_control_note": "black image is out of distribution; use only to check gross image dependence",
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
    print(json.dumps(result["summary"], indent=2))
