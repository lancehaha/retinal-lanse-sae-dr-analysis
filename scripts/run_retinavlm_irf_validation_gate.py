"""Run a fixed 13-patient paired IRF baseline gate on released RetinaVLM."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from retinavlm_binary_answer import ANSWER_FORMS, answer_token_ids, predict_answer
from retinavlm_pre_answer import pre_answer_forward
from retinavlm_released_model import load_released_specialist


QUESTION = "Is intraretinal fluid present in this OCT scan? Answer present or absent."


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(gate_dir: Path, model_dir: Path, authors_code: Path, out: Path) -> dict:
    import torch
    from PIL import Image

    receipt = json.loads((gate_dir / "selection_receipt.json").read_text(encoding="utf-8"))
    if receipt["status"] != "fixed_irf_validation_gate_13_patient_pairs":
        raise ValueError("Unexpected selection receipt")
    with (gate_dir / "selected_manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 26 or len({r["patient_key"] for r in rows}) != 13:
        raise ValueError("Expected 26 validation scans from 13 patients")
    pairs = defaultdict(set)
    for row in rows:
        pairs[row["patient_key"]].add(row["IRF"])
        if row["IRF"] not in {"0", "1"} or not row["image_archive_path"].startswith("images/"):
            raise ValueError("Invalid label or image path")
        path = gate_dir / row["image_archive_path"]
        if not path.is_file() or sha256(path) != row["sha256"]:
            raise ValueError(f"Missing or changed image: {path}")
    if any(labels != {"0", "1"} for labels in pairs.values()):
        raise ValueError("Each patient must have one positive and one negative scan")

    model, _ = load_released_specialist(model_dir, authors_code)
    tokenizer = model.model.get_tokenizer()
    absent_id, present_id = answer_token_ids(tokenizer, QUESTION)
    predictions = []
    for index, row in enumerate(rows, start=1):
        path = gate_dir / row["image_archive_path"]
        with Image.open(path) as image:
            if image.mode != "L":
                raise ValueError(f"Expected grayscale image: {path}")
            image = image.copy()
        tensor = model.convert_any_image_to_normalized_tensor(image).unsqueeze(0).to(model.model.device)
        samples = {"Image": tensor, "Question": [QUESTION], "Answer": [""]}
        with torch.inference_mode():
            logits, _ = pre_answer_forward(model.model, samples)
            answer, probability = predict_answer(logits, absent_id, present_id)
        predictions.append({
            "source_image_path": row["source_image_path"],
            "patient_key": row["patient_key"],
            "trial": row["trial"],
            "expert_IRF": int(row["IRF"]),
            "predicted_IRF": int(answer.item()),
            "conditional_present_probability": float(probability.item()),
        })
        print(f"Scored {index}/{len(rows)} validation scans", flush=True)

    by_patient = defaultdict(dict)
    confusion = Counter()
    for item in predictions:
        by_patient[item["patient_key"]][item["expert_IRF"]] = item
        confusion[f'{item["expert_IRF"]}{item["predicted_IRF"]}'] += 1
    deltas = [pair[1]["conditional_present_probability"] -
              pair[0]["conditional_present_probability"]
              for pair in by_patient.values()]
    result = {
        "status": "released_model_fixed_irf_validation_gate_complete",
        "benchmark_result": False,
        "scope": "26 selected validation scans, one positive and one negative per patient",
        "selection_receipt": receipt,
        "model_dir": str(model_dir),
        "gpu_name": torch.cuda.get_device_name(0),
        "question": QUESTION,
        "answer_token_forms": {"absent": ANSWER_FORMS[0], "present": ANSWER_FORMS[1]},
        "answer_token_ids": {"absent": absent_id, "present": present_id},
        "confusion": {"TN": confusion["00"], "FP": confusion["01"],
                      "FN": confusion["10"], "TP": confusion["11"]},
        "within_patient_positive_higher": sum(delta > 0 for delta in deltas),
        "within_patient_ties": sum(delta == 0 for delta in deltas),
        "within_patient_pairs": len(deltas),
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
    print(json.dumps({key: result[key] for key in
                      ("status", "benchmark_result", "confusion",
                       "within_patient_positive_higher", "within_patient_pairs")}, indent=2))
