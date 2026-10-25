"""Small patient-matched released-VLM gate for OLIVES B5 macular edema."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

PROMPT = "Is diffuse retinal thickening or macular edema present in this OCT scan? Answer present or absent."
SPLIT_SHA = "28690d3ced5cdd3349100448b8f471cb56bb1690b0502c28148d055a266068fe"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def select_pairs(rows):
    by_patient = defaultdict(list)
    for row in rows:
        if row["split"] == "validation" and row["B5"] in {"0", "1"}:
            by_patient[row["patient_key"]].append(row)
    rng = random.Random(42)
    selected = []
    for patient in sorted(by_patient):
        by_visit = defaultdict(lambda: defaultdict(list))
        for row in by_patient[patient]:
            by_visit[str(Path(row["source_image_path"]).parent)][row["B5"]].append(row)
        eligible = sorted(key for key, group in by_visit.items() if group["0"] and group["1"])
        if not eligible:
            continue
        visit = eligible[0]
        group = by_visit[visit]
        selected.extend(rng.choice(sorted(group[label], key=lambda r: r["source_image_path"]))
                        for label in ("0", "1"))
    if len(selected) != 22 or len({r["patient_key"] for r in selected}) != 11:
        raise ValueError("Expected 11 within-visit positive/negative patient pairs")
    return selected


def auc(labels, scores):
    positives = [s for y, s in zip(labels, scores) if y == 1]
    negatives = [s for y, s in zip(labels, scores) if y == 0]
    return sum((a > b) + 0.5 * (a == b) for a in positives for b in negatives) / (len(positives) * len(negatives))


def run(args):
    import torch
    from PIL import Image

    if sha256(args.split_csv) != SPLIT_SHA:
        raise ValueError("Frozen split changed")
    with args.split_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    selected = select_pairs(rows)
    for row in selected:
        image = args.validation_images / row["source_image_path"].lstrip("/")
        if not image.is_file() or image.is_symlink() or sha256(image) != row["sha256"]:
            raise ValueError(f"Missing or changed image: {image}")
    sys.path.insert(0, str(args.code_dir))
    from retinavlm_binary_answer import answer_token_ids
    from retinavlm_pre_answer import pre_answer_forward
    from retinavlm_released_model import load_released_specialist

    model, _ = load_released_specialist(args.model_dir, args.authors_code)
    model.eval()
    tokenizer = model.model.get_tokenizer()
    absent_id, present_id = answer_token_ids(tokenizer, PROMPT)
    def score(image):
        tensor = model.convert_any_image_to_normalized_tensor(image).unsqueeze(0).to(model.model.device)
        samples = {"Image": tensor, "Question": [PROMPT], "Answer": [""]}
        with torch.inference_mode():
            logits, _ = pre_answer_forward(model.model, samples)
        return float((logits[0, present_id] - logits[0, absent_id]).item())

    blank = score(Image.new("L", (504, 496), 0))
    predictions = []
    for index, row in enumerate(selected, 1):
        path = args.validation_images / row["source_image_path"].lstrip("/")
        with Image.open(path) as image:
            margin = score(image.convert("L"))
        predictions.append({"source_image_path": row["source_image_path"],
                            "patient_key": row["patient_key"], "B5": int(row["B5"]),
                            "image_sha256": row["sha256"], "present_minus_absent_logit": margin,
                            "predicted_present": margin >= 0})
        print(f"Scored {index}/{len(selected)} B5 scans", flush=True)
    pairs = defaultdict(dict)
    for row in predictions:
        pairs[row["patient_key"]][row["B5"]] = row["present_minus_absent_logit"]
    labels = [r["B5"] for r in predictions]
    scores = [r["present_minus_absent_logit"] for r in predictions]
    positives_higher = sum(pair[1] > pair[0] for pair in pairs.values())
    true_positive = sum(r["B5"] == 1 and r["predicted_present"] for r in predictions)
    false_positive = sum(r["B5"] == 0 and r["predicted_present"] for r in predictions)
    result = {"status": "released_b5_validation_gate_complete", "benchmark_result": False,
              "finding": "B5 diffuse retinal thickening / macular edema", "prompt": PROMPT,
              "selection": "one deterministic random positive/negative same-visit pair per eligible validation patient",
              "patients": len(pairs), "scans": len(predictions), "test_scans_used": 0,
              "answer_token_ids": {"absent": absent_id, "present": present_id},
              "blank_margin": blank, "pooled_pair_scan_AUROC": auc(labels, scores),
              "positive_higher_within_patient": positives_higher,
              "true_positive_at_zero_margin": true_positive,
              "false_positive_at_zero_margin": false_positive,
              "validation_gate_pass": bool(positives_higher >= 8 and auc(labels, scores) >= 0.65
                                           and true_positive >= 1),
              "gate_rule": "at least 8/11 patient pairs rank positive higher, pooled AUC >=0.65, and at least one true positive at the original zero margin",
              "predictions": predictions,
              "gpu": torch.cuda.get_device_name(0)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "patients", "pooled_pair_scan_AUROC",
                                                 "positive_higher_within_patient", "true_positive_at_zero_margin",
                                                 "validation_gate_pass")}, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("split-csv", "validation-images", "model-dir", "authors-code", "code-dir", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    run(p.parse_args())
