"""Held-out OLIVES F1 after matched RetinaVLM adapter alignment.

Findings use the VLM's forced-choice present/absent logits. DRSS is a separate
training-only logistic probe on frozen pre-answer Llama states, aggregated per
eye-visit. It is not a generated DRSS answer.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from retinavlm_binary_answer import answer_token_ids
from retinavlm_pre_answer import decoder_blocks, pre_answer_forward
from retinavlm_released_model import load_released_specialist
from train_retinavlm_adapter_olives import QUESTIONS, SPLIT_SHA, ADAPTED_SHA, load_rows, install_encoder, sha256
from evaluate_retinavlm_olives_f1 import (aggregate_visits, binary_metrics,
                                           choose_threshold, drss_metrics, patient_weights)

JOIN_SHA = "68b5fd90074f6acd50f609683fc08977b0ac53d85547b906d8cda42150a66229"


def image_batch(model, rows, root):
    from PIL import Image
    tensors = []
    for row in rows:
        path = root / row["source_image_path"].lstrip("/")
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing or linked image: {path}")
        with Image.open(path) as image:
            if image.mode not in {"L", "RGB"}:
                raise ValueError(f"Unexpected OCT image mode: {path}: {image.mode}")
            tensors.append(model.convert_any_image_to_normalized_tensor(image.convert("L")))
    return torch.stack(tensors).to(model.model.device)


def collect(model, rows, root, finding, batch_size, layer_index, ids):
    absent_id, present_id = ids
    margins = []
    states = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            part = rows[start:start + batch_size]
            images = image_batch(model, part, root)
            samples = {"Image": images, "Question": [QUESTIONS[finding]] * len(part),
                       "Answer": [""] * len(part)}
            logits, hidden = pre_answer_forward(model.model, samples, layer_index)
            margins.extend((logits[:, present_id] - logits[:, absent_id]).float().cpu().tolist())
            if hidden is not None:
                states.append(hidden.float().cpu().numpy())
            if (start // batch_size + 1) % 50 == 0 or start + len(part) == len(rows):
                print(json.dumps({"finding": finding, "scored": start + len(part), "total": len(rows)}), flush=True)
    return np.asarray(margins, dtype=np.float32), (np.concatenate(states) if states else None)


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    if sha256(args.drss_join) != JOIN_SHA:
        raise ValueError("DRSS join provenance mismatch")
    groups = load_rows(args.split_csv)
    with args.drss_join.open(newline="", encoding="utf-8") as file:
        joins = list(csv.DictReader(file))
    join = {r["source_image_path"]: r for r in joins}
    if len(joins) != 3822 or len(join) != len(joins):
        raise ValueError("Unexpected DRSS join size")
    checkpoint = torch.load(args.adapter_checkpoint, map_location="cpu")
    variant = checkpoint.get("variant")
    if (checkpoint.get("format") != "retinavlm_olives_adapter_alignment_v1"
            or checkpoint.get("split_sha256") != SPLIT_SHA
            or checkpoint.get("adapted_encoder_sha256") != (ADAPTED_SHA if variant == "adapted_byol" else None)
            or variant not in {"released", "adapted_byol"}):
        raise ValueError("Adapter checkpoint provenance mismatch")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    model, _ = load_released_specialist(args.model_dir, args.authors_code)
    install_encoder(model, variant, args.adapted_checkpoint)
    model.model.llama_proj.load_state_dict(checkpoint["adapter"], strict=True)
    del checkpoint
    model.eval()
    ids = answer_token_ids(model.model.get_tokenizer(), QUESTIONS["B5"])
    final_layer = len(decoder_blocks(model.model)) - 1

    # The one fixed B5 prompt yields representations for the DRSS probe.
    features = {}
    margins = {"validation": {}, "test": {}}
    for split in ("train", "validation", "test"):
        root = getattr(args, split + "_images")
        b5_margin, features[split] = collect(model, groups[split], root, "B5",
                                             args.batch_size, final_layer, ids)
        if split != "train":
            margins[split]["B5"] = b5_margin
        # Persist intermediate arrays, including after a costly scan split.
        np.savez_compressed(args.out_dir / f"{split}_b5_states.npz", features=features[split],
                            margin=b5_margin)
    for split in ("validation", "test"):
        root = getattr(args, split + "_images")
        for finding in QUESTIONS:
            if finding == "B5":
                continue
            margins[split][finding], _ = collect(model, groups[split], root,
                                                 finding, args.batch_size, None, ids)
    del model
    torch.cuda.empty_cache()

    result = {"status": "adapter_aligned_vlm_olives_f1_complete", "variant": variant,
              "inference_batch_size": args.batch_size,
              "adapter_sha256": sha256(args.adapter_checkpoint),
              "split_sha256": SPLIT_SHA, "drss_join_sha256": JOIN_SHA,
              "finding_metric_type": "VLM present/absent forced-choice logits; threshold chosen on validation",
              "DRSS_metric_type": "linear probe on frozen VLM pre-answer states, averaged by eye-visit",
              "image_preprocessing": "L/RGB input, convert to L, authors full-frame 192x192 resize and 0-1 scaling",
              "train_patients": 61, "validation_patients": 13, "test_patients": 13,
              "findings": {}, "DRSS": {}}
    predictions = []
    for finding in QUESTIONS:
        yy = {s: np.array([int(r[finding]) if r[finding] in {"0", "1"} else -1
                           for r in groups[s]], dtype=int)
              for s in ("validation", "test")}
        prob = {s: 1 / (1 + np.exp(-np.clip(margins[s][finding], -80, 80)))
                for s in ("validation", "test")}
        threshold = choose_threshold(yy["validation"], prob["validation"])
        block = {s: binary_metrics(yy[s], prob[s], threshold) for s in ("validation", "test")}
        block["test_at_unadjusted_zero_margin"] = binary_metrics(yy["test"], prob["test"], 0.5)
        result["findings"][finding] = block
        for split in ("validation", "test"):
            predictions.extend({"task": finding, "variant": variant, "split": split,
                                "source_image_path": r["source_image_path"], "patient_key": r["patient_key"],
                                "label": int(r[finding]) if r[finding] in {"0", "1"} else -1,
                                "score": float(p), "threshold": float(threshold)}
                               for r, p in zip(groups[split], prob[split]))
    result["finding_macro_F1_test"] = float(np.mean([result["findings"][f]["test"]["F1"] for f in QUESTIONS]))

    visits, vectors = {}, {}
    for split in ("train", "validation", "test"):
        visits[split], vectors[split] = aggregate_visits(groups[split], features[split], join)
    if {s: len(v) for s, v in visits.items()} != {"train": 54, "validation": 12, "test": 12}:
        raise ValueError("DRSS visit counts changed")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    y = {s: np.array([v["DRSS"] for v in visits[s]], dtype=int) for s in visits}
    scaler = StandardScaler()
    x = scaler.fit_transform(vectors["train"])
    weights = patient_weights(visits["train"])
    weights *= len(weights) / weights.sum()
    classifier = LogisticRegression(C=1, solver="lbfgs", max_iter=3000, random_state=42)
    classifier.fit(x, y["train"], sample_weight=weights)
    classes = list(map(int, classifier.classes_))
    result["DRSS"]["train_visits"] = len(visits["train"])
    result["DRSS"]["classes"] = classes
    result["DRSS"]["test_patients"] = len({v["patient_key"] for v in visits["test"]})
    for split in ("validation", "test"):
        pred = classifier.predict(scaler.transform(vectors[split]))
        result["DRSS"][split] = drss_metrics(y[split], pred, classes)
        predictions.extend({"task": "DRSS_visit_probe", "variant": variant, "split": split,
                            "source_image_path": visit["visit_key"], "patient_key": visit["patient_key"],
                            "label": int(visit["DRSS"]), "score": int(p), "threshold": ""}
                           for visit, p in zip(visits[split], pred))
    majority = Counter(y["train"]).most_common(1)[0][0]
    result["DRSS"]["majority_test"] = drss_metrics(y["test"], np.full_like(y["test"], majority), classes)
    # Keep validation and locked test predictions in separate files so later
    # map/layer selection can read validation without opening test rows.
    for split, subset in (("validation", [r for r in predictions if r["split"] == "validation"]),
                          ("test", [r for r in predictions if r["split"] == "test"]),
                          ("all", predictions)):
        filename = "f1_predictions.csv" if split == "all" else f"{split}_predictions.csv"
        path = args.out_dir / filename
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(predictions[0]))
            writer.writeheader()
            writer.writerows(subset)
        result[f"{split}_predictions_sha256"] = sha256(path)
        result[f"{split}_prediction_rows"] = len(subset)
    out = args.out_dir / "f1_metrics.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "variant": variant,
                      "finding_macro_F1_test": result["finding_macro_F1_test"],
                      "DRSS_visit_macro_F1_test": result["DRSS"]["test"]["macro_F1_all_train_classes"],
                      "metrics": str(out)}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("split-csv", "drss-join", "train-images", "validation-images",
                 "test-images", "model-dir", "authors-code", "adapted-checkpoint",
                 "adapter-checkpoint", "out-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    run(p.parse_args())
