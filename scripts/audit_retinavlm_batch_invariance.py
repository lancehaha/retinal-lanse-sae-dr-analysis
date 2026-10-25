"""Audit whether validation answer margins change with inference batch context."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import torch

from audit_retinavlm_olives_map_candidate import sha256
from evaluate_retinavlm_adapter_olives import image_batch
from retinavlm_binary_answer import answer_token_ids
from retinavlm_pre_answer import pre_answer_forward
from retinavlm_released_model import load_released_specialist
from train_retinavlm_adapter_olives import (ADAPTED_SHA, QUESTIONS, SPLIT_SHA,
                                            install_encoder, load_rows)


def run(args):
    if args.output_dir.exists() or args.batch_size < 2:
        raise ValueError("Expected new output directory and batch size at least two")
    summary = json.loads((args.map_dir / "summary.json").read_text())
    if (summary.get("finding") != args.finding
            or summary.get("split_sha256") != SPLIT_SHA
            or summary.get("adapted_encoder_sha256") != ADAPTED_SHA
            or summary.get("adapter_sha256") != sha256(args.adapter_checkpoint)
            or summary.get("test_accessed") is not False):
        raise ValueError("Map and aligned-model provenance mismatch")
    groups = load_rows(args.split_csv)
    validation = [row for row in groups["validation"] if row[args.finding] in {"0", "1"}]
    with (args.map_dir / "validation_map_scores.csv").open(newline="", encoding="utf-8") as file:
        score_rows = [row for row in csv.DictReader(file)
                      if int(row["layer"]) == args.layer]
    if ([row["source_image_path"] for row in validation]
            != [row["source_image_path"] for row in score_rows]):
        raise ValueError("Map scores do not match validation row order")
    per_patient = defaultdict(lambda: {0: [], 1: []})
    for index, row in enumerate(validation):
        per_patient[row["patient_key"]][int(row[args.finding])].append(index)
    chosen = set()
    for patient in sorted(per_patient):
        for label in (0, 1):
            choices = per_patient[patient][label]
            if choices:
                chosen.add(choices[0])
    if args.required_source:
        match = [i for i, row in enumerate(validation)
                 if row["source_image_path"] == args.required_source]
        if len(match) != 1:
            raise ValueError("Required source is not a unique validation scan")
        chosen.add(match[0])
    checkpoint = torch.load(args.adapter_checkpoint, map_location="cpu")
    if (checkpoint.get("format") != "retinavlm_olives_adapter_alignment_v1"
            or checkpoint.get("variant") != "adapted_byol"
            or checkpoint.get("step") != 10000):
        raise ValueError("Wrong aligned adapter checkpoint")
    model, _ = load_released_specialist(args.model_dir, args.authors_code)
    install_encoder(model, "adapted_byol", args.adapted_checkpoint)
    model.model.llama_proj.load_state_dict(checkpoint["adapter"], strict=True)
    del checkpoint
    model.eval()
    absent_id, present_id = answer_token_ids(model.model.get_tokenizer(),
                                             QUESTIONS[args.finding])
    batch_cache = {}
    rows = []
    with torch.inference_mode():
        for index in sorted(chosen):
            row = validation[index]
            batch_start = (index // args.batch_size) * args.batch_size
            batch_rows = validation[batch_start:batch_start + args.batch_size]
            target_index = index - batch_start
            if batch_start not in batch_cache:
                images = image_batch(model, batch_rows, args.validation_images)
                samples = {"Image": images,
                           "Question": [QUESTIONS[args.finding]] * len(batch_rows),
                           "Answer": [""] * len(batch_rows)}
                logits, _ = pre_answer_forward(model.model, samples)
                image_embeddings, _ = model.model.encode_img(images)
                batch_cache[batch_start] = (images, logits, image_embeddings)
            images, logits, image_embeddings = batch_cache[batch_start]
            singleton = images[target_index:target_index + 1]
            single_samples = {"Image": singleton,
                              "Question": [QUESTIONS[args.finding]], "Answer": [""]}
            single_logits, _ = pre_answer_forward(model.model, single_samples)
            single_embeddings, _ = model.model.encode_img(singleton)
            saved = float(score_rows[index]["final_answer_margin"])
            batched = float((logits[target_index, present_id]
                             - logits[target_index, absent_id]).float())
            alone = float((single_logits[0, present_id]
                           - single_logits[0, absent_id]).float())
            difference = (image_embeddings[target_index].float()
                          - single_embeddings[0].float()).abs()
            rows.append({"source_image_path": row["source_image_path"],
                         "patient_key": row["patient_key"],
                         "expert_label": row[args.finding],
                         "batch_start": batch_start, "batch_index": target_index,
                         "saved_margin": saved, "repeated_batch_margin": batched,
                         "single_image_margin": alone,
                         "batch_reproduction_absolute_error": abs(saved - batched),
                         "batch_vs_single_absolute_difference": abs(batched - alone),
                         "answer_flip": int((batched >= 0) != (alone >= 0)),
                         "visual_embedding_max_absolute_difference": float(difference.max()),
                         "visual_embedding_mean_absolute_difference": float(difference.mean())})
            print(json.dumps({"audited": len(rows), "total": len(chosen)}), flush=True)
    args.output_dir.mkdir(parents=True)
    output = args.output_dir / "batch_invariance_rows.csv"
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    receipt = {"status": "validation_batch_invariance_audited",
               "finding": args.finding, "layer": args.layer,
               "validation_scans_audited": len(rows),
               "patients_audited": len({row["patient_key"] for row in rows}),
               "reference_batch_size": args.batch_size,
               "saved_margin_reproduced_within_0_05": all(
                   row["batch_reproduction_absolute_error"] <= 0.05 for row in rows),
               "median_batch_vs_single_absolute_difference": statistics.median(
                   row["batch_vs_single_absolute_difference"] for row in rows),
               "max_batch_vs_single_absolute_difference": max(
                   row["batch_vs_single_absolute_difference"] for row in rows),
               "answer_flips": sum(row["answer_flip"] for row in rows),
               "max_visual_embedding_absolute_difference": max(
                   row["visual_embedding_max_absolute_difference"] for row in rows),
               "split_sha256": SPLIT_SHA,
               "adapter_sha256": sha256(args.adapter_checkpoint),
               "map_scores_sha256": sha256(args.map_dir / "validation_map_scores.csv"),
               "rows_sha256": sha256(output), "test_accessed": False}
    (args.output_dir / "batch_invariance_summary.json").write_text(
        json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("map-dir", "split-csv", "validation-images", "model-dir",
                 "authors-code", "adapted-checkpoint", "adapter-checkpoint",
                 "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--finding", default="B5", choices=list(QUESTIONS))
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--required-source", default="/Prime_FULL/01-027/W0/OS/1.tif")
    run(parser.parse_args())
