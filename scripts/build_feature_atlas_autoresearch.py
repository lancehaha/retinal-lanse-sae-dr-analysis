from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from scipy import sparse

from retina_lanse.data.embedding_shards import load_embedding_run
from retina_lanse.data.manifest import sha256_path
from retina_lanse.sae.checkpoints import load_checkpoint
from scripts.run_global_sae_search import require_external
from scripts.run_single_sae_feasibility import encode_csr, image_id


def load_labels(path: Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            key = row["image"].strip()
            value = int(row["level"])
            if key in labels or value not in range(5):
                raise RuntimeError(f"invalid or duplicate label: {key}")
            labels[key] = value
    return labels


def active_statistics(codes: sparse.csr_matrix, grades: np.ndarray) -> dict[str, np.ndarray]:
    active = codes.copy()
    active.data = np.ones_like(active.data)
    total = codes.shape[0]
    count = np.asarray(active.sum(axis=0)).ravel().astype(np.int64)
    mass = np.asarray(codes.sum(axis=0)).ravel()
    mean = mass / max(total, 1)
    positive = grades >= 2
    labeled = grades >= 0
    positive_active = np.asarray(active[labeled & positive].sum(axis=0)).ravel()
    negative_active = np.asarray(active[labeled & ~positive].sum(axis=0)).ravel()
    n_positive = max(int((labeled & positive).sum()), 1)
    n_negative = max(int((labeled & ~positive).sum()), 1)
    positive_rate = (positive_active + 0.5) / (n_positive + 1.0)
    negative_rate = (negative_active + 0.5) / (n_negative + 1.0)
    return {
        "count": count,
        "mass": mass,
        "mean": mean,
        "positive_active": positive_active,
        "negative_active": negative_active,
        "positive_rate": positive_rate,
        "negative_rate": negative_rate,
        "log_rate_ratio": np.log(positive_rate / negative_rate),
    }


def make_record(
    feature: int,
    codes: sparse.csr_matrix,
    stats: dict[str, np.ndarray],
    paths: list[str],
    patients: list[str],
    grades: np.ndarray,
    *,
    top_images: int,
    rank: int,
) -> dict:
    column = codes.getcol(feature).tocoo()
    order = np.argsort(column.data)[::-1][:top_images]
    examples = []
    for position in order:
        row = int(column.row[position])
        grade = int(grades[row]) if grades[row] >= 0 else None
        examples.append(
            {
                "relative_path": paths[row],
                "patient_id": patients[row],
                "activation": float(column.data[position]),
                "grade_if_available": grade,
                "referable_dr_if_available": bool(grade >= 2) if grade is not None else None,
            }
        )
    return {
        "feature_id": int(feature),
        "global_prominence_rank": int(rank),
        "activation_count": int(stats["count"][feature]),
        "activation_rate": float(stats["count"][feature] / codes.shape[0]),
        "activation_mass": float(stats["mass"][feature]),
        "mean_activation": float(stats["mean"][feature]),
        "labeled_dr_active_count": int(stats["positive_active"][feature]),
        "labeled_non_dr_active_count": int(stats["negative_active"][feature]),
        "labeled_dr_activation_rate": float(stats["positive_rate"][feature]),
        "labeled_non_dr_activation_rate": float(stats["negative_rate"][feature]),
        "labeled_log_rate_ratio": float(stats["log_rate_ratio"][feature]),
        "hypothesis_status": "pending_llm_or_clinician_review",
        "top_high_activation_examples": examples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a label-agnostic retinal SAE feature atlas for later LLM grounding."
    )
    parser.add_argument("--train-run", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-activations", type=int, default=50)
    parser.add_argument("--max-features", type=int, default=512)
    parser.add_argument("--top-images", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    train_root = require_external(Path(args.train_run), "train embeddings")
    labels_path = require_external(Path(args.labels), "clinical labels")
    normalization_path = require_external(Path(args.normalization), "normalization")
    checkpoint_path = require_external(Path(args.checkpoint), "SAE checkpoint")
    output = require_external(Path(args.output), "feature atlas output")
    output.mkdir(parents=True, exist_ok=True)

    run = load_embedding_run(train_root, expected_split="train", verify_sha256=True)
    labels = load_labels(labels_path)
    grades = np.asarray([labels.get(image_id(path), -1) for path in run.relative_paths], dtype=np.int64)
    normalization = torch.load(normalization_path, map_location="cpu", weights_only=False)
    if normalization["train_manifest_hash"] != run.evidence["manifest_hash"]:
        raise RuntimeError("normalization/embedding manifest mismatch")
    embeddings = (run.embeddings - normalization["mean"]) / normalization["std"]
    model, payload = load_checkpoint(checkpoint_path)
    expected_config = {"input_dim": 1024, "dictionary_size": 16384, "k": 32}
    if payload["model_config"] != expected_config:
        raise RuntimeError(f"unexpected SAE configuration: {payload['model_config']}")
    codes = encode_csr(model, embeddings, batch_size=args.batch_size)
    stats = active_statistics(codes, grades)
    eligible = stats["count"] >= args.min_activations
    ranked = np.argsort(np.where(eligible, stats["mass"], -np.inf))[::-1]
    ranked = [int(feature) for feature in ranked[: args.max_features] if eligible[feature]]

    atlas_path = output / "feature_atlas.jsonl"
    with atlas_path.open("w", encoding="utf-8") as handle:
        for rank, feature in enumerate(ranked, start=1):
            record = make_record(
                feature,
                codes,
                stats,
                run.relative_paths,
                run.patient_ids,
                grades,
                top_images=args.top_images,
                rank=rank,
            )
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    summary = {
        "schema_version": "1.0",
        "status": "complete_feature_atlas_pending_language_grounding",
        "label_agnostic_discovery": True,
        "llm_used": False,
        "images_indexed": len(run.embeddings),
        "patients_indexed": len(set(run.patient_ids)),
        "labeled_images": int((grades >= 0).sum()),
        "labeled_positive_images": int(((grades >= 0) & (grades >= 2)).sum()),
        "dictionary_size": int(codes.shape[1]),
        "nonzero_activation_entries": int(codes.nnz),
        "min_activations": args.min_activations,
        "eligible_features": int(eligible.sum()),
        "feature_cards_written": len(ranked),
        "top_images_per_feature": args.top_images,
        "ranking_rule": "descending_global_activation_mass_after_minimum_activation_count",
        "hypothesis_generation": "pending_llm_or_clinician_review",
        "train_run_evidence_sha256": sha256_path(train_root / "extraction_evidence.json"),
        "labels_sha256": sha256_path(labels_path),
        "normalization_sha256": sha256_path(normalization_path),
        "checkpoint_sha256": sha256_path(checkpoint_path),
        "atlas_jsonl": str(atlas_path),
    }
    (output / "feature_atlas_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
