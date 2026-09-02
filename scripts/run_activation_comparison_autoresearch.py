from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from scipy import sparse
from sklearn.metrics import average_precision_score, roc_auc_score

from retina_lanse.autoresearch.registry import initialize_registry
from retina_lanse.data.embedding_shards import EmbeddingRun, load_embedding_run
from retina_lanse.data.manifest import sha256_path
from retina_lanse.sae.checkpoints import load_checkpoint
from scripts.run_single_sae_feasibility import (
    encode_csr,
    labeled_subset,
    patient_bootstrap_interval,
)
from scripts.run_global_sae_search import require_external


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


def split_patients(patients: list[str], seed: int) -> np.ndarray:
    """Deterministic 80/20 patient split inside the labeled training run."""
    return np.asarray(
        [
            int(
                int.from_bytes(
                    hashlib.sha256(f"activation-ar:{seed}:{patient}".encode()).digest()[:8],
                    "big",
                )
                / 2**64
                >= 0.8
            )
            for patient in patients
        ],
        dtype=bool,
    )


def activation_stats(codes: sparse.csr_matrix, targets: np.ndarray) -> dict[str, np.ndarray]:
    active = codes.copy()
    active.data = np.ones_like(active.data)
    pos = targets == 1
    neg = ~pos
    pos_rate = np.asarray(active[pos].mean(axis=0)).ravel()
    neg_rate = np.asarray(active[neg].mean(axis=0)).ravel()
    pos_sum = np.asarray(codes[pos].sum(axis=0)).ravel()
    neg_sum = np.asarray(codes[neg].sum(axis=0)).ravel()
    pos_count = max(int(pos.sum()), 1)
    neg_count = max(int(neg.sum()), 1)
    pos_mean = pos_sum / pos_count
    neg_mean = neg_sum / neg_count
    pos_sq = np.asarray(codes[pos].power(2).sum(axis=0)).ravel() / pos_count
    neg_sq = np.asarray(codes[neg].power(2).sum(axis=0)).ravel() / neg_count
    pooled_sd = np.sqrt(np.maximum((pos_sq + neg_sq) / 2 - ((pos_mean + neg_mean) / 2) ** 2, 1e-8))
    total_active = np.asarray(active.sum(axis=0)).ravel()
    return {
        "pos_rate": pos_rate,
        "neg_rate": neg_rate,
        "log_rate_ratio": np.log((pos_rate + 1e-4) / (neg_rate + 1e-4)),
        "rate_difference": pos_rate - neg_rate,
        "mean_difference": pos_mean - neg_mean,
        "standardized_difference": (pos_mean - neg_mean) / pooled_sd,
        "total_active": total_active,
    }


def recipe_score(codes: sparse.csr_matrix, stats: dict[str, np.ndarray], recipe: dict) -> np.ndarray:
    eligible = stats["total_active"] >= recipe["minimum_activations"]
    ranking = np.where(eligible, stats[recipe["ranking_stat"]], -np.inf)
    selected = np.argsort(ranking)[::-1][: recipe["features"]]
    weights = stats[recipe["weight_stat"]][selected].astype(np.float32)
    weights = weights / max(float(np.abs(weights).sum()), 1e-8)
    score = np.asarray(codes[:, selected].dot(weights)).ravel()
    return score


def evaluate_candidate(targets: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(targets, scores)),
        "auprc": float(average_precision_score(targets, scores)),
        "objective": float(roc_auc_score(targets, scores)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Bounded AutoResearch over retinal SAE activation-comparison recipes.")
    parser.add_argument("--train-run", required=True)
    parser.add_argument("--validation-run", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=4106)
    args = parser.parse_args()

    train_root = require_external(Path(args.train_run), "train embeddings")
    validation_root = require_external(Path(args.validation_run), "validation embeddings")
    labels_path = require_external(Path(args.labels), "clinical labels")
    normalization_path = require_external(Path(args.normalization), "normalization")
    checkpoint_path = require_external(Path(args.checkpoint), "SAE checkpoint")
    output = require_external(Path(args.output), "AutoResearch output")
    if "internal_test" in str(validation_root) or "internal_test" in str(output):
        raise RuntimeError("activation AutoResearch refuses locked internal-test paths")
    output.mkdir(parents=True, exist_ok=True)
    initialize_registry(output / "autoresearch_registry.sqlite")

    train_run = load_embedding_run(train_root, expected_split="train", verify_sha256=True)
    validation_run = load_embedding_run(validation_root, expected_split="validation", verify_sha256=True)
    if set(train_run.patient_ids) & set(validation_run.patient_ids):
        raise RuntimeError("embedding patient overlap")
    if train_run.evidence["checkpoint_hash"] != validation_run.evidence["checkpoint_hash"]:
        raise RuntimeError("encoder checkpoint mismatch")
    labels = load_labels(labels_path)
    train, train_grades, train_patients, _ = labeled_subset(train_run, labels)
    validation, validation_grades, validation_patients, _ = labeled_subset(validation_run, labels)
    train_targets = (train_grades >= 2).astype(np.int64)
    validation_targets = (validation_grades >= 2).astype(np.int64)
    normalization = torch.load(normalization_path, map_location="cpu", weights_only=False)
    if normalization["train_manifest_hash"] != train_run.evidence["manifest_hash"]:
        raise RuntimeError("normalization/embedding manifest mismatch")
    train = (train - normalization["mean"]) / normalization["std"]
    validation = (validation - normalization["mean"]) / normalization["std"]
    model, payload = load_checkpoint(checkpoint_path)
    if payload["model_config"] != {"input_dim": 1024, "dictionary_size": 16384, "k": 32}:
        raise RuntimeError("unexpected SAE configuration")
    train_codes = encode_csr(model, train, batch_size=args.batch_size)
    validation_codes = encode_csr(model, validation, batch_size=args.batch_size)

    dev_mask = split_patients(train_patients, args.seed)
    dev_fit_codes, dev_eval_codes = train_codes[~dev_mask], train_codes[dev_mask]
    dev_fit_targets, dev_eval_targets = train_targets[~dev_mask], train_targets[dev_mask]
    dev_eval_patients = [patient for patient, held_out in zip(train_patients, dev_mask) if held_out]
    recipes = [
        {
            "name": "binary_log_rate_ratio_top64",
            "ranking_stat": "log_rate_ratio",
            "weight_stat": "log_rate_ratio",
            "features": 64,
            "minimum_activations": 50,
            "activation_transform": "binary_for_ranking_weighted_sum",
        },
        {
            "name": "binary_rate_difference_top64",
            "ranking_stat": "rate_difference",
            "weight_stat": "rate_difference",
            "features": 64,
            "minimum_activations": 50,
            "activation_transform": "binary_for_ranking_weighted_sum",
        },
        {
            "name": "weighted_mean_difference_top64",
            "ranking_stat": "mean_difference",
            "weight_stat": "mean_difference",
            "features": 64,
            "minimum_activations": 50,
            "activation_transform": "weighted_activation_sum",
        },
        {
            "name": "weighted_standardized_difference_top64",
            "ranking_stat": "standardized_difference",
            "weight_stat": "standardized_difference",
            "features": 64,
            "minimum_activations": 50,
            "activation_transform": "weighted_activation_sum",
        },
    ]
    candidate_results = []
    for recipe in recipes:
        stats = activation_stats(dev_fit_codes, dev_fit_targets)
        dev_scores = recipe_score(dev_eval_codes, stats, recipe)
        result = {**recipe, "development": evaluate_candidate(dev_eval_targets, dev_scores)}
        candidate_results.append(result)
    selected = max(candidate_results, key=lambda item: (item["development"]["objective"], item["development"]["auprc"]))

    full_stats = activation_stats(train_codes, train_targets)
    validation_scores = recipe_score(validation_codes, full_stats, selected)
    validation_metrics = evaluate_candidate(validation_targets, validation_scores)
    validation_metrics["auroc_patient_bootstrap"] = patient_bootstrap_interval(
        validation_targets,
        validation_scores,
        validation_patients,
        roc_auc_score,
        resamples=args.bootstrap_resamples,
        seed=args.seed + 10,
    )
    validation_metrics["auprc_patient_bootstrap"] = patient_bootstrap_interval(
        validation_targets,
        validation_scores,
        validation_patients,
        average_precision_score,
        resamples=args.bootstrap_resamples,
        seed=args.seed + 11,
    )
    report = {
        "schema_version": "1.0",
        "status": "complete_bounded_activation_comparison_search",
        "autoresearch_used": True,
        "search_scope": "four_prespecified_activation_comparison_recipes_on_patient_80_20_split_inside_labeled_train",
        "selection_rule": "highest_development_AUROC_then_AUPRC",
        "locked_internal_test_accessed": False,
        "sae_checkpoint": str(checkpoint_path),
        "sae_checkpoint_sha256": sha256_path(checkpoint_path),
        "sae_config": payload["model_config"],
        "development": {
            "images": int(dev_mask.sum()),
            "patients": len(set(dev_eval_patients)),
            "positive_images": int(dev_eval_targets.sum()),
            "patient_disjoint_from_fit": True,
        },
        "confirmatory_pilot_validation": {
            "images": len(validation_targets),
            "patients": len(set(validation_patients)),
            "positive_images": int(validation_targets.sum()),
            "previously_inspected": True,
            "metrics": validation_metrics,
        },
        "candidates": candidate_results,
        "selected_recipe": selected,
        "inputs": {
            "train_evidence_sha256": sha256_path(train_root / "extraction_evidence.json"),
            "validation_evidence_sha256": sha256_path(validation_root / "extraction_evidence.json"),
            "labels_sha256": sha256_path(labels_path),
            "normalization_sha256": sha256_path(normalization_path),
        },
        "caveat": "current validation was previously used for the fixed feasibility pilot; treat this validation result as confirmatory pilot evidence, not a pristine final estimate",
    }
    report_path = output / "activation_comparison_autoresearch.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
