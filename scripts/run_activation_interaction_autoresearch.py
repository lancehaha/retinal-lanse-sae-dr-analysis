from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from scipy import sparse
from sklearn.metrics import average_precision_score, roc_auc_score

from retina_lanse.data.embedding_shards import load_embedding_run
from retina_lanse.data.manifest import sha256_path
from retina_lanse.sae.checkpoints import load_checkpoint
from scripts.run_global_sae_search import require_external
from scripts.run_single_sae_feasibility import encode_csr, labeled_subset, patient_bootstrap_interval


def load_labels(path: Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            key, value = row["image"].strip(), int(row["level"])
            if key in labels or value not in range(5):
                raise RuntimeError(f"invalid or duplicate label: {key}")
            labels[key] = value
    return labels


def split_patients(patients: list[str], seed: int) -> np.ndarray:
    return np.asarray(
        [
            int(
                int.from_bytes(hashlib.sha256(f"activation-interaction:{seed}:{patient}".encode()).digest()[:8], "big")
                / 2**64
                >= 0.8
            )
            for patient in patients
        ],
        dtype=bool,
    )


def feature_stats(codes: sparse.csr_matrix, targets: np.ndarray) -> dict[str, np.ndarray]:
    active = codes.copy()
    active.data = np.ones_like(active.data)
    pos, neg = targets == 1, targets == 0
    pos_n, neg_n = max(int(pos.sum()), 1), max(int(neg.sum()), 1)
    pos_rate = np.asarray(active[pos].mean(axis=0)).ravel()
    neg_rate = np.asarray(active[neg].mean(axis=0)).ravel()
    pos_mean = np.asarray(codes[pos].mean(axis=0)).ravel()
    neg_mean = np.asarray(codes[neg].mean(axis=0)).ravel()
    return {
        "count": np.asarray(active.sum(axis=0)).ravel(),
        "rate_difference": pos_rate - neg_rate,
        "mean_difference": pos_mean - neg_mean,
    }


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    return weights.astype(np.float32) / max(float(np.abs(weights).sum()), 1e-8)


def score_additive(codes: sparse.csr_matrix, stats: dict[str, np.ndarray], features: int = 64) -> np.ndarray:
    eligible = stats["count"] >= 50
    ranking = np.where(eligible, stats["mean_difference"], -np.inf)
    selected = np.argsort(ranking)[::-1][:features]
    return np.asarray(codes[:, selected].dot(normalize_weights(stats["mean_difference"][selected]))).ravel()


def pair_candidates(
    codes: sparse.csr_matrix,
    targets: np.ndarray,
    *,
    base_features: int,
    pair_features: int,
    minimum_pair_activations: int,
) -> dict[str, dict]:
    stats = feature_stats(codes, targets)
    eligible = stats["count"] >= 50
    ranking = np.where(eligible, stats["mean_difference"], -np.inf)
    selected = np.argsort(ranking)[::-1][:base_features]
    dense = codes[:, selected].toarray().astype(np.float32)
    active = dense > 0
    pos, neg = targets == 1, targets == 0
    pos_n, neg_n = max(int(pos.sum()), 1), max(int(neg.sum()), 1)
    pair_rows: list[tuple[int, int, int, float, float]] = []
    for left, right in combinations(range(len(selected)), 2):
        binary = active[:, left] & active[:, right]
        count = int(binary.sum())
        if count < minimum_pair_activations:
            continue
        rate_difference = float(binary[pos].sum() / pos_n - binary[neg].sum() / neg_n)
        product = dense[:, left] * dense[:, right]
        mean_difference = float(product[pos].mean() - product[neg].mean())
        pair_rows.append((left, right, count, rate_difference, mean_difference))
    if not pair_rows:
        raise RuntimeError("no eligible feature pairs")
    return {
        "base_features": [int(x) for x in selected],
        "pairs": pair_rows,
        "pair_features": pair_features,
    }


def score_pairs(codes: sparse.csr_matrix, candidate: dict, *, statistic_index: int, continuous: bool) -> np.ndarray:
    selected = np.asarray(candidate["base_features"], dtype=np.int64)
    dense = codes[:, selected].toarray().astype(np.float32)
    pairs = sorted(candidate["pairs"], key=lambda row: row[statistic_index], reverse=True)[: candidate["pair_features"]]
    weights = normalize_weights(np.asarray([row[statistic_index] for row in pairs], dtype=np.float32))
    values = []
    for (left, right, *_), weight in zip(pairs, weights):
        product = dense[:, left] * dense[:, right]
        values.append((product if continuous else (product > 0).astype(np.float32)) * weight)
    return np.sum(np.stack(values, axis=1), axis=1)


def evaluate(targets: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(targets, scores)),
        "auprc": float(average_precision_score(targets, scores)),
        "objective": float(roc_auc_score(targets, scores)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Bounded AutoResearch over additive and pairwise retinal SAE activation rules.")
    parser.add_argument("--train-run", required=True)
    parser.add_argument("--validation-run", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=4106)
    args = parser.parse_args()

    train_root = require_external(Path(args.train_run), "train embeddings")
    validation_root = require_external(Path(args.validation_run), "validation embeddings")
    labels_path = require_external(Path(args.labels), "clinical labels")
    normalization_path = require_external(Path(args.normalization), "normalization")
    checkpoint_path = require_external(Path(args.checkpoint), "SAE checkpoint")
    output = require_external(Path(args.output), "interaction AutoResearch output")
    if "internal_test" in str(validation_root) or "internal_test" in str(output):
        raise RuntimeError("interaction AutoResearch refuses locked internal-test paths")
    output.mkdir(parents=True, exist_ok=True)

    train_run = load_embedding_run(train_root, expected_split="train", verify_sha256=True)
    validation_run = load_embedding_run(validation_root, expected_split="validation", verify_sha256=True)
    if set(train_run.patient_ids) & set(validation_run.patient_ids):
        raise RuntimeError("embedding patient overlap")
    if train_run.evidence["checkpoint_hash"] != validation_run.evidence["checkpoint_hash"]:
        raise RuntimeError("encoder checkpoint mismatch")
    labels = load_labels(labels_path)
    train, train_grades, train_patients, _ = labeled_subset(train_run, labels)
    validation, validation_grades, validation_patients, _ = labeled_subset(validation_run, labels)
    train_targets, validation_targets = (train_grades >= 2).astype(np.int64), (validation_grades >= 2).astype(np.int64)
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
    fit_codes, eval_codes = train_codes[~dev_mask], train_codes[dev_mask]
    fit_targets, eval_targets = train_targets[~dev_mask], train_targets[dev_mask]
    candidates = []
    additive_stats = feature_stats(fit_codes, fit_targets)
    additive_eval = score_additive(eval_codes, additive_stats)
    candidates.append({"name": "weighted_mean_difference_top64", "development": evaluate(eval_targets, additive_eval)})
    pair = pair_candidates(fit_codes, fit_targets, base_features=64, pair_features=128, minimum_pair_activations=20)
    for name, index, continuous in (
        ("pair_coactivation_rate_top128", 3, False),
        ("pair_activation_product_top128", 4, True),
    ):
        scores = score_pairs(eval_codes, pair, statistic_index=index, continuous=continuous)
        candidates.append({"name": name, "base_features": 64, "pair_features": 128, "minimum_pair_activations": 20, "development": evaluate(eval_targets, scores)})
    selected = max(candidates, key=lambda item: (item["development"]["objective"], item["development"]["auprc"]))

    full_stats = feature_stats(train_codes, train_targets)
    if selected["name"] == "weighted_mean_difference_top64":
        validation_scores = score_additive(validation_codes, full_stats)
    else:
        full_pair = pair_candidates(train_codes, train_targets, base_features=64, pair_features=128, minimum_pair_activations=20)
        validation_scores = score_pairs(validation_codes, full_pair, statistic_index=3 if selected["name"].startswith("pair_coactivation") else 4, continuous=selected["name"].startswith("pair_activation"))
    validation_metrics = evaluate(validation_targets, validation_scores)
    validation_metrics["auroc_patient_bootstrap"] = patient_bootstrap_interval(validation_targets, validation_scores, validation_patients, roc_auc_score, resamples=args.bootstrap_resamples, seed=args.seed + 10)
    validation_metrics["auprc_patient_bootstrap"] = patient_bootstrap_interval(validation_targets, validation_scores, validation_patients, average_precision_score, resamples=args.bootstrap_resamples, seed=args.seed + 11)
    report = {
        "schema_version": "1.0",
        "status": "complete_bounded_additive_pairwise_search",
        "autoresearch_used": True,
        "search_scope": "additive_baseline_plus_two_prespecified_pairwise_activation_rules_on_patient_80_20_split_inside_labeled_train",
        "selection_rule": "highest_development_AUROC_then_AUPRC",
        "locked_internal_test_accessed": False,
        "sae_checkpoint": str(checkpoint_path),
        "sae_checkpoint_sha256": sha256_path(checkpoint_path),
        "sae_config": payload["model_config"],
        "development": {"images": int(dev_mask.sum()), "patients": len({p for p, held in zip(train_patients, dev_mask) if held}), "positive_images": int(eval_targets.sum()), "patient_disjoint_from_fit": True},
        "candidates": candidates,
        "selected_recipe": selected,
        "confirmatory_pilot_validation": {"images": len(validation_targets), "patients": len(set(validation_patients)), "positive_images": int(validation_targets.sum()), "previously_inspected": True, "metrics": validation_metrics},
        "inputs": {"train_evidence_sha256": sha256_path(train_root / "extraction_evidence.json"), "validation_evidence_sha256": sha256_path(validation_root / "extraction_evidence.json"), "labels_sha256": sha256_path(labels_path), "normalization_sha256": sha256_path(normalization_path)},
        "caveat": "validation was previously inspected and is confirmatory pilot evidence, not a pristine final estimate",
    }
    report_path = output / "activation_interaction_autoresearch.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
