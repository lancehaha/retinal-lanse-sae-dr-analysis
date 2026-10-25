from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from retina_lanse.data.embedding_shards import EmbeddingRun, load_embedding_run
from retina_lanse.sae.checkpoints import load_checkpoint
from scripts.run_activation_interaction_autoresearch import feature_stats
from scripts.run_global_sae_search import require_external
from scripts.run_single_sae_feasibility import labeled_subset, patient_bootstrap_interval


def load_labels(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            key = row["image"].strip()
            grade = int(row["level"])
            if key in values or grade not in range(5):
                raise RuntimeError(f"invalid or duplicate label: {key}")
            values[key] = grade
    return values


def encode_and_reconstruct(model, embeddings: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, batch_size: int):
    recon_rows: list[np.ndarray] = []
    csr_rows: list[np.ndarray] = []
    csr_cols: list[np.ndarray] = []
    csr_values: list[np.ndarray] = []
    model.eval()
    offset = 0
    with torch.inference_mode():
        for start in range(0, len(embeddings), batch_size):
            batch = (embeddings[start : start + batch_size] - mean) / std
            recon, codes = model(batch)
            recon_rows.append(recon.cpu().numpy().astype(np.float32, copy=False))
            nz = codes > 0
            rows, cols = torch.where(nz)
            csr_rows.append((rows.cpu().numpy() + offset).astype(np.int64, copy=False))
            csr_cols.append(cols.cpu().numpy().astype(np.int64, copy=False))
            csr_values.append(codes[nz].cpu().numpy().astype(np.float32, copy=False))
            offset += len(batch)
    reconstruction = np.concatenate(recon_rows, axis=0)
    codes = sparse.csr_matrix(
        (np.concatenate(csr_values), (np.concatenate(csr_rows), np.concatenate(csr_cols))),
        shape=(len(embeddings), model.dictionary_size),
        dtype=np.float32,
    )
    return reconstruction, codes


def fit_probe(train_x: np.ndarray, train_y: np.ndarray, *, seed: int) -> LogisticRegression:
    # Explicit train-only standardization keeps dense and fusion arms comparable.
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    model = LogisticRegression(
        class_weight="balanced",
        C=1.0,
        max_iter=2000,
        solver="lbfgs",
        random_state=seed,
    )
    model.fit((train_x - mean) / std, train_y)
    model._retina_train_mean = mean
    model._retina_train_std = std
    return model


def predict_probe(model: LogisticRegression, x: np.ndarray) -> np.ndarray:
    mean = model._retina_train_mean
    std = model._retina_train_std
    return model.predict_proba((x - mean) / std)[:, 1]


def bootstrap_difference(targets, left, right, patients, metric, resamples, seed):
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, patient in enumerate(patients):
        grouped[str(patient)].append(index)
    ids = sorted(grouped)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(resamples):
        sampled = rng.integers(0, len(ids), size=len(ids))
        indices = np.concatenate([grouped[ids[i]] for i in sampled])
        if np.unique(targets[indices]).size == 2:
            values.append(float(metric(targets[indices], left[indices])) - float(metric(targets[indices], right[indices])))
    lo, hi = np.quantile(values, [0.025, 0.975])
    return {
        "difference": float(metric(targets, left) - metric(targets, right)),
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "confidence": 0.95,
        "resamples": int(resamples),
        "patients": len(ids),
    }


def metric_bundle(targets, scores, patients, resamples, seed):
    return {
        "auroc": patient_bootstrap_interval(targets, scores, patients, roc_auc_score, resamples=resamples, seed=seed),
        "auprc": patient_bootstrap_interval(targets, scores, patients, average_precision_score, resamples=resamples, seed=seed + 1),
        "brier": float(brier_score_loss(targets, scores)),
    }


def collect(run: EmbeddingRun, labels: dict[str, int]):
    embeddings, grades, patients, paths = labeled_subset(run, labels)
    return embeddings.numpy().astype(np.float32), grades, np.asarray(patients), paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Leakage-safe dense plus SAE fusion experiment")
    parser.add_argument("--train-run", required=True)
    parser.add_argument("--validation-run", required=True)
    parser.add_argument("--test-run", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=4106)
    args = parser.parse_args()

    roots = {name: require_external(Path(value), name) for name, value in {
        "train run": args.train_run,
        "validation run": args.validation_run,
        "locked test run": args.test_run,
        "labels": args.labels,
        "normalization": args.normalization,
        "SAE checkpoint": args.checkpoint,
        "fusion output": args.output,
    }.items()}
    output = roots["fusion output"]
    output.mkdir(parents=True, exist_ok=True)
    labels = load_labels(roots["labels"])
    train_run = load_embedding_run(roots["train run"], expected_split="train", verify_sha256=True)
    val_run = load_embedding_run(roots["validation run"], expected_split="validation", verify_sha256=True)
    test_run = load_embedding_run(roots["locked test run"], expected_split="internal_test", verify_sha256=True)
    train_dense, train_grades, train_patients, _ = collect(train_run, labels)
    val_dense, val_grades, val_patients, _ = collect(val_run, labels)
    test_dense, test_grades, test_patients, _ = collect(test_run, labels)
    train_y = (train_grades >= 2).astype(np.int64)
    val_y = (val_grades >= 2).astype(np.int64)
    test_y = (test_grades >= 2).astype(np.int64)
    if set(train_patients) & set(val_patients) or set(train_patients) & set(test_patients) or set(val_patients) & set(test_patients):
        raise RuntimeError("patient overlap detected across fusion partitions")

    normalization = torch.load(roots["normalization"], map_location="cpu", weights_only=False)
    mean = normalization["mean"].float()
    std = normalization["std"].float()
    model, _ = load_checkpoint(roots["SAE checkpoint"])
    train_recon, train_codes = encode_and_reconstruct(model, torch.from_numpy(train_dense), mean, std, args.batch_size)
    val_recon, val_codes = encode_and_reconstruct(model, torch.from_numpy(val_dense), mean, std, args.batch_size)
    test_recon, test_codes = encode_and_reconstruct(model, torch.from_numpy(test_dense), mean, std, args.batch_size)

    stats = feature_stats(train_codes, train_y)
    eligible = stats["count"] >= 50
    ranking = np.where(eligible, stats["mean_difference"], -np.inf)
    selected = np.argsort(ranking)[::-1][:64]
    train_sparse = train_codes[:, selected].toarray().astype(np.float32)
    val_sparse = val_codes[:, selected].toarray().astype(np.float32)
    test_sparse = test_codes[:, selected].toarray().astype(np.float32)
    sae_weights = stats["mean_difference"][selected].astype(np.float32)
    sae_weights /= max(float(np.abs(sae_weights).sum()), 1e-8)
    train_sae_score = np.asarray(train_codes[:, selected].dot(sae_weights)).ravel()
    val_sae_score = np.asarray(val_codes[:, selected].dot(sae_weights)).ravel()
    test_sae_score = np.asarray(test_codes[:, selected].dot(sae_weights)).ravel()

    train_dense_model = fit_probe(train_dense, train_y, seed=args.seed)
    train_recon_model = fit_probe(train_recon, train_y, seed=args.seed)
    train_sparse_model = fit_probe(train_sparse, train_y, seed=args.seed)
    train_fusion_model = fit_probe(np.concatenate([train_dense, train_sparse], axis=1), train_y, seed=args.seed)
    train_dense_recon_model = fit_probe(np.concatenate([train_dense, train_recon], axis=1), train_y, seed=args.seed)
    late_model = fit_probe(np.column_stack([predict_probe(train_dense_model, train_dense), train_sae_score]), train_y, seed=args.seed)

    predictions = {
        "dense": (predict_probe(train_dense_model, val_dense), predict_probe(train_dense_model, test_dense)),
        "sae_reconstruction": (predict_probe(train_recon_model, val_recon), predict_probe(train_recon_model, test_recon)),
        "sparse_64": (predict_probe(train_sparse_model, val_sparse), predict_probe(train_sparse_model, test_sparse)),
        "dense_plus_sparse64": (predict_probe(train_fusion_model, np.concatenate([val_dense, val_sparse], axis=1)), predict_probe(train_fusion_model, np.concatenate([test_dense, test_sparse], axis=1))),
        "dense_plus_reconstruction": (predict_probe(train_dense_recon_model, np.concatenate([val_dense, val_recon], axis=1)), predict_probe(train_dense_recon_model, np.concatenate([test_dense, test_recon], axis=1))),
        "late_dense_plus_sae": (predict_probe(late_model, np.column_stack([predict_probe(train_dense_model, val_dense), val_sae_score])), predict_probe(late_model, np.column_stack([predict_probe(train_dense_model, test_dense), test_sae_score]))),
    }
    result = {
        "schema_version": "1.0",
        "status": "complete_dense_sae_fusion",
        "encoder_frozen": True,
        "sae_frozen": True,
        "endpoint": "referable_dr_grade_ge_2",
        "cohorts": {
            "train": {"images": int(len(train_y)), "patients": len(set(train_patients)), "positive_images": int(train_y.sum())},
            "validation": {"images": int(len(val_y)), "patients": len(set(val_patients)), "positive_images": int(val_y.sum())},
            "locked_internal_test": {"images": int(len(test_y)), "patients": len(set(test_patients)), "positive_images": int(test_y.sum())},
        },
        "selected_sae_features": [int(x) for x in selected],
        "arms": {},
    }
    for name, (val_scores, test_scores) in predictions.items():
        result["arms"][name] = {
            "validation": metric_bundle(val_y, val_scores, val_patients, args.bootstrap_resamples, args.seed),
            "locked_internal_test": metric_bundle(test_y, test_scores, test_patients, args.bootstrap_resamples, args.seed + 10),
            "locked_delta_vs_dense": {
                "auroc": bootstrap_difference(test_y, test_scores, predictions["dense"][1], test_patients, roc_auc_score, args.bootstrap_resamples, args.seed + 20),
                "auprc": bootstrap_difference(test_y, test_scores, predictions["dense"][1], test_patients, average_precision_score, args.bootstrap_resamples, args.seed + 21),
            },
        }
    result["primary_comparison"] = {
        "fusion_arm": "dense_plus_sparse64",
        "reference_arm": "dense",
        "meaningful_auroc_gain": 0.03,
        "locked_internal_test_auroc_difference": result["arms"]["dense_plus_sparse64"]["locked_delta_vs_dense"]["auroc"],
    }
    result["protocol_note"] = "All model fitting, SAE feature selection, standardization, and late-fusion fitting used training data only; validation and locked internal test were evaluation cohorts."
    path = output / "dense_sae_fusion_results.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
