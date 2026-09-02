from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from retina_lanse.data.embedding_shards import EmbeddingRun, load_embedding_run
from retina_lanse.data.manifest import sha256_path
from retina_lanse.evaluation.metrics import evaluate_embeddings_batched
from retina_lanse.sae.checkpoints import load_checkpoint
from scripts.run_global_sae_search import require_external


def load_labels(path: Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            image_id = row["image"].strip()
            grade = int(row["level"])
            if image_id in labels or grade not in range(5):
                raise RuntimeError(f"invalid or duplicate label row: {image_id}")
            labels[image_id] = grade
    return labels


def image_id(relative_path: str) -> str:
    return Path(relative_path).stem


def labeled_subset(run: EmbeddingRun, labels: dict[str, int]) -> tuple[torch.Tensor, np.ndarray, list[str], list[str]]:
    indices = [index for index, path in enumerate(run.relative_paths) if image_id(path) in labels]
    if not indices:
        raise RuntimeError("embedding run contains no images with clinical labels")
    paths = [run.relative_paths[index] for index in indices]
    patients = [run.patient_ids[index] for index in indices]
    grades = np.asarray([labels[image_id(path)] for path in paths], dtype=np.int64)
    return run.embeddings[indices], grades, patients, paths


@torch.inference_mode()
def encode_csr(model, embeddings: torch.Tensor, *, batch_size: int) -> sparse.csr_matrix:
    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for start in range(0, len(embeddings), batch_size):
        batch = embeddings[start : start + batch_size]
        preactivations = torch.relu(model.encoder(batch))
        top_values, top_indices = torch.topk(preactivations, model.k, dim=1)
        nonzero = top_values > 0
        row = torch.arange(len(batch)).unsqueeze(1).expand_as(top_indices) + start
        rows.append(row[nonzero].numpy())
        columns.append(top_indices[nonzero].numpy())
        values.append(top_values[nonzero].numpy().astype(np.float32, copy=False))
    return sparse.csr_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
        shape=(len(embeddings), model.dictionary_size),
        dtype=np.float32,
    )


def fit_probe(features, targets: np.ndarray, *, sparse_input: bool, seed: int):
    scaler = StandardScaler(with_mean=not sparse_input)
    classifier = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        class_weight="balanced",
        max_iter=3000,
        tol=1e-5,
        random_state=seed,
        average=True,
    )
    model = make_pipeline(scaler, classifier)
    model.fit(features, targets)
    return model


def patient_bootstrap_interval(
    targets: np.ndarray,
    scores: np.ndarray,
    patients: list[str],
    metric,
    *,
    resamples: int,
    seed: int,
) -> dict[str, float | int]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, patient in enumerate(patients):
        grouped[str(patient)].append(index)
    patient_ids = sorted(grouped)
    rng = np.random.default_rng(seed)
    replicates: list[float] = []
    for _ in range(resamples):
        sampled = rng.integers(0, len(patient_ids), size=len(patient_ids))
        indices = np.concatenate([grouped[patient_ids[index]] for index in sampled])
        if np.unique(targets[indices]).size == 2:
            replicates.append(float(metric(targets[indices], scores[indices])))
    if len(replicates) < int(0.95 * resamples):
        raise RuntimeError("too many invalid bootstrap replicates")
    lower, upper = np.quantile(replicates, [0.025, 0.975])
    return {
        "estimate": float(metric(targets, scores)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "confidence": 0.95,
        "resamples": int(resamples),
        "patients": len(patient_ids),
    }


def patient_bootstrap_metric_difference(
    targets: np.ndarray,
    left_scores: np.ndarray,
    right_scores: np.ndarray,
    patients: list[str],
    metric,
    *,
    resamples: int,
    seed: int,
) -> dict[str, float | int]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, patient in enumerate(patients):
        grouped[str(patient)].append(index)
    patient_ids = sorted(grouped)
    rng = np.random.default_rng(seed)
    replicates: list[float] = []
    for _ in range(resamples):
        sampled = rng.integers(0, len(patient_ids), size=len(patient_ids))
        indices = np.concatenate([grouped[patient_ids[index]] for index in sampled])
        if np.unique(targets[indices]).size == 2:
            replicates.append(
                float(metric(targets[indices], left_scores[indices]))
                - float(metric(targets[indices], right_scores[indices]))
            )
    lower, upper = np.quantile(replicates, [0.025, 0.975])
    return {
        "difference": float(metric(targets, left_scores) - metric(targets, right_scores)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "confidence": 0.95,
        "resamples": int(resamples),
        "patients": len(patient_ids),
    }


def evaluate_probe(targets: np.ndarray, scores: np.ndarray, patients: list[str], *, resamples: int, seed: int) -> dict:
    predictions = scores >= 0.5
    return {
        "auroc": patient_bootstrap_interval(
            targets, scores, patients, roc_auc_score, resamples=resamples, seed=seed
        ),
        "auprc": patient_bootstrap_interval(
            targets,
            scores,
            patients,
            average_precision_score,
            resamples=resamples,
            seed=seed + 1,
        ),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(targets, predictions)),
    }


def feature_summary(
    train_codes: sparse.csr_matrix,
    train_targets: np.ndarray,
    validation_codes: sparse.csr_matrix,
    validation_targets: np.ndarray,
    validation_grades: np.ndarray,
    validation_paths: list[str],
    *,
    count: int,
    top_images: int,
) -> list[dict]:
    train_active = train_codes.copy()
    train_active.data = np.ones_like(train_active.data)
    positive = np.asarray(train_active[train_targets == 1].sum(axis=0)).ravel()
    negative = np.asarray(train_active[train_targets == 0].sum(axis=0)).ravel()
    n_positive = int((train_targets == 1).sum())
    n_negative = int((train_targets == 0).sum())
    positive_rate = (positive + 0.5) / (n_positive + 1.0)
    negative_rate = (negative + 0.5) / (n_negative + 1.0)
    log_rate_ratio = np.log(positive_rate / negative_rate)
    eligible = (positive + negative) >= 50
    ranked = np.argsort(np.where(eligible, log_rate_ratio, -np.inf))[::-1][:count]

    validation_active = validation_codes.copy()
    validation_active.data = np.ones_like(validation_active.data)
    records: list[dict] = []
    for feature in ranked:
        column = validation_codes.getcol(int(feature)).tocoo()
        order = np.argsort(column.data)[::-1][:top_images]
        image_indices = column.row[order]
        active_mask = np.asarray(validation_active.getcol(int(feature)).toarray()).ravel() > 0
        active_positive = int(validation_targets[active_mask].sum())
        active_total = int(active_mask.sum())
        records.append(
            {
                "feature": int(feature),
                "selection_rule": "largest_train_log_activation_rate_ratio_with_at_least_50_train_activations",
                "train_log_activation_rate_ratio": float(log_rate_ratio[feature]),
                "train_active_positive": int(positive[feature]),
                "train_active_negative": int(negative[feature]),
                "validation_active_images": active_total,
                "validation_active_referable_dr": active_positive,
                "validation_referable_dr_fraction": (
                    float(active_positive / active_total) if active_total else None
                ),
                "top_validation_images": [
                    {
                        "relative_path": validation_paths[int(index)],
                        "grade": int(validation_grades[int(index)]),
                        "referable_dr": bool(validation_targets[int(index)]),
                        "activation": float(column.data[order[position]]),
                    }
                    for position, index in enumerate(image_indices)
                ],
            }
        )
    return records


def make_montage(records: list[dict], dataset_root: Path, output: Path) -> None:
    tile, label_height = 220, 44
    columns = max(len(record["top_validation_images"]) for record in records)
    canvas = Image.new("RGB", (columns * tile, len(records) * (tile + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    for row, record in enumerate(records):
        for column, item in enumerate(record["top_validation_images"]):
            path = dataset_root / item["relative_path"]
            with Image.open(path) as source:
                image = source.convert("RGB")
                image.thumbnail((tile, tile))
                left = column * tile + (tile - image.width) // 2
                top = row * (tile + label_height) + label_height + (tile - image.height) // 2
                canvas.paste(image, (left, top))
        draw.text(
            (4, row * (tile + label_height) + 4),
            f"feature {record['feature']} | validation DR fraction {record['validation_referable_dr_fraction']}",
            fill="black",
        )
    canvas.save(output, quality=92)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a bounded train/validation retinal SAE feasibility evaluation.")
    parser.add_argument("--train-run", required=True)
    parser.add_argument("--validation-run", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-metrics", required=True)
    parser.add_argument("--dataset-root", required=True)
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
    metrics_path = require_external(Path(args.checkpoint_metrics), "SAE metrics")
    dataset_root = require_external(Path(args.dataset_root), "EyePACS dataset")
    output = require_external(Path(args.output), "feasibility output")
    if "internal_test" in str(validation_root) or "internal_test" in str(output):
        raise RuntimeError("this feasibility script refuses locked internal-test paths")
    output.mkdir(parents=True, exist_ok=True)

    train_run = load_embedding_run(train_root, expected_split="train", verify_sha256=True)
    validation_run = load_embedding_run(validation_root, expected_split="validation", verify_sha256=True)
    if set(train_run.patient_ids) & set(validation_run.patient_ids):
        raise RuntimeError("patient overlap between train and validation embeddings")
    if train_run.evidence["checkpoint_hash"] != validation_run.evidence["checkpoint_hash"]:
        raise RuntimeError("train and validation encoder checkpoints differ")
    if train_run.evidence["representation"] != "final_mean_patch" or validation_run.evidence["representation"] != "final_mean_patch":
        raise RuntimeError("feasibility scope is fixed to final_mean_patch")

    labels = load_labels(labels_path)
    train, train_grades, train_patients, train_paths = labeled_subset(train_run, labels)
    validation, validation_grades, validation_patients, validation_paths = labeled_subset(validation_run, labels)
    if set(train_patients) & set(validation_patients):
        raise RuntimeError("patient leakage after joining clinical labels")
    train_targets = (train_grades >= 2).astype(np.int64)
    validation_targets = (validation_grades >= 2).astype(np.int64)

    normalization = torch.load(normalization_path, map_location="cpu", weights_only=False)
    if normalization["train_manifest_hash"] != train_run.evidence["manifest_hash"]:
        raise RuntimeError("normalization was not fitted on this embedding manifest")
    train = (train - normalization["mean"]) / normalization["std"]
    validation = (validation - normalization["mean"]) / normalization["std"]

    model, payload = load_checkpoint(checkpoint_path)
    if payload["model_config"] != {"input_dim": 1024, "dictionary_size": 16384, "k": 32}:
        raise RuntimeError("checkpoint is not the frozen 16,384-feature Top-K-32 feasibility SAE")
    train_codes = encode_csr(model, train, batch_size=args.batch_size)
    validation_codes = encode_csr(model, validation, batch_size=args.batch_size)

    dense_train = train.numpy()
    dense_validation = validation.numpy()
    decoder_weight = model.decoder.weight.detach().numpy()
    decoder_bias = model.decoder.bias.detach().numpy()
    reconstructed_train = np.asarray(train_codes @ decoder_weight.T) + decoder_bias
    reconstructed_validation = np.asarray(validation_codes @ decoder_weight.T) + decoder_bias
    pca = PCA(n_components=32, svd_solver="randomized", random_state=args.seed)
    pca_train = pca.fit_transform(dense_train)
    pca_validation = pca.transform(dense_validation)
    feature_sets = {
        "dense_retfound_1024": (dense_train, dense_validation, False),
        "pca_32": (pca_train, pca_validation, False),
        "sae_reconstruction_1024": (
            reconstructed_train,
            reconstructed_validation,
            False,
        ),
        "sae_topk32_16384": (train_codes, validation_codes, True),
    }
    probes = {}
    probe_scores: dict[str, np.ndarray] = {}
    for offset, (name, (fit_x, eval_x, sparse_input)) in enumerate(feature_sets.items()):
        probe = fit_probe(fit_x, train_targets, sparse_input=sparse_input, seed=args.seed)
        scores = probe.predict_proba(eval_x)[:, 1]
        probe_scores[name] = scores
        probes[name] = evaluate_probe(
            validation_targets,
            scores,
            validation_patients,
            resamples=args.bootstrap_resamples,
            seed=args.seed + offset * 100,
        )
    dense_scores = probe_scores["dense_retfound_1024"]
    paired_differences = {
        name: {
            "auroc_model_minus_dense": patient_bootstrap_metric_difference(
                validation_targets,
                scores,
                dense_scores,
                validation_patients,
                roc_auc_score,
                resamples=args.bootstrap_resamples,
                seed=args.seed + 1000 + index * 10,
            ),
            "auprc_model_minus_dense": patient_bootstrap_metric_difference(
                validation_targets,
                scores,
                dense_scores,
                validation_patients,
                average_precision_score,
                resamples=args.bootstrap_resamples,
                seed=args.seed + 1001 + index * 10,
            ),
        }
        for index, (name, scores) in enumerate(probe_scores.items())
        if name != "dense_retfound_1024"
    }

    features = feature_summary(
        train_codes,
        train_targets,
        validation_codes,
        validation_targets,
        validation_grades,
        validation_paths,
        count=4,
        top_images=6,
    )
    montage_path = output / "top_disease_associated_features.jpg"
    make_montage(features, dataset_root, montage_path)
    checkpoint_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    report = {
        "schema_version": "1.0",
        "status": "complete_bounded_feasibility",
        "claim_boundary": "train_validation_feasibility_only_not_clinical_utility_or_external_validation",
        "autoresearch_used": False,
        "locked_internal_test_accessed": False,
        "target": "referable_diabetic_retinopathy_grade_at_least_2",
        "encoder": {"name": "RETFound_CFP", "frozen": True, "representation": "final_mean_patch", "dimensions": 1024},
        "sae": {"trained_in_this_run": False, **payload["model_config"], "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256_path(checkpoint_path)},
        "cohort": {
            "train_images": len(train_targets),
            "train_patients": len(set(train_patients)),
            "train_positive_images": int(train_targets.sum()),
            "validation_images": len(validation_targets),
            "validation_patients": len(set(validation_patients)),
            "validation_positive_images": int(validation_targets.sum()),
            "patient_overlap": 0,
        },
        "probe_protocol": {
            "model": "fixed_linear_SGD_logistic_probe",
            "class_weight": "balanced",
            "alpha": 1e-4,
            "preprocessing": "train_fitted_standard_scaling",
            "pca_components": 32,
            "hyperparameter_tuning": False,
            "validation_used_for_model_selection": False,
            "patient_cluster_bootstrap_resamples": args.bootstrap_resamples,
        },
        "sae_reconstruction_validation": checkpoint_metrics["validation"],
        "pca_32_explained_variance_ratio_sum_train": float(pca.explained_variance_ratio_.sum()),
        "clinical_signal_probes": probes,
        "paired_patient_bootstrap_vs_dense": paired_differences,
        "feature_inspection": features,
        "montage": str(montage_path),
        "inputs": {
            "train_evidence_sha256": sha256_path(train_root / "extraction_evidence.json"),
            "validation_evidence_sha256": sha256_path(validation_root / "extraction_evidence.json"),
            "labels_sha256": sha256_path(labels_path),
            "normalization_sha256": sha256_path(normalization_path),
            "checkpoint_metrics_sha256": sha256_path(metrics_path),
        },
    }
    report_path = output / "feasibility_results.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
