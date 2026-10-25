from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    roc_auc_score,
)

from retina_lanse.data.embedding_shards import EmbeddingRun, load_embedding_run
from retina_lanse.data.manifest import sha256_path
from retina_lanse.sae.checkpoints import load_checkpoint
from scripts.run_activation_interaction_autoresearch import feature_stats
from scripts.run_global_sae_search import require_external
from scripts.run_single_sae_feasibility import labeled_subset, patient_bootstrap_interval


FRACTIONS = (0.01, 0.05, 0.10, 0.25, 0.50, 1.00)
ARMS = ("dense", "sparse_64", "dense_plus_sparse64", "sae_reconstruction")


def image_key(value: str) -> str:
    return Path(value.strip()).stem.lower()


def load_eyepacs_labels(path: Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            key = image_key(row["image"])
            grade = int(row["level"])
            if key in labels or grade not in range(5):
                raise RuntimeError(f"invalid or duplicate EyePACS label: {key}")
            labels[key] = grade
    return labels


def load_idrid_manifest(path: Path) -> dict[str, tuple[int, str]]:
    result: dict[str, tuple[int, str]] = {}
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        record = json.loads(line)
        key = record["relative_path"]
        if key in result:
            raise RuntimeError(f"duplicate IDRiD relative path: {key}")
        grade = record.get("grade")
        if grade not in range(5):
            raise RuntimeError(f"IDRiD record has invalid grade: {key}")
        result[key] = (int(grade), str(record["patient_id"]))
    if not result:
        raise RuntimeError("IDRiD manifest is empty")
    return result


def collect_eyepacs(run: EmbeddingRun, labels: dict[str, int]):
    embeddings, grades, patients, paths = labeled_subset(run, labels)
    return embeddings.numpy().astype(np.float32), grades, np.asarray(patients), paths


def collect_idrid(run: EmbeddingRun, labels: dict[str, tuple[int, str]]):
    indices = [i for i, path in enumerate(run.relative_paths) if path in labels]
    if len(indices) != len(run.relative_paths):
        missing = [path for path in run.relative_paths if path not in labels]
        raise RuntimeError(f"IDRiD embedding has paths absent from clean manifest: {missing[:3]}")
    grades = np.asarray([labels[run.relative_paths[i]][0] for i in indices], dtype=np.int64)
    patients = np.asarray([labels[run.relative_paths[i]][1] for i in indices])
    return run.embeddings[indices].numpy().astype(np.float32), grades, patients, [run.relative_paths[i] for i in indices]


def encode_and_reconstruct(model, embeddings: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, batch_size: int):
    recon_rows: list[np.ndarray] = []
    csr_rows: list[np.ndarray] = []
    csr_cols: list[np.ndarray] = []
    csr_values: list[np.ndarray] = []
    offset = 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(embeddings), batch_size):
            batch = (embeddings[start : start + batch_size] - mean) / std
            recon, codes = model(batch)
            recon_rows.append(recon.cpu().numpy().astype(np.float32, copy=False))
            rows, cols = torch.where(codes > 0)
            csr_rows.append((rows.cpu().numpy() + offset).astype(np.int64, copy=False))
            csr_cols.append(cols.cpu().numpy().astype(np.int64, copy=False))
            csr_values.append(codes[codes > 0].cpu().numpy().astype(np.float32, copy=False))
            offset += len(batch)
    codes = sparse.csr_matrix(
        (np.concatenate(csr_values), (np.concatenate(csr_rows), np.concatenate(csr_cols))),
        shape=(len(embeddings), model.dictionary_size),
        dtype=np.float32,
    )
    return np.concatenate(recon_rows, axis=0), codes


def fit_probe(features: np.ndarray, targets: np.ndarray, seed: int, *, multiclass: bool = False):
    if multiclass and len(np.unique(targets)) < 5:
        return None
    model = LogisticRegression(
        class_weight="balanced",
        C=1.0,
        max_iter=2000,
        solver="lbfgs",
        multi_class="multinomial" if multiclass else "auto",
        random_state=seed,
    )
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    model.fit((features - mean) / std, targets)
    model._retina_train_mean = mean
    model._retina_train_std = std
    return model


def predict_probe(model, features: np.ndarray) -> np.ndarray:
    return model.predict_proba((features - model._retina_train_mean) / model._retina_train_std)


def fit_youden_threshold(targets: np.ndarray, scores: np.ndarray) -> float:
    """Fit a decision threshold on training data only (maximizes Youden J)."""
    order = np.argsort(scores)[::-1]
    y = targets[order]
    s = scores[order]
    positives = max(int(y.sum()), 1)
    negatives = max(int((1 - y).sum()), 1)
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    j = tp / positives - fp / negatives
    best = int(np.argmax(j))
    threshold = float(s[best])
    return float(np.clip(threshold, 0.0, 1.0))


def expected_calibration_error(targets: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = max(len(scores), 1)
    error = 0.0
    for index in range(bins):
        mask = (scores >= edges[index]) & (scores <= edges[index + 1] if index == bins - 1 else scores < edges[index + 1])
        if np.any(mask):
            error += float(mask.mean()) * abs(float(scores[mask].mean()) - float(targets[mask].mean()))
    return float(error)


def threshold_metrics(targets: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = scores >= threshold
    tp = int(np.sum(predicted & (targets == 1)))
    tn = int(np.sum(~predicted & (targets == 0)))
    fp = int(np.sum(predicted & (targets == 0)))
    fn = int(np.sum(~predicted & (targets == 1)))
    return {
        "threshold": float(threshold),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "ppv": float(tp / max(tp + fp, 1)),
        "npv": float(tn / max(tn + fn, 1)),
        "calibration_error_ece10": expected_calibration_error(targets, scores),
    }


def patient_groups(patients: np.ndarray) -> dict[str, np.ndarray]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, patient in enumerate(patients):
        groups[str(patient)].append(index)
    return {key: np.asarray(value, dtype=np.int64) for key, value in groups.items()}


def sample_patient_indices(patients: np.ndarray, grades: np.ndarray, fraction: float, seed: int):
    groups = patient_groups(patients)
    ids = sorted(groups)
    target_n = len(ids) if fraction >= 1.0 else max(5, int(round(len(ids) * fraction)))
    target_n = min(target_n, len(ids))
    rng = np.random.default_rng(seed)
    if target_n == len(ids):
        selected_ids = ids
    else:
        patient_max_grade = {pid: int(grades[groups[pid]].max()) for pid in ids}
        strata = {grade: [pid for pid in ids if patient_max_grade[pid] == grade] for grade in range(5)}
        counts = {grade: min(len(strata[grade]), int(math.floor(target_n * len(strata[grade]) / len(ids)))) for grade in range(5)}
        while sum(counts.values()) < target_n:
            candidates = [grade for grade in range(5) if counts[grade] < len(strata[grade])]
            grade = candidates[int(rng.integers(0, len(candidates)))]
            counts[grade] += 1
        selected: set[str] = set()
        for grade in range(5):
            if counts[grade]:
                selected.update(rng.choice(strata[grade], size=counts[grade], replace=False).tolist())
        # Ensure all five image grades are represented whenever the budget allows it.
        for grade in range(5):
            if target_n >= 5 and not np.any(grades[np.concatenate([groups[pid] for pid in selected])] == grade):
                candidates = [pid for pid in ids if pid not in selected and grade in set(grades[groups[pid]])]
                if not candidates:
                    continue
                replacement = next((pid for pid in sorted(selected) if counts[patient_max_grade[pid]] > 1), None)
                if replacement is None:
                    replacement = sorted(selected)[0]
                selected.remove(replacement)
                counts[patient_max_grade[replacement]] -= 1
                selected.add(candidates[int(rng.integers(0, len(candidates)))])
        selected_ids = sorted(selected)
    indices = np.concatenate([groups[pid] for pid in selected_ids])
    return np.sort(indices), selected_ids


def binary_metrics(targets: np.ndarray, scores: np.ndarray, patients: np.ndarray, bootstrap: int, seed: int):
    patient_list = [str(x) for x in patients]
    return {
        "auroc": patient_bootstrap_interval(targets, scores, patient_list, roc_auc_score, resamples=bootstrap, seed=seed),
        "auprc": patient_bootstrap_interval(targets, scores, patient_list, average_precision_score, resamples=bootstrap, seed=seed + 1),
        "brier": float(brier_score_loss(targets, scores)),
    }


def binary_difference(targets, left, right, patients, metric, bootstrap, seed):
    groups = patient_groups(np.asarray(patients))
    ids = sorted(groups)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(bootstrap):
        sampled = rng.integers(0, len(ids), size=len(ids))
        indices = np.concatenate([groups[ids[i]] for i in sampled])
        if np.unique(targets[indices]).size == 2:
            values.append(float(metric(targets[indices], left[indices])) - float(metric(targets[indices], right[indices])))
    lo, hi = np.quantile(values, [0.025, 0.975])
    p_lower = float(np.mean(np.asarray(values) <= 0.0))
    p_upper = float(np.mean(np.asarray(values) >= 0.0))
    return {"difference": float(metric(targets, left) - metric(targets, right)), "ci_lower": float(lo), "ci_upper": float(hi), "p_value": float(min(1.0, 2.0 * min(p_lower, p_upper))), "confidence": 0.95, "resamples": bootstrap, "patients": len(ids)}


def multiclass_metrics(targets: np.ndarray, probabilities: np.ndarray) -> dict:
    prediction = np.argmax(probabilities, axis=1)
    one_hot = np.eye(5, dtype=np.int64)[targets]
    return {
        "macro_auroc": float(roc_auc_score(one_hot, probabilities, average="macro", multi_class="ovr")),
        "macro_auprc": float(average_precision_score(one_hot, probabilities, average="macro")),
        "quadratic_weighted_kappa": float(cohen_kappa_score(targets, prediction, weights="quadratic")),
        "macro_f1": float(f1_score(targets, prediction, average="macro", zero_division=0)),
        "ordinal_mae": float(mean_absolute_error(targets, prediction)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="EyePACS label-efficiency and clean-IDRiD external benchmark")
    parser.add_argument("--train-run", required=True)
    parser.add_argument("--validation-run", required=True)
    parser.add_argument("--internal-test-run", required=True)
    parser.add_argument("--idrid-run", required=True)
    parser.add_argument("--idrid-manifest", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=4106)
    args = parser.parse_args()
    if args.replicates != 5:
        parser.error("the preregistered protocol requires exactly five replicates")
    if args.bootstrap_resamples < 100:
        parser.error("use at least 100 patient bootstrap resamples")
    roots = {name: require_external(Path(value), name) for name, value in {
        "train run": args.train_run,
        "validation run": args.validation_run,
        "internal test run": args.internal_test_run,
        "IDRiD run": args.idrid_run,
        "IDRiD manifest": args.idrid_manifest,
        "EyePACS labels": args.labels,
        "normalization": args.normalization,
        "SAE checkpoint": args.checkpoint,
        "output": args.output,
    }.items()}
    output = roots["output"]
    output.mkdir(parents=True, exist_ok=True)
    eyepacs_labels = load_eyepacs_labels(roots["EyePACS labels"])
    idrid_labels = load_idrid_manifest(roots["IDRiD manifest"])
    train_run = load_embedding_run(roots["train run"], expected_split="train", verify_sha256=True)
    val_run = load_embedding_run(roots["validation run"], expected_split="validation", verify_sha256=True)
    test_run = load_embedding_run(roots["internal test run"], expected_split="internal_test", verify_sha256=True)
    idrid_run = load_embedding_run(roots["IDRiD run"], expected_split="external", verify_sha256=True)
    train_dense, train_grades, train_patients, _ = collect_eyepacs(train_run, eyepacs_labels)
    val_dense, val_grades, val_patients, _ = collect_eyepacs(val_run, eyepacs_labels)
    test_dense, test_grades, test_patients, _ = collect_eyepacs(test_run, eyepacs_labels)
    idrid_dense, idrid_grades, idrid_patients, _ = collect_idrid(idrid_run, idrid_labels)
    train_y, val_y, test_y, idrid_y = [(grades >= 2).astype(np.int64) for grades in (train_grades, val_grades, test_grades, idrid_grades)]
    for left, right in ((train_patients, val_patients), (train_patients, test_patients), (val_patients, test_patients)):
        if set(left) & set(right):
            raise RuntimeError("EyePACS patient overlap detected")

    normalization = torch.load(roots["normalization"], map_location="cpu", weights_only=False)
    mean, std = normalization["mean"].float(), normalization["std"].float()
    model, _ = load_checkpoint(roots["SAE checkpoint"])
    train_recon, train_codes = encode_and_reconstruct(model, torch.from_numpy(train_dense), mean, std, args.batch_size)
    val_recon, val_codes = encode_and_reconstruct(model, torch.from_numpy(val_dense), mean, std, args.batch_size)
    test_recon, test_codes = encode_and_reconstruct(model, torch.from_numpy(test_dense), mean, std, args.batch_size)
    idrid_recon, idrid_codes = encode_and_reconstruct(model, torch.from_numpy(idrid_dense), mean, std, args.batch_size)

    result = {
        "schema_version": "1.0",
        "status": "complete_label_efficiency_external_benchmark",
        "endpoint": "referable_dr_grade_ge_2",
        "encoder_frozen": True,
        "sae_frozen": True,
        "label_fractions": list(FRACTIONS),
        "replicates": args.replicates,
        "bootstrap_resamples": args.bootstrap_resamples,
        "cohorts": {
            "train": {"images": int(len(train_y)), "patients": len(set(train_patients))},
            "validation": {"images": int(len(val_y)), "patients": len(set(val_patients))},
            "locked_internal_test": {"images": int(len(test_y)), "patients": len(set(test_patients))},
            "idrid_clean_testing": {"images": int(len(idrid_y)), "image_exam_proxy_units": len(set(idrid_patients))},
        },
        "budgets": {},
        "provenance": {
            "train_evidence_sha256": sha256_path(roots["train run"] / "extraction_evidence.json"),
            "validation_evidence_sha256": sha256_path(roots["validation run"] / "extraction_evidence.json"),
            "internal_test_evidence_sha256": sha256_path(roots["internal test run"] / "extraction_evidence.json"),
            "idrid_evidence_sha256": sha256_path(roots["IDRiD run"] / "extraction_evidence.json"),
            "idrid_manifest_sha256": sha256_path(roots["IDRiD manifest"]),
            "labels_sha256": sha256_path(roots["EyePACS labels"]),
            "normalization_sha256": sha256_path(roots["normalization"]),
            "checkpoint_sha256": sha256_path(roots["SAE checkpoint"]),
        },
        "model_specification": {"estimator": "class_balanced_logistic_regression", "C": 1.0, "solver": "lbfgs", "max_iter": 2000, "standardization": "fit_on_each_training_subset_only", "compact_sae_features": 64, "feature_minimum_training_activations": 10},
    }
    for budget_index, fraction in enumerate(FRACTIONS):
        budget_key = f"{fraction:.2f}"
        result["budgets"][budget_key] = {"fraction": fraction, "replicates": [], "ensemble": {}}
        pred_store = {arm: {"validation_binary": [], "test_binary": [], "idrid_binary": [], "validation_multi": [], "test_multi": [], "idrid_multi": [], "thresholds": []} for arm in ARMS}
        sampling_records = []
        for replicate in range(args.replicates):
            subset_seed = args.seed + budget_index * 1000 + replicate
            subset_indices, selected_patients = sample_patient_indices(train_patients, train_grades, fraction, subset_seed)
            sub_dense, sub_grades, sub_y = train_dense[subset_indices], train_grades[subset_indices], train_y[subset_indices]
            sub_recon, sub_codes = train_recon[subset_indices], train_codes[subset_indices]
            stats = feature_stats(sub_codes, sub_y)
            eligible = stats["count"] >= 10
            selected = np.argsort(np.where(eligible, stats["mean_difference"], -np.inf))[::-1][:64]
            if not np.isfinite(stats["mean_difference"][selected]).all():
                raise RuntimeError(f"fewer than 64 eligible SAE features at budget {fraction}")
            sub_sparse = sub_codes[:, selected].toarray().astype(np.float32)
            val_sparse = val_codes[:, selected].toarray().astype(np.float32)
            test_sparse = test_codes[:, selected].toarray().astype(np.float32)
            idrid_sparse = idrid_codes[:, selected].toarray().astype(np.float32)
            features = {
                "dense": (sub_dense, val_dense, test_dense, idrid_dense),
                "sparse_64": (sub_sparse, val_sparse, test_sparse, idrid_sparse),
                "dense_plus_sparse64": (np.concatenate([sub_dense, sub_sparse], 1), np.concatenate([val_dense, val_sparse], 1), np.concatenate([test_dense, test_sparse], 1), np.concatenate([idrid_dense, idrid_sparse], 1)),
                "sae_reconstruction": (sub_recon, val_recon, test_recon, idrid_recon),
            }
            replicate_result = {"seed": subset_seed, "training_images": int(len(sub_y)), "training_patients": len(selected_patients), "selected_sae_features": [int(x) for x in selected], "arms": {}}
            for arm in ARMS:
                train_x, val_x, test_x, idrid_x = features[arm]
                binary_model = fit_probe(train_x, sub_y, subset_seed, multiclass=False)
                multi_model = fit_probe(train_x, sub_grades, subset_seed, multiclass=True)
                train_prob = predict_probe(binary_model, train_x)[:, 1]
                threshold = fit_youden_threshold(sub_y, train_prob)
                val_prob = predict_probe(binary_model, val_x)[:, 1]
                test_prob = predict_probe(binary_model, test_x)[:, 1]
                idrid_prob = predict_probe(binary_model, idrid_x)[:, 1]
                pred_store[arm]["validation_binary"].append(val_prob)
                pred_store[arm]["test_binary"].append(test_prob)
                pred_store[arm]["idrid_binary"].append(idrid_prob)
                pred_store[arm]["thresholds"].append(threshold)
                arm_record = {"decision_threshold_fit_on_training": threshold, "validation_binary": {"auroc": float(roc_auc_score(val_y, val_prob)), "auprc": float(average_precision_score(val_y, val_prob))}, "locked_test_binary": {"auroc": float(roc_auc_score(test_y, test_prob)), "auprc": float(average_precision_score(test_y, test_prob))}, "idrid_binary": {"auroc": float(roc_auc_score(idrid_y, idrid_prob)), "auprc": float(average_precision_score(idrid_y, idrid_prob))}}
                arm_record["validation_binary"].update(threshold_metrics(val_y, val_prob, threshold))
                arm_record["locked_test_binary"].update(threshold_metrics(test_y, test_prob, threshold))
                arm_record["idrid_binary"].update(threshold_metrics(idrid_y, idrid_prob, threshold))
                if multi_model is None:
                    arm_record["five_class"] = {"status": "unavailable_missing_training_grade"}
                else:
                    val_multi = predict_probe(multi_model, val_x)
                    test_multi = predict_probe(multi_model, test_x)
                    idrid_multi = predict_probe(multi_model, idrid_x)
                    pred_store[arm]["validation_multi"].append(val_multi)
                    pred_store[arm]["test_multi"].append(test_multi)
                    pred_store[arm]["idrid_multi"].append(idrid_multi)
                    arm_record["five_class"] = {"validation": multiclass_metrics(val_grades, val_multi), "locked_test": multiclass_metrics(test_grades, test_multi), "idrid": multiclass_metrics(idrid_grades, idrid_multi)}
                replicate_result["arms"][arm] = arm_record
            result["budgets"][budget_key]["replicates"].append(replicate_result)
            sampling_records.append({"replicate": replicate, "seed": subset_seed, "patients": selected_patients})
        result["budgets"][budget_key]["sampling"] = sampling_records
        for arm in ARMS:
            val_avg = np.mean(np.stack(pred_store[arm]["validation_binary"]), axis=0)
            test_avg = np.mean(np.stack(pred_store[arm]["test_binary"]), axis=0)
            idrid_avg = np.mean(np.stack(pred_store[arm]["idrid_binary"]), axis=0)
            ensemble_threshold = float(np.median(pred_store[arm]["thresholds"]))
            result["budgets"][budget_key]["ensemble"][arm] = {
                "validation_binary": {**binary_metrics(val_y, val_avg, val_patients, args.bootstrap_resamples, args.seed + 100 + budget_index), **threshold_metrics(val_y, val_avg, ensemble_threshold)},
                "locked_test_binary": {**binary_metrics(test_y, test_avg, test_patients, args.bootstrap_resamples, args.seed + 200 + budget_index), **threshold_metrics(test_y, test_avg, ensemble_threshold)},
                "idrid_binary": {**binary_metrics(idrid_y, idrid_avg, idrid_patients, args.bootstrap_resamples, args.seed + 300 + budget_index), **threshold_metrics(idrid_y, idrid_avg, ensemble_threshold)},
                "locked_test_delta_vs_dense": {"auroc": binary_difference(test_y, test_avg, np.mean(np.stack(pred_store["dense"]["test_binary"]), axis=0), test_patients, roc_auc_score, args.bootstrap_resamples, args.seed + 400 + budget_index), "auprc": binary_difference(test_y, test_avg, np.mean(np.stack(pred_store["dense"]["test_binary"]), axis=0), test_patients, average_precision_score, args.bootstrap_resamples, args.seed + 500 + budget_index)},
                "idrid_delta_vs_dense": {"auroc": binary_difference(idrid_y, idrid_avg, np.mean(np.stack(pred_store["dense"]["idrid_binary"]), axis=0), idrid_patients, roc_auc_score, args.bootstrap_resamples, args.seed + 600 + budget_index), "auprc": binary_difference(idrid_y, idrid_avg, np.mean(np.stack(pred_store["dense"]["idrid_binary"]), axis=0), idrid_patients, average_precision_score, args.bootstrap_resamples, args.seed + 700 + budget_index)},
            }
            if pred_store[arm]["test_multi"]:
                result["budgets"][budget_key]["ensemble"][arm]["five_class"] = {"validation": multiclass_metrics(val_grades, np.mean(np.stack(pred_store[arm]["validation_multi"]), axis=0)), "locked_test": multiclass_metrics(test_grades, np.mean(np.stack(pred_store[arm]["test_multi"]), axis=0)), "idrid": multiclass_metrics(idrid_grades, np.mean(np.stack(pred_store[arm]["idrid_multi"]), axis=0))}
        # Explicit replicate mean/std are retained alongside the ensemble metrics.
        for arm in ARMS:
            replicate_arm = [rep["arms"][arm] for rep in result["budgets"][budget_key]["replicates"]]
            result["budgets"][budget_key].setdefault("replicate_summary", {})[arm] = {}
            for split_key in ("validation_binary", "locked_test_binary", "idrid_binary"):
                for metric_key in ("auroc", "auprc"):
                    values = [float(item[split_key][metric_key]) for item in replicate_arm]
                    result["budgets"][budget_key]["replicate_summary"][arm].setdefault(split_key, {})[metric_key] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1))}

    # Holm correction is applied to the five non-primary internal budget comparisons.
    secondary = []
    for budget_key in result["budgets"]:
        comparison = result["budgets"][budget_key]["ensemble"]["dense_plus_sparse64"]["locked_test_delta_vs_dense"]["auroc"]
        secondary.append({"budget": budget_key, "raw_p_value": float(comparison["p_value"])})
    ordered = sorted(enumerate(secondary), key=lambda pair: pair[1]["raw_p_value"])
    previous = 0.0
    for rank, (index, comparison) in enumerate(ordered, start=1):
        adjusted = min(1.0, max(previous, (len(secondary) - rank + 1) * comparison["raw_p_value"]))
        secondary[index]["holm_adjusted_p_value"] = float(adjusted)
        previous = adjusted
    result["multiple_comparison_adjustment"] = {"method": "Holm", "family": "dense_plus_sparse64_vs_dense_locked_internal_test_AUROC_by_budget", "comparisons": secondary}
    result["primary_comparisons"] = {"internal_label_efficiency_10_percent": result["budgets"]["0.10"]["ensemble"]["dense_plus_sparse64"]["locked_test_delta_vs_dense"], "external_100_percent_idrid": result["budgets"]["1.00"]["ensemble"]["dense_plus_sparse64"]["idrid_delta_vs_dense"], "meaningful_auroc_gain": 0.03}
    result["scope_boundary"] = "No retinal-specialist review or lesion-semantic claim is included; IDRiD is image-examination proxy resampling because verified patient IDs are unavailable."
    path = output / "label_efficiency_external_benchmark.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(path), "budgets": list(result["budgets"]), "primary_comparisons": result["primary_comparisons"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
