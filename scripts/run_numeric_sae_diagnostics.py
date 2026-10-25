from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retina_lanse.data.embedding_shards import load_embedding_run
from retina_lanse.data.manifest import sha256_path
from retina_lanse.sae.checkpoints import load_checkpoint
from scripts.run_global_sae_search import require_external
from scripts.run_activation_interaction_autoresearch import feature_stats, normalize_weights
from scripts.run_single_sae_feasibility import encode_csr, patient_bootstrap_interval


def labels(path: Path) -> dict[str, int]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {row["image"].strip(): int(row["level"]) for row in csv.DictReader(handle)}


def labeled(run, lookup: dict[str, int]):
    indices = [i for i, path in enumerate(run.relative_paths) if Path(path).stem in lookup]
    return run.embeddings[indices], np.asarray([lookup[Path(run.relative_paths[i]).stem] for i in indices], dtype=np.int64), [run.patient_ids[i] for i in indices]


def score_additive(codes: sparse.csr_matrix, stats: dict[str, np.ndarray], features: int = 64) -> tuple[np.ndarray, np.ndarray]:
    eligible = stats["count"] >= 50
    ranking = np.where(eligible, stats["mean_difference"], -np.inf)
    selected = np.argsort(ranking)[::-1][:features]
    weights = normalize_weights(stats["mean_difference"][selected])
    return np.asarray(codes[:, selected].dot(weights)).ravel(), selected


def operating(scores: np.ndarray, targets: np.ndarray, threshold: float) -> dict[str, float]:
    pred = scores >= threshold
    tp = int(np.sum(pred & (targets == 1)))
    tn = int(np.sum(~pred & (targets == 0)))
    fp = int(np.sum(pred & (targets == 0)))
    fn = int(np.sum(~pred & (targets == 1)))
    return {"threshold": float(threshold), "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1), "ppv": tp / max(tp + fp, 1), "npv": tn / max(tn + fn, 1), "n_positive_predictions": int(pred.sum())}


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnostics for the fixed numerical SAE DR score.")
    parser.add_argument("--train-run", required=True)
    parser.add_argument("--validation-run", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=4106)
    args = parser.parse_args()

    train_root = require_external(Path(args.train_run), "train embeddings")
    val_root = require_external(Path(args.validation_run), "validation embeddings")
    labels_path = require_external(Path(args.labels), "labels")
    norm_path = require_external(Path(args.normalization), "normalization")
    checkpoint_path = require_external(Path(args.checkpoint), "SAE checkpoint")
    output = require_external(Path(args.output), "diagnostic output")
    if "internal_test" in str(val_root) or "internal_test" in str(output):
        raise RuntimeError("numeric diagnostics refuses locked internal-test paths")
    output.mkdir(parents=True, exist_ok=True)

    train_run = load_embedding_run(train_root, expected_split="train", verify_sha256=True)
    val_run = load_embedding_run(val_root, expected_split="validation", verify_sha256=True)
    lookup = labels(labels_path)
    train_x, train_grades, train_patients = labeled(train_run, lookup)
    val_x, val_grades, val_patients = labeled(val_run, lookup)
    if set(train_patients) & set(val_patients):
        raise RuntimeError("patient leakage between labeled train and validation")
    norm = torch.load(norm_path, map_location="cpu", weights_only=False)
    train_x = (train_x - norm["mean"]) / norm["std"]
    val_x = (val_x - norm["mean"]) / norm["std"]
    model, payload = load_checkpoint(checkpoint_path)
    train_codes = encode_csr(model, train_x, batch_size=1024)
    val_codes = encode_csr(model, val_x, batch_size=1024)
    train_targets = (train_grades >= 2).astype(np.int64)
    val_targets = (val_grades >= 2).astype(np.int64)
    stats = feature_stats(train_codes, train_targets)
    train_scores, selected = score_additive(train_codes, stats)
    val_scores, _ = score_additive(val_codes, stats)

    # Fit calibration only on training scores; the validation score remains fixed.
    calibrator = LogisticRegression(class_weight="balanced", random_state=args.seed, max_iter=2000)
    calibrator.fit(train_scores.reshape(-1, 1), train_targets)
    val_prob = calibrator.predict_proba(val_scores.reshape(-1, 1))[:, 1]
    # Threshold chosen only from the train cohort by maximising Youden's J.
    thresholds = np.unique(train_scores)
    if len(thresholds) > 2000:
        thresholds = np.quantile(train_scores, np.linspace(0, 1, 2000))
    train_ops = [operating(train_scores, train_targets, float(t)) for t in thresholds]
    best = max(train_ops, key=lambda x: x["sensitivity"] + x["specificity"] - 1)
    val_ops = {
        "train_youden_threshold": operating(val_scores, val_targets, best["threshold"]),
        "train_90th_percentile_threshold": operating(val_scores, val_targets, float(np.quantile(train_scores, 0.9))),
    }
    grade_summary = {}
    for grade in range(5):
        values = val_scores[val_grades == grade]
        grade_summary[str(grade)] = {"images": int(len(values)), "mean_score": float(values.mean()) if len(values) else None, "median_score": float(np.median(values)) if len(values) else None}
    report = {
        "schema_version": "1.0",
        "status": "complete_fixed_sae_numeric_diagnostics",
        "policy": "weighted_mean_difference_top64_fit_on_labeled_train_only",
        "encoder_frozen": True,
        "locked_internal_test_accessed": False,
        "cohort": {"train_images": int(len(train_targets)), "train_patients": len(set(train_patients)), "train_positive_images": int(train_targets.sum()), "validation_images": int(len(val_targets)), "validation_patients": len(set(val_patients)), "validation_positive_images": int(val_targets.sum())},
        "selected_features": [int(x) for x in selected],
        "validation_metrics": {"auroc": patient_bootstrap_interval(val_targets, val_scores, val_patients, roc_auc_score, resamples=args.bootstrap_resamples, seed=args.seed), "auprc": patient_bootstrap_interval(val_targets, val_scores, val_patients, average_precision_score, resamples=args.bootstrap_resamples, seed=args.seed + 1), "calibrated_brier": float(brier_score_loss(val_targets, val_prob))},
        "train_selected_threshold": best,
        "validation_operating_points": val_ops,
        "validation_score_by_grade": grade_summary,
        "calibration": {"method": "balanced_logistic_regression_on_train_scores", "validation_probability_min": float(val_prob.min()), "validation_probability_max": float(val_prob.max()), "validation_probability_mean": float(val_prob.mean())},
        "inputs": {"train_evidence_sha256": sha256_path(train_root / "extraction_evidence.json"), "validation_evidence_sha256": sha256_path(val_root / "extraction_evidence.json"), "labels_sha256": sha256_path(labels_path), "normalization_sha256": sha256_path(norm_path), "checkpoint_sha256": sha256_path(checkpoint_path)},
        "caveat": "Validation was previously inspected in the pilot workflow; these are confirmatory diagnostics, not a pristine final estimate. Thresholds and calibration were not used to modify the locked test.",
    }
    path = output / "numeric_sae_diagnostics.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
