from __future__ import annotations

"""Experiment 5 stage 1: active SAE values versus frozen dense RETFound."""

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from retina_lanse.data.embedding_shards import load_embedding_run
from retina_lanse.data.manifest import sha256_path
from retina_lanse.sae.checkpoints import load_checkpoint
from scripts.run_global_sae_search import require_external
from scripts.run_label_efficiency_external_benchmark import (
    binary_difference,
    binary_metrics,
    collect_eyepacs,
    collect_idrid,
    fit_youden_threshold,
    load_idrid_manifest,
    load_eyepacs_labels,
    multiclass_metrics,
    predict_probe,
    sample_patient_indices,
    threshold_metrics,
)


SEEDS = (4106, 4107, 4108, 4109, 4110)
FRACTIONS = ("1.00", "0.10")
ARMS = ("dense", "active_values", "dense_plus_active_values")
LBFGS_TOLERANCE = 1e-3


def fit_probe(features: np.ndarray, targets: np.ndarray, seed: int, *, multiclass: bool = False):
    if multiclass and len(np.unique(targets)) < 5:
        return None
    model = LogisticRegression(
        class_weight="balanced", C=1.0, solver="lbfgs", max_iter=2000,
        tol=LBFGS_TOLERANCE, multi_class="multinomial" if multiclass else "auto",
        random_state=seed,
    )
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    model.fit((features - mean) / std, targets)
    model._retina_train_mean = mean
    model._retina_train_std = std
    model._retina_n_iter = np.asarray(model.n_iter_).tolist()
    return model


@torch.inference_mode()
def active_values_and_ids(model, embeddings: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, *, batch_size: int):
    """Return rank-ordered active magnitudes and their SAE feature IDs."""
    values = np.zeros((len(embeddings), 1024), dtype=np.float32)
    feature_ids = np.full((len(embeddings), 1024), -1, dtype=np.int32)
    model.eval()
    for start in range(0, len(embeddings), batch_size):
        batch = ((embeddings[start : start + batch_size] - mean) / std)
        pre = torch.relu(model.encoder(batch))
        top_values, top_ids = torch.topk(pre, k=model.k, dim=1, sorted=True)
        for row in range(len(batch)):
            row_values = top_values[row].cpu().numpy().astype(np.float32, copy=False)
            row_ids = top_ids[row].cpu().numpy().astype(np.int32, copy=False)
            keep = row_values > 0
            row_values, row_ids = row_values[keep], row_ids[keep]
            # Deterministic order: strongest activation first, feature ID breaks ties.
            order = np.lexsort((row_ids, -row_values))
            count = min(len(order), 1024)
            values[start + row, :count] = row_values[order[:count]]
            feature_ids[start + row, :count] = row_ids[order[:count]]
    return values, feature_ids


def metric_delta(targets, left, right, patients, metric, bootstrap, seed):
    return binary_difference(targets, left, right, patients, metric, bootstrap, seed)


def run_smoke() -> dict:
    class Dummy:
        k = 4
        dictionary_size = 8

    # The sorting/padding contract is tested independently of a model.
    vals = np.asarray([0.2, 0.9, 0.9, 0.1], dtype=np.float32)
    ids = np.asarray([5, 2, 1, 7], dtype=np.int32)
    order = np.lexsort((ids, -vals))
    output = np.zeros(8, dtype=np.float32)
    output[: len(order)] = vals[order]
    ids_out = np.full(8, -1, dtype=np.int32)
    ids_out[: len(order)] = ids[order]
    assert output.shape == (8,) and ids_out.shape == (8,)
    assert ids_out[:4].tolist() == [1, 2, 5, 7]
    assert np.all(output[4:] == 0)
    return {"status": "smoke_passed", "vector_length": 1024, "tie_break": "feature_id_ascending", "classifier_receives_feature_ids": False}


def main() -> int:
    parser = argparse.ArgumentParser(description="Active-value-only SAE fusion benchmark at K=1024")
    parser.add_argument("--train-run", required=True)
    parser.add_argument("--validation-run", required=True)
    parser.add_argument("--internal-test-run", required=True)
    parser.add_argument("--idrid-run", required=True)
    parser.add_argument("--idrid-manifest", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=4106)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.replicates != 5:
        parser.error("Experiment 5 requires exactly five SAE seeds")
    roots = {name: require_external(Path(value), name) for name, value in {
        "train run": args.train_run, "validation run": args.validation_run,
        "internal test run": args.internal_test_run, "IDRiD run": args.idrid_run,
        "IDRiD manifest": args.idrid_manifest, "labels": args.labels,
        "normalization": args.normalization, "checkpoint directory": args.checkpoint_dir,
        "output": args.output,
    }.items()}
    output = roots["output"]
    output.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        smoke = run_smoke()
        (output / "stage1_smoke.json").write_text(json.dumps(smoke, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(smoke, indent=2))
        return 0

    labels = load_eyepacs_labels(roots["labels"])
    idrid_labels = load_idrid_manifest(roots["IDRiD manifest"])
    train_run = load_embedding_run(roots["train run"], expected_split="train", verify_sha256=True)
    val_run = load_embedding_run(roots["validation run"], expected_split="validation", verify_sha256=True)
    test_run = load_embedding_run(roots["internal test run"], expected_split="internal_test", verify_sha256=True)
    idrid_run = load_embedding_run(roots["IDRiD run"], expected_split="external", verify_sha256=True)
    train_dense, train_grades, train_patients, train_paths = collect_eyepacs(train_run, labels)
    val_dense, val_grades, val_patients, val_paths = collect_eyepacs(val_run, labels)
    test_dense, test_grades, test_patients, test_paths = collect_eyepacs(test_run, labels)
    idrid_dense, idrid_grades, idrid_patients, idrid_paths = collect_idrid(idrid_run, idrid_labels)
    for left, right in ((train_patients, val_patients), (train_patients, test_patients), (val_patients, test_patients)):
        if set(left) & set(right):
            raise RuntimeError("EyePACS patient overlap detected")
    normalization = torch.load(roots["normalization"], map_location="cpu", weights_only=False)
    mean, std = normalization["mean"].float(), normalization["std"].float()
    checkpoints = {}
    for seed in SEEDS:
        path = roots["checkpoint directory"] / f"seed_{seed}" / "checkpoint.pt"
        if not path.is_file():
            raise RuntimeError(f"missing K=1024 checkpoint: {path}")
        model, payload = load_checkpoint(path)
        if payload["model_config"] != {"input_dim": 1024, "dictionary_size": 16384, "k": 1024}:
            raise RuntimeError(f"unexpected checkpoint config: {payload['model_config']}")
        checkpoints[seed] = (path, model)

    y = {"1.00": (train_grades >= 2).astype(np.int64), "0.10": None}
    low_indices, low_patients = sample_patient_indices(train_patients, train_grades, 0.10, args.seed)
    y["0.10"] = (train_grades[low_indices] >= 2).astype(np.int64)
    index_by_fraction = {"1.00": np.arange(len(train_grades), dtype=np.int64), "0.10": low_indices}
    targets = {"validation": (val_grades >= 2).astype(np.int64), "test": (test_grades >= 2).astype(np.int64), "idrid": (idrid_grades >= 2).astype(np.int64)}
    patients = {"validation": np.asarray(val_patients), "test": np.asarray(test_patients), "idrid": np.asarray(idrid_patients)}
    prediction_store = {fraction: {arm: {split: [] for split in ("validation", "test", "idrid")} for arm in ARMS} for fraction in FRACTIONS}
    replicate_store = {fraction: [] for fraction in FRACTIONS}
    trace_dir = output / "active_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    for seed in SEEDS:
        checkpoint_path, model = checkpoints[seed]
        train_values, train_ids = active_values_and_ids(model, torch.from_numpy(train_dense), mean, std, batch_size=args.batch_size)
        val_values, val_ids = active_values_and_ids(model, torch.from_numpy(val_dense), mean, std, batch_size=args.batch_size)
        test_values, test_ids = active_values_and_ids(model, torch.from_numpy(test_dense), mean, std, batch_size=args.batch_size)
        idrid_values, idrid_ids = active_values_and_ids(model, torch.from_numpy(idrid_dense), mean, std, batch_size=args.batch_size)
        if int(np.max(np.sum(train_ids >= 0, axis=1))) > 1024:
            raise RuntimeError("active-value vector exceeded 1024 entries")
        np.savez_compressed(trace_dir / f"seed_{seed}.npz", train_values=train_values, train_feature_ids=train_ids, validation_values=val_values, validation_feature_ids=val_ids, test_values=test_values, test_feature_ids=test_ids, idrid_values=idrid_values, idrid_feature_ids=idrid_ids)
        traces = {"dense": train_dense, "active_values": train_values, "dense_plus_active_values": np.concatenate([train_dense, train_values], axis=1)}
        eval_traces = {
            "dense": (val_dense, test_dense, idrid_dense),
            "active_values": (val_values, test_values, idrid_values),
            "dense_plus_active_values": (np.concatenate([val_dense, val_values], axis=1), np.concatenate([test_dense, test_values], axis=1), np.concatenate([idrid_dense, idrid_values], axis=1)),
        }
        seed_record = {"seed": seed, "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256_path(checkpoint_path), "active_vector_length": 1024, "classifier_feature_dimensions": {"dense": 1024, "active_values": 1024, "dense_plus_active_values": 2048}, "fractions": {}}
        for fraction in FRACTIONS:
            indices = index_by_fraction[fraction]
            fraction_record = {"training_images": int(len(indices)), "training_patients": int(len(set(train_patients[indices]))), "arms": {}}
            for arm in ARMS:
                train_features = traces[arm][indices]
                fitted = fit_probe(train_features, y[fraction], seed, multiclass=False)
                train_prob = predict_probe(fitted, train_features)[:, 1]
                threshold = fit_youden_threshold(y[fraction], train_prob)
                eval_features = eval_traces[arm]
                probs = [predict_probe(fitted, feature)[:, 1] for feature in eval_features]
                for split, values in zip(("validation", "test", "idrid"), probs):
                    prediction_store[fraction][arm][split].append(values)
                arm_record = {"threshold": threshold}
                for split, values in zip(("validation", "test", "idrid"), probs):
                    arm_record[split] = {**binary_metrics(targets[split], values, patients[split], args.bootstrap_resamples, args.seed + seed), **threshold_metrics(targets[split], values, threshold)}
                multi = fit_probe(train_features, train_grades[indices], seed, multiclass=True)
                if multi is None:
                    arm_record["five_class"] = {"status": "unavailable_missing_training_grade"}
                else:
                    arm_record["five_class"] = {split: multiclass_metrics(grades, predict_probe(multi, feature)) for split, grades, feature in zip(("validation", "locked_test", "idrid"), (val_grades, test_grades, idrid_grades), eval_features)}
                fraction_record["arms"][arm] = arm_record
            seed_record["fractions"][fraction] = fraction_record
        replicate_store["1.00"].append(seed_record)
        replicate_store["0.10"].append(seed_record)

    ensembles = {fraction: {} for fraction in FRACTIONS}
    ensemble_predictions = {}
    for fraction in FRACTIONS:
        for arm in ARMS:
            validation_avg = np.mean(np.stack(prediction_store[fraction][arm]["validation"]), axis=0)
            test_avg = np.mean(np.stack(prediction_store[fraction][arm]["test"]), axis=0)
            idrid_avg = np.mean(np.stack(prediction_store[fraction][arm]["idrid"]), axis=0)
            thresholds = [rep["fractions"][fraction]["arms"][arm]["threshold"] for rep in replicate_store[fraction]]
            threshold = float(np.median(thresholds))
            record = {
                "validation": {**binary_metrics(targets["validation"], validation_avg, patients["validation"], args.bootstrap_resamples, args.seed + 100), **threshold_metrics(targets["validation"], validation_avg, threshold)},
                "locked_test": {**binary_metrics(targets["test"], test_avg, patients["test"], args.bootstrap_resamples, args.seed + 200), **threshold_metrics(targets["test"], test_avg, threshold)},
                "idrid": {**binary_metrics(targets["idrid"], idrid_avg, patients["idrid"], args.bootstrap_resamples, args.seed + 300), **threshold_metrics(targets["idrid"], idrid_avg, threshold)},
            }
            ensemble_predictions[f"{fraction}_{arm}_validation"] = validation_avg.astype(np.float32)
            ensemble_predictions[f"{fraction}_{arm}_test"] = test_avg.astype(np.float32)
            ensemble_predictions[f"{fraction}_{arm}_idrid"] = idrid_avg.astype(np.float32)
            dense_test = np.mean(np.stack(prediction_store[fraction]["dense"]["test"]), axis=0)
            dense_idrid = np.mean(np.stack(prediction_store[fraction]["dense"]["idrid"]), axis=0)
            record["locked_test_delta_vs_dense"] = {"auroc": metric_delta(targets["test"], test_avg, dense_test, patients["test"], roc_auc_score, args.bootstrap_resamples, args.seed + 400), "auprc": metric_delta(targets["test"], test_avg, dense_test, patients["test"], average_precision_score, args.bootstrap_resamples, args.seed + 401)}
            record["idrid_delta_vs_dense"] = {"auroc": metric_delta(targets["idrid"], idrid_avg, dense_idrid, patients["idrid"], roc_auc_score, args.bootstrap_resamples, args.seed + 500), "auprc": metric_delta(targets["idrid"], idrid_avg, dense_idrid, patients["idrid"], average_precision_score, args.bootstrap_resamples, args.seed + 501)}
            ensembles[fraction][arm] = record
    prediction_path = output / "stage1_predictions.npz"
    np.savez_compressed(prediction_path, **ensemble_predictions)
    result = {
        "schema_version": "1.0", "status": "complete_active_value_fusion_1024", "endpoint": "referable_dr_grade_ge_2", "sae_frozen": True, "encoder_frozen": True, "seeds": list(SEEDS), "label_fractions": list(FRACTIONS),
        "model_specification": {"dense_dim": 1024, "active_value_dim": 1024, "fusion_dim": 2048, "classifier": "class_balanced_logistic_C1_lbfgs_max_iter_2000_tol_1e-3", "lbfgs_tolerance": LBFGS_TOLERANCE, "standardization": "train_only", "feature_ids_in_classifier": False, "active_order": "descending_activation_then_feature_id", "bootstrap_resamples": args.bootstrap_resamples},
        "cohorts": {"sae_train": {"images": len(train_run.embeddings), "patients": len(set(train_run.patient_ids))}, "downstream_train": {"images": len(train_grades), "patients": len(set(train_patients))}, "validation": {"images": len(val_grades), "patients": len(set(val_patients))}, "locked_internal_test": {"images": len(test_grades), "patients": len(set(test_patients))}, "idrid_clean_testing": {"images": len(idrid_grades), "image_exam_proxy_units": len(set(idrid_patients))}},
        "label_sampling": {"fraction_0.10_seed": args.seed, "fraction_0.10_patients": int(len(low_patients)), "fraction_0.10_images": int(len(low_indices))},
        "provenance": {"train_evidence_sha256": sha256_path(roots["train run"] / "extraction_evidence.json"), "validation_evidence_sha256": sha256_path(roots["validation run"] / "extraction_evidence.json"), "internal_test_evidence_sha256": sha256_path(roots["internal test run"] / "extraction_evidence.json"), "idrid_evidence_sha256": sha256_path(roots["IDRiD run"] / "extraction_evidence.json"), "idrid_manifest_sha256": sha256_path(roots["IDRiD manifest"]), "labels_sha256": sha256_path(roots["labels"]), "normalization_sha256": sha256_path(roots["normalization"])},
        "ensembles": ensembles, "replicates": replicate_store, "active_trace_directory": str(trace_dir), "prediction_artifact": str(prediction_path), "locked_test_used_for_model_selection": False, "scope_boundary": "This stage tests active magnitudes only; AutoResearch feature meaning is a subsequent frozen stage.",
    }
    path = output / "stage1_active_value_fusion.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(path), "primary": {"dense": ensembles["1.00"]["dense"]["locked_test"], "fusion": ensembles["1.00"]["dense_plus_active_values"]["locked_test"], "delta": ensembles["1.00"]["dense_plus_active_values"]["locked_test_delta_vs_dense"]["auroc"]}}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
