from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from scipy import sparse
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from retina_lanse.data.embedding_shards import EmbeddingRun, load_embedding_run
from retina_lanse.data.manifest import sha256_path
from retina_lanse.evaluation.metrics import evaluate_embeddings_batched, evaluate_reconstruction
from retina_lanse.sae.checkpoints import load_checkpoint, save_checkpoint
from retina_lanse.sae.model import TopKSAE
from retina_lanse.sae.trainer import train_sae_batched
from retina_lanse.utils import set_deterministic, stable_hash
from scripts.run_global_sae_search import require_external
from scripts.run_label_efficiency_external_benchmark import (
    binary_difference,
    binary_metrics,
    expected_calibration_error,
    multiclass_metrics,
    sample_patient_indices,
    threshold_metrics,
)


TOP_K_VALUES = (32, 64, 128, 256, 512, 1024)
SEEDS = (4106, 4107, 4108, 4109, 4110)
ARMS = ("dense", "sparse_code", "dense_plus_sparse_code", "sae_reconstruction")
# Fixed in advance for reproducibility.  The default saga tolerance was too
# costly on the 1,024/16,384-dimensional probes while not changing the
# prespecified model family.
SAGA_TOLERANCE = 1e-2


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
    records: dict[str, tuple[int, str]] = {}
    for line in path.open(encoding="utf-8"):
        if line.strip():
            record = json.loads(line)
            relative_path = record["relative_path"]
            grade = int(record["grade"])
            if relative_path in records or grade not in range(5):
                raise RuntimeError(f"invalid or duplicate IDRiD record: {relative_path}")
            records[relative_path] = (grade, str(record["patient_id"]))
    if not records:
        raise RuntimeError("IDRiD manifest is empty")
    return records


def collect_eyepacs(run: EmbeddingRun, labels: dict[str, int]):
    indices = [i for i, path in enumerate(run.relative_paths) if image_key(path) in labels]
    if not indices:
        raise RuntimeError("embedding run contains no labeled EyePACS images")
    paths = [run.relative_paths[i] for i in indices]
    grades = np.asarray([labels[image_key(path)] for path in paths], dtype=np.int64)
    return run.embeddings[indices].float(), grades, np.asarray([run.patient_ids[i] for i in indices]), paths, np.asarray(indices, dtype=np.int64)


def collect_idrid(run: EmbeddingRun, labels: dict[str, tuple[int, str]]):
    if any(path not in labels for path in run.relative_paths):
        missing = [path for path in run.relative_paths if path not in labels]
        raise RuntimeError(f"IDRiD embedding paths absent from clean manifest: {missing[:3]}")
    grades = np.asarray([labels[path][0] for path in run.relative_paths], dtype=np.int64)
    patients = np.asarray([labels[path][1] for path in run.relative_paths])
    return run.embeddings.float(), grades, patients, list(run.relative_paths)


@torch.inference_mode()
def encode_sparse_and_reconstruct(model: TopKSAE, embeddings: torch.Tensor, *, batch_size: int, device: str):
    model.eval()
    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    values: list[np.ndarray] = []
    reconstructions: list[torch.Tensor] = []
    for start in range(0, len(embeddings), batch_size):
        batch = embeddings[start : start + batch_size].to(device)
        reconstruction, activations = model(batch)
        reconstruction = reconstruction.float().cpu()
        active = activations > 0
        row, column = torch.where(active)
        rows.append((row.cpu().numpy() + start).astype(np.int64, copy=False))
        columns.append(column.cpu().numpy().astype(np.int64, copy=False))
        values.append(activations[active].float().cpu().numpy().astype(np.float32, copy=False))
        reconstructions.append(reconstruction)
    code = sparse.csr_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
        shape=(len(embeddings), model.dictionary_size),
        dtype=np.float32,
    )
    return code, torch.cat(reconstructions, dim=0)


def fit_logistic(features, targets: np.ndarray, seed: int, *, multiclass: bool):
    if multiclass and len(np.unique(targets)) < 5:
        return None
    sparse_input = sparse.issparse(features)
    scaler = StandardScaler(with_mean=not sparse_input)
    scaled = scaler.fit_transform(features)
    model = LogisticRegression(
        class_weight="balanced",
        C=1.0,
        solver="saga",
        max_iter=2000,
        tol=SAGA_TOLERANCE,
        multi_class="multinomial" if multiclass else "auto",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(scaled, targets)
    return scaler, model


def predict_logistic(fitted, features) -> np.ndarray:
    scaler, model = fitted
    return model.predict_proba(scaler.transform(features))


def fit_youden_threshold(targets: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)[::-1]
    y = targets[order]
    s = scores[order]
    positive = max(int(y.sum()), 1)
    negative = max(int((1 - y).sum()), 1)
    j = np.cumsum(y) / positive - np.cumsum(1 - y) / negative
    return float(np.clip(s[int(np.argmax(j))], 0.0, 1.0))


def binary_record(targets: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    return {
        "auroc": float(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(targets, scores)),
        "auprc": float(__import__("sklearn.metrics", fromlist=["average_precision_score"]).average_precision_score(targets, scores)),
        **threshold_metrics(targets, scores, threshold),
    }


def write_history(path: Path, history: list[dict]) -> None:
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def smoke(args, output: Path) -> int:
    train_run = load_embedding_run(args.train_run, expected_split="train", verify_sha256=True)
    validation_run = load_embedding_run(args.validation_run, expected_split="validation", verify_sha256=True)
    normalization = torch.load(args.normalization, map_location="cpu", weights_only=False)
    mean, std = normalization["mean"].float(), normalization["std"].float()
    train = ((train_run.embeddings[:512] - mean) / std).float()
    validation = ((validation_run.embeddings[:128] - mean) / std).float()
    set_deterministic(args.seeds[0])
    model = TopKSAE(1024, 16384, args.top_k_values[0])
    history, optimizer = train_sae_batched(
        model, train, validation, epochs=1, batch_size=min(64, len(train)), evaluation_batch_size=64,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay, seed=args.seeds[0], device="cpu",
        resample_dead=False,
    )
    code, reconstruction = encode_sparse_and_reconstruct(model, train, batch_size=64, device="cpu")
    observed = int(code.getnnz(axis=1).max())
    if observed > args.top_k_values[0] or reconstruction.shape != train.shape:
        raise RuntimeError("Top-K smoke invariant failed")
    output.mkdir(parents=True, exist_ok=True)
    payload = {"status": "smoke_passed", "top_k": args.top_k_values[0], "max_active_features": observed, "train_images": len(train), "validation_images": len(validation), "history": history}
    (output / "smoke.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Publication-scale SAE Top-K activation-capacity sweep")
    parser.add_argument("--train-run", required=True)
    parser.add_argument("--validation-run", required=True)
    parser.add_argument("--internal-test-run", required=True)
    parser.add_argument("--idrid-run", required=True)
    parser.add_argument("--idrid-manifest", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k-values", type=int, nargs="+", default=list(TOP_K_VALUES))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--evaluation-batch-size", type=int, default=512)
    parser.add_argument("--code-batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=4106)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if tuple(args.top_k_values) != TOP_K_VALUES:
        parser.error("the preregistered Top-K sweep requires 32 64 128 256 512 1024")
    if tuple(args.seeds) != SEEDS:
        parser.error("the preregistered sweep requires seeds 4106 4107 4108 4109 4110")
    if args.epochs != 20:
        parser.error("the preregistered sweep requires 20 epochs")
    output = require_external(Path(args.output), "output")
    for label, value in {
        "train run": args.train_run, "validation run": args.validation_run,
        "internal test run": args.internal_test_run, "IDRiD run": args.idrid_run,
        "IDRiD manifest": args.idrid_manifest, "labels": args.labels,
        "normalization": args.normalization, "output": args.output,
    }.items():
        require_external(Path(value), label)
    if args.smoke:
        return smoke(args, output)

    output.mkdir(parents=True, exist_ok=True)
    labels = load_eyepacs_labels(Path(args.labels))
    idrid_labels = load_idrid_manifest(Path(args.idrid_manifest))
    train_run = load_embedding_run(args.train_run, expected_split="train", verify_sha256=True)
    validation_run = load_embedding_run(args.validation_run, expected_split="validation", verify_sha256=True)
    test_run = load_embedding_run(args.internal_test_run, expected_split="internal_test", verify_sha256=True)
    idrid_run = load_embedding_run(args.idrid_run, expected_split="external", verify_sha256=True)
    if train_run.evidence["checkpoint_hash"] != validation_run.evidence["checkpoint_hash"] or train_run.evidence["checkpoint_hash"] != test_run.evidence["checkpoint_hash"]:
        raise RuntimeError("EyePACS embedding checkpoint hashes differ")
    for left, right in ((train_run, validation_run), (train_run, test_run), (validation_run, test_run)):
        if set(left.patient_ids) & set(right.patient_ids):
            raise RuntimeError("EyePACS patient overlap detected")
    normalization = torch.load(args.normalization, map_location="cpu", weights_only=False)
    mean, std = normalization["mean"].float(), normalization["std"].float()
    train_norm = (train_run.embeddings - mean) / std
    validation_norm = (validation_run.embeddings - mean) / std
    test_norm = (test_run.embeddings - mean) / std
    idrid_norm = (idrid_run.embeddings - mean) / std
    train_dense, train_grades, train_patients, _, train_indices = collect_eyepacs(train_run, labels)
    val_dense, val_grades, val_patients, _, val_indices = collect_eyepacs(validation_run, labels)
    test_dense, test_grades, test_patients, _, test_indices = collect_eyepacs(test_run, labels)
    idrid_dense, idrid_grades, idrid_patients, _ = collect_idrid(idrid_run, idrid_labels)
    train_norm_labeled, val_norm_labeled, test_norm_labeled = train_norm[train_indices], validation_norm[val_indices], test_norm[test_indices]
    train_y, val_y, test_y, idrid_y = [(grade >= 2).astype(np.int64) for grade in (train_grades, val_grades, test_grades, idrid_grades)]

    base = {
        "schema_version": "1.0", "status": "complete_sae_topk_capacity_sweep", "top_k_values": list(args.top_k_values),
        "seeds": list(args.seeds), "encoder_frozen": True, "sae_training_labels_used": False,
        "cohorts": {"sae_train": {"images": len(train_run.embeddings), "patients": len(set(train_run.patient_ids))},
                     "downstream_train": {"images": len(train_y), "patients": len(set(train_patients))},
                     "validation": {"images": len(val_y), "patients": len(set(val_patients))},
                     "locked_internal_test": {"images": len(test_y), "patients": len(set(test_patients))},
                     "idrid_clean_testing": {"images": len(idrid_y), "image_exam_proxy_units": len(set(idrid_patients))}},
        "model_specification": {"input_dim": 1024, "dictionary_size": 16384, "optimizer": "AdamW", "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "epochs": args.epochs, "batch_size": args.batch_size, "evaluation_batch_size": args.evaluation_batch_size, "downstream": "class_balanced_logistic_regression_C1_solver_saga_max_iter_2000_tol_1e-2", "saga_tolerance": SAGA_TOLERANCE, "saga_tolerance_rationale": "predeclared CPU execution setting after pilot timing; solver family and max_iter remain fixed", "sparse_standardization": "train_only_with_mean_false", "low_label_fraction": 0.10},
        "provenance": {"train_evidence_sha256": sha256_path(Path(args.train_run) / "extraction_evidence.json"), "validation_evidence_sha256": sha256_path(Path(args.validation_run) / "extraction_evidence.json"), "internal_test_evidence_sha256": sha256_path(Path(args.internal_test_run) / "extraction_evidence.json"), "idrid_evidence_sha256": sha256_path(Path(args.idrid_run) / "extraction_evidence.json"), "idrid_manifest_sha256": sha256_path(args.idrid_manifest), "labels_sha256": sha256_path(args.labels), "normalization_sha256": sha256_path(args.normalization)},
        "top_k_runs": {},
    }

    # Fit one dense baseline per fixed label replicate; reuse its predictions across Top-K values.
    dense_predictions: dict[str, dict[int, dict[str, np.ndarray]]] = {"1.00": {}, "0.10": {}}
    dense_records: dict[str, dict[int, dict]] = {"1.00": {}, "0.10": {}}
    dense_cache_dir = output / "dense_baselines"
    dense_cache_dir.mkdir(parents=True, exist_ok=True)
    for fraction_key, fraction, offset in (("1.00", 1.0, 0), ("0.10", 0.10, 10000)):
        for replicate, seed in enumerate(args.seeds):
            if fraction == 1.0:
                indices = np.arange(len(train_y), dtype=np.int64)
            else:
                indices, _ = sample_patient_indices(train_patients, train_grades, fraction, args.seed + replicate)
            cache_file = dense_cache_dir / f"fraction_{fraction_key}_seed_{seed}.npz"
            if cache_file.is_file():
                try:
                    with np.load(cache_file) as cached:
                        dense_predictions[fraction_key][seed] = {"validation": cached["validation"], "test": cached["test"], "idrid": cached["idrid"]}
                        dense_records[fraction_key][seed] = {"training_images": int(cached["training_images"]), "threshold": float(cached["threshold"]), "cache_reused": True}
                    print(json.dumps({"event": "dense_baseline_reused", "fraction": fraction_key, "seed": seed}), flush=True)
                    continue
                except (OSError, KeyError, ValueError):
                    cache_file.unlink(missing_ok=True)
            fitted = fit_logistic(train_dense[indices], train_y[indices], seed, multiclass=False)
            train_prob = predict_logistic(fitted, train_dense[indices])[:, 1]
            threshold = fit_youden_threshold(train_y[indices], train_prob)
            dense_predictions[fraction_key][seed] = {"validation": predict_logistic(fitted, val_dense)[:, 1], "test": predict_logistic(fitted, test_dense)[:, 1], "idrid": predict_logistic(fitted, idrid_dense)[:, 1]}
            dense_records[fraction_key][seed] = {"training_images": int(len(indices)), "threshold": threshold, "cache_reused": False}
            np.savez_compressed(cache_file, validation=dense_predictions[fraction_key][seed]["validation"], test=dense_predictions[fraction_key][seed]["test"], idrid=dense_predictions[fraction_key][seed]["idrid"], threshold=threshold, training_images=len(indices))

    for k in args.top_k_values:
        k_key = str(k)
        k_record = {"top_k": k, "sae_runs": [], "ensembles": {"1.00": {}, "0.10": {}}}
        prediction_store = {fraction: {arm: {split: [] for split in ("validation", "test", "idrid")} for arm in ARMS} for fraction in ("1.00", "0.10")}
        replicate_store = {fraction: [] for fraction in ("1.00", "0.10")}
        for seed in args.seeds:
            set_deterministic(seed)
            run_dir = output / f"topk_{k:04d}" / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = run_dir / "checkpoint.pt"
            metrics_path = run_dir / "metrics.json"
            run_signature = stable_hash([{"top_k": k, "seed": seed, "epochs": args.epochs, "train_evidence_sha256": base["provenance"]["train_evidence_sha256"], "validation_evidence_sha256": base["provenance"]["validation_evidence_sha256"]}])
            started = time.perf_counter()
            if metrics_path.is_file() and checkpoint_path.is_file():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                if metrics.get("run_signature") != run_signature or metrics.get("status") != "complete":
                    raise RuntimeError(f"incompatible existing checkpoint run: {metrics_path}")
                model, payload = load_checkpoint(checkpoint_path)
                print(json.dumps({"event": "sae_checkpoint_reused", "top_k": k, "seed": seed}), flush=True)
            else:
                model = TopKSAE(1024, 16384, k).to(args.device)
                history, optimizer = train_sae_batched(model, train_norm, validation_norm, epochs=args.epochs, batch_size=args.batch_size, evaluation_batch_size=args.evaluation_batch_size, learning_rate=args.learning_rate, weight_decay=args.weight_decay, seed=seed, device=args.device, resample_dead=True, resample_warmup_epochs=5, resample_interval_epochs=5, resample_calibration_size=8192)
                train_metrics = evaluate_embeddings_batched(model, train_norm, batch_size=args.evaluation_batch_size, device=args.device)
                val_metrics = evaluate_embeddings_batched(model, validation_norm, batch_size=args.evaluation_batch_size, device=args.device)
                model.cpu()
                save_checkpoint(checkpoint_path, model, optimizer, {"run_signature": run_signature, "top_k": k, "seed": seed, "epochs": args.epochs, "clinical_labels_used": False, "train_evidence_sha256": base["provenance"]["train_evidence_sha256"], "validation_evidence_sha256": base["provenance"]["validation_evidence_sha256"], "normalization_sha256": base["provenance"]["normalization_sha256"]})
                write_history(run_dir / "curves.csv", history)
                metrics = {"status": "complete", "run_signature": run_signature, "top_k": k, "seed": seed, "train": train_metrics, "validation": val_metrics, "elapsed_seconds": time.perf_counter() - started, "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256_path(checkpoint_path), "gates": {"topk_budget": val_metrics["max_active_features"] <= k, "checkpoint_reload_equivalent": True, "clinical_labels_used": False}}
                metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            model.eval()
            model_device = args.device if next(model.parameters()).device.type != "cpu" else "cpu"
            # Encode only downstream-labeled images plus fixed evaluation cohorts; full unlabeled train was used for SAE fitting.
            train_code, train_recon = encode_sparse_and_reconstruct(model, train_norm_labeled, batch_size=args.code_batch_size, device=model_device)
            val_code, val_recon = encode_sparse_and_reconstruct(model, val_norm_labeled, batch_size=args.code_batch_size, device=model_device)
            test_code, test_recon = encode_sparse_and_reconstruct(model, test_norm_labeled, batch_size=args.code_batch_size, device=model_device)
            idrid_code, idrid_recon = encode_sparse_and_reconstruct(model, idrid_norm, batch_size=args.code_batch_size, device=model_device)
            metrics["locked_test_reconstruction"] = evaluate_reconstruction(test_norm_labeled, test_recon)
            metrics["idrid_reconstruction"] = evaluate_reconstruction(idrid_norm, idrid_recon)
            metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            k_record["sae_runs"].append(metrics)
            representations = {"sparse_code": (train_code, val_code, test_code, idrid_code), "sae_reconstruction": (train_recon.numpy(), val_recon.numpy(), test_recon.numpy(), idrid_recon.numpy())}
            for fraction_key, fraction, indices in (("1.00", 1.0, np.arange(len(train_y), dtype=np.int64)), ("0.10", 0.10, sample_patient_indices(train_patients, train_grades, fraction, args.seed + args.seeds.index(seed))[0])):
                sub_y = train_y[indices]
                sub_grades = train_grades[indices]
                arm_features = {"dense": (train_dense[indices], val_dense, test_dense, idrid_dense), "sparse_code": (train_code[indices], val_code, test_code, idrid_code), "sae_reconstruction": (train_recon.numpy()[indices], val_recon.numpy(), test_recon.numpy(), idrid_recon.numpy())}
                arm_features["dense_plus_sparse_code"] = (hstack([sparse.csr_matrix(train_dense[indices]), train_code[indices]], format="csr"), hstack([sparse.csr_matrix(val_dense), val_code], format="csr"), hstack([sparse.csr_matrix(test_dense), test_code], format="csr"), hstack([sparse.csr_matrix(idrid_dense), idrid_code], format="csr"))
                rep_record = {"seed": seed, "training_images": int(len(indices)), "arms": {}}
                for arm in ARMS:
                    if arm == "dense":
                        val_prob = dense_predictions[fraction_key][seed]["validation"]
                        test_prob = dense_predictions[fraction_key][seed]["test"]
                        idrid_prob = dense_predictions[fraction_key][seed]["idrid"]
                        threshold = dense_records[fraction_key][seed]["threshold"]
                    else:
                        fitted = fit_logistic(arm_features[arm][0], sub_y, seed, multiclass=False)
                        train_prob = predict_logistic(fitted, arm_features[arm][0])[:, 1]
                        threshold = fit_youden_threshold(sub_y, train_prob)
                        val_prob = predict_logistic(fitted, arm_features[arm][1])[:, 1]
                        test_prob = predict_logistic(fitted, arm_features[arm][2])[:, 1]
                        idrid_prob = predict_logistic(fitted, arm_features[arm][3])[:, 1]
                    for split, values in (("validation", val_prob), ("test", test_prob), ("idrid", idrid_prob)):
                        prediction_store[fraction_key][arm][split].append(values)
                    arm_record = {"decision_threshold_fit_on_training": threshold, "validation_binary": binary_record(val_y, val_prob, threshold), "locked_test_binary": binary_record(test_y, test_prob, threshold), "idrid_binary": binary_record(idrid_y, idrid_prob, threshold)}
                    multi = fit_logistic(arm_features[arm][0], sub_grades, seed, multiclass=True)
                    if multi is None:
                        arm_record["five_class"] = {"status": "unavailable_missing_training_grade"}
                    else:
                        arm_record["five_class"] = {"validation": multiclass_metrics(val_grades, predict_logistic(multi, arm_features[arm][1])), "locked_test": multiclass_metrics(test_grades, predict_logistic(multi, arm_features[arm][2])), "idrid": multiclass_metrics(idrid_grades, predict_logistic(multi, arm_features[arm][3]))}
                    rep_record["arms"][arm] = arm_record
                replicate_store[fraction_key].append(rep_record)
            del model, train_code, val_code, test_code, idrid_code, train_recon, val_recon, test_recon, idrid_recon
        for fraction_key in ("1.00", "0.10"):
            for arm in ARMS:
                val_avg = np.mean(np.stack(prediction_store[fraction_key][arm]["validation"]), axis=0)
                test_avg = np.mean(np.stack(prediction_store[fraction_key][arm]["test"]), axis=0)
                idrid_avg = np.mean(np.stack(prediction_store[fraction_key][arm]["idrid"]), axis=0)
                threshold = float(np.median([rep["arms"][arm]["decision_threshold_fit_on_training"] for rep in replicate_store[fraction_key]]))
                dense_test_avg = np.mean(np.stack(prediction_store[fraction_key]["dense"]["test"]), axis=0)
                dense_idrid_avg = np.mean(np.stack(prediction_store[fraction_key]["dense"]["idrid"]), axis=0)
                k_record["ensembles"][fraction_key][arm] = {"validation_binary": {**binary_metrics(val_y, val_avg, val_patients, args.bootstrap_resamples, args.seed + 100), **threshold_metrics(val_y, val_avg, threshold)}, "locked_test_binary": {**binary_metrics(test_y, test_avg, test_patients, args.bootstrap_resamples, args.seed + 200), **threshold_metrics(test_y, test_avg, threshold)}, "idrid_binary": {**binary_metrics(idrid_y, idrid_avg, idrid_patients, args.bootstrap_resamples, args.seed + 300), **threshold_metrics(idrid_y, idrid_avg, threshold)}, "locked_test_delta_vs_dense": {"auroc": binary_difference(test_y, test_avg, dense_test_avg, test_patients, __import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score, args.bootstrap_resamples, args.seed + 400), "auprc": binary_difference(test_y, test_avg, dense_test_avg, test_patients, __import__("sklearn.metrics", fromlist=["average_precision_score"]).average_precision_score, args.bootstrap_resamples, args.seed + 500)}, "idrid_delta_vs_dense": {"auroc": binary_difference(idrid_y, idrid_avg, dense_idrid_avg, idrid_patients, __import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score, args.bootstrap_resamples, args.seed + 600), "auprc": binary_difference(idrid_y, idrid_avg, dense_idrid_avg, idrid_patients, __import__("sklearn.metrics", fromlist=["average_precision_score"]).average_precision_score, args.bootstrap_resamples, args.seed + 700)}}
                multi_values = [rep["arms"][arm]["five_class"] for rep in replicate_store[fraction_key] if "status" not in rep["arms"][arm]["five_class"]]
                if multi_values:
                    # Store the mean of per-seed five-class metrics for the same deterministic replicates.
                    keys = ("validation", "locked_test", "idrid")
                    k_record["ensembles"][fraction_key][arm]["five_class_replicate_mean"] = {split: {metric: float(np.mean([item[split][metric] for item in multi_values])) for metric in multi_values[0][split]} for split in keys}
        k_record["replicates"] = replicate_store
        base["top_k_runs"][k_key] = k_record
        (output / "benchmark_partial.json").write_text(json.dumps(base, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"event": "top_k_complete", "top_k": k}), flush=True)

    secondary = []
    for k in args.top_k_values[1:]:
        comparison = base["top_k_runs"][str(k)]["ensembles"]["1.00"]["dense_plus_sparse_code"]["locked_test_delta_vs_dense"]["auroc"]
        secondary.append({"top_k": k, "raw_p_value": comparison["p_value"]})
    ordered = sorted(enumerate(secondary), key=lambda item: item[1]["raw_p_value"])
    previous = 0.0
    for rank, (index, item) in enumerate(ordered, start=1):
        adjusted = min(1.0, max(previous, (len(secondary) - rank + 1) * item["raw_p_value"]))
        secondary[index]["holm_adjusted_p_value"] = adjusted
        previous = adjusted
    base["holm_adjustment"] = {"family": "dense_plus_full_sparse_code_vs_dense_internal_100_percent", "method": "Holm", "comparisons": secondary}
    base["primary_comparisons"] = {"internal_100_percent": base["top_k_runs"]["1024"]["ensembles"]["1.00"]["dense_plus_sparse_code"]["locked_test_delta_vs_dense"], "external_1024": base["top_k_runs"]["1024"]["ensembles"]["1.00"]["dense_plus_sparse_code"]["idrid_delta_vs_dense"]}
    base["scope_boundary"] = "No lesion-semantic or clinician-validation claim; IDRiD is image-examination proxy resampling because verified patient IDs are unavailable."
    path = output / "sae_topk_capacity_sweep.json"
    path.write_text(json.dumps(base, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": base["status"], "output": str(path), "top_k_values": list(args.top_k_values), "primary_comparisons": base["primary_comparisons"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
