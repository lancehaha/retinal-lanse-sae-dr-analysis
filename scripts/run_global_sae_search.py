from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from retina_lanse.config import load_config
from retina_lanse.data.embedding_shards import load_embedding_run
from retina_lanse.data.manifest import sha256_path
from retina_lanse.evaluation.metrics import evaluate_embeddings_batched
from retina_lanse.sae.model import TopKSAE
from retina_lanse.sae.trainer import train_sae_batched
from retina_lanse.utils import set_deterministic, stable_hash


EXTERNAL_ROOT = Path("/Volumes/Elements/Retina-LanSE")


def require_external(path: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(EXTERNAL_ROOT)
    except ValueError as exc:
        raise RuntimeError(f"{label} must stay under {EXTERNAL_ROOT}: {resolved}") from exc
    return resolved


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def candidate_id(expansion: int, k: int, learning_rate: float, resample: bool) -> str:
    lr = f"{learning_rate:.0e}".replace("-", "m")
    return f"exp{expansion:02d}_k{k:02d}_lr{lr}_resample{int(resample)}"


def write_history(path: Path, history: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def dominates(left: dict, right: dict) -> bool:
    left_values = (
        -left["validation"]["explained_variance"],
        left["validation"]["mse"],
        -left["validation"]["cosine_similarity"],
        left["validation"]["dead_feature_fraction"],
        left["validation"]["saturated_feature_fraction"],
    )
    right_values = (
        -right["validation"]["explained_variance"],
        right["validation"]["mse"],
        -right["validation"]["cosine_similarity"],
        right["validation"]["dead_feature_fraction"],
        right["validation"]["saturated_feature_fraction"],
    )
    return all(a <= b for a, b in zip(left_values, right_values)) and any(
        a < b for a, b in zip(left_values, right_values)
    )


def metrics_equal(left: dict, right: dict, tolerance: float = 1e-8) -> bool:
    if set(left) != set(right):
        return False
    for key in left:
        if isinstance(left[key], int) or isinstance(right[key], int):
            if left[key] != right[key]:
                return False
        elif abs(float(left[key]) - float(right[key])) > tolerance:
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the label-free publication global-SAE screen.")
    parser.add_argument("--config", default="configs/paper_global_sae.yaml")
    parser.add_argument("--train-run", required=True)
    parser.add_argument("--validation-run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit-configs", type=int)
    parser.add_argument("--epochs", type=int, help="Software-check override; marks output partial.")
    parser.add_argument("--skip-shard-sha256", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    output = require_external(Path(args.output), "search output")
    train_root = require_external(Path(args.train_run), "train embedding run")
    validation_root = require_external(Path(args.validation_run), "validation embedding run")
    output.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    train_run = load_embedding_run(
        train_root, expected_split="train", verify_sha256=not args.skip_shard_sha256
    )
    validation_run = load_embedding_run(
        validation_root, expected_split="validation", verify_sha256=not args.skip_shard_sha256
    )
    if train_run.evidence["checkpoint_hash"] != validation_run.evidence["checkpoint_hash"]:
        raise RuntimeError("train and validation encoder checkpoint hashes differ")
    if train_run.evidence["representation"] != validation_run.evidence["representation"]:
        raise RuntimeError("train and validation representations differ")
    overlap = set(train_run.patient_ids) & set(validation_run.patient_ids)
    if overlap:
        raise RuntimeError(f"patient leakage between embedding runs: {len(overlap)} patients")

    train_raw = train_run.embeddings
    validation_raw = validation_run.embeddings
    normalization_mean = train_raw.mean(dim=0)
    normalization_std = train_raw.std(dim=0, unbiased=False).clamp_min(1e-6)
    train = (train_raw - normalization_mean) / normalization_std
    validation = (validation_raw - normalization_mean) / normalization_std
    normalization_path = output / "train_only_normalization.pt"
    normalization_payload = {
        "method": "train_split_zscore",
        "mean": normalization_mean,
        "std": normalization_std,
        "train_manifest_hash": train_run.evidence["manifest_hash"],
        "validation_manifest_hash": validation_run.evidence["manifest_hash"],
    }
    if normalization_path.is_file():
        prior_normalization = torch.load(normalization_path, map_location="cpu", weights_only=False)
        if (
            prior_normalization.get("method") != normalization_payload["method"]
            or prior_normalization.get("train_manifest_hash")
            != normalization_payload["train_manifest_hash"]
            or prior_normalization.get("validation_manifest_hash")
            != normalization_payload["validation_manifest_hash"]
            or not torch.equal(prior_normalization["mean"], normalization_mean)
            or not torch.equal(prior_normalization["std"], normalization_std)
        ):
            raise RuntimeError(f"existing normalization artifact is incompatible: {normalization_path}")
    else:
        torch.save(normalization_payload, normalization_path)

    search = config["search"]
    combinations = list(
        itertools.product(
            search["dictionary_expansion"],
            search["top_k"],
            search["learning_rate"],
            search["dead_feature_resampling"],
        )
    )
    full_grid_size = len(combinations)
    if args.limit_configs is not None:
        if args.limit_configs < 1:
            parser.error("--limit-configs must be positive")
        combinations = combinations[: args.limit_configs]
    training = config["training"]
    epochs = args.epochs if args.epochs is not None else int(training["screen_epochs"])
    partial = len(combinations) != full_grid_size or args.epochs is not None
    config_hash = stable_hash([config])
    results: list[dict] = []

    for index, (expansion, k, learning_rate, resample) in enumerate(combinations, start=1):
        run_id = candidate_id(int(expansion), int(k), float(learning_rate), bool(resample))
        run_dir = output / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = run_dir / "metrics.json"
        run_signature = stable_hash(
            [
                {
                    "config_hash": config_hash,
                    "scope": "partial_software_check" if partial else "publication_screen",
                    "epochs": epochs,
                    "dictionary_expansion": int(expansion),
                    "top_k": int(k),
                    "learning_rate": float(learning_rate),
                    "dead_feature_resampling": bool(resample),
                    "train_evidence_sha256": sha256_path(train_root / "extraction_evidence.json"),
                    "validation_evidence_sha256": sha256_path(
                        validation_root / "extraction_evidence.json"
                    ),
                }
            ]
        )
        if metrics_path.is_file():
            prior = json.loads(metrics_path.read_text(encoding="utf-8"))
            if prior.get("status") != "complete" or prior.get("run_signature") != run_signature:
                raise RuntimeError(f"incompatible existing candidate evidence: {metrics_path}")
            results.append(prior)
            print(json.dumps({"event": "candidate_reused", "candidate": run_id}), flush=True)
            continue

        seed = int(config["seed_screen"])
        set_deterministic(seed)
        model = TopKSAE(1024, 1024 * int(expansion), int(k)).to(device)
        initial = evaluate_embeddings_batched(
            model,
            validation,
            batch_size=int(training["evaluation_batch_size"]),
            device=device,
        )
        started = time.perf_counter()
        history, optimizer = train_sae_batched(
            model,
            train,
            validation,
            epochs=epochs,
            batch_size=int(training["batch_size"]),
            evaluation_batch_size=int(training["evaluation_batch_size"]),
            learning_rate=float(learning_rate),
            weight_decay=float(training["weight_decay"]),
            seed=seed,
            device=device,
            resample_dead=bool(resample),
            resample_warmup_epochs=int(training["resample_warmup_epochs"]),
            resample_interval_epochs=int(training["resample_interval_epochs"]),
            resample_calibration_size=int(training["resample_calibration_size"]),
        )
        elapsed = time.perf_counter() - started
        final_train = evaluate_embeddings_batched(
            model,
            train,
            batch_size=int(training["evaluation_batch_size"]),
            device=device,
        )
        final_validation = evaluate_embeddings_batched(
            model,
            validation,
            batch_size=int(training["evaluation_batch_size"]),
            device=device,
        )
        model.cpu()
        checkpoint_path = run_dir / "checkpoint.pt"
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "model_config": {
                    "input_dim": 1024,
                    "dictionary_size": 1024 * int(expansion),
                    "k": int(k),
                },
                "normalization_path": str(normalization_path),
                "seed": seed,
                "epochs": epochs,
            },
            checkpoint_path,
        )
        reloaded_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        reloaded_model = TopKSAE(**reloaded_payload["model_config"]).to(device)
        reloaded_model.load_state_dict(reloaded_payload["model_state"])
        reload_validation = evaluate_embeddings_batched(
            reloaded_model,
            validation,
            batch_size=int(training["evaluation_batch_size"]),
            device=device,
        )
        reloaded_model.cpu()
        reload_equivalent = metrics_equal(final_validation, reload_validation)
        write_history(run_dir / "curves.csv", history)
        result = {
            "status": "complete",
            "scope": "partial_software_check" if partial else "publication_screen",
            "candidate": run_id,
            "candidate_index": index,
            "config_hash": config_hash,
            "run_signature": run_signature,
            "configuration": {
                "dictionary_expansion": int(expansion),
                "dictionary_size": 1024 * int(expansion),
                "top_k": int(k),
                "learning_rate": float(learning_rate),
                "dead_feature_resampling": bool(resample),
                "seed": seed,
                "epochs": epochs,
            },
            "input": {
                "train_images": len(train),
                "validation_images": len(validation),
                "checkpoint_hash": train_run.evidence["checkpoint_hash"],
                "representation": train_run.evidence["representation"],
                "train_evidence": str(train_root / "extraction_evidence.json"),
                "validation_evidence": str(validation_root / "extraction_evidence.json"),
                "normalization_sha256": sha256_path(normalization_path),
            },
            "initial_validation": initial,
            "train": final_train,
            "validation": final_validation,
            "reload_validation": reload_validation,
            "elapsed_seconds": elapsed,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_path(checkpoint_path),
            "gates": {
                "topk_budget": final_validation["max_active_features"] <= int(k),
                "dead_fraction_at_most_0_5": final_validation["dead_feature_fraction"] <= 0.5,
                "clinical_labels_used": False,
                "patient_overlap": 0,
                "checkpoint_reload_equivalent": reload_equivalent,
            },
        }
        metrics_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        results.append(result)
        print(
            json.dumps(
                {
                    "event": "candidate_complete",
                    "candidate": run_id,
                    "completed": index,
                    "total": len(combinations),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    feasible = [
        item
        for item in results
        if item["gates"]["topk_budget"] and item["gates"]["dead_fraction_at_most_0_5"]
        and item["gates"]["checkpoint_reload_equivalent"]
    ]
    pareto = [item for item in feasible if not any(dominates(other, item) for other in feasible if other is not item)]
    pareto.sort(
        key=lambda item: (
            -item["validation"]["explained_variance"],
            item["validation"]["dead_feature_fraction"],
            item["validation"]["mse"],
            -item["validation"]["cosine_similarity"],
            item["validation"]["saturated_feature_fraction"],
            item["configuration"]["top_k"],
            item["configuration"]["dictionary_expansion"],
        )
    )
    finalists = [item["candidate"] for item in pareto[: int(config["selection"]["finalists"])]]
    summary = {
        "status": "partial_software_check" if partial else "complete_publication_screen",
        "config_hash": config_hash,
        "full_grid_size": full_grid_size,
        "executed_configurations": len(results),
        "feasible_configurations": len(feasible),
        "pareto_candidates": [item["candidate"] for item in pareto],
        "screen_finalists": finalists,
        "selection_uses_clinical_labels": False,
        "device": str(device),
    }
    (output / "search_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
