from __future__ import annotations

import argparse
import json
import math
import statistics
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
from scripts.run_global_sae_search import choose_device, metrics_equal, require_external, write_history


def aggregate(values: list[float]) -> dict[str, float]:
    standard_deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "mean": statistics.mean(values),
        "standard_deviation": standard_deviation,
        "standard_error": standard_deviation / math.sqrt(len(values)),
        "minimum": min(values),
        "maximum": max(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run five-seed publication finalist SAEs and freeze selection.")
    parser.add_argument("--config", default="configs/paper_global_sae.yaml")
    parser.add_argument("--screen-output", required=True)
    parser.add_argument("--train-run", required=True)
    parser.add_argument("--validation-run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = load_config(args.config)
    screen_root = require_external(Path(args.screen_output), "screen output")
    train_root = require_external(Path(args.train_run), "train embedding run")
    validation_root = require_external(Path(args.validation_run), "validation embedding run")
    output = require_external(Path(args.output), "finalist output")
    output.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    summary_path = screen_root / "search_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete_publication_screen":
        raise RuntimeError(f"finalists require a complete publication screen: {summary_path}")
    finalists = summary.get("screen_finalists", [])
    required_finalists = int(config["selection"]["finalists"])
    if len(finalists) != required_finalists:
        raise RuntimeError(f"expected {required_finalists} screen finalists, observed {len(finalists)}")

    train_run = load_embedding_run(train_root, expected_split="train", verify_sha256=True)
    validation_run = load_embedding_run(
        validation_root, expected_split="validation", verify_sha256=True
    )
    if train_run.evidence["checkpoint_hash"] != validation_run.evidence["checkpoint_hash"]:
        raise RuntimeError("train and validation checkpoint hashes differ")
    overlap = set(train_run.patient_ids) & set(validation_run.patient_ids)
    if overlap:
        raise RuntimeError(f"patient leakage between finalist inputs: {len(overlap)}")

    normalization_path = screen_root / "train_only_normalization.pt"
    normalization = torch.load(normalization_path, map_location="cpu", weights_only=False)
    if (
        normalization["train_manifest_hash"] != train_run.evidence["manifest_hash"]
        or normalization["validation_manifest_hash"] != validation_run.evidence["manifest_hash"]
    ):
        raise RuntimeError("screen normalization does not match finalist embedding manifests")
    train = (train_run.embeddings - normalization["mean"]) / normalization["std"]
    validation = (validation_run.embeddings - normalization["mean"]) / normalization["std"]
    training = config["training"]
    epochs = int(training["finalist_epochs"])
    seeds = [int(seed) for seed in config["final_seeds"]]
    config_hash = stable_hash([config])
    finalist_results: list[dict] = []

    for finalist in finalists:
        screen_metrics_path = screen_root / finalist / "metrics.json"
        screen_metrics = json.loads(screen_metrics_path.read_text(encoding="utf-8"))
        configuration = screen_metrics["configuration"]
        finalist_configuration = {**configuration, "epochs": epochs}
        seed_results: list[dict] = []
        for seed in seeds:
            run_dir = output / finalist / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = run_dir / "metrics.json"
            run_signature = stable_hash(
                [
                    {
                        "config_hash": config_hash,
                        "screen_summary_sha256": sha256_path(summary_path),
                        "finalist": finalist,
                        "seed": seed,
                        "epochs": epochs,
                        "train_evidence_sha256": sha256_path(
                            train_root / "extraction_evidence.json"
                        ),
                        "validation_evidence_sha256": sha256_path(
                            validation_root / "extraction_evidence.json"
                        ),
                    }
                ]
            )
            if metrics_path.is_file():
                prior = json.loads(metrics_path.read_text(encoding="utf-8"))
                if prior.get("status") != "complete" or prior.get("run_signature") != run_signature:
                    raise RuntimeError(f"incompatible existing finalist run: {metrics_path}")
                seed_results.append(prior)
                print(json.dumps({"event": "finalist_seed_reused", "finalist": finalist, "seed": seed}), flush=True)
                continue

            set_deterministic(seed)
            model = TopKSAE(
                1024,
                int(configuration["dictionary_size"]),
                int(configuration["top_k"]),
            ).to(device)
            started = time.perf_counter()
            history, optimizer = train_sae_batched(
                model,
                train,
                validation,
                epochs=epochs,
                batch_size=int(training["batch_size"]),
                evaluation_batch_size=int(training["evaluation_batch_size"]),
                learning_rate=float(configuration["learning_rate"]),
                weight_decay=float(training["weight_decay"]),
                seed=seed,
                device=device,
                resample_dead=bool(configuration["dead_feature_resampling"]),
                resample_warmup_epochs=int(training["resample_warmup_epochs"]),
                resample_interval_epochs=int(training["resample_interval_epochs"]),
                resample_calibration_size=int(training["resample_calibration_size"]),
            )
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
                        "dictionary_size": int(configuration["dictionary_size"]),
                        "k": int(configuration["top_k"]),
                    },
                    "normalization_path": str(normalization_path),
                    "seed": seed,
                    "epochs": epochs,
                },
                checkpoint_path,
            )
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            reloaded = TopKSAE(**payload["model_config"]).to(device)
            reloaded.load_state_dict(payload["model_state"])
            reload_validation = evaluate_embeddings_batched(
                reloaded,
                validation,
                batch_size=int(training["evaluation_batch_size"]),
                device=device,
            )
            reload_equivalent = metrics_equal(final_validation, reload_validation)
            write_history(run_dir / "curves.csv", history)
            result = {
                "status": "complete",
                "scope": "publication_finalist",
                "finalist": finalist,
                "seed": seed,
                "epochs": epochs,
                "configuration": finalist_configuration,
                "run_signature": run_signature,
                "train": final_train,
                "validation": final_validation,
                "reload_validation": reload_validation,
                "elapsed_seconds": time.perf_counter() - started,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256_path(checkpoint_path),
                "gates": {
                    "topk_budget": final_validation["max_active_features"]
                    <= int(configuration["top_k"]),
                    "dead_fraction_at_most_0_5": final_validation["dead_feature_fraction"]
                    <= 0.5,
                    "checkpoint_reload_equivalent": reload_equivalent,
                    "clinical_labels_used": False,
                    "patient_overlap": 0,
                },
            }
            metrics_path.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            seed_results.append(result)
            print(json.dumps({"event": "finalist_seed_complete", "finalist": finalist, "seed": seed}), flush=True)

        metric_names = [
            "mse",
            "explained_variance",
            "cosine_similarity",
            "dead_feature_fraction",
            "saturated_feature_fraction",
        ]
        aggregated = {
            metric: aggregate([float(item["validation"][metric]) for item in seed_results])
            for metric in metric_names
        }
        all_gates_pass = all(
            item["gates"]["topk_budget"]
            and item["gates"]["dead_fraction_at_most_0_5"]
            and item["gates"]["checkpoint_reload_equivalent"]
            for item in seed_results
        )
        finalist_results.append(
            {
                "finalist": finalist,
                "configuration": finalist_configuration,
                "seeds": seeds,
                "validation_aggregate": aggregated,
                "all_seed_gates_pass": all_gates_pass,
            }
        )

    feasible = [item for item in finalist_results if item["all_seed_gates_pass"]]
    if not feasible:
        raise RuntimeError("no finalist passed all gates across all five seeds")
    best = max(feasible, key=lambda item: item["validation_aggregate"]["explained_variance"]["mean"])
    best_ev = best["validation_aggregate"]["explained_variance"]
    threshold = best_ev["mean"] - best_ev["standard_error"]
    within_one_se = [
        item
        for item in feasible
        if item["validation_aggregate"]["explained_variance"]["mean"] >= threshold
    ]
    within_one_se.sort(
        key=lambda item: (
            int(item["configuration"]["top_k"]),
            item["validation_aggregate"]["dead_feature_fraction"]["mean"],
            int(item["configuration"]["dictionary_expansion"]),
            item["validation_aggregate"]["mse"]["mean"],
        )
    )
    selected = within_one_se[0]
    final = {
        "status": "selection_frozen",
        "scope": "publication_finalist_selection",
        "screen_summary": str(summary_path),
        "screen_summary_sha256": sha256_path(summary_path),
        "normalization": str(normalization_path),
        "normalization_sha256": sha256_path(normalization_path),
        "finalists": finalist_results,
        "best_mean_explained_variance_finalist": best["finalist"],
        "one_standard_error_threshold": threshold,
        "within_one_standard_error": [item["finalist"] for item in within_one_se],
        "selected_finalist": selected["finalist"],
        "selected_configuration": selected["configuration"],
        "selection_rule": config["selection"]["rule"],
        "selection_uses_clinical_labels": False,
        "locked_internal_test_accessed": False,
    }
    final_path = output / "final_selection.json"
    final_path.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
