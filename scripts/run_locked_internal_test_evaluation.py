from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import numpy as np

from retina_lanse.data.embedding_shards import load_embedding_run
from retina_lanse.data.manifest import sha256_path
from retina_lanse.evaluation.metrics import (
    evaluate_embeddings_batched,
    evaluate_reconstruction,
    reconstruct_embeddings_batched,
    sparse_coordinate_baseline,
)
from retina_lanse.evaluation.statistics import paired_patient_bootstrap_difference
from retina_lanse.sae.model import TopKSAE
from retina_lanse.utils import stable_hash
from scripts.run_global_sae_search import choose_device, require_external


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
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen five-seed SAE selection once on the locked internal test split."
    )
    parser.add_argument("--freeze-manifest", required=True)
    parser.add_argument("--test-run", required=True)
    parser.add_argument("--train-run", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--baseline-report", required=True)
    parser.add_argument("--representation", choices=("final_mean_patch", "final_cls"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--evaluation-batch-size", type=int, default=512)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    args = parser.parse_args()

    freeze_path = require_external(Path(args.freeze_manifest), "freeze manifest")
    test_root = require_external(Path(args.test_run), "internal-test embedding run")
    train_root = require_external(Path(args.train_run), "train embedding run")
    selection_path = require_external(Path(args.selection), "frozen selection")
    baseline_path = require_external(Path(args.baseline_report), "internal-test baseline report")
    output = require_external(Path(args.output), "locked-test output")
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "locked_internal_test_evaluation.json"
    if report_path.exists():
        raise RuntimeError(f"locked-test report already exists; refusing a second evaluation: {report_path}")

    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "test_access_authorized":
        raise RuntimeError("locked-test evaluation requires an authorized freeze manifest")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("status") != "selection_frozen":
        raise RuntimeError("selection is not frozen")
    selection_hash_key = (
        "mean_patch_selection_sha256"
        if args.representation == "final_mean_patch"
        else "cls_selection_sha256"
    )
    if freeze.get(selection_hash_key) != sha256_path(selection_path):
        raise RuntimeError("selection differs from the freeze manifest")

    test_run = load_embedding_run(test_root, expected_split="internal_test", verify_sha256=True)
    train_run = load_embedding_run(train_root, expected_split="train", verify_sha256=True)
    if test_run.evidence["representation"] != args.representation:
        raise RuntimeError("internal-test embedding representation mismatch")
    if test_run.evidence["manifest_hash"] != freeze["dataset_manifest_sha256"]:
        raise RuntimeError("internal-test embeddings use a different dataset manifest")
    if (
        train_run.evidence["representation"] != args.representation
        or train_run.evidence["checkpoint_hash"] != test_run.evidence["checkpoint_hash"]
        or set(train_run.patient_ids) & set(test_run.patient_ids)
    ):
        raise RuntimeError("train/internal-test representation, checkpoint, or patient isolation failed")

    normalization_path = Path(selection["normalization"]).resolve()
    if sha256_path(normalization_path) != selection["normalization_sha256"]:
        raise RuntimeError("frozen normalization hash mismatch")
    normalization = torch.load(normalization_path, map_location="cpu", weights_only=False)
    train = (train_run.embeddings - normalization["mean"]) / normalization["std"]
    test = (test_run.embeddings - normalization["mean"]) / normalization["std"]

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if (
        baseline.get("status") != "complete"
        or baseline.get("target_split") != "internal_test"
        or baseline.get("selection_sha256") != sha256_path(selection_path)
        or baseline.get("clinical_labels_used") is not False
        or baseline.get("target_evidence_sha256")
        != sha256_path(test_root / "extraction_evidence.json")
    ):
        raise RuntimeError("internal-test baseline report is incompatible")

    train_mean = train.mean(dim=0, keepdim=True)
    baseline_reconstructions = {"train_mean": train_mean.expand_as(test)}
    coordinate_reconstruction, _ = sparse_coordinate_baseline(
        test, train_mean, int(selection["selected_configuration"]["top_k"])
    )
    baseline_reconstructions["matched_k_input_coordinate"] = coordinate_reconstruction
    pca_path = Path(baseline["pca_artifact"]).resolve()
    if sha256_path(pca_path) != baseline["pca_artifact_sha256"]:
        raise RuntimeError("internal-test baseline PCA artifact hash mismatch")
    pca = torch.load(pca_path, map_location="cpu", weights_only=False)
    centered_test = test - pca["mean"]
    baseline_reconstructions["matched_k_pca"] = (
        centered_test @ pca["components"] @ pca["components"].T + pca["mean"]
    )
    for name, reconstruction in baseline_reconstructions.items():
        observed = evaluate_reconstruction(test, reconstruction)
        expected = baseline["baselines"][name]
        for metric in ("mse", "explained_variance", "cosine_similarity"):
            if abs(float(observed[metric]) - float(expected[metric])) > 1e-6:
                raise RuntimeError(f"baseline reconstruction does not reproduce report: {name}/{metric}")
    baseline_per_image = {
        name: {
            "mse": ((test - reconstruction).double() ** 2).mean(dim=1).numpy(),
            "cosine": F.cosine_similarity(reconstruction, test, dim=1).double().numpy(),
        }
        for name, reconstruction in baseline_reconstructions.items()
    }

    frozen_checkpoints = [
        item
        for item in freeze["selected_seed_checkpoints"]
        if item["representation"] == args.representation
    ]
    if len(frozen_checkpoints) != 5:
        raise RuntimeError(f"expected five frozen checkpoints, observed {len(frozen_checkpoints)}")
    device = choose_device(args.device)
    seed_results = []
    seed_per_image_mse = []
    seed_per_image_cosine = []
    for item in sorted(frozen_checkpoints, key=lambda record: int(record["seed"])):
        checkpoint = Path(item["checkpoint"]).resolve()
        if sha256_path(checkpoint) != item["checkpoint_sha256"]:
            raise RuntimeError(f"frozen checkpoint hash mismatch: {checkpoint}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = TopKSAE(**payload["model_config"]).to(device)
        model.load_state_dict(payload["model_state"])
        metrics = evaluate_embeddings_batched(
            model,
            test,
            batch_size=args.evaluation_batch_size,
            device=device,
        )
        if metrics["max_active_features"] > int(payload["model_config"]["k"]):
            raise RuntimeError(f"Top-K invariant failed for seed {item['seed']}")
        reconstruction = reconstruct_embeddings_batched(
            model,
            test,
            batch_size=args.evaluation_batch_size,
            device=device,
        )
        per_image_mse = ((test - reconstruction).double() ** 2).mean(dim=1).numpy()
        per_image_cosine = F.cosine_similarity(reconstruction, test, dim=1).double().numpy()
        seed_per_image_mse.append(per_image_mse)
        seed_per_image_cosine.append(per_image_cosine)
        paired = {}
        for baseline_index, (name, baseline_reconstruction) in enumerate(
            baseline_reconstructions.items()
        ):
            baseline_mse = baseline_per_image[name]["mse"]
            baseline_cosine = baseline_per_image[name]["cosine"]
            bootstrap_seed = 4106 + int(item["seed"]) * 10 + baseline_index
            paired[name] = {
                "mse_sae_minus_baseline": paired_patient_bootstrap_difference(
                    per_image_mse,
                    baseline_mse,
                    test_run.patient_ids,
                    resamples=args.bootstrap_resamples,
                    seed=bootstrap_seed,
                ),
                "cosine_sae_minus_baseline": paired_patient_bootstrap_difference(
                    per_image_cosine,
                    baseline_cosine,
                    test_run.patient_ids,
                    resamples=args.bootstrap_resamples,
                    seed=bootstrap_seed + 100000,
                ),
            }
        seed_results.append(
            {"seed": int(item["seed"]), "metrics": metrics, "paired_patient_bootstrap": paired}
        )

    mean_seed_mse = np.stack(seed_per_image_mse, axis=0).mean(axis=0)
    mean_seed_cosine = np.stack(seed_per_image_cosine, axis=0).mean(axis=0)
    seed_averaged_paired = {}
    for baseline_index, name in enumerate(baseline_reconstructions):
        bootstrap_seed = 94106 + baseline_index
        seed_averaged_paired[name] = {
            "mse_sae_minus_baseline": paired_patient_bootstrap_difference(
                mean_seed_mse,
                baseline_per_image[name]["mse"],
                test_run.patient_ids,
                resamples=args.bootstrap_resamples,
                seed=bootstrap_seed,
            ),
            "cosine_sae_minus_baseline": paired_patient_bootstrap_difference(
                mean_seed_cosine,
                baseline_per_image[name]["cosine"],
                test_run.patient_ids,
                resamples=args.bootstrap_resamples,
                seed=bootstrap_seed + 100000,
            ),
        }

    metric_names = [
        "mse",
        "explained_variance",
        "cosine_similarity",
        "mean_active_features",
        "dead_feature_fraction",
        "saturated_feature_fraction",
    ]
    aggregate_metrics = {
        metric: aggregate([float(item["metrics"][metric]) for item in seed_results])
        for metric in metric_names
    }
    comparisons = {}
    for name, metrics in baseline["baselines"].items():
        comparisons[name] = {
            "explained_variance_delta": aggregate(
                [
                    float(item["metrics"]["explained_variance"])
                    - float(metrics["explained_variance"])
                    for item in seed_results
                ]
            ),
            "mse_delta": aggregate(
                [float(item["metrics"]["mse"]) - float(metrics["mse"]) for item in seed_results]
            ),
        }

    report = {
        "schema_version": "1.0",
        "status": "complete_locked_internal_test",
        "single_access_evaluation": True,
        "clinical_labels_used": False,
        "patient_resampling_unit": True,
        "bootstrap_resamples": args.bootstrap_resamples,
        "representation": args.representation,
        "images": len(test),
        "patients": len(set(test_run.patient_ids)),
        "patient_overlap_train_internal_test": 0,
        "freeze_manifest": str(freeze_path),
        "freeze_manifest_sha256": sha256_path(freeze_path),
        "analysis_bundle_sha256": freeze["analysis_bundle_sha256"],
        "selection": str(selection_path),
        "selection_sha256": sha256_path(selection_path),
        "test_evidence": str(test_root / "extraction_evidence.json"),
        "test_evidence_sha256": sha256_path(test_root / "extraction_evidence.json"),
        "train_evidence": str(train_root / "extraction_evidence.json"),
        "train_evidence_sha256": sha256_path(train_root / "extraction_evidence.json"),
        "baseline_report": str(baseline_path),
        "baseline_report_sha256": sha256_path(baseline_path),
        "seed_results": seed_results,
        "seed_averaged_paired_patient_bootstrap": seed_averaged_paired,
        "aggregate": aggregate_metrics,
        "baseline_comparisons": comparisons,
    }
    report["run_signature"] = stable_hash(
        [
            {
                "freeze_manifest_sha256": report["freeze_manifest_sha256"],
                "selection_sha256": report["selection_sha256"],
                "test_evidence_sha256": report["test_evidence_sha256"],
                "baseline_report_sha256": report["baseline_report_sha256"],
                "bootstrap_resamples": args.bootstrap_resamples,
            }
        ]
    )
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
