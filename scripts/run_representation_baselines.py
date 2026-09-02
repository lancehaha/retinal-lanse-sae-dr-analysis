from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from retina_lanse.data.embedding_shards import load_embedding_run
from retina_lanse.data.manifest import sha256_path
from retina_lanse.evaluation.metrics import evaluate_reconstruction, sparse_coordinate_baseline
from retina_lanse.utils import set_deterministic
from scripts.run_global_sae_search import require_external


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit/evaluate label-free reconstruction baselines.")
    parser.add_argument("--train-run", required=True)
    parser.add_argument("--target-run", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-split", choices=("validation", "internal_test"), required=True)
    parser.add_argument("--pca-artifact")
    parser.add_argument("--seed", type=int, default=4106)
    args = parser.parse_args()

    train_root = require_external(Path(args.train_run), "train embedding run")
    target_root = require_external(Path(args.target_run), "target embedding run")
    selection_path = require_external(Path(args.selection), "selection")
    output = require_external(Path(args.output), "baseline output")
    output.mkdir(parents=True, exist_ok=True)

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("status") != "selection_frozen":
        raise RuntimeError(f"baseline evaluation requires frozen selection: {selection_path}")
    selected = selection["selected_configuration"]
    k = int(selected["top_k"])
    train_run = load_embedding_run(train_root, expected_split="train", verify_sha256=True)
    target_run = load_embedding_run(
        target_root, expected_split=args.target_split, verify_sha256=True
    )
    if train_run.evidence["checkpoint_hash"] != target_run.evidence["checkpoint_hash"]:
        raise RuntimeError("train and target embedding checkpoint hashes differ")
    if train_run.evidence["representation"] != target_run.evidence["representation"]:
        raise RuntimeError("train and target embedding representations differ")
    if set(train_run.patient_ids) & set(target_run.patient_ids):
        raise RuntimeError("patient leakage between baseline train and target runs")

    normalization_path = Path(selection["normalization"]).resolve()
    if (
        not normalization_path.is_file()
        or sha256_path(normalization_path) != selection["normalization_sha256"]
    ):
        raise RuntimeError("frozen selection normalization hash mismatch")
    normalization = torch.load(normalization_path, map_location="cpu", weights_only=False)
    if normalization["train_manifest_hash"] != train_run.evidence["manifest_hash"]:
        raise RuntimeError("baseline train run does not match frozen normalization")
    train = (train_run.embeddings - normalization["mean"]) / normalization["std"]
    target = (target_run.embeddings - normalization["mean"]) / normalization["std"]
    train_mean = train.mean(dim=0, keepdim=True)

    mean_reconstruction = train_mean.expand_as(target)
    mean_metrics = evaluate_reconstruction(target, mean_reconstruction)
    _, coordinate_metrics = sparse_coordinate_baseline(target, train_mean, k)

    if args.pca_artifact:
        pca_path = require_external(Path(args.pca_artifact), "PCA artifact")
        payload = torch.load(pca_path, map_location="cpu", weights_only=False)
        if payload["k"] != k:
            raise RuntimeError(f"PCA artifact K differs from frozen selection: {pca_path}")
        if payload["train_evidence_sha256"] != sha256_path(
            train_root / "extraction_evidence.json"
        ):
            raise RuntimeError("PCA artifact was fitted on a different train embedding run")
        pca_mean, components = payload["mean"], payload["components"]
    else:
        if args.target_split != "validation":
            parser.error("internal_test evaluation requires a validation-fitted --pca-artifact")
        set_deterministic(args.seed)
        pca_mean = train.mean(dim=0, keepdim=True)
        centered_train = train - pca_mean
        _, _, components = torch.pca_lowrank(centered_train, q=k, center=False, niter=4)
        pca_path = output / "pca_fit.pt"
        torch.save(
            {
                "schema_version": "1.0",
                "k": k,
                "seed": args.seed,
                "mean": pca_mean,
                "components": components,
                "train_evidence": str(train_root / "extraction_evidence.json"),
                "train_evidence_sha256": sha256_path(
                    train_root / "extraction_evidence.json"
                ),
                "normalization": str(normalization_path),
                "normalization_sha256": sha256_path(normalization_path),
            },
            pca_path,
        )
    centered_target = target - pca_mean
    pca_reconstruction = centered_target @ components @ components.T + pca_mean
    pca_metrics = evaluate_reconstruction(target, pca_reconstruction)

    report = {
        "schema_version": "1.0",
        "status": "complete",
        "scope": f"label_free_{args.target_split}_representation_baselines",
        "target_split": args.target_split,
        "images": len(target),
        "k": k,
        "selection": str(selection_path),
        "selection_sha256": sha256_path(selection_path),
        "train_evidence": str(train_root / "extraction_evidence.json"),
        "train_evidence_sha256": sha256_path(train_root / "extraction_evidence.json"),
        "target_evidence": str(target_root / "extraction_evidence.json"),
        "target_evidence_sha256": sha256_path(target_root / "extraction_evidence.json"),
        "normalization": str(normalization_path),
        "normalization_sha256": sha256_path(normalization_path),
        "pca_artifact": str(pca_path),
        "pca_artifact_sha256": sha256_path(pca_path),
        "baselines": {
            "train_mean": mean_metrics,
            "matched_k_input_coordinate": coordinate_metrics,
            "matched_k_pca": pca_metrics,
        },
        "clinical_labels_used": False,
    }
    report_path = output / f"{args.target_split}_baselines.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
