"""Run the frozen one-time external Stage 6 image-level evaluation.

The evaluator is intentionally downstream of ``check_stage6_external_gate``.
It refuses to run until official access metadata, a verified external
manifest, the frozen local release, and independent review are complete. It
does not fine-tune the encoder/SAE, select features on external labels, or
change the train-derived threshold.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retina_lanse.data.embedding_shards import load_embedding_run
from retina_lanse.data.manifest import sha256_path
from retina_lanse.sae.checkpoints import load_checkpoint
from scripts.check_stage6_external_gate import check_stage6_gate, under
from scripts.run_activation_interaction_autoresearch import feature_stats, normalize_weights
from scripts.run_single_sae_feasibility import encode_csr, patient_bootstrap_interval


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError(f"external manifest line {line_number} is not an object")
            records.append(row)
    return records


def train_labels(path: Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            image = row["image"].strip()
            level = int(row["level"])
            if image in labels or level not in range(5):
                raise RuntimeError(f"invalid or duplicate EyePACS label: {image}")
            labels[image] = level
    return labels


def image_stem(path: str) -> str:
    return Path(path).stem


def external_target(row: dict[str, Any]) -> int:
    """Read a preregistered binary target from a verified external row."""

    if "referable_dr" in row:
        value = row["referable_dr"]
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)) and value in (0, 1):
            return int(value)
    for key in ("grade", "dr_grade", "level", "label"):
        if key not in row:
            continue
        value = row[key]
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            number = int(value)
            if number in (0, 1) and key == "label":
                return number
            if 0 <= number <= 4:
                return int(number >= 2)
    raise RuntimeError("external row has no supported binary referable-DR target")


def external_targets(records: list[dict[str, Any]], relative_paths: list[str]) -> tuple[np.ndarray, list[str]]:
    by_path: dict[str, dict[str, Any]] = {}
    for row in records:
        path = row.get("relative_path") or row.get("image_path") or row.get("path") or row.get("filename")
        if not isinstance(path, str) or not path.strip():
            raise RuntimeError("external manifest row has no relative image path")
        if path in by_path:
            raise RuntimeError(f"duplicate external manifest path: {path}")
        by_path[path] = row
    if set(relative_paths) != set(by_path):
        missing = sorted(set(relative_paths) - set(by_path))[:5]
        extra = sorted(set(by_path) - set(relative_paths))[:5]
        raise RuntimeError(f"embedding/manifest path mismatch; missing={missing}, extra={extra}")
    targets = np.asarray([external_target(by_path[path]) for path in relative_paths], dtype=np.int64)
    patients: list[str] = []
    for path in relative_paths:
        patient = by_path[path].get("patient_id")
        if not isinstance(patient, str) or not patient.strip():
            raise RuntimeError(f"missing patient_id for external image: {path}")
        patients.append(patient)
    if np.unique(targets).size != 2:
        raise RuntimeError("external manifest must contain both referable-DR classes")
    return targets, patients


def score_policy(train_codes, train_targets: np.ndarray, external_codes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stats = feature_stats(train_codes, train_targets)
    eligible = stats["count"] >= 50
    ranking = np.where(eligible, stats["mean_difference"], -np.inf)
    selected = np.argsort(ranking)[::-1][:64]
    weights = normalize_weights(stats["mean_difference"][selected])
    train_scores = np.asarray(train_codes[:, selected].dot(weights)).ravel()
    external_scores = np.asarray(external_codes[:, selected].dot(weights)).ravel()
    return train_scores, external_scores, selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the frozen one-time Stage 6 external retinal evaluation.")
    parser.add_argument("--dataset", choices=("idrid", "fgadr"), required=True)
    parser.add_argument("--external-root", default="/Volumes/Elements/Retina-LanSE")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--approval", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--release-manifest", required=True)
    parser.add_argument("--clinician-status", required=True)
    parser.add_argument("--gate-output", required=True)
    parser.add_argument("--train-run", required=True)
    parser.add_argument("--train-labels", required=True)
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--numeric-policy", required=True, help="Frozen numeric_sae_diagnostics.json")
    parser.add_argument("--external-embedding-run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=4106)
    args = parser.parse_args()

    external_root = Path(args.external_root).resolve()
    output = under(external_root, Path(args.output), "Stage 6 output")
    gate_output = under(external_root, Path(args.gate_output), "Stage 6 gate output")
    if output.exists():
        raise RuntimeError(f"refusing to overwrite one-time Stage 6 output: {output}")
    gate = check_stage6_gate(
        dataset=args.dataset,
        external_root=external_root,
        dataset_root=Path(args.dataset_root),
        approval_path=Path(args.approval),
        manifest_path=Path(args.manifest),
        release_manifest_path=Path(args.release_manifest),
        clinician_status_path=Path(args.clinician_status),
    )
    gate_output.parent.mkdir(parents=True, exist_ok=True)
    gate_output.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if gate["status"] != "ready_for_one_time_run":
        print(json.dumps({"status": "blocked_before_inference", "gate": gate}, indent=2, sort_keys=True))
        return 2

    paths = [
        Path(args.train_run), Path(args.train_labels), Path(args.normalization),
        Path(args.checkpoint), Path(args.numeric_policy), Path(args.external_embedding_run), Path(args.manifest),
    ]
    for path in paths:
        under(external_root, path, "Stage 6 input")
    if "internal_test" in str(args.train_run) or "internal_test" in str(args.external_embedding_run):
        raise RuntimeError("Stage 6 evaluator refuses internal-test embedding paths")

    train_run = load_embedding_run(Path(args.train_run), expected_split="train", verify_sha256=True)
    external_run = load_embedding_run(Path(args.external_embedding_run), expected_split="external", verify_sha256=True)
    records = read_jsonl(Path(args.manifest))
    external_y, external_patients = external_targets(records, external_run.relative_paths)
    lookup = train_labels(Path(args.train_labels))
    train_indices = [i for i, path in enumerate(train_run.relative_paths) if image_stem(path) in lookup]
    if not train_indices:
        raise RuntimeError("train embedding run contains no EyePACS labels")
    train_x = train_run.embeddings[train_indices]
    train_y = np.asarray([lookup[image_stem(train_run.relative_paths[i])] >= 2 for i in train_indices], dtype=np.int64)
    train_patients = [train_run.patient_ids[i] for i in train_indices]
    normalization_path = Path(args.normalization)
    policy_path = Path(args.numeric_policy)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if policy.get("status") != "complete_fixed_sae_numeric_diagnostics":
        raise RuntimeError("numeric policy is not a completed frozen diagnostic artifact")
    policy_inputs = policy.get("inputs", {})
    if policy_inputs.get("checkpoint_sha256") != sha256_path(Path(args.checkpoint)):
        raise RuntimeError("numeric policy checkpoint hash does not match Stage 6 checkpoint")
    if policy_inputs.get("normalization_sha256") != sha256_path(normalization_path):
        raise RuntimeError("numeric policy normalization hash does not match Stage 6 normalization")
    normalization = torch.load(normalization_path, map_location="cpu", weights_only=False)
    train_x = (train_x - normalization["mean"]) / normalization["std"]
    external_x = (external_run.embeddings - normalization["mean"]) / normalization["std"]
    model, checkpoint_payload = load_checkpoint(Path(args.checkpoint))
    train_codes = encode_csr(model, train_x, batch_size=1024)
    external_codes = encode_csr(model, external_x, batch_size=1024)
    train_scores, external_scores, selected = score_policy(train_codes, train_y, external_codes)
    policy_selected = np.asarray(policy.get("selected_features", []), dtype=np.int64)
    if not np.array_equal(selected, policy_selected):
        raise RuntimeError("recomputed train-derived feature set does not match frozen numeric policy")
    threshold_record = policy.get("train_selected_threshold", {}).get("threshold")
    if not isinstance(threshold_record, (int, float)):
        raise RuntimeError("frozen numeric policy has no train-only threshold")
    threshold = float(threshold_record)
    metrics = {
        "auroc": patient_bootstrap_interval(external_y, external_scores, external_patients, roc_auc_score, resamples=args.bootstrap_resamples, seed=args.seed),
        "auprc": patient_bootstrap_interval(external_y, external_scores, external_patients, average_precision_score, resamples=args.bootstrap_resamples, seed=args.seed + 1),
    }
    report = {
        "schema_version": "1.0",
        "status": "complete_stage6_external_evaluation",
        "dataset": args.dataset,
        "inference_performed": True,
        "encoder_frozen": True,
        "sae_retrained": False,
        "external_labels_used_for_selection": False,
        "threshold": {"value": threshold, "source": "frozen train-only Youden threshold from numeric_sae_diagnostics.json"},
        "cohort": {"images": int(len(external_y)), "patients": len(set(external_patients)), "positive_images": int(external_y.sum()), "negative_images": int((1 - external_y).sum())},
        "metrics": metrics,
        "selected_features": [int(x) for x in selected],
        "inputs": {
            "manifest_sha256": sha256_path(Path(args.manifest)),
            "external_embedding_evidence_sha256": sha256_path(Path(args.external_embedding_run) / "extraction_evidence.json"),
            "train_embedding_evidence_sha256": sha256_path(Path(args.train_run) / "extraction_evidence.json"),
            "train_labels_sha256": sha256_path(Path(args.train_labels)),
            "normalization_sha256": sha256_path(normalization_path),
            "checkpoint_sha256": sha256_path(Path(args.checkpoint)),
            "checkpoint_payload_status": checkpoint_payload.get("status") if isinstance(checkpoint_payload, dict) else None,
        },
        "gate": gate,
        "caveat": "This runner is dormant until the Stage 6 gate passes; no external result is valid unless the gate record and this immutable output are retained together.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
