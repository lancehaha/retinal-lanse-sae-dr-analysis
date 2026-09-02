from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retina_lanse.data.manifest import sha256_path
from retina_lanse.utils import verify_file_hash_bundle
from scripts.run_global_sae_search import require_external


EXPECTED_BASELINES = {"train_mean", "matched_k_input_coordinate", "matched_k_pca"}


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"required evidence is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_cohort(path: Path) -> tuple[dict[str, int], dict[str, int]]:
    images: Counter[str] = Counter()
    patients: dict[str, set[str]] = defaultdict(set)
    patient_split: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            split, patient = str(record["split"]), str(record["patient_id"])
            prior = patient_split.setdefault(patient, split)
            if prior != split:
                raise RuntimeError(f"patient crosses publication splits: {patient}")
            images[split] += 1
            patients[split].add(patient)
    required = {"train", "validation", "internal_test"}
    if set(images) != required:
        raise RuntimeError(f"publication manifest split set is invalid: {sorted(images)}")
    return dict(images), {split: len(values) for split, values in patients.items()}


def check_embedding(
    path: Path, *, representation: str, split: str, manifest_hash: str, expected_images: int
) -> dict:
    evidence = read_json(path / "extraction_evidence.json")
    if (
        evidence.get("status") != "complete"
        or evidence.get("failures") != 0
        or evidence.get("images") != expected_images
        or evidence.get("representation") != representation
        or evidence.get("manifest_hash") != manifest_hash
    ):
        raise RuntimeError(f"embedding evidence failed audit: {path}")
    shard_images = sum(int(item["images"]) for item in evidence["shards"])
    if shard_images != expected_images:
        raise RuntimeError(f"embedding shard image count failed audit: {path}")
    return {
        "path": str(path / "extraction_evidence.json"),
        "sha256": sha256_path(path / "extraction_evidence.json"),
        "images": evidence["images"],
        "shards": len(evidence["shards"]),
    }


def check_representation(
    root: Path,
    freeze: dict,
    representation: str,
    short: str,
    expected_split_images: dict[str, int],
    expected_split_patients: dict[str, int],
) -> dict:
    embedding_root = root / "embeddings" / f"global_{representation}"
    experiment_root = root / "experiments"
    screen_path = experiment_root / f"paper_global_sae_screen_{short}" / "search_summary.json"
    screen = read_json(screen_path)
    if (
        screen.get("status") != "complete_publication_screen"
        or screen.get("full_grid_size") != 54
        or screen.get("executed_configurations") != 54
        or len(screen.get("screen_finalists", [])) != 3
        or screen.get("selection_uses_clinical_labels") is not False
    ):
        raise RuntimeError(f"screen audit failed for {representation}")
    screen_metric_paths = sorted(screen_path.parent.glob("*/metrics.json"))
    if len(screen_metric_paths) != 54:
        raise RuntimeError(f"expected 54 screen metric files for {representation}")
    screen_candidates = set()
    for path in screen_metric_paths:
        metrics = read_json(path)
        checkpoint = Path(metrics["checkpoint"])
        if (
            metrics.get("status") != "complete"
            or metrics.get("scope") != "publication_screen"
            or metrics.get("config_hash") != screen["config_hash"]
            or metrics.get("gates", {}).get("clinical_labels_used") is not False
            or metrics.get("gates", {}).get("patient_overlap") != 0
            or not checkpoint.is_file()
            or sha256_path(checkpoint) != metrics.get("checkpoint_sha256")
        ):
            raise RuntimeError(f"screen run evidence failed audit: {path}")
        screen_candidates.add(metrics["candidate"])
    if len(screen_candidates) != 54:
        raise RuntimeError(f"screen candidate identifiers are not unique: {representation}")

    finalist_root = experiment_root / f"paper_global_sae_finalists_{short}"
    selection_path = finalist_root / "final_selection.json"
    selection = read_json(selection_path)
    if (
        selection.get("status") != "selection_frozen"
        or len(selection.get("finalists", [])) != 3
        or selection.get("selection_uses_clinical_labels") is not False
    ):
        raise RuntimeError(f"selection audit failed for {representation}")
    all_seed_metrics = sorted(finalist_root.glob("*/seed_*/metrics.json"))
    if len(all_seed_metrics) != 15:
        raise RuntimeError(f"expected 15 finalist seed runs: {representation}")
    finalist_seed_pairs = set()
    for path in all_seed_metrics:
        metrics = read_json(path)
        checkpoint = Path(metrics["checkpoint"])
        if (
            metrics.get("status") != "complete"
            or metrics.get("scope") != "publication_finalist"
            or metrics.get("gates", {}).get("clinical_labels_used") is not False
            or metrics.get("gates", {}).get("patient_overlap") != 0
            or not all(
                metrics["gates"][key]
                for key in (
                    "topk_budget",
                    "dead_fraction_at_most_0_5",
                    "checkpoint_reload_equivalent",
                )
            )
            or not checkpoint.is_file()
            or sha256_path(checkpoint) != metrics.get("checkpoint_sha256")
        ):
            raise RuntimeError(f"selected seed gate failed: {path}")
        finalist_seed_pairs.add((metrics["finalist"], int(metrics["seed"])))
    if len(finalist_seed_pairs) != 15:
        raise RuntimeError(f"finalist/seed pairs are not unique: {representation}")
    if any(
        not item.get("all_seed_gates_pass") or len(item.get("seeds", [])) != 5
        for item in selection["finalists"]
    ):
        raise RuntimeError(f"a frozen finalist aggregate is incomplete: {representation}")
    selected = finalist_root / selection["selected_finalist"]
    selected_seed_metrics = sorted(selected.glob("seed_*/metrics.json"))
    if len(selected_seed_metrics) != 5:
        raise RuntimeError(f"selected finalist does not have five seeds: {representation}")

    validation_baseline_path = (
        experiment_root / f"paper_representation_baselines_{short}" / "validation_baselines.json"
    )
    test_baseline_path = (
        experiment_root
        / f"paper_representation_baselines_{short}_test"
        / "internal_test_baselines.json"
    )
    for split, path in (("validation", validation_baseline_path), ("internal_test", test_baseline_path)):
        baseline = read_json(path)
        if (
            baseline.get("status") != "complete"
            or baseline.get("target_split") != split
            or baseline.get("clinical_labels_used") is not False
            or set(baseline.get("baselines", {})) != EXPECTED_BASELINES
        ):
            raise RuntimeError(f"baseline audit failed: {path}")

    locked_path = (
        experiment_root
        / f"paper_locked_internal_test_{short}"
        / "locked_internal_test_evaluation.json"
    )
    locked = read_json(locked_path)
    if (
        locked.get("status") != "complete_locked_internal_test"
        or locked.get("single_access_evaluation") is not True
        or locked.get("clinical_labels_used") is not False
        or locked.get("images") != expected_split_images["internal_test"]
        or locked.get("patients") != expected_split_patients["internal_test"]
        or locked.get("patient_overlap_train_internal_test") != 0
        or len(locked.get("seed_results", [])) != 5
        or locked.get("bootstrap_resamples") != 2000
        or locked.get("analysis_bundle_sha256") != freeze["analysis_bundle_sha256"]
    ):
        raise RuntimeError(f"locked-test audit failed for {representation}")
    for seed in locked["seed_results"]:
        if set(seed.get("paired_patient_bootstrap", {})) != EXPECTED_BASELINES:
            raise RuntimeError(f"locked-test baseline family incomplete for {representation}")
        for comparison in seed["paired_patient_bootstrap"].values():
            for metric in comparison.values():
                if (
                    metric.get("resamples") != 2000
                    or metric.get("patients") != expected_split_patients["internal_test"]
                ):
                    raise RuntimeError(f"patient bootstrap audit failed for {representation}")
    seed_averaged = locked.get("seed_averaged_paired_patient_bootstrap", {})
    if set(seed_averaged) != EXPECTED_BASELINES:
        raise RuntimeError(f"seed-averaged comparison family is incomplete: {representation}")
    for comparison in seed_averaged.values():
        for metric in comparison.values():
            if (
                metric.get("resamples") != 2000
                or metric.get("patients") != expected_split_patients["internal_test"]
            ):
                raise RuntimeError(f"seed-averaged patient bootstrap failed: {representation}")

    return {
        "screen": {"path": str(screen_path), "sha256": sha256_path(screen_path), "configurations": 54},
        "selection": {"path": str(selection_path), "sha256": sha256_path(selection_path)},
        "screen_metric_files": len(screen_metric_paths),
        "finalist_seed_metric_files": len(all_seed_metrics),
        "selected_seed_metrics": [str(path) for path in selected_seed_metrics],
        "embeddings": {
            split: check_embedding(
                embedding_root / f"{split}_cpu_fp32",
                representation=representation,
                split=split,
                manifest_hash=freeze["dataset_manifest_sha256"],
                expected_images=expected_split_images[split],
            )
            for split in ("train", "validation", "internal_test")
        },
        "validation_baselines": str(validation_baseline_path),
        "internal_test_baselines": str(test_baseline_path),
        "locked_test": {"path": str(locked_path), "sha256": sha256_path(locked_path)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit all Experiment 1 publication evidence.")
    parser.add_argument("--external-root", default="/Volumes/Elements/Retina-LanSE")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = require_external(Path(args.external_root), "external root")
    output = require_external(Path(args.output), "audit output")
    freeze_path = root / "experiments/internal_test_freeze_manifest.json"
    freeze = read_json(freeze_path)
    if freeze.get("status") != "test_access_authorized":
        raise RuntimeError("freeze manifest is not authorized")
    verified, reason = verify_file_hash_bundle(
        freeze.get("analysis_files", {}), freeze.get("analysis_bundle_sha256", "")
    )
    if not verified:
        raise RuntimeError(reason)
    selection_verified, selection_reason = verify_file_hash_bundle(
        freeze.get("selection_evidence_files", {}),
        freeze.get("selection_evidence_bundle_sha256", ""),
    )
    if not selection_verified:
        raise RuntimeError(selection_reason)
    if any("autoresearch" in path.lower() for path in freeze["analysis_files"]):
        raise RuntimeError("AutoResearch source unexpectedly entered the frozen analysis bundle")

    dataset_manifest = Path(freeze["dataset_manifest"])
    if sha256_path(dataset_manifest) != freeze["dataset_manifest_sha256"]:
        raise RuntimeError("dataset manifest changed after freeze")
    expected_split_images, expected_split_patients = manifest_cohort(dataset_manifest)
    runtime_snapshot = Path(freeze["runtime_snapshot"])
    if (
        not runtime_snapshot.is_file()
        or sha256_path(runtime_snapshot) != freeze["runtime_snapshot_sha256"]
    ):
        raise RuntimeError("runtime snapshot changed after freeze")
    repeatability_report = Path(freeze["sae_repeatability_report"])
    if (
        not repeatability_report.is_file()
        or sha256_path(repeatability_report) != freeze["sae_repeatability_report_sha256"]
        or read_json(repeatability_report).get("status") != "pass"
    ):
        raise RuntimeError("SAE repeatability evidence changed or failed")
    near_duplicate_audit = Path(freeze["near_duplicate_audit"])
    if (
        not near_duplicate_audit.is_file()
        or sha256_path(near_duplicate_audit) != freeze["near_duplicate_audit_sha256"]
        or read_json(near_duplicate_audit).get("status") != "pass"
    ):
        raise RuntimeError("near-duplicate leakage evidence changed or failed")
    remediation_path = Path(freeze["near_duplicate_remediation"])
    remediation = read_json(remediation_path)
    source_failed_audit = Path(remediation.get("source_failed_audit", ""))
    if (
        sha256_path(remediation_path) != freeze["near_duplicate_remediation_sha256"]
        or remediation.get("status") != "complete"
        or remediation.get("output_manifest_sha256") != freeze["dataset_manifest_sha256"]
        or remediation.get("clinical_labels_used") is not False
        or remediation.get("patients_reassigned") != 0
        or not source_failed_audit.is_file()
        or sha256_path(source_failed_audit) != remediation.get("source_failed_audit_sha256")
        or read_json(source_failed_audit).get("status") != "fail"
    ):
        raise RuntimeError("near-duplicate remediation evidence changed or failed")
    representations = {
        "final_mean_patch": check_representation(
            root,
            freeze,
            "final_mean_patch",
            "mean_patch",
            expected_split_images,
            expected_split_patients,
        ),
        "final_cls": check_representation(
            root,
            freeze,
            "final_cls",
            "cls",
            expected_split_images,
            expected_split_patients,
        ),
    }
    audit = {
        "schema_version": "1.0",
        "status": "experiment_1_complete",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_manifest": str(dataset_manifest),
        "dataset_manifest_sha256": freeze["dataset_manifest_sha256"],
        "split_images": expected_split_images,
        "split_patients": expected_split_patients,
        "freeze_manifest": str(freeze_path),
        "freeze_manifest_sha256": sha256_path(freeze_path),
        "analysis_bundle_sha256": freeze["analysis_bundle_sha256"],
        "selection_evidence_bundle_sha256": freeze["selection_evidence_bundle_sha256"],
        "runtime_snapshot": str(runtime_snapshot),
        "runtime_snapshot_sha256": freeze["runtime_snapshot_sha256"],
        "sae_repeatability_report": str(repeatability_report),
        "sae_repeatability_report_sha256": freeze["sae_repeatability_report_sha256"],
        "near_duplicate_audit": str(near_duplicate_audit),
        "near_duplicate_audit_sha256": freeze["near_duplicate_audit_sha256"],
        "near_duplicate_remediation": str(remediation_path),
        "near_duplicate_remediation_sha256": freeze["near_duplicate_remediation_sha256"],
        "clinical_labels_used": False,
        "autoresearch_evaluator_used": False,
        "representations": representations,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
