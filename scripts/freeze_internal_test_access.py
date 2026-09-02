from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retina_lanse.data.manifest import sha256_path
from retina_lanse.utils import stable_hash


EXTERNAL_ROOT = Path("/Volumes/Elements/Retina-LanSE")


def require_external(path: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(EXTERNAL_ROOT)
    except ValueError as exc:
        raise RuntimeError(f"{label} must stay under {EXTERNAL_ROOT}: {resolved}") from exc
    return resolved


def validate_selection(path: Path, representation: str) -> tuple[dict, list[dict]]:
    selection = json.loads(path.read_text(encoding="utf-8"))
    if selection.get("status") != "selection_frozen":
        raise RuntimeError(f"selection is not frozen: {path}")
    if selection.get("selection_uses_clinical_labels") is not False:
        raise RuntimeError(f"selection does not attest label-free choice: {path}")
    if selection.get("locked_internal_test_accessed") is not False:
        raise RuntimeError(f"selection indicates prior locked-test access: {path}")
    normalization = Path(selection["normalization"]).resolve()
    if (
        not normalization.is_file()
        or sha256_path(normalization) != selection["normalization_sha256"]
    ):
        raise RuntimeError(f"selection normalization hash mismatch: {normalization}")
    screen_summary = Path(selection["screen_summary"]).resolve()
    if (
        not screen_summary.is_file()
        or sha256_path(screen_summary) != selection["screen_summary_sha256"]
    ):
        raise RuntimeError(f"selection screen-summary hash mismatch: {screen_summary}")
    selected = selection["selected_finalist"]
    run_root = path.parent / selected
    checkpoints: list[dict] = []
    for metrics_path in sorted(run_root.glob("seed_*/metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("status") != "complete" or metrics.get("scope") != "publication_finalist":
            raise RuntimeError(f"invalid finalist metrics: {metrics_path}")
        checkpoint = Path(metrics["checkpoint"]).resolve()
        if not checkpoint.is_file() or sha256_path(checkpoint) != metrics["checkpoint_sha256"]:
            raise RuntimeError(f"finalist checkpoint hash mismatch: {checkpoint}")
        if not all(
            metrics["gates"][key]
            for key in (
                "topk_budget",
                "dead_fraction_at_most_0_5",
                "checkpoint_reload_equivalent",
            )
        ):
            raise RuntimeError(f"finalist seed failed a required gate: {metrics_path}")
        checkpoints.append(
            {
                "representation": representation,
                "seed": metrics["seed"],
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": metrics["checkpoint_sha256"],
                "metrics": str(metrics_path),
                "metrics_sha256": sha256_path(metrics_path),
            }
        )
    if len(checkpoints) != 5:
        raise RuntimeError(f"expected five selected-seed checkpoints for {representation}, found {len(checkpoints)}")
    return selection, checkpoints


def validate_baselines(path: Path, selection_path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "complete" or report.get("target_split") != "validation":
        raise RuntimeError(f"baseline report is not a completed validation run: {path}")
    if report.get("clinical_labels_used") is not False:
        raise RuntimeError(f"baseline report does not attest label-free evaluation: {path}")
    if report.get("selection_sha256") != sha256_path(selection_path):
        raise RuntimeError(f"baseline report selection hash mismatch: {path}")
    pca = Path(report["pca_artifact"]).resolve()
    if not pca.is_file() or sha256_path(pca) != report["pca_artifact_sha256"]:
        raise RuntimeError(f"baseline PCA artifact hash mismatch: {pca}")
    return {
        "report": str(path),
        "report_sha256": sha256_path(path),
        "pca_artifact": str(pca),
        "pca_artifact_sha256": report["pca_artifact_sha256"],
    }


def collect_selection_evidence(selection_paths: list[Path]) -> tuple[dict[str, str], str]:
    evidence_paths: list[Path] = []
    for selection_path in selection_paths:
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        screen_summary = require_external(
            Path(selection["screen_summary"]), "screen summary"
        )
        screen_root = screen_summary.parent
        finalist_root = selection_path.parent
        screen_metrics = sorted(screen_root.glob("*/metrics.json"))
        screen_curves = sorted(screen_root.glob("*/curves.csv"))
        finalist_metrics = sorted(finalist_root.glob("*/seed_*/metrics.json"))
        finalist_curves = sorted(finalist_root.glob("*/seed_*/curves.csv"))
        if not (
            len(screen_metrics) == 54
            and len(screen_curves) == 54
            and len(finalist_metrics) == 15
            and len(finalist_curves) == 15
        ):
            raise RuntimeError(f"model-selection evidence tree is incomplete: {selection_path}")
        evidence_paths.extend(
            [
                screen_summary,
                screen_root / "train_only_normalization.pt",
                selection_path,
                *screen_metrics,
                *screen_curves,
                *finalist_metrics,
                *finalist_curves,
            ]
        )
    hashes = {str(path): sha256_path(path) for path in sorted(set(evidence_paths))}
    bundle = stable_hash(
        [{"path": path, "sha256": digest} for path, digest in sorted(hashes.items())]
    )
    return hashes, bundle


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze analysis state before authorizing EyePACS internal-test access.")
    parser.add_argument("--mean-patch-selection", required=True)
    parser.add_argument("--cls-selection", required=True)
    parser.add_argument("--mean-patch-baselines", required=True)
    parser.add_argument("--cls-baselines", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--runtime-snapshot", required=True)
    parser.add_argument("--sae-repeatability-report", required=True)
    parser.add_argument("--near-duplicate-audit", required=True)
    parser.add_argument("--near-duplicate-remediation", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    mean_path = require_external(Path(args.mean_patch_selection), "mean-patch selection")
    cls_path = require_external(Path(args.cls_selection), "CLS selection")
    mean_baselines_path = require_external(Path(args.mean_patch_baselines), "mean-patch baselines")
    cls_baselines_path = require_external(Path(args.cls_baselines), "CLS baselines")
    dataset_manifest = require_external(Path(args.dataset_manifest), "dataset manifest")
    output = require_external(Path(args.output), "freeze output")
    runtime_snapshot_path = require_external(Path(args.runtime_snapshot), "runtime snapshot")
    repeatability_path = require_external(
        Path(args.sae_repeatability_report), "SAE repeatability report"
    )
    near_duplicate_path = require_external(
        Path(args.near_duplicate_audit), "near-duplicate audit"
    )
    remediation_path = require_external(
        Path(args.near_duplicate_remediation), "near-duplicate remediation"
    )
    protocol = Path(args.protocol).resolve()
    config = Path(args.config).resolve()
    mean_selection, mean_checkpoints = validate_selection(mean_path, "final_mean_patch")
    cls_selection, cls_checkpoints = validate_selection(cls_path, "final_cls")
    mean_baselines = validate_baselines(mean_baselines_path, mean_path)
    cls_baselines = validate_baselines(cls_baselines_path, cls_path)
    selection_evidence_files, selection_evidence_bundle_sha256 = collect_selection_evidence(
        [mean_path, cls_path]
    )
    runtime_snapshot = json.loads(runtime_snapshot_path.read_text(encoding="utf-8"))
    if (
        runtime_snapshot.get("status") != "complete"
        or runtime_snapshot.get("dataset_manifest_sha256") != sha256_path(dataset_manifest)
        or runtime_snapshot.get("checkpoint_sha256")
        != "e1e4f66a1b792eeb6e2efaf158f33be35c8255f36b3d17ed67cd5129da246485"
        or not runtime_snapshot.get("retfound_upstream_commit")
    ):
        raise RuntimeError("runtime snapshot is incomplete or incompatible")
    repeatability = json.loads(repeatability_path.read_text(encoding="utf-8"))
    if repeatability.get("status") != "pass" or repeatability.get("pass") is not True:
        raise RuntimeError("real-embedding SAE training repeatability gate did not pass")
    for path_string, expected_hash in repeatability.get("analysis_files", {}).items():
        path = Path(path_string).resolve()
        if not path.is_file() or sha256_path(path) != expected_hash:
            raise RuntimeError(f"SAE training code changed after repeatability check: {path}")
    near_duplicate = json.loads(near_duplicate_path.read_text(encoding="utf-8"))
    remediation = json.loads(remediation_path.read_text(encoding="utf-8"))
    manifest_records = sum(1 for line in dataset_manifest.open(encoding="utf-8") if line.strip())
    source_failed_audit = Path(remediation.get("source_failed_audit", "")).resolve()
    if (
        remediation.get("status") != "complete"
        or remediation.get("clinical_labels_used") is not False
        or remediation.get("patients_reassigned") != 0
        or remediation.get("output_manifest_sha256") != sha256_path(dataset_manifest)
        or remediation.get("retained_records") != manifest_records
        or not source_failed_audit.is_file()
        or sha256_path(source_failed_audit) != remediation.get("source_failed_audit_sha256")
        or json.loads(source_failed_audit.read_text(encoding="utf-8")).get("status") != "fail"
    ):
        raise RuntimeError("near-duplicate remediation evidence is incomplete or incompatible")
    if (
        near_duplicate.get("status") != "pass"
        or near_duplicate.get("pass") is not True
        or near_duplicate.get("records") != manifest_records
        or near_duplicate.get("manifest_sha256") != sha256_path(dataset_manifest)
        or near_duplicate.get("confirmed_cross_split_near_duplicates") != 0
    ):
        raise RuntimeError("high-specificity cross-split near-duplicate audit did not pass")

    project_root = Path(__file__).resolve().parents[1]
    analysis_files = [
        project_root / "scripts/extract_retfound_embeddings.py",
        project_root / "scripts/freeze_internal_test_access.py",
        project_root / "scripts/run_global_sae_search.py",
        project_root / "scripts/run_global_sae_finalists.py",
        project_root / "scripts/run_representation_baselines.py",
        project_root / "scripts/run_locked_internal_test_evaluation.py",
        project_root / "scripts/audit_experiment_1.py",
        project_root / "scripts/generate_experiment_1_report.py",
        project_root / "scripts/capture_runtime_snapshot.py",
        project_root / "scripts/compare_sae_runs.py",
        project_root / "scripts/audit_near_duplicate_leakage.py",
        project_root / "scripts/remediate_near_duplicate_manifest.py",
        project_root / "scripts/run_post_train_pipeline_terminal.zsh",
        protocol,
        config,
    ]
    analysis_files.extend(
        path
        for path in sorted((project_root / "retina_lanse").rglob("*.py"))
        if "autoresearch" not in path.parts
    )
    analysis_files = sorted(set(analysis_files))
    analysis_hashes = {str(path): sha256_path(path) for path in analysis_files}
    analysis_bundle_sha256 = stable_hash(
        [{"path": path, "sha256": digest} for path, digest in sorted(analysis_hashes.items())]
    )
    freeze = {
        "schema_version": "1.0",
        "status": "test_access_authorized",
        "authorized_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_manifest": str(dataset_manifest),
        "dataset_manifest_sha256": sha256_path(dataset_manifest),
        "mean_patch_selection": str(mean_path),
        "mean_patch_selection_sha256": sha256_path(mean_path),
        "mean_patch_selected_finalist": mean_selection["selected_finalist"],
        "cls_selection": str(cls_path),
        "cls_selection_sha256": sha256_path(cls_path),
        "cls_selected_finalist": cls_selection["selected_finalist"],
        "mean_patch_validation_baselines": mean_baselines,
        "cls_validation_baselines": cls_baselines,
        "runtime_snapshot": str(runtime_snapshot_path),
        "runtime_snapshot_sha256": sha256_path(runtime_snapshot_path),
        "sae_repeatability_report": str(repeatability_path),
        "sae_repeatability_report_sha256": sha256_path(repeatability_path),
        "near_duplicate_audit": str(near_duplicate_path),
        "near_duplicate_audit_sha256": sha256_path(near_duplicate_path),
        "near_duplicate_remediation": str(remediation_path),
        "near_duplicate_remediation_sha256": sha256_path(remediation_path),
        "selected_seed_checkpoints": mean_checkpoints + cls_checkpoints,
        "selection_evidence_files": selection_evidence_files,
        "selection_evidence_bundle_sha256": selection_evidence_bundle_sha256,
        "analysis_files": analysis_hashes,
        "analysis_bundle_sha256": analysis_bundle_sha256,
        "selection_uses_clinical_labels": False,
        "internal_test_access_before_freeze": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(freeze, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
