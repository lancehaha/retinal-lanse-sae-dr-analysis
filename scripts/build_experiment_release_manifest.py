from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXTERNAL_ROOT = Path("/Volumes/Elements/Retina-LanSE")


def require_external(path: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(EXTERNAL_ROOT)
    except ValueError as exc:
        raise RuntimeError(f"{label} must stay under {EXTERNAL_ROOT}: {resolved}") from exc
    return resolved


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a release manifest for the executed retinal SAE experiment.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = require_external(Path(args.output), "release manifest")
    output.parent.mkdir(parents=True, exist_ok=True)
    root = EXTERNAL_ROOT / "experiments"
    entries = {
        "numeric_sae_diagnostics": root / "numeric_sae_diagnostics/numeric_sae_diagnostics.json",
        "activation_comparison": root / "activation_comparison_autoresearch/activation_comparison_autoresearch.json",
        "activation_interaction": root / "activation_interaction_autoresearch/activation_interaction_autoresearch.json",
        "feature_atlas_summary": root / "feature_atlas_autoresearch/feature_atlas_summary.json",
        "grounding_summary_16": root / "feature_grounding_qwen25vl_expanded16/feature_grounding_summary.json",
        "grounding_audit_16": root / "feature_grounding_verification_qwen25vl_expanded16/feature_grounding_verification.json",
        "hypothesis_consistency_64": root / "feature_hypothesis_verification_qwen25vl_expanded16/hypothesis_verification_summary.json",
        "vlm_image_only_64": root / "vlm_inference_ablation_qwen25vl_64/vlm_inference_summary.json",
        "vlm_feature_card_64": root / "vlm_featurecard_ablation_qwen25vl_64/vlm_featurecard_summary.json",
        "vlm_paired_64": root / "vlm_ablation_comparison_qwen25vl_64/paired_vlm_ablation_comparison.json",
        "vlm_image_only_128": root / "vlm_inference_ablation_qwen25vl_128/vlm_inference_summary.json",
        "vlm_feature_card_128": root / "vlm_featurecard_ablation_qwen25vl_128/vlm_featurecard_summary.json",
        "vlm_paired_128": root / "vlm_ablation_comparison_qwen25vl_128/paired_vlm_ablation_comparison.json",
        "vlm_diagnostics_128": root / "vlm_ablation_diagnostics_qwen25vl_128/vlm_ablation_diagnostics.json",
        "vlm_random_card_128": root / "vlm_randomcard_ablation_qwen25vl_128/vlm_featurecard_summary.json",
        "vlm_random_card_paired_128": root / "vlm_randomcard_comparison_qwen25vl_128/paired_vlm_ablation_comparison.json",
        "vlm_random_card_diagnostics_128": root / "vlm_randomcard_diagnostics_qwen25vl_128/vlm_ablation_diagnostics.json",
        "vlm_active_neutral_128": root / "vlm_active_neutral_ablation_qwen25vl_128/vlm_featurecard_summary.json",
        "vlm_random_neutral_128": root / "vlm_random_neutral_ablation_qwen25vl_128/vlm_featurecard_summary.json",
        "vlm_active_neutral_paired_128": root / "vlm_active_neutral_comparison_qwen25vl_128/paired_vlm_ablation_comparison.json",
        "vlm_random_neutral_paired_128": root / "vlm_random_neutral_comparison_qwen25vl_128/paired_vlm_ablation_comparison.json",
        "vlm_active_random_neutral_paired_128": root / "vlm_active_vs_random_neutral_comparison_qwen25vl_128/paired_vlm_ablation_comparison.json",
        "vlm_active_neutral_diagnostics_128": root / "vlm_active_neutral_diagnostics_qwen25vl_128/vlm_ablation_diagnostics.json",
        "vlm_random_neutral_diagnostics_128": root / "vlm_random_neutral_diagnostics_qwen25vl_128/vlm_ablation_diagnostics.json",
        "vlm_image_only_256": root / "vlm_inference_ablation_qwen25vl_256/vlm_inference_summary.json",
        "vlm_feature_card_256": root / "vlm_featurecard_ablation_qwen25vl_256/vlm_featurecard_summary.json",
        "vlm_paired_256": root / "vlm_ablation_comparison_qwen25vl_256/paired_vlm_ablation_comparison.json",
        "vlm_diagnostics_256": root / "vlm_ablation_diagnostics_qwen25vl_256/vlm_ablation_diagnostics.json",
        "vlm_image_only_1024": root / "vlm_inference_ablation_qwen25vl_1024/vlm_inference_summary.json",
        "vlm_feature_card_1024_atlas16384": root / "vlm_featurecard_ablation_qwen25vl_1024_atlas16384/vlm_featurecard_summary.json",
        "vlm_paired_1024": root / "vlm_ablation_comparison_qwen25vl_1024/paired_vlm_ablation_comparison.json",
        "vlm_diagnostics_1024": root / "vlm_ablation_diagnostics_qwen25vl_1024/vlm_ablation_diagnostics.json",
        "vlm_provenance_audit_1024": root / "vlm_ablation_provenance_audit_qwen25vl_1024/provenance_audit.json",
        "clinician_packet": root / "clinician_review_packet_qwen25vl_expanded16/review_packet_summary.json",
        "clinician_review_templates": root / "clinician_review_templates_qwen25vl_expanded16/review_template_summary.json",
        "clinician_review_scoring": root / "clinician_review_scoring_qwen25vl_expanded16/clinician_review_scoring.json",
        "clinician_review_handoff_manifest": root / "clinician_review_handoff_qwen25vl_expanded16/clinician_review_handoff_manifest.json",
        "clinician_review_handoff_archive": root / "clinician_review_handoff_qwen25vl_expanded16/clinician_review_handoff.zip",
        "dataset_preparation": EXTERNAL_ROOT / "datasets/eyepacs/preparation_evidence.json",
        "internal_test_freeze": root / "internal_test_freeze_manifest.json",
        "experiment_1_completion_audit": root / "experiment_1_completion_audit.json",
        "experiment_1_full_audit": root / "experiment_1_full_audit_20260905.json",
        "mean_patch_internal_test_baselines": root / "paper_representation_baselines_mean_patch_test/internal_test_baselines.json",
        "mean_patch_locked_internal_test": root / "paper_locked_internal_test_mean_patch/locked_internal_test_evaluation.json",
        "cls_internal_test_baselines": root / "paper_representation_baselines_cls_test/internal_test_baselines.json",
        "cls_locked_internal_test": root / "paper_locked_internal_test_cls/locked_internal_test_evaluation.json",
        "experiment_1_results_report": EXTERNAL_ROOT / "reports/EXPERIMENT_1_RESULTS.md",
        "final_consolidated_report": EXTERNAL_ROOT / "reports/RETINA_LANSE_CONSOLIDATED_REPORT_20260905.md",
        "stage6_external_gate_idrid": root / "stage6_external_gate_idrid_20260905.json",
    }
    artifacts = []
    for name, path in entries.items():
        if not path.is_file():
            raise RuntimeError(f"missing required artifact: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = None
        artifacts.append({"name": name, "path": str(path), "sha256": sha256(path), "status": payload.get("status") if isinstance(payload, dict) else "non_json"})
    report = {
        "schema_version": "1.0",
        "status": "complete_release_manifest_local_experiment",
        "created_by": "scripts/build_experiment_release_manifest.py",
        "external_storage_root": str(EXTERNAL_ROOT),
        "locked_internal_test_accessed": True,
        "all_heavy_outputs_on_external_drive": True,
        "artifacts": artifacts,
        "paper_claim_boundary": "Label-free frozen-representation reconstruction and exploratory ablation evidence are complete; Stage 6 is recorded as fail-closed and remains blocked pending approved external data and independent ophthalmologist review.",
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
