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


def read_json(path: Path, label: str) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing {label}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return payload


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the completed retinal SAE release package.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = require_external(Path(args.output), "audit output")
    experiments = EXTERNAL_ROOT / "experiments"
    failures: list[str] = []
    evidence: dict[str, object] = {}

    image_path = experiments / "vlm_inference_ablation_qwen25vl_1024/vlm_inference_summary.json"
    card_path = experiments / "vlm_featurecard_ablation_qwen25vl_1024_atlas16384/vlm_featurecard_summary.json"
    paired_path = experiments / "vlm_ablation_comparison_qwen25vl_1024/paired_vlm_ablation_comparison.json"
    diagnostics_path = experiments / "vlm_ablation_diagnostics_qwen25vl_1024/vlm_ablation_diagnostics.json"
    provenance_path = experiments / "vlm_ablation_provenance_audit_qwen25vl_1024/provenance_audit.json"
    clinician_path = experiments / "clinician_review_scoring_qwen25vl_expanded16/clinician_review_scoring.json"
    stage6_gate_path = experiments / "stage6_external_gate_idrid_20260905.json"
    full_experiment_audit_path = experiments / "experiment_1_full_audit_20260905.json"
    manifest_path = experiments / "retina_lanse_release_manifest_20260905.json"
    report_paths = [
        Path("/Users/hgh54913/Documents/ChatGPT/CP4106 SAE ON RETINAL IMAGE ANALYSIS/research/RESULTS_SUMMARY.md"),
        Path("/Users/hgh54913/Documents/ChatGPT/CP4106 SAE ON RETINAL IMAGE ANALYSIS/research/EXPERIMENT_REPORT_TO_DATE.md"),
        Path("/Users/hgh54913/Documents/ChatGPT/CP4106 SAE ON RETINAL IMAGE ANALYSIS/research/VLM_INFERENCE_ABLATION_REPORT.md"),
        EXTERNAL_ROOT / "reports/RETINA_LANSE_CONSOLIDATED_REPORT_20260905.md",
    ]

    image = read_json(image_path, "image-only summary")
    card = read_json(card_path, "feature-card summary")
    paired = read_json(paired_path, "paired comparison")
    diagnostics = read_json(diagnostics_path, "diagnostics")
    provenance = read_json(provenance_path, "provenance audit")
    clinician = read_json(clinician_path, "clinician scoring status")
    stage6_gate = read_json(stage6_gate_path, "Stage 6 external gate")
    full_experiment_audit = read_json(full_experiment_audit_path, "full Experiment 1 audit")
    manifest = read_json(manifest_path, "release manifest")

    check(image.get("status") == "complete_vlm_inference_pilot", "image-only arm is not complete", failures)
    check(card.get("status") == "complete_vlm_featurecard_pilot", "feature-card arm is not complete", failures)
    check(image.get("images_completed") == 1024 and image.get("images_requested") == 1024, "image-only arm count is not 1024/1024", failures)
    check(card.get("images_completed") == 1024 and card.get("images_requested") == 1024, "feature-card arm count is not 1024/1024", failures)
    check(card.get("mean_grounded_cards_per_image", 0) > 0, "accepted feature-card arm has no grounded cards", failures)
    check(paired.get("status") == "complete_paired_vlm_ablation_comparison", "paired comparison is not complete", failures)
    check(paired.get("images") == 1024 and paired.get("patients") == 808, "paired sample counts are inconsistent", failures)
    check(diagnostics.get("status") == "complete_vlm_ablation_diagnostics", "diagnostics are not complete", failures)
    check(provenance.get("status") == "provenance_audit_complete", "provenance audit is not complete", failures)
    check(provenance.get("accepted_run", {}).get("decision") == "accepted_for_matched_analysis", "accepted-run decision missing", failures)
    check(provenance.get("accepted_run", {}).get("dictionary_size") == 16384, "accepted dictionary size is not 16384", failures)
    check(provenance.get("rejected_run", {}).get("decision") == "rejected_checkpoint_feature_atlas_mismatch", "rejected-run decision missing", failures)
    check(provenance.get("rejected_run", {}).get("mean_grounded_cards_per_image") == 0.0, "rejected run is not documented as zero-card", failures)
    check(clinician.get("status") == "pending_clinician_reviews", "clinician gate status changed unexpectedly", failures)
    check(clinician.get("complete_rows") == 0 and clinician.get("expected_rows") == 192, "clinician review row counts changed unexpectedly", failures)
    check(stage6_gate.get("status") == "blocked_external_access", "Stage 6 gate must remain explicitly blocked until approved inputs arrive", failures)
    check(stage6_gate.get("inference_performed") is False, "Stage 6 gate must not claim inference was performed", failures)
    check(any("approved dataset directory" in reason for reason in stage6_gate.get("reasons", [])), "Stage 6 gate does not record the missing external dataset", failures)
    check(full_experiment_audit.get("status") == "experiment_1_complete", "full Experiment 1 audit is not complete", failures)
    check(full_experiment_audit.get("clinical_labels_used") is False, "full Experiment 1 audit records clinical-label use", failures)
    check(full_experiment_audit.get("autoresearch_evaluator_used") is False, "full Experiment 1 audit records AutoResearch use", failures)
    check(manifest.get("all_heavy_outputs_on_external_drive") is True, "manifest does not confirm external-only heavy outputs", failures)
    check(manifest.get("locked_internal_test_accessed") is True, "manifest does not record the authorized internal-test access", failures)
    names = {item.get("name") for item in manifest.get("artifacts", []) if isinstance(item, dict)}
    for required in ("vlm_image_only_1024", "vlm_feature_card_1024_atlas16384", "vlm_paired_1024", "vlm_diagnostics_1024", "vlm_provenance_audit_1024"):
        check(required in names, f"release manifest missing {required}", failures)
    for path in report_paths:
        check(path.is_file(), f"missing reader-facing report: {path}", failures)
    evidence["image_only_sha256"] = sha256(image_path)
    evidence["feature_card_sha256"] = sha256(card_path)
    evidence["paired_sha256"] = sha256(paired_path)
    evidence["diagnostics_sha256"] = sha256(diagnostics_path)
    evidence["provenance_sha256"] = sha256(provenance_path)
    evidence["clinician_status_sha256"] = sha256(clinician_path)
    evidence["manifest_sha256"] = sha256(manifest_path)
    evidence["final_consolidated_report_sha256"] = sha256(EXTERNAL_ROOT / "reports/RETINA_LANSE_CONSOLIDATED_REPORT_20260905.md")
    evidence["metrics"] = {
        "image_only": image.get("metrics"),
        "feature_card": card.get("metrics"),
        "paired_difference": paired.get("paired_difference_feature_card_minus_image_only"),
        "grounded_cards_per_image": card.get("mean_grounded_cards_per_image"),
    }
    evidence["external_gate"] = {
        "status": stage6_gate.get("status"),
        "sha256": sha256(stage6_gate_path),
        "reasons": stage6_gate.get("reasons"),
    }
    evidence["full_experiment_audit_sha256"] = sha256(full_experiment_audit_path)
    payload = {
        "schema_version": "1.0",
        "status": "release_audit_passed" if not failures else "release_audit_failed",
        "scope": "local_retinal_sae_experiment_and_1024_image_vlm_ablation",
        "failures": failures,
        "evidence": evidence,
        "paper_claim_boundary": "Exploratory frozen-representation reconstruction and provenance-audited VLM ablation are complete; Stage 6 remains fail-closed pending approved external clinical data, and clinician review is pending.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
