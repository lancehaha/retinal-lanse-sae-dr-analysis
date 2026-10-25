"""Build the reader-facing report from authoritative external-drive artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXTERNAL_ROOT = Path("/Volumes/Elements/Retina-LanSE")


def read_json(path: Path, label: str) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing {label}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} is not a JSON object: {path}")
    return payload


def pct(value: float) -> str:
    return f"{value:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the consolidated Retina-LanSE experiment report.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    try:
        output.relative_to(EXTERNAL_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError(f"report must be written under {EXTERNAL_ROOT}: {output}") from exc

    root = EXTERNAL_ROOT / "experiments"
    numeric = read_json(root / "numeric_sae_diagnostics/numeric_sae_diagnostics.json", "numeric diagnostics")
    paired = read_json(root / "vlm_ablation_comparison_qwen25vl_1024/paired_vlm_ablation_comparison.json", "1,024-image paired ablation")
    image_only = read_json(root / "vlm_inference_ablation_qwen25vl_1024/vlm_inference_summary.json", "image-only arm")
    cards = read_json(root / "vlm_featurecard_ablation_qwen25vl_1024_atlas16384/vlm_featurecard_summary.json", "feature-card arm")
    mean_test = read_json(root / "paper_locked_internal_test_mean_patch/locked_internal_test_evaluation.json", "mean-patch internal test")
    cls_test = read_json(root / "paper_locked_internal_test_cls/locked_internal_test_evaluation.json", "CLS internal test")
    gate = read_json(root / "stage6_external_gate_idrid_20260905.json", "Stage 6 gate")
    audit = read_json(root / "release_audit_20260905.json", "release audit")
    full_audit = read_json(root / "experiment_1_full_audit_20260905.json", "full Experiment 1 audit")
    if audit.get("status") != "release_audit_passed":
        raise RuntimeError("release audit is not passed")
    if full_audit.get("status") != "experiment_1_complete":
        raise RuntimeError("full Experiment 1 audit is not complete")
    if gate.get("status") != "blocked_external_access" or gate.get("inference_performed") is not False:
        raise RuntimeError("Stage 6 gate is not the expected fail-closed record")

    numeric_metrics = numeric["validation_metrics"]
    paired_diff = paired["paired_difference_feature_card_minus_image_only"]
    mean_agg = mean_test["aggregate"]
    cls_agg = cls_test["aggregate"]
    reasons_text = "\n".join(f"- {reason}" for reason in gate.get("reasons", []))
    audit_status = audit["status"]
    full_audit_status = full_audit["status"]
    val_images = numeric["cohort"]["validation_images"]
    val_patients = numeric["cohort"]["validation_patients"]
    num_auroc = numeric_metrics["auroc"]["estimate"]
    num_auroc_lo = numeric_metrics["auroc"]["ci_lower"]
    num_auroc_hi = numeric_metrics["auroc"]["ci_upper"]
    num_auprc = numeric_metrics["auprc"]["estimate"]
    num_auprc_lo = numeric_metrics["auprc"]["ci_lower"]
    num_auprc_hi = numeric_metrics["auprc"]["ci_upper"]
    image_auroc = image_only["metrics"]["auroc"]
    image_auprc = image_only["metrics"]["auprc"]
    card_auroc = cards["metrics"]["auroc"]
    card_auprc = cards["metrics"]["auprc"]
    diff_auroc = paired_diff["auroc"]["difference_left_minus_right"]
    diff_auroc_lo = paired_diff["auroc"]["ci_lower"]
    diff_auroc_hi = paired_diff["auroc"]["ci_upper"]
    diff_auprc = paired_diff["auprc"]["difference_left_minus_right"]
    diff_auprc_lo = paired_diff["auprc"]["ci_lower"]
    diff_auprc_hi = paired_diff["auprc"]["ci_upper"]
    mean_ev = mean_agg["explained_variance"]["mean"]
    mean_cos = mean_agg["cosine_similarity"]["mean"]
    cls_ev = cls_agg["explained_variance"]["mean"]
    cls_cos = cls_agg["cosine_similarity"]["mean"]
    report = f"""# Retina-LanSE consolidated experiment report

Date: 2026-09-05

## Executive status

The local, frozen-representation experiment and the internal SAE/LLM ablation
package are complete and reproducible. The canonical Stage 6 external clinical
evaluation is **not complete**: its fail-closed gate recorded
`blocked_external_access`, so no IDRiD/FGADR inference was performed. This is
an evidence boundary, not a missing or invented result.

The full machine-readable release audit is `release_audit_20260905.json` and
has status `{audit_status}`. The full frozen Experiment 1 audit is
`experiment_1_full_audit_20260905.json` and has status `{full_audit_status}`;
it checked both 54-configuration screens, all 15 finalist seed runs per
representation, leakage gates, baselines, and the one-time locked-test
evaluations.

## Study design

- Dataset: decontaminated EyePACS cohort, 88,694 images from 44,347 patients;
  71,134 train, 8,800 validation, and 8,760 locked internal-test images.
- Encoder: RETFound-CFP, frozen; no encoder fine-tuning.
- SAE: Top-K-32, 16,384 dictionary features; SAE weights trained on frozen
  embeddings only.
- Primary internal clinical association endpoint: referable DR (EyePACS grade
  >=2), with patient-level grouping and bootstrap resampling.
- LLM: local Qwen2.5-VL 3B; labels were hidden from prompts in the ablations.
- Heavy data, checkpoints, embeddings, and outputs were stored under
  `/Volumes/Elements/Retina-LanSE`.

## Main results

| Analysis | Cohort | Result | Interpretation |
|---|---:|---:|---|
| Numerical SAE activation score | {val_images} validation images / {val_patients} patients | AUROC {num_auroc:.3f} (95% CI {num_auroc_lo:.3f}--{num_auroc_hi:.3f}); AUPRC {num_auprc:.3f} (CI {num_auprc_lo:.3f}--{num_auprc_hi:.3f}) | Measurable internal association; not external clinical validation |
| VLM image-only | 1,024 balanced images / 808 patients | AUROC {image_auroc:.3f}; AUPRC {image_auprc:.3f} | Baseline ablation arm |
| VLM + active SAE cards | 1,024 matched balanced images / 808 patients | AUROC {card_auroc:.3f}; AUPRC {card_auprc:.3f} | No improvement over image-only |
| Active cards minus image-only | Same 1,024 images | AUROC difference {diff_auroc:.4f} (CI {diff_auroc_lo:.4f}--{diff_auroc_hi:.4f}); AUPRC difference {diff_auprc:.4f} (CI {diff_auprc_lo:.4f}--{diff_auprc_hi:.4f}) | No reliable SAE-card gain detected |
| Mean-patch SAE reconstruction | 8,760 internal-test images / 4,380 patients | Explained variance {mean_ev:.4f}; cosine {mean_cos:.4f} | Label-free reconstruction only |
| CLS SAE reconstruction | 8,760 internal-test images / 4,380 patients | Explained variance {cls_ev:.4f}; cosine {cls_cos:.4f} | Label-free reconstruction only |

## What the AutoResearch component tested

The bounded search compared prespecified activation aggregation rules and then
tested pairwise co-activation rules. The additive weighted-mean rule remained
the selected policy. The feature atlas ranked prominent activations without
using DR labels for feature selection. The LLM/card comparison was an ablation:
it asked whether adding SAE-derived feature context improves image-level
inference. The 1,024-image paired confidence intervals include zero, so the
current evidence does not support an improvement claim.

## What is trained and what is frozen

The RETFound encoder is frozen. The SAE is trained on the label-free training
embeddings. The downstream numerical score uses train-derived feature
statistics and a train-only threshold. The VLM is run locally; its generated
feature descriptions are hypotheses, not validated lesion concepts.

## Stage 6 gate and remaining work

The gate record is `{gate["status"]}`. It lists:

{reasons_text}

The evaluator was exercised against this state and stopped before inference.
To finish the clinical study, the project still needs approved IDRiD and/or
FGADR access, a verified image/label/patient manifest, and three independent
ophthalmology reviewers to return the blinded packet (currently 0/192 rows).
After those inputs are present, the frozen evaluator must be run exactly once;
no external labels may be used to select features, prompts, or thresholds.

## Evidence index

- Full execution ledger: repository `research/EXPERIMENT_EXECUTION_LOG.md`
- Detailed report: repository `research/EXPERIMENT_REPORT_TO_DATE.md`
- Numeric diagnostics: `experiments/numeric_sae_diagnostics/numeric_sae_diagnostics.json`
- 1,024-image paired ablation: `experiments/vlm_ablation_comparison_qwen25vl_1024/paired_vlm_ablation_comparison.json`
- Frozen reconstruction evaluations:
  `experiments/paper_locked_internal_test_mean_patch/` and
  `experiments/paper_locked_internal_test_cls/`
- Stage 6 gate: `experiments/stage6_external_gate_idrid_20260905.json`
- Full frozen SAE audit: `experiments/experiment_1_full_audit_20260905.json`
- Release audit: `experiments/release_audit_20260905.json`

Claim boundary: this package supports a reproducible frozen-representation
feasibility study and exploratory ablation result. It does not establish
clinical safety, lesion semantics, causality, or external generalization.
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(json.dumps({"status": "complete_final_experiment_report", "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
