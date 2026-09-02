from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retina_lanse.data.manifest import sha256_path
from scripts.run_global_sae_search import require_external


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def number(value: float) -> str:
    return f"{float(value):.6f}"


def aggregate_cell(summary: dict) -> str:
    return f"{number(summary['mean'])} +/- {number(summary['standard_deviation'])}"


def representation_section(root: Path, short: str, title: str) -> list[str]:
    experiments = root / "experiments"
    selection_path = experiments / f"paper_global_sae_finalists_{short}/final_selection.json"
    selection = read_json(selection_path)
    locked_path = (
        experiments
        / f"paper_locked_internal_test_{short}/locked_internal_test_evaluation.json"
    )
    locked = read_json(locked_path)
    validation_baseline = read_json(
        experiments
        / f"paper_representation_baselines_{short}/validation_baselines.json"
    )
    test_baseline = read_json(
        experiments
        / f"paper_representation_baselines_{short}_test/internal_test_baselines.json"
    )
    selected = next(
        item for item in selection["finalists"] if item["finalist"] == selection["selected_finalist"]
    )
    configuration = selection["selected_configuration"]
    lines = [
        f"## {title}",
        "",
        (
            f"Selected `{selection['selected_finalist']}`: expansion {configuration['dictionary_expansion']}x, "
            f"dictionary width {configuration['dictionary_size']}, K={configuration['top_k']}, "
            f"learning rate {configuration['learning_rate']}, dead-feature resampling "
            f"{configuration['dead_feature_resampling']}."
        ),
        "",
        "| Endpoint | Validation, five seeds mean +/- SD | Internal test, five seeds mean +/- SD |",
        "|---|---:|---:|",
    ]
    for metric in ("mse", "explained_variance", "cosine_similarity", "dead_feature_fraction"):
        lines.append(
            f"| {metric.replace('_', ' ')} | "
            f"{aggregate_cell(selected['validation_aggregate'][metric])} | "
            f"{aggregate_cell(locked['aggregate'][metric])} |"
        )
    lines.extend(
        [
            "",
            "### Label-free reconstruction baselines",
            "",
            "| Baseline | Validation MSE | Validation EV | Test MSE | Test EV |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for baseline_name in ("train_mean", "matched_k_input_coordinate", "matched_k_pca"):
        validation = validation_baseline["baselines"][baseline_name]
        test = test_baseline["baselines"][baseline_name]
        lines.append(
            f"| {baseline_name.replace('_', ' ')} | {number(validation['mse'])} | "
            f"{number(validation['explained_variance'])} | {number(test['mse'])} | "
            f"{number(test['explained_variance'])} |"
        )
    lines.extend(
        [
            "",
            "### Five-seed-averaged paired comparisons",
            "",
            "The SAE metric is averaged across the five fixed seeds for each image before resampling patients. The preregistered primary inference is mean-patch SAE versus matched-K PCA for MSE; all other intervals are secondary.",
            "",
            "| Baseline | MSE difference [95% CI] | Cosine difference [95% CI] |",
            "|---|---:|---:|",
        ]
    )
    for baseline_name, comparison in locked["seed_averaged_paired_patient_bootstrap"].items():
        mse = comparison["mse_sae_minus_baseline"]
        cosine = comparison["cosine_sae_minus_baseline"]
        lines.append(
            f"| {baseline_name.replace('_', ' ')} | "
            f"{number(mse['difference'])} [{number(mse['ci_lower'])}, {number(mse['ci_upper'])}] | "
            f"{number(cosine['difference'])} [{number(cosine['ci_lower'])}, {number(cosine['ci_upper'])}] |"
        )
    lines.extend(
        [
            "",
            "### Patient-clustered paired comparisons",
            "",
            f"Differences are SAE minus baseline. Negative MSE and positive cosine differences favor the SAE. Each confidence interval uses 2,000 paired resamples of the {locked['patients']:,} internal-test patients.",
            "",
            "| Seed | Baseline | MSE difference [95% CI] | Cosine difference [95% CI] |",
            "|---:|---|---:|---:|",
        ]
    )
    for seed_result in locked["seed_results"]:
        for baseline_name, comparison in seed_result["paired_patient_bootstrap"].items():
            mse = comparison["mse_sae_minus_baseline"]
            cosine = comparison["cosine_sae_minus_baseline"]
            lines.append(
                f"| {seed_result['seed']} | {baseline_name.replace('_', ' ')} | "
                f"{number(mse['difference'])} [{number(mse['ci_lower'])}, {number(mse['ci_upper'])}] | "
                f"{number(cosine['difference'])} [{number(cosine['ci_lower'])}, {number(cosine['ci_upper'])}] |"
            )
    lines.extend(
        [
            "",
            f"Selection SHA-256: `{sha256_path(selection_path)}`  ",
            f"Locked-test report SHA-256: `{sha256_path(locked_path)}`",
            "",
        ]
    )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the audited Experiment 1 results report.")
    parser.add_argument("--external-root", default="/Volumes/Elements/Retina-LanSE")
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = require_external(Path(args.external_root), "external root")
    audit_path = require_external(Path(args.audit), "completion audit")
    output = require_external(Path(args.output), "results report")
    audit = read_json(audit_path)
    if audit.get("status") != "experiment_1_complete":
        raise RuntimeError("refusing to generate a results report before the completion audit passes")

    split_images = audit["split_images"]
    split_patients = audit["split_patients"]
    total_images = sum(split_images.values())
    total_patients = sum(split_patients.values())

    lines = [
        "# Retina-LanSE Experiment 1 audited results",
        "",
        "Status: complete and evidence-audited.",
        "",
        f"This label-free representation experiment used a decontaminated cohort of {total_images:,} EyePACS images from {total_patients:,} patients: {split_images['train']:,} train, {split_images['validation']:,} validation, and {split_images['internal_test']:,} internal test. The source cohort contained 88,702 images; a pre-selection label-free near-duplicate gate excluded all images of four lower-priority holdout patients (eight images total), without patient reassignment. RETFound-CFP remained frozen; no clinical labels or AutoResearch evaluator were used.",
        "",
        "A total of 108 screening runs and 30 five-seed finalist runs were executed across final mean-patch and final CLS pooling. Before the full search, two independent same-seed trainings on real embeddings produced exact model tensors, optimizer state, curves, and metrics. The internal test was opened only after both selections, baselines, checkpoints, protocol, configuration, dataset manifest, runtime, and analysis code were frozen.",
        "",
    ]
    lines.extend(representation_section(root, "mean_patch", "Final mean-patch pooling (primary)"))
    lines.extend(representation_section(root, "cls", "Final CLS pooling (preregistered ablation)"))
    lines.extend(
        [
            "## Integrity and interpretation boundary",
            "",
            f"Dataset manifest SHA-256: `{audit['dataset_manifest_sha256']}`  ",
            f"Near-duplicate remediation SHA-256: `{audit['near_duplicate_remediation_sha256']}`  ",
            f"Analysis bundle SHA-256: `{audit['analysis_bundle_sha256']}`  ",
            f"Completion audit SHA-256: `{sha256_path(audit_path)}`",
            "",
            "These results support only label-free reconstruction claims for frozen RETFound representations. They do not establish lesion semantics, localization, diagnostic preservation, cross-dataset generalization, or clinical safety.",
            "",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "complete", "report": str(output), "sha256": sha256_path(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
