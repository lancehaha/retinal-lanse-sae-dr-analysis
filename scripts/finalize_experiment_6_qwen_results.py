#!/usr/bin/env python3
"""Fail-closed finalization for Experiment 6 Qwen semantic-control results."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path


ARMS = ("real_descriptions", "description_free", "shuffled_descriptions")
SPLITS = ("locked_test", "idrid")
EXPECTED_IMAGES = {"locked_test": 3518, "idrid": 102}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def format_metric(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    report = args.report.resolve()
    if not str(output).startswith("/Volumes/Elements/Retina-LanSE/"):
        raise RuntimeError("Experiment output must remain on the approved external drive")

    result_path = output / "experiment_6_feature_card_prediction.json"
    prediction_path = output / "llm_predictions.jsonl"
    raw_path = output / "llm_raw_batches.jsonl"
    if not result_path.is_file() or not prediction_path.is_file() or not raw_path.is_file():
        raise RuntimeError("missing Experiment 6 result, prediction, or raw-response artifact")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    rows = load_jsonl(prediction_path)
    raw_rows = load_jsonl(raw_path)
    keys = [(row.get("arm"), row.get("split"), row.get("relative_path")) for row in rows]
    duplicate_keys = [key for key, count in collections.Counter(keys).items() if count > 1]
    if duplicate_keys:
        raise RuntimeError(f"duplicate LLM prediction keys detected: {duplicate_keys[:3]}")

    counts: dict[str, dict[str, dict[str, int]]] = {}
    missing: list[str] = []
    for arm in ARMS:
        counts[arm] = {}
        for split in SPLITS:
            selected = [row for row in rows if row.get("arm") == arm and row.get("split") == split]
            status = collections.Counter(str(row.get("status")) for row in selected)
            counts[arm][split] = {"rows": len(selected), **dict(status)}
            if len(selected) != EXPECTED_IMAGES[split]:
                missing.append(f"{arm}/{split}: expected {EXPECTED_IMAGES[split]}, found {len(selected)}")
    if result.get("status") != "complete":
        missing.append(f"result status is {result.get('status')!r}, not 'complete'")
    llm_metrics = result.get("llm_metrics", {})
    for arm in ARMS:
        for split in SPLITS:
            if split not in llm_metrics.get(arm, {}):
                missing.append(f"missing metrics for {arm}/{split}")
    if missing:
        audit = {
            "status": "incomplete",
            "missing": missing,
            "prediction_counts": counts,
            "result_sha256": sha256(result_path),
            "prediction_sha256": sha256(prediction_path),
            "raw_batch_sha256": sha256(raw_path),
        }
        (output / "experiment_6_completion_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise RuntimeError("Experiment 6 finalization refused: " + "; ".join(missing))

    def metric_row(arm: str, split: str) -> tuple[float, float, float, float]:
        record = llm_metrics[arm][split]
        return (
            float(record["completion_rate"]),
            float(record["binary"]["auroc"]["estimate"]),
            float(record["binary"]["auprc"]["estimate"]),
            float(record["five_grade"]["quadratic_weighted_kappa"]),
        )

    report_lines = [
        "# Experiment 6: Feature-Card DR Prediction Results",
        "",
        "Status: **complete**. All Qwen predictions were generated locally with the frozen K=1024, seed-4106 SAE and fixed feature-card evidence. RETFound, SAE, card associations, prompts, and downstream numerical specifications were frozen before the LLM ablation.",
        "",
        "## Numerical arms",
        "",
        "| Arm | Locked AUROC | Locked AUPRC | Locked QWK | IDRiD AUROC | IDRiD QWK |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    numerical = result["numeric_metrics"]
    for arm in ("dense", "direct_card_evidence", "direct_card_evidence_251", "learned_card_probe"):
        locked = numerical[arm]["locked_test"]
        idrid = numerical[arm]["idrid"]
        report_lines.append(
            f"| {arm} | {locked['binary']['auroc']['estimate']:.4f} | {locked['binary']['auprc']['estimate']:.4f} | {locked['five_grade']['quadratic_weighted_kappa']:.4f} | {idrid['binary']['auroc']['estimate']:.4f} | {idrid['five_grade']['quadratic_weighted_kappa']:.4f} |"
        )
    report_lines += [
        "",
        "## Qwen semantic-text controls",
        "",
        "| Arm | Locked completion | Locked AUROC | Locked AUPRC | Locked QWK | IDRiD completion | IDRiD AUROC | IDRiD AUPRC | IDRiD QWK |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        locked = metric_row(arm, "locked_test")
        idrid = metric_row(arm, "idrid")
        report_lines.append(
            f"| {arm} | {locked[0]:.3f} | {locked[1]:.4f} | {locked[2]:.4f} | {locked[3]:.4f} | {idrid[0]:.3f} | {idrid[1]:.4f} | {idrid[2]:.4f} | {idrid[3]:.4f} |"
        )
    comparisons = result.get("llm_comparisons", {})
    report_lines += ["", "## Semantic comparison", ""]
    if comparisons:
        report_lines += ["| Comparison | Locked AUROC difference | 95% CI | Holm-adjusted p |", "|---|---:|---:|---:|"]
        for name, value in comparisons.items():
            report_lines.append(
                f"| {name} | {value['difference']:+.4f} | [{value['ci_lower']:+.4f}, {value['ci_upper']:+.4f}] | {value.get('holm_adjusted_p_value', float('nan')):.4f} |"
            )
    else:
        report_lines.append("No LLM comparison record was found.")
    report_lines += [
        "",
        "Interpretation: this is an association-based, same-model feature-card analysis. It does not establish lesion identity, causality, DR specificity, clinical deployment utility, or equivalence to ophthalmologist review.",
        "",
    ]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(report_lines), encoding="utf-8")
    audit = {
        "status": "complete",
        "prediction_counts": counts,
        "raw_batch_statuses": dict(collections.Counter(str(row.get("status")) for row in raw_rows)),
        "result_sha256": sha256(result_path),
        "prediction_sha256": sha256(prediction_path),
        "raw_batch_sha256": sha256(raw_path),
        "report": str(report),
        "report_sha256": sha256(report),
    }
    (output / "experiment_6_completion_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
