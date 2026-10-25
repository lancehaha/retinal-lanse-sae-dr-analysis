from __future__ import annotations

"""Compute fixed calibration and subgroup diagnostics for paired VLM arms.

This is post-response analysis only: the two prediction files already contain
the persisted hidden-label audit fields. No prompt, model, feature card, or
threshold is changed here.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

EXTERNAL_ROOT = Path("/Volumes/Elements/Retina-LanSE")


def require_external(path: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(EXTERNAL_ROOT)
    except ValueError as exc:
        raise RuntimeError(f"{label} must stay under {EXTERNAL_ROOT}: {resolved}") from exc
    return resolved


def read(path: Path, key: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if item.get("status") == "complete":
                rows[item["relative_path"]] = item
    if not rows:
        raise RuntimeError(f"no complete rows in {key}")
    return rows


def ece(targets: np.ndarray, scores: np.ndarray, bins: int = 5) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(scores)
    value = 0.0
    for index in range(bins):
        mask = (scores >= edges[index]) & (scores <= edges[index + 1] if index == bins - 1 else scores < edges[index + 1])
        if np.any(mask):
            value += float(mask.mean()) * abs(float(scores[mask].mean()) - float(targets[mask].mean()))
    return value


def metric_bundle(targets: np.ndarray, scores: np.ndarray) -> dict[str, float | int | None]:
    result: dict[str, float | int | None] = {
        "images": int(len(targets)),
        "positives": int(targets.sum()),
        "prevalence": float(targets.mean()) if len(targets) else None,
        "brier": float(brier_score_loss(targets, scores)),
        "ece_5bin": ece(targets, scores, 5),
    }
    if np.unique(targets).size == 2:
        result["auroc"] = float(roc_auc_score(targets, scores))
        result["auprc"] = float(average_precision_score(targets, scores))
    else:
        result["auroc"] = None
        result["auprc"] = None
    return result


def bootstrap_ci(targets: np.ndarray, scores: np.ndarray, patients: list[str], seed: int, resamples: int) -> dict[str, float | int | None]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, patient in enumerate(patients):
        groups[str(patient)].append(index)
    patient_ids = sorted(groups)
    rng = np.random.default_rng(seed)
    aurocs: list[float] = []
    auprcs: list[float] = []
    for _ in range(resamples):
        selected = rng.integers(0, len(patient_ids), size=len(patient_ids))
        indices = np.concatenate([groups[patient_ids[index]] for index in selected])
        if np.unique(targets[indices]).size == 2:
            aurocs.append(float(roc_auc_score(targets[indices], scores[indices])))
            auprcs.append(float(average_precision_score(targets[indices], scores[indices])))
    if not aurocs:
        return {"patients": len(patient_ids), "resamples": resamples, "auroc_ci_lower": None, "auroc_ci_upper": None, "auprc_ci_lower": None, "auprc_ci_upper": None}
    return {"patients": len(patient_ids), "resamples": resamples, "auroc_ci_lower": float(np.quantile(aurocs, 0.025)), "auroc_ci_upper": float(np.quantile(aurocs, 0.975)), "auprc_ci_lower": float(np.quantile(auprcs, 0.025)), "auprc_ci_upper": float(np.quantile(auprcs, 0.975))}


def subgroup(rows: list[dict], key_fn) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(key_fn(row))].append(row)
    result = {}
    for key, items in sorted(groups.items()):
        targets = np.asarray([int(item["target"]) for item in items], dtype=np.int64)
        scores = np.asarray([float(item["parsed_response"]["referable_probability"]) for item in items], dtype=float)
        result[key] = metric_bundle(targets, scores)
    return result


def arm_diagnostics(rows: list[dict], include_cards: bool, seed: int, resamples: int) -> dict:
    targets = np.asarray([int(item["target"]) for item in rows], dtype=np.int64)
    scores = np.asarray([float(item["parsed_response"]["referable_probability"]) for item in rows], dtype=float)
    grades = np.asarray([int(item["grade_used_after_response"]) for item in rows], dtype=np.int64)
    predicted = np.asarray([int(item["parsed_response"].get("predicted_grade", -1)) for item in rows], dtype=np.int64)
    patients = [str(item.get("patient_id", item["relative_path"])) for item in rows]
    output = {
        "overall": metric_bundle(targets, scores),
        "patient_bootstrap": bootstrap_ci(targets, scores, patients, seed, resamples),
        "by_reference_grade": subgroup(rows, lambda item: item["grade_used_after_response"]),
        "by_image_quality": subgroup(rows, lambda item: item["parsed_response"].get("image_quality", "missing")),
        "by_uncertainty": subgroup(rows, lambda item: item["parsed_response"].get("uncertainty", "missing")),
        "mean_predicted_grade": float(predicted.mean()),
        "mean_absolute_grade_error": float(np.abs(predicted - grades).mean()),
    }
    if include_cards:
        output["by_grounded_card_count"] = subgroup(rows, lambda item: item.get("cards_available", 0))
        output["grounded_card_count"] = {
            "mean": float(np.mean([item.get("cards_available", 0) for item in rows])),
            "median": float(np.median([item.get("cards_available", 0) for item in rows])),
            "zero_count": int(sum(item.get("cards_available", 0) == 0 for item in rows)),
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibration and subgroup diagnostics for paired VLM ablation outputs.")
    parser.add_argument("--image-only", required=True)
    parser.add_argument("--feature-card", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=4106)
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()
    image_path = require_external(Path(args.image_only), "image-only predictions")
    card_path = require_external(Path(args.feature_card), "feature-card predictions")
    output = require_external(Path(args.output), "diagnostics output")
    output.mkdir(parents=True, exist_ok=True)
    image_rows, card_rows = read(image_path, "image-only"), read(card_path, "feature-card")
    if set(image_rows) != set(card_rows):
        raise RuntimeError("VLM arms do not contain exactly the same image set")
    paths = sorted(image_rows)
    if any(image_rows[path]["target"] != card_rows[path]["target"] for path in paths):
        raise RuntimeError("VLM arms disagree on post-response targets")
    image_ordered = [image_rows[path] for path in paths]
    card_ordered = [card_rows[path] for path in paths]
    report = {
        "schema_version": "1.0",
        "status": "complete_vlm_ablation_diagnostics",
        "images": len(paths),
        "labels_hidden_from_prompt": True,
        "image_only": arm_diagnostics(image_ordered, False, args.seed, args.resamples),
        "feature_card": arm_diagnostics(card_ordered, True, args.seed + 1, args.resamples),
        "image_only_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "feature_card_sha256": hashlib.sha256(card_path.read_bytes()).hexdigest(),
        "caveat": "Balanced internal pilot diagnostics; no subgroup or calibration value is a clinical validation estimate.",
    }
    path = output / "vlm_ablation_diagnostics.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
