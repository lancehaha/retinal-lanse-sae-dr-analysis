from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

EXTERNAL_ROOT = Path("/Volumes/Elements/Retina-LanSE")


def require_external(path: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(EXTERNAL_ROOT)
    except ValueError as exc:
        raise RuntimeError(f"{label} must stay under {EXTERNAL_ROOT}: {resolved}") from exc
    return resolved


def read(path: Path, key: str) -> dict[str, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if item.get("status") == "complete":
                rows[item["relative_path"]] = item
    if not rows:
        raise RuntimeError(f"no complete rows in {key}")
    return rows


def paired_ci(targets: np.ndarray, left: np.ndarray, right: np.ndarray, patients: list[str], metric, seed: int, resamples: int) -> dict:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, patient in enumerate(patients):
        groups[str(patient)].append(index)
    patient_ids = sorted(groups)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(resamples):
        selected = rng.integers(0, len(patient_ids), size=len(patient_ids))
        indices = np.concatenate([groups[patient_ids[index]] for index in selected])
        if np.unique(targets[indices]).size == 2:
            values.append(float(metric(targets[indices], left[indices]) - metric(targets[indices], right[indices])))
    lower, upper = np.quantile(values, [0.025, 0.975])
    observed = float(metric(targets, left) - metric(targets, right))
    return {"difference_left_minus_right": observed, "ci_lower": float(lower), "ci_upper": float(upper), "confidence": 0.95, "resamples": resamples, "patients": len(patient_ids)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Paired patient-bootstrap comparison of two blinded VLM arms.")
    parser.add_argument("--image-only", required=True)
    parser.add_argument("--feature-card", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=4106)
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()
    left_path = require_external(Path(args.image_only), "image-only predictions")
    right_path = require_external(Path(args.feature_card), "feature-card predictions")
    output = require_external(Path(args.output), "comparison output")
    output.mkdir(parents=True, exist_ok=True)
    left, right = read(left_path, "image-only"), read(right_path, "feature-card")
    paths = sorted(set(left) & set(right))
    if set(left) != set(right):
        raise RuntimeError("VLM arms do not contain exactly the same image set")
    if any(left[p]["target"] != right[p]["target"] for p in paths):
        raise RuntimeError("VLM arms disagree on post-response targets")
    targets = np.asarray([left[p]["target"] for p in paths], dtype=np.int64)
    image_scores = np.asarray([left[p]["parsed_response"]["referable_probability"] for p in paths], dtype=float)
    card_scores = np.asarray([right[p]["parsed_response"]["referable_probability"] for p in paths], dtype=float)
    patients = [left[p].get("patient_id", p) for p in paths]
    report = {
        "schema_version": "1.0",
        "status": "complete_paired_vlm_ablation_comparison",
        "images": len(paths),
        "patients": len(set(patients)),
        "positive_images": int(targets.sum()),
        "image_only": {"auroc": float(roc_auc_score(targets, image_scores)), "auprc": float(average_precision_score(targets, image_scores))},
        "feature_card": {"auroc": float(roc_auc_score(targets, card_scores)), "auprc": float(average_precision_score(targets, card_scores))},
        "paired_difference_feature_card_minus_image_only": {
            "auroc": paired_ci(targets, card_scores, image_scores, patients, roc_auc_score, args.seed, args.resamples),
            "auprc": paired_ci(targets, card_scores, image_scores, patients, average_precision_score, args.seed + 1, args.resamples),
        },
        "labels_hidden_from_prompt": True,
        "image_only_sha256": hashlib.sha256(left_path.read_bytes()).hexdigest(),
        "feature_card_sha256": hashlib.sha256(right_path.read_bytes()).hexdigest(),
        "caveat": "This is a small balanced pilot; the paired interval is not a clinical validation estimate.",
    }
    path = output / "paired_vlm_ablation_comparison.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
