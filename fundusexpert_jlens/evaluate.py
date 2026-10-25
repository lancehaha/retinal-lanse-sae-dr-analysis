"""Locked-test evaluation and available-ID cluster bootstrap.

DDR does not provide patient identifiers in the public archive. In that
setting the bootstrap unit is the image, which must be reported explicitly.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .protocol import audit_manifest, manifest_sha256, parse_report, read_jsonl


def score(rows: list[dict]) -> dict:
    verified = [r for r in rows if r["label"] is not None]
    negatives = [r for r in verified if r["label"] == 0]
    positives = [r for r in verified if r["label"] == 1]
    false_claims = sum(r["finding"] == "present" for r in negatives)
    true_claims = sum(r["finding"] == "present" for r in positives)
    claims = sum(r["finding"] == "present" for r in verified)
    graded = [r for r in rows if r["grade"] in {"0", "1", "2", "3", "4"}]
    located = [r for r in positives if r["location_truth"] is not None and r["finding"] == "present"]
    return {
        "images": len(rows), "verified_negatives": len(negatives), "verified_positives": len(positives),
        "unsupported_claims": false_claims, "unsupported_claims_per_image": false_claims / len(rows) if rows else None,
        "hard_exudate_recall": true_claims / len(positives) if positives else None,
        "hard_exudate_precision": true_claims / claims if claims else None,
        "lesion_claim_count": claims, "invalid_report_count": sum(not r["valid"] for r in rows),
        "invalid_report_rate": sum(not r["valid"] for r in rows) / len(rows) if rows else None,
        "grade_accuracy": sum(r["grade_pred"] == r["grade"] for r in graded) / len(graded) if graded else None,
        "location_exact_agreement": sum(r["location_pred"] == r["location_truth"] for r in located) / len(located) if located else None,
    }


def load_arm(path: str, truth: dict[str, dict]) -> list[dict]:
    predictions = read_jsonl(path)
    if len(predictions) != len(truth) or {p["image_id"] for p in predictions} != set(truth):
        raise ValueError(f"Prediction IDs must exactly equal locked test IDs: {path}")
    rows = []
    for pred in predictions:
        item = truth[pred["image_id"]]
        parsed = parse_report(pred["report"])
        label = 1 if item["hard_exudates"] == "present" else 0 if item["hard_exudates"] == "absent" else None
        rows.append({"image_id": pred["image_id"],
                     "patient": item.get("patient_id") or item.get("duplicate_group") or item["image_id"],
                     "label": label, "grade": str(item.get("dr_grade", "unknown")),
                     "location_truth": item.get("location"), "location_pred": parsed.get("location") if parsed else None,
                     "grade_pred": str(parsed.get("dr_grade")) if parsed else None,
                     "finding": parsed.get("hard_exudates") if parsed else None, "valid": parsed is not None,
                     "report": pred["report"]})
    return rows


def paired_bootstrap(a: list[dict], b: list[dict], metric: str, n: int = 2000, seed: int = 4106) -> dict:
    aa = {r["image_id"]: r for r in a}
    bb = {r["image_id"]: r for r in b}
    if set(aa) != set(bb):
        raise ValueError("Paired bootstrap requires identical images")
    groups = defaultdict(list)
    for row in a:
        groups[row["patient"]].append(row["image_id"])
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    left_full, right_full = score(a)[metric], score(b)[metric]
    if left_full is None or right_full is None:
        return {"difference": None, "ci95": None, "n_bootstrap": 0}
    observed = left_full - right_full
    draws = []
    for _ in range(n):
        ids = [i for key in rng.choice(keys, size=len(keys), replace=True) for i in groups[key]]
        left, right = score([aa[i] for i in ids])[metric], score([bb[i] for i in ids])[metric]
        if left is not None and right is not None:
            draws.append(left - right)
    return {"difference": observed, "ci95": np.quantile(draws, [0.025, 0.975]).tolist() if draws else None,
            "n_bootstrap": len(draws)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--lock", required=True)
    p.add_argument("--prediction", action="append", required=True,
                   help="arm:seed:/absolute/path/predictions.jsonl")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    lock = json.loads(Path(args.lock).read_text())
    if lock.get("manifest_sha256") != manifest_sha256(args.manifest) or not lock.get("frozen_before_test"):
        raise ValueError("Lock must be frozen against this manifest before test scoring")
    manifest = read_jsonl(args.manifest)
    audit_manifest(manifest)
    truth = {r["image_id"]: r for r in manifest if r["split"] == "test"}
    arms = {}
    for entry in args.prediction:
        arm, seed, path = entry.split(":", 2)
        key = f"{arm}:{seed}"
        if key in arms:
            raise ValueError(f"Duplicate prediction: {key}")
        arms[key] = load_arm(path, truth)
    result = {"lock": lock, "metrics": {key: score(rows) for key, rows in arms.items()}, "comparisons": {}}
    for seed in ("4106", "4107", "4108"):
        d = arms.get(f"D:{seed}")
        if d is None:
            continue
        for other in ("B", "C"):
            base = arms.get(f"{other}:{seed}")
            if base is None:
                continue
            result["comparisons"][f"D-{other}:{seed}"] = {
                metric: paired_bootstrap(d, base, metric)
                for metric in ("unsupported_claims_per_image", "hard_exudate_recall", "grade_accuracy",
                               "invalid_report_rate")}
    decisions = {}
    for key, comparison in result["comparisons"].items():
        claim_ci = comparison["unsupported_claims_per_image"]["ci95"]
        recall_ci = comparison["hard_exudate_recall"]["ci95"]
        grade_ci = comparison["grade_accuracy"]["ci95"]
        decisions[key] = bool(claim_ci and recall_ci and grade_ci
                              and claim_ci[1] < 0
                              and recall_ci[0] >= -lock["recall_margin"]
                              and grade_ci[0] >= -lock["grade_margin"]
                              and comparison["invalid_report_rate"]["difference"] <= 0)
    result["decision"] = {"per_comparison": decisions,
                          "j_lens_specific_benefit": len(decisions) == 6 and all(decisions.values())}
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
