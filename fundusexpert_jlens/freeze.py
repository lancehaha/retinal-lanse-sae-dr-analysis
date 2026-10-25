"""Choose candidates on validation predictions and freeze the test protocol."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .evaluate import load_arm, score
from .protocol import audit_manifest, manifest_sha256, read_jsonl


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--candidate", action="append", required=True,
                   help="arm:seed:weight:prediction_path:adapter_path")
    p.add_argument("--recall-margin", type=float, required=True)
    p.add_argument("--grade-margin", type=float, required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if not (0 <= args.recall_margin <= 1 and 0 <= args.grade_margin <= 1):
        raise ValueError("Margins must be fractions from 0 to 1")
    rows = read_jsonl(args.manifest)
    audit_manifest(rows)
    truth = {r["image_id"]: r for r in rows if r["split"] == "validation"}
    candidates = []
    for entry in args.candidate:
        arm, seed, weight, prediction_path, adapter_path = entry.split(":", 4)
        metrics = score(load_arm(prediction_path, truth))
        candidates.append({"arm": arm, "seed": seed, "weight": float(weight),
                           "prediction_path": prediction_path, "adapter_path": adapter_path,
                           "validation": metrics})
    references = [c for c in candidates if c["arm"] == "F"]
    if len(references) != 1:
        raise ValueError("Supply exactly one F validation prediction as a recall/grade reference")
    reference = references[0]["validation"]

    def eligible(candidate: dict) -> bool:
        metrics = candidate["validation"]
        return (metrics["hard_exudate_recall"] is not None
                and reference["hard_exudate_recall"] is not None
                and metrics["hard_exudate_recall"] >= reference["hard_exudate_recall"] - args.recall_margin
                and metrics["grade_accuracy"] is not None
                and reference["grade_accuracy"] is not None
                and metrics["grade_accuracy"] >= reference["grade_accuracy"] - args.grade_margin)

    chosen = {}
    for arm in ("B", "C", "D"):
        for seed in ("4106", "4107", "4108"):
            subset = [c for c in candidates if c["arm"] == arm and c["seed"] == seed and eligible(c)]
            if not subset:
                raise ValueError(f"No validation candidate meets F recall/grade margins for {arm}:{seed}")
            chosen[f"{arm}:{seed}"] = min(
                subset, key=lambda c: (c["validation"]["unsupported_claims_per_image"],
                                       -(c["validation"]["hard_exudate_recall"] or 0),
                                       -(c["validation"]["grade_accuracy"] or 0)))
    eligibility = {}
    for seed in ("4106", "4107", "4108"):
        b, d = chosen[f"B:{seed}"]["validation"], chosen[f"D:{seed}"]["validation"]
        eligibility[seed] = {
            "recall_noninferior_on_validation": d["hard_exudate_recall"] is not None and b["hard_exudate_recall"] is not None
            and d["hard_exudate_recall"] >= b["hard_exudate_recall"] - args.recall_margin,
            "grade_noninferior_on_validation": d["grade_accuracy"] is not None and b["grade_accuracy"] is not None
            and d["grade_accuracy"] >= b["grade_accuracy"] - args.grade_margin,
        }
    lock = {"frozen_before_test": True, "created_utc": datetime.now(timezone.utc).isoformat(),
            "manifest_sha256": manifest_sha256(args.manifest), "primary_metric": "unsupported_claims_per_image",
            "parser": "fundusexpert_jlens.protocol.parse_report:v1", "seed_policy": [4106, 4107, 4108],
            "recall_margin": args.recall_margin, "grade_margin": args.grade_margin,
            "reference_F": references[0], "chosen": chosen, "validation_eligibility": eligibility}
    Path(args.out).write_text(json.dumps(lock, indent=2))
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
