from __future__ import annotations

"""Validate independent clinician forms and compute agreement statistics."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

EXTERNAL_ROOT = Path("/Volumes/Elements/Retina-LanSE")
NUISANCE = {"blur", "illumination", "camera/border", "artifact", "none"}


def require_external(path: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(EXTERNAL_ROOT)
    except ValueError as exc:
        raise RuntimeError(f"{label} must stay under {EXTERNAL_ROOT}: {resolved}") from exc
    return resolved


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fleiss_kappa(ratings: list[list[int]], categories: tuple[int, ...] = (0, 1, 2, 3)) -> float | None:
    if not ratings or len(ratings[0]) < 2:
        return None
    n = len(ratings[0])
    counts = []
    for row in ratings:
        counts.append([sum(value == category for value in row) for category in categories])
    total = len(ratings) * n
    proportions = [sum(row[index] for row in counts) / total for index in range(len(categories))]
    expected = sum(value * value for value in proportions)
    observed = sum((sum(value * value for value in row) - n) / (n * (n - 1)) for row in counts) / len(counts)
    if expected >= 1.0:
        return None
    return float((observed - expected) / (1.0 - expected))


def pairwise_exact_agreement(ratings: list[list[int]]) -> float | None:
    if not ratings or len(ratings[0]) < 2:
        return None
    comparisons = 0
    matches = 0
    for row in ratings:
        for left in range(len(row)):
            for right in range(left + 1, len(row)):
                comparisons += 1
                matches += int(row[left] == row[right])
    return float(matches / comparisons) if comparisons else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Score independent clinician review forms.")
    parser.add_argument("--review-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-reviewers", type=int, default=3)
    args = parser.parse_args()
    review_dir = require_external(Path(args.review_dir), "review directory")
    output = require_external(Path(args.output), "review scoring output")
    output.mkdir(parents=True, exist_ok=True)
    files = sorted(review_dir.glob("reviewer_*.jsonl"))
    if len(files) < args.min_reviewers:
        raise RuntimeError(f"found {len(files)} reviewer files; need at least {args.min_reviewers}")
    forms = [read_jsonl(path) for path in files]
    ids = [[row.get("review_id") for row in rows] for rows in forms]
    if any(len(set(row_ids)) != len(row_ids) for row_ids in ids):
        raise RuntimeError("review IDs are not unique within a reviewer form")
    reference = ids[0]
    if any(row_ids != reference for row_ids in ids[1:]):
        raise RuntimeError("reviewer forms do not contain the same ordered review IDs")
    forbidden = {"target", "grade", "activation", "role", "patient_id"}
    for rows in forms:
        for row in rows:
            if forbidden.intersection(row):
                raise RuntimeError("review form contains forbidden label/key fields")
            for field in ("review_match_score_0_to_3", "review_specificity_0_to_3"):
                value = row.get(field)
                if value is not None and (not isinstance(value, int) or not 0 <= value <= 3):
                    raise RuntimeError(f"invalid {field} in {row['review_id']}")
            nuisance = row.get("review_nuisance_flag")
            if nuisance is not None and nuisance not in NUISANCE:
                raise RuntimeError(f"invalid nuisance flag in {row['review_id']}: {nuisance}")
    complete = all(
        row.get("review_match_score_0_to_3") is not None
        and row.get("review_specificity_0_to_3") is not None
        and row.get("review_nuisance_flag") is not None
        for rows in forms for row in rows
    )
    report: dict = {
        "schema_version": "1.0",
        "status": "complete_clinician_review_scoring" if complete else "pending_clinician_reviews",
        "reviewers": [rows[0].get("reviewer_id") for rows in forms],
        "reviewer_count": len(forms),
        "items": len(reference),
        "complete_rows": sum(
            row.get("review_match_score_0_to_3") is not None
            and row.get("review_specificity_0_to_3") is not None
            and row.get("review_nuisance_flag") is not None
            for rows in forms for row in rows
        ),
        "expected_rows": len(reference) * len(forms),
        "complete": complete,
    }
    if complete:
        match = [[forms[reviewer][item]["review_match_score_0_to_3"] for reviewer in range(len(forms))] for item in range(len(reference))]
        specificity = [[forms[reviewer][item]["review_specificity_0_to_3"] for reviewer in range(len(forms))] for item in range(len(reference))]
        report["agreement"] = {
            "match_fleiss_kappa": fleiss_kappa(match),
            "specificity_fleiss_kappa": fleiss_kappa(specificity),
            "match_pairwise_exact_agreement": pairwise_exact_agreement(match),
            "specificity_pairwise_exact_agreement": pairwise_exact_agreement(specificity),
        }
        by_feature: dict[str, list[int]] = defaultdict(list)
        for reviewer_rows in forms:
            for row in reviewer_rows:
                by_feature[str(row["feature_id"])].append(row["review_match_score_0_to_3"])
        report["feature_match_means"] = {feature: sum(values) / len(values) for feature, values in sorted(by_feature.items())}
    else:
        report["agreement"] = None
        report["next_action"] = "Return all reviewer JSONL files with every score and nuisance flag filled."
    path = output / "clinician_review_scoring.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
