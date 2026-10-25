from __future__ import annotations

"""Create separate blinded JSONL forms for independent ophthalmology reviewers."""

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


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare independent clinician review forms.")
    parser.add_argument("--packet", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reviewers", default="reviewer_01,reviewer_02,reviewer_03")
    args = parser.parse_args()
    packet_path = require_external(Path(args.packet), "blinded packet")
    output = require_external(Path(args.output), "template output")
    output.mkdir(parents=True, exist_ok=True)
    packet = read_jsonl(packet_path)
    if not packet:
        raise RuntimeError("blinded packet is empty")
    required = {"review_id", "feature_id", "hypothesis", "relative_path"}
    if any(not required.issubset(row) for row in packet):
        raise RuntimeError("packet is missing required blinded fields")
    forbidden = {"target", "grade", "activation", "role", "patient_id"}
    if any(forbidden.intersection(row) for row in packet):
        raise RuntimeError("packet contains forbidden label/key fields")
    reviewers = [item.strip() for item in args.reviewers.split(",") if item.strip()]
    if len(set(reviewers)) != len(reviewers) or len(reviewers) < 3:
        raise RuntimeError("at least three unique reviewer IDs are required")
    ids = [row["review_id"] for row in packet]
    if len(set(ids)) != len(ids):
        raise RuntimeError("packet review IDs are not unique")
    paths = []
    for reviewer in reviewers:
        rows = []
        for row in packet:
            rows.append({
                "reviewer_id": reviewer,
                "review_id": row["review_id"],
                "feature_id": row["feature_id"],
                "hypothesis": row["hypothesis"],
                "image_index": row["image_index"],
                "relative_path": row["relative_path"],
                "review_match_score_0_to_3": None,
                "review_specificity_0_to_3": None,
                "review_nuisance_flag": None,
                "reviewer_comment": None,
            })
        path = output / f"{reviewer}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        paths.append(str(path))
    instructions = output / "README_REVIEWERS.md"
    instructions.write_text(
        "# Independent review forms\n\n"
        "Each reviewer must edit only their own JSONL file. Fill the two 0--3\n"
        "scores, nuisance flag, and concise comment for every row. Do not add\n"
        "DR labels, patient identifiers, activation values, or analysis keys.\n"
        "Return the completed file unchanged in schema. The scoring runner will\n"
        "reject incomplete or mismatched forms.\n",
        encoding="utf-8",
    )
    summary = {
        "schema_version": "1.0",
        "status": "complete_blinded_reviewer_templates",
        "reviewers": reviewers,
        "images_per_reviewer": len(packet),
        "packet_sha256": hashlib.sha256(packet_path.read_bytes()).hexdigest(),
        "template_paths": paths,
        "labels_in_templates": False,
        "activation_values_in_templates": False,
        "analysis_key_in_templates": False,
        "caveat": "Templates enable independent review but contain no completed clinician judgments.",
    }
    (output / "review_template_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
