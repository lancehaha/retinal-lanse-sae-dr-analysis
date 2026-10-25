from __future__ import annotations

"""Join blinded VLM feature hypotheses to a post-hoc label-side audit.

The VLM response is generated before this script reads the DR labels.  The audit
does not validate clinical meaning; it reports whether the feature's activation
is associated with referable DR in the labeled training cohort and flags an
unparsed/uncertain language response for human review.
"""

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


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit grounded feature cards against held-out labels.")
    parser.add_argument("--atlas", required=True)
    parser.add_argument("--grounding", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    atlas_path = require_external(Path(args.atlas), "feature atlas")
    grounding_path = require_external(Path(args.grounding), "grounding JSONL")
    output = require_external(Path(args.output), "verification output")
    output.mkdir(parents=True, exist_ok=True)

    atlas = {}
    for line in atlas_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            atlas[int(item["feature_id"])] = item
    grounded = [json.loads(line) for line in grounding_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = []
    for item in grounded:
        card = atlas.get(int(item["feature_id"]))
        if card is None:
            rows.append({"feature_id": item["feature_id"], "status": "missing_atlas_card"})
            continue
        ratio = float(card["labeled_dr_activation_rate"]) / max(float(card["labeled_non_dr_activation_rate"]), 1e-12)
        log_ratio = float(card["labeled_log_rate_ratio"])
        direction = "positive" if log_ratio > 0.1 else "negative" if log_ratio < -0.1 else "near_neutral"
        parsed = item.get("parsed_response")
        rows.append({
            "feature_id": int(item["feature_id"]),
            "grounding_status": item.get("status"),
            "hypothesis": parsed.get("hypothesis") if isinstance(parsed, dict) else None,
            "evidence_strength": parsed.get("evidence_strength") if isinstance(parsed, dict) else None,
            "llm_uncertain": parsed.get("uncertain") if isinstance(parsed, dict) else None,
            "activation_count": int(card["activation_count"]),
            "labeled_dr_active_count": int(card["labeled_dr_active_count"]),
            "labeled_non_dr_active_count": int(card["labeled_non_dr_active_count"]),
            "labeled_dr_activation_rate": float(card["labeled_dr_activation_rate"]),
            "labeled_non_dr_activation_rate": float(card["labeled_non_dr_activation_rate"]),
            "activation_rate_ratio_dr_to_non_dr": ratio,
            "labeled_log_rate_ratio": log_ratio,
            "association_direction": direction,
            "requires_clinician_review": True,
        })
    report = {
        "schema_version": "1.0",
        "status": "complete_posthoc_label_side_audit",
        "grounding_labels_hidden_during_generation": True,
        "clinical_semantics_validated": False,
        "cards_audited": len(rows),
        "cards_with_parseable_hypothesis": sum(bool(row.get("hypothesis")) for row in rows),
        "positive_association_cards": sum(row.get("association_direction") == "positive" for row in rows),
        "negative_association_cards": sum(row.get("association_direction") == "negative" for row in rows),
        "near_neutral_cards": sum(row.get("association_direction") == "near_neutral" for row in rows),
        "atlas_sha256": hashlib.sha256(atlas_path.read_bytes()).hexdigest(),
        "grounding_sha256": hashlib.sha256(grounding_path.read_bytes()).hexdigest(),
        "rows": rows,
        "caveat": "Activation association is not semantic or causal validation; all cards remain subject to blinded ophthalmologist review.",
    }
    path = output / "feature_grounding_verification.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
