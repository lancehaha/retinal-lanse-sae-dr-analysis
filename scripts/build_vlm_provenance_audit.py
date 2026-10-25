from __future__ import annotations

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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Record provenance acceptance/rejection for matched VLM arms.")
    parser.add_argument("--rejected-summary", required=True)
    parser.add_argument("--accepted-summary", required=True)
    parser.add_argument("--atlas-summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rejected = require_external(Path(args.rejected_summary), "rejected summary")
    accepted = require_external(Path(args.accepted_summary), "accepted summary")
    atlas = require_external(Path(args.atlas_summary), "atlas summary")
    output = require_external(Path(args.output), "audit output")
    rejected_payload = json.loads(rejected.read_text(encoding="utf-8"))
    accepted_payload = json.loads(accepted.read_text(encoding="utf-8"))
    atlas_payload = json.loads(atlas.read_text(encoding="utf-8"))
    rejected_cards = float(rejected_payload.get("mean_grounded_cards_per_image", -1))
    atlas_dictionary = int(atlas_payload["dictionary_size"])
    accepted_grounding = int(accepted_payload["grounding_cards_available"])
    if rejected_cards != 0:
        raise RuntimeError("the proposed rejected run does not have zero grounded cards")
    if atlas_dictionary != 16384 or accepted_grounding != 16:
        raise RuntimeError("accepted run or atlas provenance is not the expected 16,384-feature setup")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "status": "provenance_audit_complete",
        "accepted_run": {
            "summary": str(accepted),
            "summary_sha256": sha256(accepted),
            "dictionary_size": atlas_dictionary,
            "grounding_cards_available": accepted_grounding,
            "decision": "accepted_for_matched_analysis",
        },
        "rejected_run": {
            "summary": str(rejected),
            "summary_sha256": sha256(rejected),
            "mean_grounded_cards_per_image": rejected_cards,
            "decision": "rejected_checkpoint_feature_atlas_mismatch",
            "reason": "The run used a 4,096-feature checkpoint while the grounded atlas was generated from a 16,384-feature checkpoint; no active feature IDs could map to grounded cards.",
        },
        "atlas_summary": str(atlas),
        "atlas_summary_sha256": sha256(atlas),
        "labels_hidden_from_prompt": bool(accepted_payload["labels_hidden_from_prompt"]),
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
