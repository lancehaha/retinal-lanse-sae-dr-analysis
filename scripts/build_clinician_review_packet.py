from __future__ import annotations

"""Create a label-blinded clinician review packet for grounded SAE features."""

import argparse
import hashlib
import json
import random
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
    parser = argparse.ArgumentParser(description="Build a blinded review packet for SAE feature hypotheses.")
    parser.add_argument("--atlas", required=True)
    parser.add_argument("--grounding", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-cards", type=int, default=16)
    parser.add_argument("--examples-per-card", type=int, default=2)
    parser.add_argument("--controls-per-card", type=int, default=2)
    parser.add_argument("--seed", type=int, default=4106)
    args = parser.parse_args()

    atlas_path = require_external(Path(args.atlas), "feature atlas")
    grounding_path = require_external(Path(args.grounding), "grounding JSONL")
    manifest_path = require_external(Path(args.manifest), "manifest")
    output = require_external(Path(args.output), "review packet")
    output.mkdir(parents=True, exist_ok=True)

    atlas = {int(x["feature_id"]): x for x in read_jsonl(atlas_path)}
    grounding = {}
    for item in read_jsonl(grounding_path):
        parsed = item.get("parsed_response")
        if item.get("status") == "complete" and isinstance(parsed, dict) and parsed.get("hypothesis"):
            grounding[int(item["feature_id"])] = parsed
    cards = [atlas[feature_id] for feature_id in grounding if feature_id in atlas][: args.max_cards]
    if not cards:
        raise RuntimeError("no parseable grounded cards")

    manifest = read_jsonl(manifest_path)
    all_paths = sorted({item["relative_path"] for item in manifest})
    used: set[str] = set()
    rng = random.Random(args.seed)
    packet: list[dict] = []
    key: list[dict] = []
    for card in cards:
        feature_id = int(card["feature_id"])
        high = [x["relative_path"] for x in card["top_high_activation_examples"] if x["relative_path"] in all_paths]
        high = [x for x in high if x not in used][: args.examples_per_card]
        used.update(high)
        available = [x for x in all_paths if x not in used]
        if len(available) < args.controls_per_card:
            raise RuntimeError("not enough deterministic control images")
        controls = rng.sample(available, args.controls_per_card)
        used.update(controls)
        images = high + controls
        rng.shuffle(images)
        for image_index, relative_path in enumerate(images, start=1):
            review_id = f"feature_{feature_id:05d}_image_{image_index:02d}"
            packet.append({
                "review_id": review_id,
                "feature_id": feature_id,
                "hypothesis": grounding[feature_id]["hypothesis"],
                "image_index": image_index,
                "relative_path": relative_path,
                "review_match_score_0_to_3": None,
                "review_specificity_0_to_3": None,
                "review_nuisance_flag": None,
                "reviewer_comment": None,
            })
            key.append({"review_id": review_id, "role": "high_activation" if relative_path in high else "control", "activation": next((x["activation"] for x in card["top_high_activation_examples"] if x["relative_path"] == relative_path), None)})

    packet_path = output / "blinded_review_packet.jsonl"
    with packet_path.open("w", encoding="utf-8") as handle:
        for row in packet:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    key_path = output / "analysis_key.jsonl"
    with key_path.open("w", encoding="utf-8") as handle:
        for row in key:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    instructions = output / "REVIEW_INSTRUCTIONS.md"
    instructions.write_text(
        "# Blinded SAE feature review\n\n"
        "Review the linked retinal image for the feature hypothesis shown. Do not use\n"
        "external DR labels or infer the hidden role of an image. For each image, fill:\n\n"
        "- `review_match_score_0_to_3`: 0 absent, 1 weak, 2 moderate, 3 clear shared pattern.\n"
        "- `review_specificity_0_to_3`: 0 nonspecific/nuisance, 3 specific visible structure.\n"
        "- `review_nuisance_flag`: blur, illumination, camera/border, artifact, or none.\n"
        "- `reviewer_comment`: concise evidence or alternative explanation.\n\n"
        "The packet is not a diagnostic task. A feature becomes a clinically meaningful\n"
        "concept only after independent review by at least three ophthalmology reviewers.\n",
        encoding="utf-8",
    )
    summary = {
        "schema_version": "1.0",
        "status": "complete_blinded_review_packet",
        "cards": len(cards),
        "images": len(packet),
        "examples_per_card": args.examples_per_card,
        "controls_per_card": args.controls_per_card,
        "labels_in_packet": False,
        "activation_values_in_packet": False,
        "packet_jsonl": str(packet_path),
        "analysis_key": str(key_path),
        "atlas_sha256": hashlib.sha256(atlas_path.read_bytes()).hexdigest(),
        "grounding_sha256": hashlib.sha256(grounding_path.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "caveat": "This packet enables review but does not constitute clinician validation until completed and adjudicated.",
    }
    (output / "review_packet_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
