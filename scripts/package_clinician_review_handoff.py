from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

EXTERNAL_ROOT = Path("/Volumes/Elements/Retina-LanSE")
PACKET_ROOT = EXTERNAL_ROOT / "experiments/clinician_review_packet_qwen25vl_expanded16"
TEMPLATE_ROOT = EXTERNAL_ROOT / "experiments/clinician_review_templates_qwen25vl_expanded16"
DATA_ROOT = EXTERNAL_ROOT / "datasets/eyepacs"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Package the blinded clinician-review handoff on the external drive.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest-output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    manifest_output = Path(args.manifest_output).resolve()
    for label, path in (("output", output), ("manifest", manifest_output)):
        try:
            path.relative_to(EXTERNAL_ROOT)
        except ValueError as exc:
            raise RuntimeError(f"{label} must stay under {EXTERNAL_ROOT}: {path}") from exc

    packet_path = PACKET_ROOT / "blinded_review_packet.jsonl"
    rows = [json.loads(line) for line in packet_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    relative_paths = sorted({row["relative_path"] for row in rows})
    image_sources: list[tuple[str, Path]] = []
    for relative_path in relative_paths:
        source = DATA_ROOT / relative_path
        if not source.is_file():
            raise RuntimeError(f"missing review image: {source}")
        image_sources.append((f"images/{relative_path}", source))

    files: list[tuple[str, Path]] = [
        ("REVIEW_INSTRUCTIONS.md", PACKET_ROOT / "REVIEW_INSTRUCTIONS.md"),
        ("blinded_review_packet.jsonl", packet_path),
        ("README_REVIEWERS.md", TEMPLATE_ROOT / "README_REVIEWERS.md"),
        ("reviewer_forms/reviewer_01.jsonl", TEMPLATE_ROOT / "reviewer_01.jsonl"),
        ("reviewer_forms/reviewer_02.jsonl", TEMPLATE_ROOT / "reviewer_02.jsonl"),
        ("reviewer_forms/reviewer_03.jsonl", TEMPLATE_ROOT / "reviewer_03.jsonl"),
    ]
    for archive_name, source in files:
        if not source.is_file():
            raise RuntimeError(f"missing handoff file: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    readme = (
        "# Clinician review handoff\n\n"
        "This archive contains a blinded, non-diagnostic review packet for independent ophthalmology review.\n"
        "It includes 16 SAE feature cards, 64 retinal images, and three blank reviewer forms.\n"
        "Do not add labels, patient identifiers, activation values, or analysis keys to returned forms.\n"
        "Return one completed JSONL form per reviewer; the scoring runner will validate completeness and agreement.\n"
    ).encode("utf-8")
    contents: dict[str, str] = {}
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr("HANDOFF_README.md", readme)
        contents["HANDOFF_README.md"] = sha256_bytes(readme)
        for archive_name, source in files:
            archive.write(source, archive_name)
            contents[archive_name] = sha256_file(source)
        for archive_name, source in image_sources:
            archive.write(source, archive_name)
            contents[archive_name] = sha256_file(source)
    manifest = {
        "schema_version": "1.0",
        "status": "complete_blinded_clinician_review_handoff",
        "archive": str(output),
        "archive_sha256": sha256_file(output),
        "cards": len({row["feature_id"] for row in rows}),
        "review_rows": len(rows),
        "unique_images": len(relative_paths),
        "reviewer_forms": 3,
        "labels_included": False,
        "activation_values_included": False,
        "analysis_key_included": False,
        "contents_sha256": contents,
        "caveat": "Packaging enables independent review; it is not clinician validation until completed and adjudicated.",
    }
    manifest_output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
