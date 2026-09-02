from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path


EXTERNAL_ROOT = Path("/Volumes/Elements/Retina-LanSE")
EXPECTED_IMAGE_COUNTS = {"train": 35_126, "test": 53_576}


def require_external(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(EXTERNAL_ROOT)
    except ValueError as exc:
        raise RuntimeError(f"publication-scale path must stay under {EXTERNAL_ROOT}: {resolved}") from exc
    return resolved


def sha256_path(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def validate_inventory(archive_dir: Path, manifest_path: Path) -> list[dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validated: list[dict] = []
    for item in manifest["files"]:
        path = archive_dir / item["name"]
        if not path.is_file():
            raise RuntimeError(f"missing locked EyePACS file: {path}")
        observed = path.stat().st_size
        expected = int(item["expected_bytes"])
        if observed != expected:
            raise RuntimeError(f"size mismatch for {path.name}: expected {expected}, observed {observed}")
        validated.append({"name": path.name, "bytes": observed})
    return validated


def reconstruct(parts: list[Path], destination: Path) -> dict:
    expected_bytes = sum(path.stat().st_size for path in parts)
    if destination.is_file() and destination.stat().st_size == expected_bytes:
        return {
            "path": str(destination),
            "bytes": expected_bytes,
            "sha256": sha256_path(destination),
            "reused": True,
        }

    partial = destination.with_suffix(destination.suffix + ".partial")
    completed_part_bytes = 0
    if partial.exists():
        partial_bytes = partial.stat().st_size
        cumulative = 0
        resume_index = None
        for index, part in enumerate(parts):
            if cumulative == partial_bytes:
                resume_index = index
                break
            cumulative += part.stat().st_size
        if cumulative == partial_bytes:
            resume_index = len(parts)
        if resume_index is None:
            raise RuntimeError(
                f"partial archive ends mid-part ({partial_bytes} bytes); move it aside and rerun: {partial}"
            )
        completed_part_bytes = partial_bytes
    else:
        resume_index = 0

    destination.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("ab") as output:
        for part in parts[resume_index:]:
            print(f"Appending {part.name} ({part.stat().st_size} bytes)", flush=True)
            with part.open("rb") as source:
                shutil.copyfileobj(source, output, length=16 * 1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
            completed_part_bytes += part.stat().st_size
            print(f"Reconstructed {completed_part_bytes}/{expected_bytes} bytes", flush=True)

    if partial.stat().st_size != expected_bytes:
        raise RuntimeError(f"reconstructed size mismatch for {partial}")
    partial.replace(destination)
    return {
        "path": str(destination),
        "bytes": expected_bytes,
        "sha256": sha256_path(destination),
        "reused": False,
    }


def safe_extract(archive: Path, dataset_root: Path) -> dict:
    with zipfile.ZipFile(archive) as handle:
        members = handle.infolist()
        for member in members:
            target = (dataset_root / member.filename).resolve()
            try:
                target.relative_to(dataset_root)
            except ValueError as exc:
                raise RuntimeError(f"unsafe ZIP member in {archive}: {member.filename}") from exc
        print(f"Extracting {len(members)} members from {archive.name}", flush=True)
        handle.extractall(dataset_root)
    return {"archive": str(archive), "members": len(members)}


def count_images(dataset_root: Path, partition: str) -> int:
    root = dataset_root / partition
    return sum(
        1
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpeg", ".jpg", ".png"}
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate, reconstruct, and safely extract the official multipart EyePACS archives."
    )
    parser.add_argument("--archive-dir", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest", default="research/eyepacs_kaggle_manifest.json")
    args = parser.parse_args()

    archive_dir = require_external(Path(args.archive_dir))
    dataset_root = require_external(Path(args.dataset_root))
    manifest_path = Path(args.manifest).resolve()
    if not archive_dir.is_dir():
        parser.error(f"archive directory does not exist: {archive_dir}")

    inventory = validate_inventory(archive_dir, manifest_path)
    dataset_root.mkdir(parents=True, exist_ok=True)
    reconstructed_dir = archive_dir / "reconstructed"

    reconstructed: dict[str, dict] = {}
    extracted: dict[str, dict] = {}
    for partition in ("train", "test"):
        parts = sorted(archive_dir.glob(f"{partition}.zip.[0-9][0-9][0-9]"))
        if not parts:
            raise RuntimeError(f"no multipart files found for {partition}")
        combined = reconstructed_dir / f"{partition}.zip"
        reconstructed[partition] = reconstruct(parts, combined)
        extracted[partition] = safe_extract(combined, dataset_root)

    labels_archive = archive_dir / "trainLabels.csv.zip"
    extracted["labels"] = safe_extract(labels_archive, dataset_root)

    observed_counts = {
        partition: count_images(dataset_root, partition)
        for partition in EXPECTED_IMAGE_COUNTS
    }
    if observed_counts != EXPECTED_IMAGE_COUNTS:
        raise RuntimeError(
            f"EyePACS image-count gate failed: expected {EXPECTED_IMAGE_COUNTS}, observed {observed_counts}"
        )

    labels = dataset_root / "trainLabels.csv"
    if not labels.is_file():
        raise RuntimeError(f"missing extracted label table: {labels}")

    evidence = {
        "schema_version": "1.0",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Kaggle competition diabetic-retinopathy-detection",
        "archive_dir": str(archive_dir),
        "dataset_root": str(dataset_root),
        "locked_inventory": inventory,
        "reconstructed_archives": reconstructed,
        "extraction": extracted,
        "image_counts": observed_counts,
        "total_images": sum(observed_counts.values()),
        "labels_bytes": labels.stat().st_size,
        "storage_policy": "external_only",
    }
    evidence_path = dataset_root / "preparation_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
