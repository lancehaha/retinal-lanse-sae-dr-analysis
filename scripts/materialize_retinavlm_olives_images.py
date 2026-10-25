"""Extract only official expert-labeled OLIVES OCT scans after archive verification.

The official OLIVES.zip contains two nested ZIP archives. This script validates
its MD5, extracts those nested ZIPs, requires a unique archive member for every
expert-label path, then writes only the selected OCT files and an image audit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import zipfile
import zlib
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


OFFICIAL_MD5 = "e08f4198aa919d6900bd30525c137933"
OFFICIAL_LABELS_SHA256 = "8c63ef84541b747c91de3c754801012e0742e8672243a219bc49f809c78b9362"
INNER_BY_TRIAL = {"TREX DME": "TREX_DME.zip", "Prime_FULL": "Prime_FULL.zip"}


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file:
        while block := file.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def file_crc32(path: Path) -> int:
    checksum = 0
    with path.open("rb") as file:
        while block := file.read(8 * 1024 * 1024):
            checksum = zlib.crc32(block, checksum)
    return checksum


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def decoded_pixel_sha256(image: Image.Image) -> str:
    """Hash dimensions, mode, and decoded pixels, independent of TIFF encoding."""
    image.load()
    digest = hashlib.sha256()
    digest.update(f"{image.width}x{image.height}:{image.mode}\0".encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def pixel_extrema(image: Image.Image) -> tuple[float, float]:
    """Record decoded intensity range before the authors' integer /255 scaling."""
    extrema = image.getextrema()
    bands = extrema if extrema and isinstance(extrema[0], tuple) else (extrema,)
    return min(float(pair[0]) for pair in bands), max(float(pair[1]) for pair in bands)


def verify_label_audit(manifest: Path, audit_path: Path, expected_rows: int) -> str:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    manifest_hash = file_sha256(manifest)
    if (audit.get("status") != "labels_audited_images_unverified"
            or audit.get("source_sha256") != OFFICIAL_LABELS_SHA256
            or audit.get("manifest_sha256") != manifest_hash
            or audit.get("source_rows") != expected_rows
            or audit.get("valid_rows") != expected_rows
            or audit.get("split_seed") != 4106):
        raise ValueError("Official label manifest/audit binding failed")
    return manifest_hash


def safe_parts(path: str) -> tuple[str, ...]:
    parts = tuple(path.strip("/").split("/"))
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"Unsafe path: {path!r}")
    return parts


def suffix_key(path: str) -> str:
    parts = safe_parts(path)
    if len(parts) < 4:
        raise ValueError(f"Path has fewer than four components: {path!r}")
    return "/".join(parts[-4:]).casefold()


def extract_nested(outer: Path, out_dir: Path) -> dict[str, Path]:
    nested_dir = out_dir / "nested_archives"
    nested_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    with zipfile.ZipFile(outer) as archive:
        for trial, name in INNER_BY_TRIAL.items():
            candidates = [info for info in archive.infolist() if Path(info.filename).name == name]
            if len(candidates) != 1:
                raise ValueError(f"Expected one {name}, found {len(candidates)}")
            info = candidates[0]
            destination = nested_dir / name
            if destination.exists() and destination.stat().st_size == info.file_size:
                # A same-size file left by an interrupted run is not sufficient
                # evidence that the nested archive matches the official ZIP.
                if file_crc32(destination) == info.CRC:
                    result[trial] = destination
                    continue
            temporary = destination.with_suffix(".extracting")
            with archive.open(info) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
            if temporary.stat().st_size != info.file_size:
                raise OSError(f"Incomplete nested archive: {name}")
            os.replace(temporary, destination)
            result[trial] = destination
    return result


def load_labels(manifest: Path, expected_rows: int) -> list[dict[str, str]]:
    with manifest.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    required = {"source_image_path", "trial", "Patient_ID", "split", "fluid"}
    if len(rows) != expected_rows or not required.issubset(rows[0] if rows else {}):
        raise ValueError("Label manifest is incomplete or has unexpected row count")
    paths = [row["source_image_path"] for row in rows]
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate source paths in label manifest")
    return rows


def run(outer: Path, labels_manifest: Path, label_audit: Path,
        out_dir: Path, expected_rows: int) -> dict:
    if file_md5(outer) != OFFICIAL_MD5:
        raise ValueError("Official archive MD5 mismatch")
    label_manifest_sha256 = verify_label_audit(labels_manifest, label_audit, expected_rows)
    labels = load_labels(labels_manifest, expected_rows)
    nested = extract_nested(outer, out_dir)
    images_dir = out_dir / "expert_images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows_by_trial = defaultdict(list)
    for row in labels:
        if row["trial"] not in nested:
            raise ValueError(f"Unexpected trial: {row['trial']}")
        rows_by_trial[row["trial"]].append(row)

    selected = []
    for trial, trial_rows in rows_by_trial.items():
        with zipfile.ZipFile(nested[trial]) as archive:
            catalog = defaultdict(list)
            for info in archive.infolist():
                if not info.is_dir() and info.filename.casefold().endswith(
                    (".tif", ".tiff", ".jpg", ".jpeg", ".png")
                ):
                    catalog[suffix_key(info.filename)].append(info)
            pairs = []
            for row in trial_rows:
                candidates = catalog.get(suffix_key(row["source_image_path"]), [])
                if len(candidates) != 1:
                    raise ValueError(
                        f"Expected one image for {row['source_image_path']}, found {len(candidates)}"
                    )
                pairs.append((row, candidates[0]))
            # No image is materialized until every path in this trial matches.
            for row, info in pairs:
                relative = Path(*safe_parts(row["source_image_path"]))
                destination = images_dir / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if (destination.exists()
                        and destination.stat().st_size == info.file_size
                        and file_crc32(destination) == info.CRC):
                    image_sha256 = file_sha256(destination)
                else:
                    digest = hashlib.sha256()
                    temporary = destination.with_suffix(destination.suffix + ".extracting")
                    with archive.open(info) as source, temporary.open("wb") as target:
                        while block := source.read(1024 * 1024):
                            target.write(block)
                            digest.update(block)
                    os.replace(temporary, destination)
                    image_sha256 = digest.hexdigest()
                with Image.open(destination) as image:
                    width, height, mode = image.width, image.height, image.mode
                    pixel_sha256 = decoded_pixel_sha256(image)
                    pixel_min, pixel_max = pixel_extrema(image)
                selected.append({
                    **row,
                    "archive_member": info.filename,
                    "local_image_path": str(destination),
                    "sha256": image_sha256,
                    "pixel_sha256": pixel_sha256,
                    "width": width,
                    "height": height,
                    "mode": mode,
                    "pixel_min": pixel_min,
                    "pixel_max": pixel_max,
                })

    if len(selected) != expected_rows:
        raise ValueError(f"Extracted {len(selected)} images, expected {expected_rows}")
    file_hash_groups = defaultdict(list)
    pixel_hash_groups = defaultdict(list)
    for row in selected:
        file_hash_groups[row["sha256"]].append(row)
        pixel_hash_groups[row["pixel_sha256"]].append(row)
    cross_split_duplicates = [
        digest for digest, group in pixel_hash_groups.items()
        if len({row["split"] for row in group}) > 1
    ]
    if cross_split_duplicates:
        raise ValueError(f"Decoded pixel duplicates cross splits: {len(cross_split_duplicates)}")
    conflicting_labels = [
        digest for digest, group in pixel_hash_groups.items()
        if len({row["fluid"] for row in group}) > 1
    ]
    if conflicting_labels:
        raise ValueError(f"Decoded pixel duplicates have conflicting fluid labels: {len(conflicting_labels)}")
    fields = list(selected[0])
    image_manifest = out_dir / "official_expert_image_manifest.csv"
    with image_manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
    report = {
        "status": "official_images_matched_and_exact_duplicate_audited",
        "source_archive_md5": OFFICIAL_MD5,
        "source_label_manifest_sha256": label_manifest_sha256,
        "source_label_audit_sha256": file_sha256(label_audit),
        "matched_images": len(selected),
        "patients": len({row["Patient_ID"] for row in selected}),
        "dimensions": dict(Counter(f"{row['width']}x{row['height']}" for row in selected)),
        "modes": dict(Counter(row["mode"] for row in selected)),
        "images_with_values_above_255": sum(row["pixel_max"] > 255 for row in selected),
        "constant_intensity_images": sum(row["pixel_min"] == row["pixel_max"] for row in selected),
        "unique_file_hashes": len(file_hash_groups),
        "unique_decoded_pixel_hashes": len(pixel_hash_groups),
        "cross_split_decoded_pixel_duplicate_hashes": 0,
        "conflicting_label_decoded_pixel_duplicate_hashes": 0,
        "manifest_sha256": hashlib.sha256(image_manifest.read_bytes()).hexdigest(),
        "near_duplicate_gate": "not_run",
        "fovea_crop_and_scanner_gate": "not_run",
    }
    (out_dir / "official_image_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("official_archive", type=Path)
    parser.add_argument("labels_manifest", type=Path)
    parser.add_argument("label_audit", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--expected-rows", type=int, default=9408)
    args = parser.parse_args()
    print(json.dumps(run(args.official_archive, args.labels_manifest, args.label_audit,
                         args.out_dir, args.expected_rows), indent=2))
