"""Load OLIVES images only through the official, hash-bound expert manifest.

The test partition requires an explicit selection lock receipt. The receipt
does not prove model quality; it records that the train/validation choices
were frozen before test predictions were generated.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image


ARCHIVE_MD5 = "e08f4198aa919d6900bd30525c137933"
EXPERT_IMAGES = 9408
SPLITS = {"train", "validation", "test"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def source_parts(source_path: str) -> tuple[str, ...]:
    if not source_path.startswith("/") or "\\" in source_path:
        raise ValueError(f"Unsafe source path: {source_path!r}")
    parts = tuple(source_path[1:].split("/"))
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"Unsafe source path: {source_path!r}")
    return parts


def load_verified_manifest(manifest: Path, audit_path: Path) -> list[dict[str, str]]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("images_with_values_above_255", 0) != 0:
        raise ValueError("Integer OCT pixels exceed 255; review preprocessing before inference")
    if audit.get("near_duplicate_gate") != "reviewed_pass":
        raise ValueError("Cross-split near-duplicate screen and review have not passed")
    if audit.get("fovea_crop_and_scanner_gate") != "reviewed_input_compatible_with_domain_shift":
        raise ValueError("OCT geometry and scanner-domain review has not passed")
    if (audit.get("status") != "official_images_matched_and_exact_duplicate_audited"
            or audit.get("source_archive_md5") != ARCHIVE_MD5
            or audit.get("matched_images") != EXPERT_IMAGES
            or audit.get("manifest_sha256") != sha256(manifest)
            or audit.get("cross_split_decoded_pixel_duplicate_hashes") != 0
            or audit.get("conflicting_label_decoded_pixel_duplicate_hashes") != 0):
        raise ValueError("Official image provenance or pixel-duplicate audit failed")
    with manifest.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    required = {"source_image_path", "sha256", "pixel_sha256", "Patient_ID", "split", "trial", "fluid"}
    if len(rows) != EXPERT_IMAGES or not required.issubset(rows[0] if rows else {}):
        raise ValueError("Official image manifest is incomplete")
    paths = set()
    patient_splits = defaultdict(set)
    for row in rows:
        source = row["source_image_path"]
        source_parts(source)
        if source in paths or row["split"] not in SPLITS or row["fluid"] not in {"0", "1"}:
            raise ValueError(f"Invalid or repeated image row: {source}")
        if not row["Patient_ID"] or len(row["sha256"]) != 64 or len(row["pixel_sha256"]) != 64:
            raise ValueError(f"Incomplete provenance for image: {source}")
        paths.add(source)
        patient_splits[row["Patient_ID"]].add(row["split"])
    if any(len(splits) != 1 for splits in patient_splits.values()):
        raise ValueError("Patient crosses train/validation/test partitions")
    return rows


def select_split(rows: list[dict[str, str]], split: str, *,
                 manifest_digest: str | None = None,
                 selection_lock: Path | None = None) -> list[dict[str, str]]:
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    if split == "test":
        if selection_lock is None:
            raise ValueError("Test split requires a frozen selection lock")
        lock = json.loads(selection_lock.read_text(encoding="utf-8"))
        if (lock.get("status") != "validation_selection_frozen"
                or manifest_digest is None
                or lock.get("manifest_sha256") != manifest_digest
                or not isinstance(lock.get("selected_configuration"), dict)
                or not lock["selected_configuration"]):
            raise ValueError("Selection lock is incomplete")
    selected = [row for row in rows if row["split"] == split]
    if not selected:
        raise ValueError(f"No images for {split}")
    return selected


def open_verified_image(row: dict[str, str], image_root: Path) -> Image.Image:
    path = image_root.joinpath(*source_parts(row["source_image_path"]))
    if not path.is_file() or sha256(path) != row["sha256"]:
        raise ValueError(f"Image missing or changed: {row['source_image_path']}")
    with Image.open(path) as image:
        image.load()
        if image.mode == "L":
            return image.copy()
        if image.mode == "RGB" and row.get("mode") == "RGB":
            # One official OCT TIFF is RGB because of its colored scale marker.
            # The released encoder expects a single channel; preserve and hash
            # the official file above, then convert only the model input.
            return image.convert("L")
        raise ValueError(f"Unsupported verified image mode: {image.mode}")
