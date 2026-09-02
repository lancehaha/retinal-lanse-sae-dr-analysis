from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
from PIL import Image


EYEPACS_NAME = re.compile(r"^(?P<patient_id>\d+)_(?P<laterality>left|right)$", re.IGNORECASE)


def parse_eyepacs_identity(path: str | Path) -> tuple[str, str]:
    match = EYEPACS_NAME.fullmatch(Path(path).stem)
    if not match:
        raise ValueError(f"Not an EyePACS image filename: {Path(path).name}")
    return match.group("patient_id"), match.group("laterality").lower()


def stable_patient_split(
    patient_id: str,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> str:
    if not 0.0 < train_fraction < 1.0 or not 0.0 < validation_fraction < 1.0:
        raise ValueError("train and validation fractions must be between zero and one")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train plus validation fractions must leave a non-empty internal test fraction")
    payload = f"{seed}:{patient_id}".encode("utf-8")
    unit_interval = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / 2**64
    if unit_interval < train_fraction:
        return "train"
    if unit_interval < train_fraction + validation_fraction:
        return "validation"
    return "internal_test"


def sha256_path(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def difference_hash(image: Image.Image) -> str:
    """Return a deterministic 64-bit perceptual dHash for duplicate screening."""
    gray = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
    bits = gray[:, 1:] > gray[:, :-1]
    value = 0
    for bit in bits.flat:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def eyepacs_image_record(
    path: str | Path,
    dataset_root: str | Path,
    *,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
    include_sha256: bool = True,
) -> dict:
    image_path = Path(path).resolve()
    root = Path(dataset_root).resolve()
    patient_id, laterality = parse_eyepacs_identity(image_path)
    with Image.open(image_path) as image:
        width, height = image.size
        mode, image_format = image.mode, image.format
        perceptual_hash = difference_hash(image)
    return {
        "dataset": "EyePACS-Kaggle-DR",
        "source_partition": image_path.parent.name,
        "relative_path": image_path.relative_to(root).as_posix(),
        "patient_id": patient_id,
        "laterality": laterality,
        "split": stable_patient_split(patient_id, seed, train_fraction, validation_fraction),
        "width": width,
        "height": height,
        "mode": mode,
        "format": image_format,
        "camera": None,
        "camera_source": "unavailable_in_kaggle_distribution",
        "bytes": image_path.stat().st_size,
        "sha256": sha256_path(image_path) if include_sha256 else None,
        "perceptual_hash_algorithm": "dhash-9x8-v1",
        "perceptual_hash": perceptual_hash,
    }
