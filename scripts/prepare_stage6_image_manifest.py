"""Prepare a verified image/grade manifest for Stage 6 without downloading data.

This is deliberately an intake tool, not a scraper. It expects an archive
obtained through the official access route and a user-supplied CSV containing
one row per image. The output stays under the external drive and is suitable
for the fail-closed Stage 6 gate and frozen evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from retina_lanse.data.manifest import difference_hash, sha256_path


EXTERNAL_ROOT = Path("/Volumes/Elements/Retina-LanSE")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
GRADE_COLUMNS = ("grade", "dr_grade", "level", "retinopathy_grade")


def under(root: Path, path: Path, label: str) -> Path:
    root = root.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"{label} must stay under {root}: {resolved}") from exc
    return resolved


def image_key(value: str) -> str:
    return Path(value.strip()).stem.lower()


def normalize_column(value: str) -> str:
    return "_".join(value.strip().lower().replace("-", " ").split())


def infer_official_split(path: Path) -> str:
    text = str(path).lower()
    if "training" in text or "train" in path.stem.lower():
        return "training"
    if "testing" in text or "test" in path.stem.lower():
        return "testing"
    return "external"


def parse_grade(row: dict[str, str], row_number: int) -> tuple[int | None, int]:
    for key in ("referable_dr", "label"):
        value = row.get(key, "").strip().lower()
        if value in {"0", "1", "true", "false", "yes", "no"}:
            target = int(value in {"1", "true", "yes"})
            return None, target
    for key in GRADE_COLUMNS:
        value = row.get(key, "").strip()
        if value:
            try:
                grade = int(value)
            except ValueError as exc:
                raise RuntimeError(f"row {row_number} has non-integer {key}: {value}") from exc
            if grade not in range(5):
                raise RuntimeError(f"row {row_number} has grade outside 0..4: {grade}")
            return grade, int(grade >= 2)
    raise RuntimeError(f"row {row_number} has no grade/referable_dr/label column value")


def load_labels(path: Path, *, official_split: str | None = None) -> dict[tuple[str, str], dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise RuntimeError("labels CSV has no header")
        field_map = {normalize_column(field): field for field in reader.fieldnames if field is not None}
        image_columns = ("image", "image_name", "filename", "file")
        image_column = next((field_map[key] for key in image_columns if key in field_map), None)
        if image_column is None:
            raise RuntimeError("labels CSV needs one of: image, image name, filename, file")
        normalized_rows: list[dict[str, str]] = []
        for row in reader:
            normalized_rows.append({normalize_column(key): (value or "") for key, value in row.items() if key is not None})
        labels: dict[tuple[str, str], dict[str, Any]] = {}
        split = official_split or infer_official_split(path)
        for row_number, row in enumerate(normalized_rows, start=2):
            raw_name = row.get(normalize_column(image_column), "")
            key = image_key(raw_name)
            if not key:
                raise RuntimeError(f"row {row_number} has an empty image name")
            label_key = (split, key)
            if label_key in labels:
                raise RuntimeError(f"duplicate label for image stem/split: {split}/{key}")
            grade, target = parse_grade(row, row_number)
            patient_value = row.get("patient_id", "").strip()
            patient = patient_value or key
            labels[label_key] = {
                "source_name": raw_name.strip(),
                "patient_id": patient,
                "patient_id_source": "csv" if patient_value else "partition_qualified_image_stem_fallback",
                "grade": grade,
                "referable_dr": target,
                "official_split": split,
            }
    if not labels:
        raise RuntimeError("labels CSV contains no rows")
    return labels


def build_records(dataset: str, dataset_root: Path, image_root: Path, labels: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    images = sorted(path for path in image_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise RuntimeError(f"no images found under {image_root}")
    seen: set[str] = set()
    used_labels: set[tuple[str, str]] = set()
    records: list[dict[str, Any]] = []
    for path in images:
        key = path.stem.lower()
        official_split = infer_official_split(path)
        label_key = (official_split, key)
        if label_key not in labels and ("external", key) in labels:
            label_key = ("external", key)
        if label_key not in labels:
            raise RuntimeError(f"image has no label row: {path.relative_to(dataset_root)}")
        if str(path.relative_to(dataset_root)) in seen:
            raise RuntimeError(f"duplicate image path in image root: {path.relative_to(dataset_root)}")
        seen.add(str(path.relative_to(dataset_root)))
        used_labels.add(label_key)
        label = labels[label_key]
        with Image.open(path) as image:
            width, height = image.size
            image_format = image.format
            perceptual_hash = difference_hash(image)
        records.append(
            {
                "dataset": dataset,
                "relative_path": path.relative_to(dataset_root).as_posix(),
                "patient_id": str(label["patient_id"] if label["patient_id_source"] == "csv" else f"{dataset}:{label['official_split']}:{label['patient_id']}"),
                "patient_id_source": label["patient_id_source"],
                "split": "external",
                "official_split": label["official_split"],
                "source_partition": path.parent.name,
                "width": width,
                "height": height,
                "format": image_format,
                "bytes": path.stat().st_size,
                "sha256": sha256_path(path),
                "perceptual_hash_algorithm": "dhash-9x8-v1",
                "perceptual_hash": perceptual_hash,
                "grade": label["grade"],
                "referable_dr": label["referable_dr"],
            }
        )
    extra = sorted(set(labels) - used_labels)
    if extra:
        raise RuntimeError(f"labels have no matching image files (first five): {extra[:5]}")
    patients = [record["patient_id"] for record in records]
    if any(not patient.strip() for patient in patients):
        raise RuntimeError("empty patient_id in prepared manifest")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a verified external Stage 6 image/grade manifest.")
    parser.add_argument("--dataset", choices=("idrid", "fgadr"), required=True)
    parser.add_argument("--external-root", default=str(EXTERNAL_ROOT))
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--labels-csv", action="append", required=True, help="Repeat for official training/testing label CSVs")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    external_root = Path(args.external_root).resolve()
    if not external_root.is_dir():
        parser.error(f"external root is not mounted: {external_root}")
    dataset_root = under(external_root, Path(args.dataset_root), "dataset root")
    image_root = under(dataset_root, Path(args.image_root), "image root")
    output = under(external_root, Path(args.output), "manifest output")
    if not dataset_root.is_dir() or not image_root.is_dir():
        parser.error("dataset root and image root must exist")
    labels_paths = [under(dataset_root, Path(value), "labels CSV") for value in args.labels_csv]
    for labels_path in labels_paths:
        if not labels_path.is_file():
            parser.error(f"labels CSV does not exist: {labels_path}")
    labels: dict[tuple[str, str], dict[str, Any]] = {}
    for labels_path in labels_paths:
        labels.update(load_labels(labels_path))
    records = build_records(args.dataset, dataset_root, image_root, labels)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    with partial.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    partial.replace(output)
    counts = Counter(record["referable_dr"] for record in records)
    print(json.dumps({"status": "complete_stage6_manifest", "dataset": args.dataset, "manifest": str(output), "images": len(records), "patients": len({r["patient_id"] for r in records}), "official_splits": dict(sorted(Counter(r["official_split"] for r in records).items())), "positive_images": counts[1], "negative_images": counts[0], "labels_sha256": {str(path): sha256_path(path) for path in labels_paths}}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
