"""Build the fixed DDR segmentation manifest and a machine-readable audit.

DDR's segmentation partitions are the study partitions. The grading task has
different partitions, so its labels are joined by exact filename only. No
patient identifier is supplied in the public archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image

from .protocol import audit_manifest, target_report, write_jsonl

SPLITS = {"train": "train", "valid": "validation", "test": "test"}
EXPECTED = {"train": 383, "validation": 149, "test": 225}
MASK_FOLDER = {"train": "label", "valid": "segmentation label", "test": "label"}
QUADRANTS = ("upper-left", "upper-right", "lower-left", "lower-right")


def phash(path: Path) -> int:
    """Center-cropped DCT hash; a candidate finder, never proof of identity."""
    n = 32
    with Image.open(path) as image:
        width, height = image.size
        margin = int(min(width, height) * 0.1)
        if width >= height:
            box = ((width - height) // 2 + margin, margin,
                   (width + height) // 2 - margin, height - margin)
        else:
            box = (margin, (height - width) // 2 + margin,
                   width - margin, (height + width) // 2 - margin)
        pixels = np.asarray(image.convert("L").crop(box).resize((n, n), Image.Resampling.BICUBIC),
                            dtype=np.float32)
    x = np.arange(n)
    cosine = np.cos(np.pi * (2 * x[None, :] + 1) * np.arange(n)[:, None] / (2 * n))
    cosine[0] *= np.sqrt(1 / n)
    cosine[1:] *= np.sqrt(2 / n)
    coefficients = cosine @ pixels @ cosine.T
    low = coefficients[:8, :8].copy()
    low[0, 0] = np.nan
    bits = low > np.nanmedian(low)
    value = 0
    for bit in bits.flat:
        value = (value << 1) | bool(bit)
    return value


def location_from_mask(mask: np.ndarray) -> str | None:
    if not np.any(mask):
        return None
    height, width = mask.shape
    parts = (mask[: height // 2, : width // 2],
             mask[: height // 2, width // 2 :],
             mask[height // 2 :, : width // 2],
             mask[height // 2 :, width // 2 :])
    counts = np.array([int(part.sum()) for part in parts])
    keep = counts >= max(1, int(np.ceil(counts.sum() * 0.05)))
    if not np.any(keep):
        keep[np.argmax(counts)] = True
    return "image-relative " + ", ".join(name for name, present in zip(QUADRANTS, keep) if present)


def grades_from_archive(root: Path) -> tuple[dict[str, str], dict[str, str]]:
    grades, grade_splits = {}, {}
    for split in SPLITS:
        path = root / "DR_grading" / f"{split}.txt"
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            name, grade = line.split()
            if grade not in {"0", "1", "2", "3", "4"}:
                continue
            if name in grades and grades[name] != grade:
                raise ValueError(f"Conflicting DR grade for {name}")
            grades[name], grade_splits[name] = grade, split
    return grades, grade_splits


def build(root: Path, output: Path, audit_path: Path, *, phash_threshold: int = 4) -> dict:
    if not (0 <= phash_threshold <= 8):
        raise ValueError("pHash candidate threshold must be 0..8")
    grades, grade_splits = grades_from_archive(root)
    rows, hashes, digests = [], {}, {}
    for source_split, split in SPLITS.items():
        base = root / "lesion_segmentation" / source_split
        images = {p.stem: p for p in (base / "image").glob("*.jpg")}
        masks = {p.stem: p for p in (base / MASK_FOLDER[source_split] / "EX").glob("*.tif")}
        if len(images) != EXPECTED[split] or set(images) != set(masks):
            raise ValueError(f"DDR {split} image/mask coverage differs from published segmentation set")
        for image_id, image_path in sorted(images.items()):
            mask_path = masks[image_id]
            with Image.open(image_path) as image, Image.open(mask_path) as mask_image:
                if image.size != mask_image.size or mask_image.mode != "L":
                    raise ValueError(f"Image/mask shape or mode mismatch: {image_id}")
                mask = np.asarray(mask_image)
                values = set(np.unique(mask).tolist())
                if not values <= {0, 255}:
                    raise ValueError(f"Unexpected EX mask values for {image_id}: {values}")
                binary = mask > 0
            hashes[image_id] = phash(image_path)
            digests[image_id] = hashlib.sha256(image_path.read_bytes()).hexdigest()
            rows.append({
                "image_id": image_id,
                "image_path": str(image_path.resolve()),
                "hard_exudate_mask_path": str(mask_path.resolve()),
                "hard_exudates": "present" if np.any(binary) else "absent",
                "exhaustive_hard_exudate_annotation": True,
                "positive_mask_pixels": int(binary.sum()),
                "patient_id": None,
                "duplicate_group": None,
                "split": split,
                "dr_grade": grades.get(image_path.name, "unknown"),
                "location": location_from_mask(binary),
                "source": "DDR",
            })
    by_id = {row["image_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Image IDs overlap across segmentation splits")
    exact = []
    candidates = []
    for a, b in combinations(rows, 2):
        if a["split"] == b["split"]:
            continue
        aid, bid = a["image_id"], b["image_id"]
        if digests[aid] == digests[bid]:
            exact.append([aid, bid])
        distance = (hashes[aid] ^ hashes[bid]).bit_count()
        if distance <= phash_threshold:
            candidates.append({"a": aid, "a_split": a["split"],
                               "b": bid, "b_split": b["split"], "distance": distance})
    # Conservatively remove lookalikes from the less protected partition.
    # This uses images only, before any model result or threshold selection.
    priority = {"train": 0, "validation": 1, "test": 2}
    excluded = set()
    for pair in candidates:
        a, b = pair["a"], pair["b"]
        excluded.add(a if priority[pair["a_split"]] < priority[pair["b_split"]] else b)
    kept = [row for row in rows if row["image_id"] not in excluded]
    remaining = [pair for pair in candidates if pair["a"] not in excluded and pair["b"] not in excluded]
    if remaining:
        raise AssertionError("Cross-split pHash candidates remain after pruning")
    if exact:
        raise ValueError(f"Exact cross-split duplicate images found: {exact}")
    result = audit_manifest(kept)
    sample_targets = []
    for split in ("train", "validation", "test"):
        for status in ("present", "absent"):
            sample = next((r for r in kept if r["split"] == split and r["hard_exudates"] == status), None)
            if sample is not None:
                sample_targets.append({"image_id": sample["image_id"], "target": target_report(sample)})
    result.update({
        "source": "Official OIA-DDR ZIP; segmentation partitions",
        "source_paper_doi": "10.1016/j.ins.2019.06.011",
        "annotation_policy": "All 757 segmentation images have class-specific EX masks; original paper reports 486 EX-positive images; empty masks are DDR dataset-level negatives, not clinical proof of no lesion.",
        "original_count": len(rows),
        "original_ex_positive": sum(r["hard_exudates"] == "present" for r in rows),
        "original_ex_empty": sum(r["hard_exudates"] == "absent" for r in rows),
        "patient_ids_available": False,
        "grade_joined_original": sum(r["dr_grade"] != "unknown" for r in rows),
        "grade_split_mismatches_original": sum(
            r["dr_grade"] != "unknown"
            and SPLITS[grade_splits[Path(r["image_path"]).name]] != r["split"]
            for r in rows),
        "phash_policy": f"center-cropped 64-bit DCT hash, Hamming distance <= {phash_threshold}; cross-split candidates excluded from lower-priority train/validation partitions",
        "cross_split_phash_candidates": candidates,
        "excluded_image_ids": sorted(excluded),
        "location_policy": "Image-relative quadrants with >=5% of annotated EX pixels; no anatomical left/right inference.",
        "sample_targets": sample_targets,
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, kept)
    audit_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="extracted DDR-dataset directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.root, args.output, args.audit), indent=2))


if __name__ == "__main__":
    main()
