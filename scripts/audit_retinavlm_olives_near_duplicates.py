"""Screen official expert OCT images for possible cross-split near duplicates.

This is a candidate finder, not an automatic leakage verdict. Compare the
reported pairs visually and with source metadata before accepting the split.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
from itertools import combinations
from pathlib import Path

from PIL import Image

from retinavlm_verified_images import ARCHIVE_MD5, EXPERT_IMAGES, sha256, source_parts


def difference_hash(path: Path) -> int:
    with Image.open(path) as image:
        if len(image.getbands()) > 1:
            image = image.convert("L")
        small = image.convert("F").resize((9, 8), Image.Resampling.BILINEAR)
        values = list(small.getdata())
    result = 0
    for y in range(8):
        for x in range(8):
            result = (result << 1) | int(values[y * 9 + x] > values[y * 9 + x + 1])
    return result


def audit(manifest: Path, image_audit: Path, max_distance: int, max_reported: int) -> dict:
    if not 0 <= max_distance <= 64 or max_reported < 1:
        raise ValueError("Invalid distance or report limit")
    provenance = json.loads(image_audit.read_text(encoding="utf-8"))
    manifest_digest = sha256(manifest)
    if (provenance.get("status") != "official_images_matched_and_exact_duplicate_audited"
            or provenance.get("source_archive_md5") != ARCHIVE_MD5
            or provenance.get("manifest_sha256") != manifest_digest
            or provenance.get("matched_images") != EXPERT_IMAGES
            or provenance.get("cross_split_decoded_pixel_duplicate_hashes") != 0):
        raise ValueError("Near-duplicate screen requires the complete official image audit")
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"source_image_path", "local_image_path", "Patient_ID", "split", "fluid", "sha256"}
    if len(rows) != EXPERT_IMAGES or not required.issubset(rows[0] if rows else {}):
        raise ValueError("Official image manifest is missing required columns")
    seen_paths = set()
    patient_splits = {}
    records = []
    for row in rows:
        source_path = row["source_image_path"]
        source_parts(source_path)
        if source_path in seen_paths:
            raise ValueError(f"Repeated source path: {source_path}")
        seen_paths.add(source_path)
        patient = row["Patient_ID"]
        split = row["split"]
        if patient in patient_splits and patient_splits[patient] != split:
            raise ValueError(f"Patient {patient} crosses splits")
        patient_splits[patient] = split
        image_path = Path(row["local_image_path"])
        if not image_path.is_file() or sha256(image_path) != row["sha256"]:
            raise ValueError(f"Image missing or changed: {source_path}")
        records.append((row, difference_hash(image_path)))
    candidate_heap = []
    candidate_count = 0
    for (left, left_hash), (right, right_hash) in combinations(records, 2):
        if left["split"] == right["split"]:
            continue
        distance = (left_hash ^ right_hash).bit_count()
        if distance <= max_distance:
            candidate_count += 1
            if len(candidate_heap) < max_reported or distance < -candidate_heap[0][0]:
                candidate = {
                    "distance": distance,
                    "left_source": left["source_image_path"],
                    "right_source": right["source_image_path"],
                    "left_image": left["local_image_path"],
                    "right_image": right["local_image_path"],
                    "left_patient": left["Patient_ID"],
                    "right_patient": right["Patient_ID"],
                    "left_split": left["split"],
                    "right_split": right["split"],
                    "left_fluid": left["fluid"],
                    "right_fluid": right["fluid"],
                    "same_exact_hash": left["sha256"] == right["sha256"],
                }
                entry = (-distance, -candidate_count, candidate)
                if len(candidate_heap) < max_reported:
                    heapq.heappush(candidate_heap, entry)
                else:
                    heapq.heapreplace(candidate_heap, entry)
    candidates = [entry[2] for entry in candidate_heap]
    candidates.sort(key=lambda item: (item["distance"], item["left_source"], item["right_source"]))
    return {
        "status": "candidate_screen_only_review_required",
        "source_manifest": str(manifest),
        "source_manifest_sha256": manifest_digest,
        "source_image_audit_sha256": sha256(image_audit),
        "images": len(records),
        "patients": len(patient_splits),
        "max_difference_hash_distance": max_distance,
        "candidate_pair_count": candidate_count,
        "reported_pair_count": len(candidates),
        "truncated": candidate_count > len(candidates),
        "candidates": candidates,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("official_image_manifest", type=Path)
    parser.add_argument("--image-audit", type=Path, required=True)
    parser.add_argument("--max-distance", type=int, default=4)
    parser.add_argument("--max-reported", type=int, default=1000)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = audit(args.official_image_manifest, args.image_audit,
                   args.max_distance, args.max_reported)
    payload = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
