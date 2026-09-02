from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from PIL import Image

from retina_lanse.data.manifest import sha256_path
from scripts.run_global_sae_search import require_external


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def phash255(path: Path) -> bytes:
    with Image.open(path) as image:
        image.draft("L", (64, 64))
        grayscale = np.asarray(image.convert("L").resize((64, 64), Image.Resampling.LANCZOS))
    coefficients = cv2.dct(grayscale.astype(np.float32))[:16, :16].reshape(-1)[1:]
    bits = coefficients > np.median(coefficients)
    return np.packbits(bits, bitorder="big").tobytes()


def correlation(path_a: Path, path_b: Path) -> tuple[float, bool]:
    arrays = []
    for path in (path_a, path_b):
        with Image.open(path) as image:
            image.draft("L", (128, 128))
            array = np.asarray(
                image.convert("L").resize((128, 128), Image.Resampling.LANCZOS),
                dtype=np.float32,
            )
        array = (array - array.mean()) / max(float(array.std()), 1e-6)
        arrays.append(array)
    direct = float(np.mean(arrays[0] * arrays[1]))
    flipped = float(np.mean(arrays[0] * np.fliplr(arrays[1])))
    return (flipped, True) if flipped > direct else (direct, False)


def lsh_bands(signature: bytes, bands: int = 13, bits: int = 255) -> list[int]:
    """Partition all hash bits; bands > threshold guarantees one exact band."""
    value = int.from_bytes(signature, "big") >> 1  # Drop the one packbits padding bit.
    base, remainder = divmod(bits, bands)
    values = []
    offset = 0
    for band in range(bands):
        width = base + (1 if band < remainder else 0)
        shift = bits - offset - width
        values.append((value >> shift) & ((1 << width) - 1))
        offset += width
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description="High-specificity cross-split near-duplicate audit.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--hamming-threshold", type=int, default=12)
    parser.add_argument("--confirmation-hamming", type=int, default=8)
    parser.add_argument("--correlation-threshold", type=float, default=0.995)
    args = parser.parse_args()
    dataset_root = require_external(Path(args.dataset_root), "dataset root")
    manifest_path = require_external(Path(args.manifest), "manifest")
    output = require_external(Path(args.output), "near-duplicate audit output")
    if args.workers < 1:
        parser.error("--workers must be positive")
    records = read_jsonl(manifest_path)
    if args.max_images is not None:
        records = records[: args.max_images]
    paths = [(dataset_root / record["relative_path"]).resolve() for record in records]
    started = time.perf_counter()
    signatures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, signature in enumerate(executor.map(phash255, paths), start=1):
            signatures.append(signature)
            if index % 5000 == 0 or index == len(records):
                print(
                    json.dumps(
                        {
                            "event": "near_duplicate_hash_progress",
                            "images": index,
                            "total": len(records),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    hashing_elapsed = time.perf_counter() - started

    band_count = args.hamming_threshold + 1
    buckets: dict[tuple[int, int], list[int]] = {}
    lsh_candidate_occurrences = 0
    hamming_pair_distances: dict[tuple[int, int], int] = {}
    for index, signature in enumerate(signatures):
        for band, band_value in enumerate(lsh_bands(signature, bands=band_count)):
            key = (band, band_value)
            for other in buckets.get(key, []):
                if records[other]["split"] != records[index]["split"]:
                    lsh_candidate_occurrences += 1
                    pair = (other, index)
                    if pair not in hamming_pair_distances:
                        distance = int.from_bytes(
                            bytes(a ^ b for a, b in zip(signatures[other], signature)), "big"
                        ).bit_count()
                        if distance <= args.hamming_threshold:
                            hamming_pair_distances[pair] = distance
            buckets.setdefault(key, []).append(index)
        if (index + 1) % 10000 == 0 or index + 1 == len(records):
            print(
                json.dumps(
                    {
                        "event": "near_duplicate_lsh_progress",
                        "images": index + 1,
                        "total": len(records),
                        "candidate_occurrences": lsh_candidate_occurrences,
                        "hamming_candidates": len(hamming_pair_distances),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    hamming_candidates = [
        (left, right, distance)
        for (left, right), distance in sorted(hamming_pair_distances.items())
    ]

    confirmed = []
    reviewed_examples = []
    for left, right, distance in hamming_candidates:
        similarity, horizontally_flipped = correlation(paths[left], paths[right])
        item = {
            "left": records[left]["relative_path"],
            "left_patient": records[left]["patient_id"],
            "left_split": records[left]["split"],
            "right": records[right]["relative_path"],
            "right_patient": records[right]["patient_id"],
            "right_split": records[right]["split"],
            "phash255_hamming": distance,
            "normalized_pixel_correlation": similarity,
            "horizontal_flip_used": horizontally_flipped,
        }
        if len(reviewed_examples) < 200:
            reviewed_examples.append(item)
        if distance <= args.confirmation_hamming and similarity >= args.correlation_threshold:
            confirmed.append(item)

    report = {
        "schema_version": "1.0",
        "status": "pass" if not confirmed else "fail",
        "pass": not confirmed,
        "scope": "cross_split_high_specificity_near_duplicate_audit",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_path(manifest_path),
        "records": len(records),
        "workers": args.workers,
        "elapsed_seconds": time.perf_counter() - started,
        "hashing_elapsed_seconds": hashing_elapsed,
        "hashing_images_per_second": len(records) / max(hashing_elapsed, 1e-12),
        "phash_bits": 255,
        "lsh_bands": band_count,
        "lsh_band_bits_minimum": 255 // band_count,
        "lsh_band_bits_maximum": (255 + band_count - 1) // band_count,
        "lsh_recall_guarantee_hamming_at_most": band_count - 1,
        "hamming_candidate_threshold": args.hamming_threshold,
        "confirmation_hamming_threshold": args.confirmation_hamming,
        "confirmation_correlation_threshold": args.correlation_threshold,
        "lsh_candidate_occurrences": lsh_candidate_occurrences,
        "hamming_candidate_pairs": len(hamming_candidates),
        "confirmed_cross_split_near_duplicates": len(confirmed),
        "confirmed": confirmed,
        "reviewed_examples_first_200": reviewed_examples,
        "clinical_labels_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "records",
                    "lsh_candidate_occurrences",
                    "hamming_candidate_pairs",
                    "confirmed_cross_split_near_duplicates",
                )
            },
            sort_keys=True,
        )
    )
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
