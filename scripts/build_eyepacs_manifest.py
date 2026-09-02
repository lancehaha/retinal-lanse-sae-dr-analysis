from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retina_lanse.data.manifest import eyepacs_image_record


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a patient-grouped EyePACS provenance manifest.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=4106)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--source-partition", choices=("train", "test", "all"), default="all")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--skip-sha256", action="store_true")
    args = parser.parse_args()

    root, output = Path(args.dataset_root).resolve(), Path(args.output).resolve()
    if not root.is_dir():
        parser.error(f"dataset root does not exist: {root}")
    images = sorted(path for path in root.rglob("*") if path.suffix.lower() in {".jpeg", ".jpg", ".png"})
    if args.source_partition != "all":
        images = [path for path in images if path.parent.name == args.source_partition]
    if args.max_images is not None:
        if args.max_images < 1:
            parser.error("--max-images must be positive")
        images = images[: args.max_images]
    if not images:
        parser.error(f"no retinal images found under {root}")
    if args.workers < 1:
        parser.error("--workers must be positive")

    def build_record(path: Path) -> dict:
        return eyepacs_image_record(
            path,
            root,
            seed=args.seed,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
            include_sha256=not args.skip_sha256,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    split_counts: Counter[str] = Counter()
    patient_splits: dict[str, str] = {}
    partial = output.with_suffix(output.suffix + ".partial")
    with partial.open("w", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for record in executor.map(build_record, images):
                prior = patient_splits.setdefault(record["patient_id"], record["split"])
                if prior != record["split"]:
                    raise RuntimeError(f"patient leakage detected for {record['patient_id']}")
                split_counts[record["split"]] += 1
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    partial.replace(output)

    print(json.dumps({
        "manifest": str(output),
        "images": len(images),
        "patients": len(patient_splits),
        "split_images": dict(sorted(split_counts.items())),
        "sha256_included": not args.skip_sha256,
        "workers": args.workers,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
