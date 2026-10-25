"""Build an audited manifest from reviewed DDR annotation records.

The input CSV is deliberately explicit: a missing mask is never evidence of
absence. DDR archive layouts differ between its tasks, so paths are supplied
by the reviewer rather than guessed from filenames.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .protocol import assign_splits, audit_manifest, write_jsonl


def build_rows(csv_path: str) -> list[dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for source in csv.DictReader(handle):
            image_path = Path(source["image_path"]).expanduser().resolve()
            mask_name = source.get("hard_exudate_mask_path", "").strip()
            mask_path = Path(mask_name).expanduser().resolve() if mask_name else None
            exhaustive = source.get("exhaustive_hard_exudate_annotation", "").lower() == "true"
            if mask_path:
                if not mask_path.is_file():
                    raise FileNotFoundError(mask_path)
                mask = np.asarray(Image.open(mask_path))
                if mask.ndim == 3:
                    mask = mask[..., :3].max(axis=-1)
                positive_pixels = int(np.count_nonzero(mask))
                status = "present" if positive_pixels else "absent" if exhaustive else "unknown"
            else:
                positive_pixels = None
                status = "unknown"
            row = {
                "image_id": source["image_id"],
                "image_path": str(image_path),
                "hard_exudate_mask_path": str(mask_path) if mask_path else None,
                "hard_exudates": status,
                "exhaustive_hard_exudate_annotation": exhaustive,
                "positive_mask_pixels": positive_pixels,
                "patient_id": source.get("patient_id") or None,
                "duplicate_group": source.get("duplicate_group") or None,
                "split": source.get("split") or None,
                "dr_grade": source.get("dr_grade") or "unknown",
                "location": source.get("location") or None,
                "source": "DDR",
            }
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviewed-csv", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = assign_splits(build_rows(args.reviewed_csv))
    audit = audit_manifest(rows)
    write_jsonl(args.output, rows)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
