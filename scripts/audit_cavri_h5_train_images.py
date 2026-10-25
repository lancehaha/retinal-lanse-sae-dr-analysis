#!/usr/bin/env python3
"""Read CAVRI-H5 training TIFFs in-place; leave validation and test unread."""

import argparse
import collections
import hashlib
import io
import json
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import tifffile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Refusing to overwrite geometry audit"
    receipt_path = args.source_root / "archive_audit_complete.json"
    receipt = json.loads(receipt_path.read_text())
    archive = args.source_root / receipt["filename"]
    assert sha256(archive) == receipt["archive_sha256"]
    rows = []
    with zipfile.ZipFile(archive) as source:
        paths = sorted(name for name in source.namelist() if name.lower().endswith(".tiff") and PurePosixPath(name).parts[1] == "train")
        assert len(paths) == 509
        for name in paths:
            raw = source.read(name)
            image = tifffile.imread(io.BytesIO(raw))
            assert image.ndim in (2, 3) and image.size > 0
            assert image.ndim == 2 or image.shape[2] in (3, 4)
            group = PurePosixPath(name).parts[2]
            channels = 1 if image.ndim == 2 else image.shape[2]
            rgb_max_difference = None
            if channels >= 3:
                rgb = image[..., :3].astype(np.int32)
                rgb_max_difference = int(max(np.max(np.abs(rgb[..., 0] - rgb[..., 1])), np.max(np.abs(rgb[..., 0] - rgb[..., 2]))))
            rows.append({
                "archive_path": name,
                "group": group,
                "member_sha256": hashlib.sha256(raw).hexdigest(),
                "height": int(image.shape[0]),
                "width": int(image.shape[1]),
                "channels": channels,
                "rgb_max_difference": rgb_max_difference,
                "dtype": str(image.dtype),
                "minimum": int(np.min(image)),
                "maximum": int(np.max(image)),
                "mean": float(np.mean(image)),
            })
    counts = collections.Counter(row["group"] for row in rows)
    assert counts == {"CNV": 76, "DME": 79, "DRUSEN": 117, "NORMAL": 122, "VMA-VMT": 115}
    result = {
        "status": "train_only_source_geometry_audit_no_model_fitting",
        "archive_sha256": sha256(archive),
        "source_receipt_sha256": sha256(receipt_path),
        "train_images": len(rows),
        "class_counts": dict(sorted(counts.items())),
        "geometry_counts": {str(key): count for key, count in sorted(collections.Counter((row["height"], row["width"], row["channels"], row["dtype"]) for row in rows).items())},
        "records": rows,
        "validation_images_read": False,
        "test_images_read": False,
        "extracted": False,
        "olives_accessed": False,
        "claim_boundary": "External dominant-pathology labels are not OLIVES B2/B3 diagnoses or J-Lens gate evidence",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}))


if __name__ == "__main__":
    main()
