"""Pure protocol logic shared by data preparation, training, and evaluation."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

PROMPT = (
    "Describe this color fundus image using this exact JSON schema: "
    '{"hard_exudates":"present|absent|uncertain",'
    '"location":"image-relative quadrant description or null",'
    '"dr_grade":"0|1|2|3|4|unknown"}. '
    "For location, use only upper-left, upper-right, lower-left, and lower-right image quadrants. "
    "State only findings supported by the image."
)
SEEDS = (4106, 4107, 4108)


def verified_label(row: dict) -> int | None:
    """Unknown masks never become negative labels."""
    status = row.get("hard_exudates")
    if status == "present":
        return 1
    if status == "absent" and row.get("exhaustive_hard_exudate_annotation") is True:
        return 0
    return None


def target_report(row: dict) -> str:
    status = row.get("hard_exudates", "unknown")
    finding = status if verified_label(row) is not None else "uncertain"
    grade = str(row.get("dr_grade", "unknown"))
    if grade not in {"0", "1", "2", "3", "4"}:
        grade = "unknown"
    location = row.get("location") if finding == "present" else None
    return json.dumps(
        {"hard_exudates": finding, "location": location, "dr_grade": grade},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def parse_report(text: str) -> dict | None:
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        data = json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or data.get("hard_exudates") not in {"present", "absent", "uncertain"}:
        return None
    return data


def assign_splits(rows: list[dict], seed: int = 4106) -> list[dict]:
    """Keep patients and supplied duplicate groups together; preserve official splits."""
    official = {r.get("split") for r in rows}
    if official <= {"train", "validation", "test"} and None not in official:
        return rows
    parent = list(range(len(rows)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen: dict[tuple[str, str], int] = {}
    for i, row in enumerate(rows):
        for field in ("patient_id", "duplicate_group"):
            if row.get(field) is not None:
                key = (field, str(row[field]))
                if key in seen:
                    parent[root(i)] = root(seen[key])
                else:
                    seen[key] = i
    groups: dict[int, list[dict]] = {}
    for i, row in enumerate(rows):
        groups.setdefault(root(i), []).append(row)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n = len(keys)
    train_end, val_end = round(n * 0.70), round(n * 0.85)
    for index, key in enumerate(keys):
        split = "train" if index < train_end else "validation" if index < val_end else "test"
        for row in groups[key]:
            row["split"] = split
    return rows


def audit_manifest(rows: list[dict]) -> dict:
    ids = [r["image_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate image_id in manifest")
    for row in rows:
        if row.get("source") != "DDR":
            raise ValueError(f"Manifest source must be DDR: {row['image_id']}")
        if not Path(row["image_path"]).is_file():
            raise FileNotFoundError(row["image_path"])
        if row.get("split") not in {"train", "validation", "test"}:
            raise ValueError(f"Invalid split: {row['image_id']}")
        if row.get("hard_exudates") not in {"present", "absent", "unknown"}:
            raise ValueError(f"Invalid lesion label: {row['image_id']}")
        if row.get("hard_exudates") in {"present", "absent"}:
            mask_path = row.get("hard_exudate_mask_path")
            if not mask_path or not Path(mask_path).is_file():
                raise ValueError(f"Verified lesion label lacks a mask: {row['image_id']}")
            pixels = row.get("positive_mask_pixels")
            if pixels is None or (row["hard_exudates"] == "present") != (pixels > 0):
                raise ValueError(f"Lesion label contradicts mask pixels: {row['image_id']}")
        if row.get("hard_exudates") == "absent" and not row.get("exhaustive_hard_exudate_annotation"):
            raise ValueError(f"Unverified negative: {row['image_id']}")
    for field in ("patient_id", "duplicate_group"):
        seen: dict[str, str] = {}
        for row in rows:
            key = row.get(field)
            if key is None:
                continue
            old = seen.setdefault(str(key), row["split"])
            if old != row["split"]:
                raise ValueError(f"{field} crosses splits: {key}")
    counts = {split: sum(r["split"] == split for r in rows) for split in ("train", "validation", "test")}
    labels = {label: sum(verified_label(r) == label for r in rows) for label in (0, 1)}
    return {"images": len(rows), "splits": counts, "verified_labels": labels,
            "unknown_labels": sum(verified_label(r) is None for r in rows)}


def manifest_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
