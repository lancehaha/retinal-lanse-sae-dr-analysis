"""Find candidate OLIVES image rows by matching all available metadata fields.

Reads label columns only. A mirror row is a metadata candidate only when its
full key has the same multiplicity in the official expert CSV and HF mirror.
Image identity is still unverified until pixels or original paths are matched.
Keys with extra mirror rows remain ambiguous and are not selected.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import urllib.request
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq
import requests

from audit_retinavlm_olives_hf_images import HTTPRangeFile, REPO, TREE, CSV_IN_ZIP


def norm(value: object) -> str | None:
    if value is None or value == "":
        return None
    number = float(value)
    if math.isnan(number):
        return None
    return format(number, ".12g")


def key(row: dict, fields: list[str]) -> tuple[str | None, ...]:
    return tuple(norm(row[field]) for field in fields)


def read_shard(path: str, fields: list[str], cache_dir: Path) -> list[dict]:
    cache = cache_dir / (Path(path).name + ".json")
    if cache.exists():
        return json.loads(cache.read_text())
    url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{path}"
    response = requests.head(url, allow_redirects=True, timeout=30)
    response.raise_for_status()
    parquet = pq.ParquetFile(HTTPRangeFile(url, int(response.headers["content-length"])))
    table = parquet.read(columns=fields)
    values = table.to_pylist()
    output = []
    for i, row in enumerate(values):
        if norm(row["Fluid (IRF)"]) not in {"0", "1"}:
            continue
        if norm(row["Fluid (SRF)"]) not in {"0", "1"}:
            continue
        output.append({"parquet_path": path, "row_in_shard": i, "key": key(row, fields)})
    cache.write_text(json.dumps(output, separators=(",", ":")) + "\n")
    return output


def run(labels_zip: Path, out_dir: Path, workers: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(labels_zip) as archive:
        with archive.open(CSV_IN_ZIP) as binary:
            reader = csv.DictReader(io.TextIOWrapper(binary, encoding="utf-8-sig"))
            fields = [field for field in reader.fieldnames or [] if not field.startswith("Path (")]
            official = Counter(key(row, fields) for row in reader)
    tree = json.load(urllib.request.urlopen(TREE, timeout=30))
    shards = sorted(
        item["path"] for item in tree
        if item["path"].startswith("disease_classification/train-")
        and item["path"].endswith(".parquet")
    )
    cache_dir = out_dir / "shard_full_metadata_cache"
    cache_dir.mkdir(exist_ok=True)
    all_rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(read_shard, path, fields, cache_dir): path for path in shards}
        for future in as_completed(futures):
            result = future.result()
            all_rows.extend(result)
            print(f"{futures[future]}: {len(result)} nonnull rows", flush=True)
    by_key = defaultdict(list)
    for row in all_rows:
        by_key[tuple(row["key"])].append(row)
    selected = []
    matched_official = 0
    ambiguous_official = 0
    missing_official = 0
    for metadata_key, count in official.items():
        mirror_rows = by_key.get(metadata_key, [])
        if len(mirror_rows) == count:
            selected.extend(mirror_rows)
            matched_official += count
        elif len(mirror_rows) > count:
            ambiguous_official += count
        else:
            missing_official += count - len(mirror_rows)
            ambiguous_official += len(mirror_rows)
    report = {
        "status": "metadata_matching_complete_pixels_unverified",
        "official_rows": sum(official.values()),
        "hf_nonnull_rows": len(all_rows),
        "exact_multiplicity_candidate_official_rows": matched_official,
        "ambiguous_official_rows": ambiguous_official,
        "missing_official_rows": missing_official,
        "candidate_image_row_references": len(selected),
        "candidate_patients": len({row["key"][fields.index("Patient_ID")] for row in selected}),
        "key_fields": fields,
    }
    (out_dir / "hf_full_metadata_match_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    if selected:
        selected.sort(key=lambda row: (row["parquet_path"], row["row_in_shard"]))
        with (out_dir / "hf_metadata_candidate_image_rows.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["parquet_path", "row_in_shard", *fields])
            for row in selected:
                writer.writerow([row["parquet_path"], row["row_in_shard"], *row["key"]])
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels_zip", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run(args.labels_zip, args.out_dir, args.workers), indent=2))
