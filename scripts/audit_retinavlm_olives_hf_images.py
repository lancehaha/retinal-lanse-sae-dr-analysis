"""Compare Hugging Face OLIVES image-row labels with the expert-label archive.

Reads only Parquet label columns by HTTP byte range. This does not download OCT
pixels or inspect test images. Install `pyarrow` and `requests` first.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import time
import urllib.request
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq
import requests


REPO = "gOLIVES/OLIVES_Dataset"
TREE = f"https://huggingface.co/api/datasets/{REPO}/tree/main?recursive=true&limit=1000"
CSV_IN_ZIP = (
    "OLIVES_Dataset_Labels/ml_centric_labels/"
    "Biomarker_Clinical_Data_Images.csv"
)
COLUMNS = ["Fluid (IRF)", "Fluid (SRF)", "Patient_ID", "Eye_ID", "Scan (n/49)"]


class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url: str, size: int):
        self.url = url
        self.size = size
        self.position = 0
        self.session = requests.Session()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        self.position = (
            offset if whence == 0 else
            self.position + offset if whence == 1 else
            self.size + offset
        )
        return self.position

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self.size - self.position
        if size == 0:
            return b""
        first = self.position
        last = min(self.size - 1, first + size - 1)
        for attempt in range(4):
            try:
                response = self.session.get(
                    self.url,
                    headers={"Range": f"bytes={first}-{last}"},
                    timeout=45,
                )
                response.raise_for_status()
                if response.status_code != 206 or len(response.content) != last - first + 1:
                    raise OSError(f"Unexpected HTTP range response: {response.status_code}")
                self.position += len(response.content)
                return response.content
            except (requests.RequestException, OSError):
                if attempt == 3:
                    raise
                time.sleep(1 + attempt)
        raise AssertionError("unreachable")


def binary(value: float | None) -> int | None:
    return int(value) if value in (0, 1) else None


def official_counter(labels_zip: Path) -> Counter:
    with zipfile.ZipFile(labels_zip) as archive:
        with archive.open(CSV_IN_ZIP) as binary_file:
            rows = csv.DictReader(io.TextIOWrapper(binary_file, encoding="utf-8-sig"))
            return Counter(
                (
                    int(row["Patient_ID"]), int(row["Eye_ID"]),
                    int(row["Scan (n/49)"]),
                    int(row["Fluid (IRF)"]), int(row["Fluid (SRF)"]),
                )
                for row in rows
            )


def read_shard(path: str, cache_dir: Path) -> list[dict]:
    cache = cache_dir / (Path(path).name + ".json")
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{path}"
    head = requests.head(url, allow_redirects=True, timeout=30)
    head.raise_for_status()
    size = int(head.headers["content-length"])
    parquet = pq.ParquetFile(HTTPRangeFile(url, size))
    table = parquet.read(columns=COLUMNS)
    columns = table.to_pydict()
    matched = []
    for index in range(len(table)):
        irf, srf = binary(columns["Fluid (IRF)"][index]), binary(columns["Fluid (SRF)"][index])
        if irf is None or srf is None:
            continue
        patient, eye, scan = (
            columns["Patient_ID"][index],
            columns["Eye_ID"][index],
            columns["Scan (n/49)"][index],
        )
        if None in (patient, eye, scan):
            raise ValueError(f"Missing identity field at {path} row {index}")
        matched.append({
            "parquet_path": path,
            "row_in_shard": index,
            "Patient_ID": int(patient),
            "Eye_ID": int(eye),
            "scan_n": int(scan),
            "IRF": irf,
            "SRF": srf,
            "fluid": int(bool(irf or srf)),
        })
    cache.write_text(json.dumps(matched, separators=(",", ":")) + "\n", encoding="utf-8")
    return matched


def run(labels_zip: Path, out_dir: Path, workers: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    tree = json.load(urllib.request.urlopen(TREE, timeout=30))
    shards = sorted(
        item["path"] for item in tree
        if item["path"].startswith("disease_classification/train-")
        and item["path"].endswith(".parquet")
    )
    if not shards:
        raise RuntimeError("No disease_classification Parquet shards found")
    cache_dir = out_dir / "shard_label_cache"
    cache_dir.mkdir(exist_ok=True)
    all_rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(read_shard, path, cache_dir): path for path in shards}
        for future in as_completed(futures):
            rows = future.result()
            all_rows.extend(rows)
            print(f"{futures[future]}: {len(rows)} nonnull fluid-label rows", flush=True)
    all_rows.sort(key=lambda row: (row["parquet_path"], row["row_in_shard"]))
    actual = Counter(
        (row["Patient_ID"], row["Eye_ID"], row["scan_n"], row["IRF"], row["SRF"])
        for row in all_rows
    )
    expected = official_counter(labels_zip)
    report = {
        "hf_repo": REPO,
        "shards": len(shards),
        "hf_nonnull_fluid_rows": len(all_rows),
        "official_expert_rows": sum(expected.values()),
        "hf_patients": len({row["Patient_ID"] for row in all_rows}),
        "label_tuple_multiset_match": actual == expected,
        "missing_official_tuple_instances": sum((expected - actual).values()),
        "extra_hf_tuple_instances": sum((actual - expected).values()),
        "status": (
            "label_tuple_multiset_matched_pixels_not_downloaded"
            if actual == expected else
            "hf_nonnull_labels_exceed_official_expert_set_exact_image_identity_unverified"
        ),
    }
    (out_dir / "hf_image_row_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    if actual != expected:
        return report
    fields = ["parquet_path", "row_in_shard", "Patient_ID", "Eye_ID", "scan_n", "IRF", "SRF", "fluid"]
    with (out_dir / "hf_expert_image_rows.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels_zip", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run(args.labels_zip, args.out_dir, args.workers), indent=2))
