"""Resume and verify the official OLIVES.zip archive from Zenodo.

Downloads bounded HTTP ranges into a dedicated directory, then assembles and
checks the published MD5. Partial chunks are kept after transient failures.
Use a drive with at least 70 GB free during assembly.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import time
from pathlib import Path

import requests


# Both official Zenodo endpoints returned identical bytes for an overlapping
# range; a timed range probe found this direct endpoint faster on this host.
URL = "https://zenodo.org/records/7105232/files/OLIVES.zip?download=1"
EXPECTED_SIZE = 33_913_178_281
EXPECTED_MD5 = "e08f4198aa919d6900bd30525c137933"
CHUNK_BYTES = 64 * 1024 * 1024


def chunk_bounds(index: int) -> tuple[int, int]:
    first = index * CHUNK_BYTES
    return first, min(EXPECTED_SIZE - 1, first + CHUNK_BYTES - 1)


def chunk_path(parts_dir: Path, index: int) -> Path:
    return parts_dir / f"{index:04d}.part"


def fetch_chunk(index: int, parts_dir: Path) -> int:
    first, last = chunk_bounds(index)
    final = chunk_path(parts_dir, index)
    expected = last - first + 1
    if final.exists() and final.stat().st_size == expected:
        return index
    temporary = final.with_suffix(".download")
    if temporary.exists() and temporary.stat().st_size > expected:
        temporary.unlink()
    for attempt in range(5):
        try:
            present = temporary.stat().st_size if temporary.exists() else 0
            if present == expected:
                os.replace(temporary, final)
                return index
            resume_first = first + present
            with requests.get(
                URL,
                headers={"Range": f"bytes={resume_first}-{last}"},
                stream=True,
                timeout=(30, 180),
            ) as response:
                response.raise_for_status()
                expected_range = f"bytes {resume_first}-{last}/{EXPECTED_SIZE}"
                if response.status_code != 206 or response.headers.get("Content-Range") != expected_range:
                    raise OSError(f"Unexpected range response for chunk {index}: {response.status_code}")
                with temporary.open("ab") as file:
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        file.write(block)
            if temporary.stat().st_size != expected:
                raise OSError(f"Incomplete chunk {index}: {temporary.stat().st_size} != {expected}")
            os.replace(temporary, final)
            return index
        except (requests.RequestException, OSError):
            if attempt == 4:
                raise
            time.sleep(min(30, 2 ** attempt))
    raise AssertionError("unreachable")


def assemble(destination: Path, parts_dir: Path, count: int) -> str:
    assembled = destination.with_suffix(".assembling")
    md5 = hashlib.md5()
    with assembled.open("wb") as output:
        for index in range(count):
            part = chunk_path(parts_dir, index)
            first, last = chunk_bounds(index)
            if part.stat().st_size != last - first + 1:
                raise OSError(f"Incorrect part size: {part}")
            with part.open("rb") as source:
                while block := source.read(8 * 1024 * 1024):
                    output.write(block)
                    md5.update(block)
    if assembled.stat().st_size != EXPECTED_SIZE or md5.hexdigest() != EXPECTED_MD5:
        raise OSError(f"Archive checksum mismatch: {md5.hexdigest()}")
    os.replace(assembled, destination)
    return md5.hexdigest()


def run(destination_dir: Path, workers: int, check_only: bool) -> dict:
    destination_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = destination_dir / "OLIVES.parts"
    parts_dir.mkdir(exist_ok=True)
    destination = destination_dir / "OLIVES.zip"
    count = math.ceil(EXPECTED_SIZE / CHUNK_BYTES)
    present = [i for i in range(count) if chunk_path(parts_dir, i).exists()
               and chunk_path(parts_dir, i).stat().st_size == chunk_bounds(i)[1] - chunk_bounds(i)[0] + 1]
    state = {
        "source": URL,
        "workers": workers,
        "expected_size": EXPECTED_SIZE,
        "expected_md5": EXPECTED_MD5,
        "total_parts": count,
        "complete_parts": len(present),
        "archive_verified": False,
    }
    if destination.exists() and destination.stat().st_size == EXPECTED_SIZE:
        md5 = hashlib.md5()
        with destination.open("rb") as file:
            while block := file.read(8 * 1024 * 1024):
                md5.update(block)
        if md5.hexdigest() == EXPECTED_MD5:
            state["archive_verified"] = True
            state["complete_parts"] = count
            return state
    if check_only:
        return state
    remaining = [i for i in range(count) if i not in set(present)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_chunk, i, parts_dir): i for i in remaining}
        for number, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            future.result()
            state["complete_parts"] = len(present) + number
            if number % 10 == 0 or number == len(remaining):
                print(f"parts complete: {state['complete_parts']}/{count}", flush=True)
    state["actual_md5"] = assemble(destination, parts_dir, count)
    state["archive_verified"] = True
    (destination_dir / "download_receipt.json").write_text(json.dumps(state, indent=2) + "\n")
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination_dir", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.destination_dir, args.workers, args.check_only), indent=2))
