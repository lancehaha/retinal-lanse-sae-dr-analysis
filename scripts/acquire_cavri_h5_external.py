#!/usr/bin/env python3
"""Acquire and verify the public CAVRI-H5 archive without extracting it."""

import argparse
import hashlib
import json
import stat
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

DATASET_URL = "https://data.mendeley.com/public-api/datasets/gfkdznwyvj"
EXPECTED_SHA256 = "8c0cb58bd3e6c32d4200cdde05b884a1dd42be044b17d3a5af51a405e794f4c0"
EXPECTED_BYTES = 174985279
EXPECTED_FILE_ID = "c7aec88e-b542-447e-aac0-c0ef49604300"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    archive = args.output_root / "CAVRI-H5_CLAN_tiff.zip"
    part = args.output_root / "CAVRI-H5_CLAN_tiff.zip.part"
    receipt = args.output_root / "archive_audit_complete.json"
    inventory = args.output_root / "archive_members.json"
    assert not any(p.exists() for p in (archive, part, receipt, inventory)), "Refusing to overwrite source or receipt"
    request = urllib.request.Request(DATASET_URL, headers={"User-Agent": "RetinaVLM-research/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        metadata = json.load(response)
    assert metadata["version"] == 1 and metadata["doi"]["id"] == "10.17632/gfkdznwyvj.1"
    assert metadata["data_licence"]["short_name"] == "CC BY 4.0"
    assert len(metadata["files"]) == 1
    item = metadata["files"][0]
    assert item["id"] == EXPECTED_FILE_ID and item["size"] == EXPECTED_BYTES
    assert item["content_details"]["sha256_hash"] == EXPECTED_SHA256
    url = item["content_details"]["download_url"]
    assert url.startswith("https://data.mendeley.com/public-files/datasets/gfkdznwyvj/")
    request = urllib.request.Request(url, headers={"User-Agent": "RetinaVLM-research/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, part.open("wb") as output:
        for chunk in iter(lambda: response.read(1024 * 1024), b""):
            output.write(chunk)
    assert part.stat().st_size == EXPECTED_BYTES
    assert sha256(part) == EXPECTED_SHA256
    members = []
    with zipfile.ZipFile(part) as source:
        for item in source.infolist():
            path = PurePosixPath(item.filename)
            assert not path.is_absolute() and ".." not in path.parts and not item.filename.startswith("\\")
            assert not stat.S_ISLNK(item.external_attr >> 16)
            members.append({"name": item.filename, "compressed_bytes": item.compress_size, "uncompressed_bytes": item.file_size, "crc32": item.CRC})
    assert len(members) >= 609
    assert sum(x["uncompressed_bytes"] for x in members) < 2_000_000_000
    part.rename(archive)
    inventory.write_text(json.dumps(members, indent=2) + "\n")
    result = {
        "status": "public_external_archive_sha_verified_no_extraction",
        "dataset_url": DATASET_URL,
        "doi": metadata["doi"]["id"],
        "license": metadata["data_licence"]["short_name"],
        "filename": archive.name,
        "file_id": EXPECTED_FILE_ID,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256(archive),
        "member_count": len(members),
        "member_inventory_sha256": sha256(inventory),
        "uncompressed_bytes": sum(x["uncompressed_bytes"] for x in members),
        "extracted": False,
        "olives_accessed": False,
        "claim_boundary": "VMA-VMT single-label external images are not OLIVES B2/B3 labels or gate confirmation",
    }
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
