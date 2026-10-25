"""Run the official OLIVES image and near-duplicate audits after download.

Wait for the downloader's verified receipt rather than opening a partial ZIP.
The materializer independently rechecks the full archive MD5 before extracting
expert-labeled scans. Candidate near duplicates still require visual review.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from audit_retinavlm_olives_near_duplicates import audit as near_duplicate_audit
from materialize_retinavlm_olives_images import OFFICIAL_MD5, run as materialize


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def wait_for_archive(download_dir: Path, download_pid: int,
                     interval_seconds: int = 30) -> Path:
    receipt = download_dir / "download_receipt.json"
    archive = download_dir / "OLIVES.zip"
    while not receipt.is_file():
        if not process_alive(download_pid):
            time.sleep(2)  # allow an atomic receipt write at process exit
            if not receipt.is_file():
                raise RuntimeError("Downloader stopped without a verified archive receipt")
        else:
            time.sleep(interval_seconds)
    state = json.loads(receipt.read_text(encoding="utf-8"))
    if (state.get("archive_verified") is not True
            or state.get("actual_md5") != OFFICIAL_MD5
            or not archive.is_file()):
        raise ValueError("Download receipt does not attest the official archive")
    return archive


def run(download_dir: Path, download_pid: int, labels_manifest: Path,
        label_audit: Path, output_dir: Path) -> dict:
    archive = wait_for_archive(download_dir, download_pid)
    print("Official archive receipt found; verifying and matching expert images", flush=True)
    image_report = materialize(archive, labels_manifest, label_audit,
                               output_dir, expected_rows=9408)
    print(json.dumps({"image_audit": image_report}, indent=2), flush=True)
    manifest = output_dir / "official_expert_image_manifest.csv"
    audit_receipt = output_dir / "official_image_audit.json"
    candidate_report = near_duplicate_audit(manifest, audit_receipt, 4, 1000)
    destination = output_dir / "cross_split_near_duplicate_candidates.json"
    temporary = destination.with_suffix(destination.suffix + ".writing")
    temporary.write_text(json.dumps(candidate_report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    print(json.dumps({
        "status": "official_image_audit_and_candidate_screen_complete",
        "matched_images": image_report["matched_images"],
        "candidate_pairs": candidate_report["candidate_pair_count"],
        "candidate_report_truncated": candidate_report["truncated"],
        "review_required": candidate_report["candidate_pair_count"] > 0,
        "candidate_report": str(destination),
    }, indent=2), flush=True)
    return candidate_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("download_dir", type=Path)
    parser.add_argument("download_pid", type=int)
    parser.add_argument("labels_manifest", type=Path)
    parser.add_argument("label_audit", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    run(args.download_dir, args.download_pid, args.labels_manifest,
        args.label_audit, args.output_dir)
