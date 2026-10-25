"""Fail-closed readiness check for the one-time external Stage 6 evaluation.

This module deliberately does not download data or run model inference.  It
checks the provenance and freeze conditions that must be true before a
licensed IDRiD/FGADR evaluation is opened.  A blocked result is useful
evidence: it records exactly which external input is still missing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_EXTERNAL_ROOT = Path("/Volumes/Elements/Retina-LanSE")
ALLOWED_DATASETS = {"idrid", "fgadr"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def under(root: Path, path: Path, label: str) -> Path:
    root = root.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"{label} must stay under {root}: {resolved}") from exc
    return resolved


def read_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_access_record(record: dict[str, Any]) -> list[str]:
    """Return missing/invalid fields in an institution-approved access record."""

    required_nonempty = {
        "source_url": "official source URL",
        "license": "license or data-use terms",
        "agreement_reference": "agreement/reference",
        "archive_sha256": "archive SHA-256",
        "access_date": "access date",
    }
    reasons: list[str] = []
    if record.get("approved") is not True:
        reasons.append("access record does not state approved=true")
    for key, label in required_nonempty.items():
        value = record.get(key)
        if not isinstance(value, str) or not value.strip():
            reasons.append(f"missing {label} ({key})")
    for key, label in (
        ("labels_verified", "verified labels"),
        ("patient_grouping_verified", "verified patient grouping"),
        ("preprocessing_recorded", "recorded preprocessing"),
    ):
        if record.get(key) is not True:
            reasons.append(f"{label} is not confirmed ({key}=true required)")
    archive_sha = record.get("archive_sha256")
    if isinstance(archive_sha, str) and archive_sha and len(archive_sha) != 64:
        reasons.append("archive_sha256 is not a 64-character SHA-256 digest")
    return reasons


def _row_image_path(row: dict[str, Any]) -> str | None:
    for key in ("relative_path", "image_path", "path", "filename"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def check_manifest(manifest_path: Path, dataset_root: Path) -> tuple[list[str], dict[str, Any]]:
    """Validate the minimum image/label/patient manifest contract.

    The manifest is intentionally required even for image-level evaluation:
    patient grouping and immutable file hashes are necessary to make a
    patient-clustered confidence interval reproducible.
    """

    reasons: list[str] = []
    rows = 0
    patients: set[str] = set()
    labels = 0
    hashes = 0
    if not manifest_path.is_file():
        return [f"missing external manifest: {manifest_path}"], {
            "rows": 0,
            "patients": 0,
            "labels": 0,
            "sha256": 0,
        }
    try:
        handle = manifest_path.open("r", encoding="utf-8")
    except OSError as exc:
        return [f"cannot open external manifest: {exc}"], {
            "rows": 0,
            "patients": 0,
            "labels": 0,
            "sha256": 0,
        }
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                reasons.append(f"manifest line {line_number} is invalid JSON: {exc.msg}")
                continue
            if not isinstance(row, dict):
                reasons.append(f"manifest line {line_number} is not an object")
                continue
            rows += 1
            image_value = _row_image_path(row)
            if image_value is None:
                reasons.append(f"manifest line {line_number} has no image path")
            else:
                image_path = (dataset_root / image_value).resolve()
                try:
                    image_path.relative_to(dataset_root.resolve())
                except ValueError:
                    reasons.append(f"manifest line {line_number} image escapes dataset root")
                else:
                    if not image_path.is_file():
                        reasons.append(f"manifest line {line_number} image is missing: {image_value}")
                    elif image_path.suffix.lower() not in IMAGE_SUFFIXES:
                        reasons.append(f"manifest line {line_number} is not an image: {image_value}")
            patient = row.get("patient_id")
            if not isinstance(patient, str) or not patient.strip():
                reasons.append(f"manifest line {line_number} has no patient_id")
            else:
                patients.add(patient)
            if any(key in row for key in ("label", "grade", "dr_grade", "lesion_masks", "masks")):
                labels += 1
            if isinstance(row.get("sha256"), str) and len(row["sha256"]) == 64:
                hashes += 1
            else:
                reasons.append(f"manifest line {line_number} has no valid image SHA-256")
    if rows == 0:
        reasons.append("external manifest contains no records")
    if labels != rows:
        reasons.append(f"labels/masks are missing for {rows - labels} manifest rows")
    if hashes != rows:
        reasons.append(f"image SHA-256 is missing for {rows - hashes} manifest rows")
    return reasons, {"rows": rows, "patients": len(patients), "labels": labels, "sha256": hashes}


def check_stage6_gate(
    *,
    dataset: str,
    external_root: Path,
    dataset_root: Path,
    approval_path: Path,
    manifest_path: Path,
    release_manifest_path: Path,
    clinician_status_path: Path,
) -> dict[str, Any]:
    """Return a deterministic, JSON-serialisable Stage 6 gate report."""

    if dataset not in ALLOWED_DATASETS:
        raise ValueError(f"dataset must be one of {sorted(ALLOWED_DATASETS)}")
    external_root = external_root.resolve()
    dataset_root = under(external_root, dataset_root, "dataset root")
    approval_path = under(external_root, approval_path, "access record")
    manifest_path = under(external_root, manifest_path, "external manifest")
    release_manifest_path = under(external_root, release_manifest_path, "release manifest")
    clinician_status_path = under(external_root, clinician_status_path, "clinician status")

    reasons: list[str] = []
    evidence: dict[str, Any] = {
        "external_root": str(external_root),
        "dataset_root": str(dataset_root),
        "dataset": dataset,
        "dataset_root_exists": dataset_root.is_dir(),
        "approval_path": str(approval_path),
        "manifest_path": str(manifest_path),
        "release_manifest_path": str(release_manifest_path),
        "clinician_status_path": str(clinician_status_path),
    }
    if not external_root.is_dir():
        reasons.append(f"external drive is not mounted: {external_root}")
    if not dataset_root.is_dir():
        reasons.append(f"approved dataset directory is missing: {dataset_root}")

    if approval_path.is_file():
        try:
            access = read_object(approval_path, "access record")
            access_reasons = check_access_record(access)
            reasons.extend(access_reasons)
            evidence["access_record"] = {
                "approved": access.get("approved"),
                "source_url": access.get("source_url"),
                "license": access.get("license"),
                "access_date": access.get("access_date"),
                "archive_sha256": access.get("archive_sha256"),
                "checks": access_reasons,
            }
        except RuntimeError as exc:
            reasons.append(str(exc))
    else:
        reasons.append(f"missing approved access record: {approval_path}")

    manifest_reasons, manifest_stats = check_manifest(manifest_path, dataset_root)
    reasons.extend(manifest_reasons)
    evidence["manifest"] = manifest_stats

    if release_manifest_path.is_file():
        try:
            release = read_object(release_manifest_path, "release manifest")
            evidence["release_manifest"] = {
                "all_heavy_outputs_on_external_drive": release.get("all_heavy_outputs_on_external_drive"),
                "locked_internal_test_accessed": release.get("locked_internal_test_accessed"),
                "paper_claim_boundary": release.get("paper_claim_boundary"),
            }
            if release.get("all_heavy_outputs_on_external_drive") is not True:
                reasons.append("release manifest does not confirm external-only heavy outputs")
            if release.get("locked_internal_test_accessed") is not True:
                reasons.append("release manifest does not record the frozen internal-test access")
        except RuntimeError as exc:
            reasons.append(str(exc))
    else:
        reasons.append(f"missing frozen release manifest: {release_manifest_path}")

    if clinician_status_path.is_file():
        try:
            clinician = read_object(clinician_status_path, "clinician status")
            evidence["clinician_review"] = {
                "status": clinician.get("status"),
                "complete_rows": clinician.get("complete_rows"),
                "expected_rows": clinician.get("expected_rows"),
            }
            if clinician.get("status") != "complete_clinician_reviews":
                reasons.append("independent clinician review is not complete")
        except RuntimeError as exc:
            reasons.append(str(exc))
    else:
        reasons.append(f"missing clinician review status: {clinician_status_path}")

    return {
        "schema_version": "1.0",
        "status": "ready_for_one_time_run" if not reasons else "blocked_external_access",
        "dataset": dataset,
        "checked_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "inference_performed": False,
        "reasons": reasons,
        "evidence": evidence,
        "required_next_action": (
            "Run the frozen external evaluator exactly once and append immutable metrics to the release manifest."
            if not reasons
            else "Resolve every recorded gate reason; this check intentionally ran no inference."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check readiness for the one-time Stage 6 external evaluation.")
    parser.add_argument("--dataset", choices=sorted(ALLOWED_DATASETS), required=True)
    parser.add_argument("--external-root", default=str(DEFAULT_EXTERNAL_ROOT))
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--approval", required=True, help="Institutional access/license record JSON")
    parser.add_argument("--manifest", required=True, help="Verified external JSONL image/label manifest")
    parser.add_argument("--release-manifest", required=True)
    parser.add_argument("--clinician-status", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    external_root = Path(args.external_root)
    output = under(external_root, Path(args.output), "gate output")
    report = check_stage6_gate(
        dataset=args.dataset,
        external_root=external_root,
        dataset_root=Path(args.dataset_root),
        approval_path=Path(args.approval),
        manifest_path=Path(args.manifest),
        release_manifest_path=Path(args.release_manifest),
        clinician_status_path=Path(args.clinician_status),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "ready_for_one_time_run" else 2


if __name__ == "__main__":
    raise SystemExit(main())
