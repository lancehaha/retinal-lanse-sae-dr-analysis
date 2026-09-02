from __future__ import annotations

import argparse
from collections import Counter
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retina_lanse.data.manifest import sha256_path


SHARD_PATTERN = re.compile(r"embeddings_(\d{6})_(\d{6})\.pt$")


def json_status(path: Path, expected_status: str) -> bool:
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == expected_status
    except (json.JSONDecodeError, OSError):
        return False


def embedding_status(path: Path, split: str, expected_images: int, manifest_hash: str) -> dict:
    evidence_path = path / "extraction_evidence.json"
    evidence = {}
    if evidence_path.is_file():
        try:
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            evidence = {}
    complete = (
        evidence.get("status") == "complete"
        and evidence.get("images") == expected_images
        and evidence.get("manifest_hash") == manifest_hash
    )
    images = 0
    shards = 0
    shard_records = []
    for shard in path.glob("embeddings_*.pt") if path.is_dir() else ():
        match = SHARD_PATTERN.match(shard.name)
        if match:
            shard_images = int(match.group(2)) - int(match.group(1)) + 1
            images += shard_images
            shards += 1
            shard_records.append((shard.stat().st_mtime, shard_images))
    shard_records.sort()
    recent_rate = None
    eta_seconds = None
    if len(shard_records) >= 2:
        recent = shard_records[-min(6, len(shard_records)) :]
        elapsed = recent[-1][0] - recent[0][0]
        if elapsed > 0:
            recent_rate = sum(item[1] for item in recent[1:]) / elapsed
            eta_seconds = max(expected_images - images, 0) / recent_rate
    status = {
        "complete": complete,
        "images": images,
        "expected_images": expected_images,
        "percent": round(100.0 * images / expected_images, 2) if expected_images else 0.0,
        "shards": shards,
        "evidence": str(evidence_path),
        "latest_shard_utc": (
            datetime.fromtimestamp(shard_records[-1][0], timezone.utc).isoformat()
            if shard_records
            else None
        ),
        "recent_images_per_second": round(recent_rate, 3) if recent_rate else None,
        "estimated_seconds_remaining": round(eta_seconds) if eta_seconds is not None else None,
    }
    return status


def active_stage(
    representations: dict,
    *,
    near_duplicates_passed: bool,
    frozen: bool,
    audited: bool,
    report_exists: bool,
) -> str:
    mean = representations["final_mean_patch"]
    cls = representations["final_cls"]
    if audited and report_exists:
        return "complete"
    if not mean["embeddings"]["train"]["complete"]:
        return "mean_patch_train_extraction"
    if not near_duplicates_passed:
        return "near_duplicate_leakage_audit"
    if not mean["embeddings"]["validation"]["complete"]:
        return "mean_patch_validation_extraction"
    if not mean["screen_complete"]:
        return "mean_patch_sae_screen"
    if not mean["selection_frozen"]:
        return "mean_patch_sae_finalists"
    if not mean["validation_baselines_complete"]:
        return "mean_patch_validation_baselines"
    if not cls["embeddings"]["train"]["complete"]:
        return "cls_train_extraction"
    if not cls["embeddings"]["validation"]["complete"]:
        return "cls_validation_extraction"
    if not cls["screen_complete"]:
        return "cls_sae_screen"
    if not cls["selection_frozen"]:
        return "cls_sae_finalists"
    if not cls["validation_baselines_complete"]:
        return "cls_validation_baselines"
    if not frozen:
        return "protocol_freeze"
    if not mean["embeddings"]["internal_test"]["complete"]:
        return "mean_patch_internal_test_extraction"
    if not mean["locked_test_complete"]:
        return "mean_patch_locked_test_evaluation"
    if not cls["embeddings"]["internal_test"]["complete"]:
        return "cls_internal_test_extraction"
    if not cls["locked_test_complete"]:
        return "cls_locked_test_evaluation"
    if not audited:
        return "completion_audit"
    return "results_report"


def count_complete_metrics(path: Path, scope: str) -> int:
    count = 0
    for metrics_path in path.glob("*/metrics.json") if path.is_dir() else ():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if metrics.get("status") == "complete" and metrics.get("scope") == scope:
            count += 1
    return count


def count_finalist_seeds(path: Path) -> int:
    count = 0
    for metrics_path in path.glob("*/seed_*/metrics.json") if path.is_dir() else ():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if metrics.get("status") == "complete" and metrics.get("scope") == "publication_finalist":
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only Experiment 1 progress report.")
    parser.add_argument("--external-root", default="/Volumes/Elements/Retina-LanSE")
    args = parser.parse_args()
    root = Path(args.external_root).resolve()
    if not root.is_dir():
        parser.error(f"external root is not mounted: {root}")
    embeddings = root / "embeddings"
    experiments = root / "experiments"
    manifest = root / "datasets/eyepacs/manifest_publication_v1.jsonl"
    if not manifest.is_file():
        manifest = root / "datasets/eyepacs/manifest.jsonl"
    if not manifest.is_file():
        raise RuntimeError(f"dataset manifest is missing: {manifest}")
    expected_images = Counter()
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                expected_images[json.loads(line)["split"]] += 1
    manifest_hash = sha256_path(manifest)
    representations = {}
    for representation, short in (("final_mean_patch", "mean_patch"), ("final_cls", "cls")):
        run_root = embeddings / f"global_{representation}"
        screen_root = experiments / f"paper_global_sae_screen_{short}"
        finalist_root = experiments / f"paper_global_sae_finalists_{short}"
        representations[representation] = {
            "embeddings": {
                split: embedding_status(
                    run_root / f"{split}_cpu_fp32",
                    split,
                    expected_images[split],
                    manifest_hash,
                )
                for split in ("train", "validation", "internal_test")
            },
            "screen_runs_complete": count_complete_metrics(screen_root, "publication_screen"),
            "screen_runs_expected": 54,
            "screen_complete": json_status(
                screen_root / "search_summary.json", "complete_publication_screen"
            ),
            "finalist_seed_runs_complete": count_finalist_seeds(finalist_root),
            "finalist_seed_runs_expected": 15,
            "selection_frozen": json_status(
                finalist_root / "final_selection.json", "selection_frozen"
            ),
            "validation_baselines_complete": json_status(
                experiments
                / f"paper_representation_baselines_{short}/validation_baselines.json",
                "complete",
            ),
            "internal_test_baselines_complete": json_status(
                experiments
                / f"paper_representation_baselines_{short}_test/internal_test_baselines.json",
                "complete",
            ),
            "locked_test_complete": json_status(
                experiments
                / f"paper_locked_internal_test_{short}/locked_internal_test_evaluation.json",
                "complete_locked_internal_test",
            ),
        }
    frozen = json_status(
        experiments / "internal_test_freeze_manifest.json", "test_access_authorized"
    )
    audited = json_status(
        experiments / "experiment_1_completion_audit.json", "experiment_1_complete"
    )
    report_exists = (root / "reports/EXPERIMENT_1_RESULTS.md").is_file()
    near_duplicates_passed = json_status(
        root / "datasets/eyepacs/near_duplicate_audit.json", "pass"
    )
    storage = shutil.disk_usage(root)
    payload = {
        "experiment": "experiment_1_global_sae_reconstruction",
        "external_root": str(root),
        "dataset_manifest": str(manifest),
        "dataset_manifest_sha256": manifest_hash,
        "expected_split_images": dict(sorted(expected_images.items())),
        "representations": representations,
        "runtime_snapshot_complete": json_status(
            experiments / "experiment_1_runtime_snapshot.json", "complete"
        ),
        "test_access_frozen": frozen,
        "completion_audit_passed": audited,
        "paper_report_exists": report_exists,
        "near_duplicate_audit_passed": near_duplicates_passed,
        "external_storage": {
            "total_bytes": storage.total,
            "used_bytes": storage.used,
            "free_bytes": storage.free,
        },
    }
    payload["active_stage"] = active_stage(
        representations,
        near_duplicates_passed=near_duplicates_passed,
        frozen=frozen,
        audited=audited,
        report_exists=report_exists,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
