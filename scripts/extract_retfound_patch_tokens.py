from __future__ import annotations

"""Extract audited 14x14 frozen RETFound patch tokens to external storage."""

import argparse
import json
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from retina_lanse.data.embedding_shards import load_embedding_run
from retina_lanse.data.manifest import sha256_path
from retina_lanse.encoders import load_retfound_cfp
from retina_lanse.utils import verify_file_hash_bundle
from scripts.extract_retfound_embeddings import (
    ManifestImages,
    PREPROCESSING_VERSION,
    assert_under_external,
    choose_device,
    read_jsonl,
)


EXTERNAL_ROOT = Path("/Volumes/Elements/Retina-LanSE")
GRID_ROWS = 14
GRID_COLS = 14
TOKENS = GRID_ROWS * GRID_COLS
TOKEN_DIM = 1024


def load_selected_patients(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    groups = payload.get("patients") if isinstance(payload, dict) else None
    if not isinstance(groups, dict):
        raise RuntimeError("selected-patients file must contain a patients mapping")
    selected: set[str] = set()
    for group in ("sae_train", "development", "pilot_holdout"):
        values = groups.get(group)
        if not isinstance(values, list) or not values:
            raise RuntimeError(f"selected-patients file lacks nonempty {group} list")
        selected.update(str(value) for value in values)
    if len(selected) != 2000:
        raise RuntimeError(f"Experiment 8 requires 2,000 selected patients, found {len(selected)}")
    return selected


def verify_global_parity(
    paths: list[str],
    pooled: torch.Tensor,
    global_lookup: dict[str, torch.Tensor],
    tolerance: float,
) -> dict:
    expected = torch.stack([global_lookup[path] for path in paths])
    difference = (pooled.cpu() - expected).abs()
    maximum = float(difference.max().item())
    if maximum > tolerance:
        raise RuntimeError(f"patch/global pooling parity failed: max_abs={maximum} > {tolerance}")
    return {"max_abs_difference": maximum, "mean_abs_difference": float(difference.mean().item()), "tolerance": tolerance}


def smoke() -> dict:
    ids = torch.arange(TOKENS).reshape(GRID_ROWS, GRID_COLS)
    flattened = ids.reshape(-1)
    if not torch.equal(flattened, torch.arange(TOKENS)):
        raise RuntimeError("row-major patch ordering failed")
    row, column = divmod(137, GRID_COLS)
    if (row, column) != (9, 11):
        raise RuntimeError("patch coordinate conversion failed")
    return {"status": "smoke_passed", "grid": [GRID_ROWS, GRID_COLS], "token_shape": [TOKENS, TOKEN_DIM], "row_major": True, "coordinate_137": [row, column]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract frozen RETFound final patch tokens with global-pooling parity checks.")
    parser.add_argument("--dataset-root")
    parser.add_argument("--manifest")
    parser.add_argument("--checkpoint")
    parser.add_argument("--upstream-repo", default="/Volumes/Elements/Retina-LanSE/runtime/RETFound")
    parser.add_argument("--selected-patients")
    parser.add_argument("--global-run")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--parity-tolerance", type=float, default=2e-5,
                        help="maximum absolute pooling difference; 2e-5 accommodates CPU floating-point reduction order")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    output = assert_under_external(Path(args.output), EXTERNAL_ROOT, "output")
    output.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        record = smoke()
        (output / "patch_extraction_smoke.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(record, sort_keys=True))
        return 0
    required = ("dataset_root", "manifest", "checkpoint", "selected_patients", "global_run")
    absent = [name for name in required if not getattr(args, name)]
    if absent:
        parser.error("missing: " + ", ".join(absent))
    dataset_root = assert_under_external(Path(args.dataset_root), EXTERNAL_ROOT, "dataset root")
    manifest = assert_under_external(Path(args.manifest), EXTERNAL_ROOT, "manifest")
    checkpoint = assert_under_external(Path(args.checkpoint), EXTERNAL_ROOT, "checkpoint")
    upstream = assert_under_external(Path(args.upstream_repo), EXTERNAL_ROOT, "upstream repo")
    selected_path = assert_under_external(Path(args.selected_patients), EXTERNAL_ROOT, "selected patients")
    global_root = assert_under_external(Path(args.global_run), EXTERNAL_ROOT, "global run")
    selected = load_selected_patients(selected_path)
    records = [record for record in read_jsonl(manifest) if record["split"] == "train" and str(record["patient_id"]) in selected]
    if not records or {str(record["patient_id"]) for record in records} != selected:
        raise RuntimeError("selected Experiment 8 patients are absent from the train manifest")
    if any(record["split"] != "train" for record in records):
        raise RuntimeError("Experiment 8 patch extraction rejects non-training splits")
    print(json.dumps({"event": "selected_records", "images": len(records), "patients": len(selected)}), flush=True)
    global_run = load_embedding_run(global_root, expected_split="train", verify_sha256=True)
    print(json.dumps({"event": "global_run_verified", "images": len(global_run.relative_paths)}), flush=True)
    global_lookup = {path: embedding for path, embedding in zip(global_run.relative_paths, global_run.embeddings)}
    if not all(record["relative_path"] in global_lookup for record in records):
        raise RuntimeError("global dense run does not cover selected patch-extraction records")
    if global_run.evidence.get("representation") != "final_mean_patch":
        raise RuntimeError("global parity requires final_mean_patch embeddings")

    source_evidence_path = global_root / "extraction_evidence.json"
    source_evidence = json.loads(source_evidence_path.read_text(encoding="utf-8"))
    if Path(source_evidence.get("checkpoint", "")).resolve() != checkpoint.resolve() or not isinstance(source_evidence.get("checkpoint_hash"), str):
        raise RuntimeError("global extraction evidence does not attest to this exact RETFound checkpoint path and hash")
    device = choose_device(args.device)
    model = load_retfound_cfp(checkpoint, upstream, representation="final_mean_patch", recorded_checkpoint_hash=source_evidence["checkpoint_hash"]).to(device).eval()
    print(json.dumps({"event": "retfound_loaded", "device": str(device)}), flush=True)
    evidence_path = output / "patch_extraction_evidence.json"
    if evidence_path.is_file():
        prior = json.loads(evidence_path.read_text(encoding="utf-8"))
        if prior.get("status") == "complete" and prior.get("manifest_sha256") == sha256_path(manifest):
            print(json.dumps({"status": "reused", "output": str(evidence_path)}))
            return 0
        raise RuntimeError(f"incompatible existing output: {evidence_path}")

    started = time.perf_counter()
    shard_evidence = []
    parity_values = []
    for start in range(0, len(records), args.shard_size):
        shard_records = records[start : start + args.shard_size]
        end = start + len(shard_records) - 1
        shard_path = output / f"patch_tokens_{start:06d}_{end:06d}.pt"
        loader = DataLoader(ManifestImages(dataset_root, shard_records), batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        tokens, globals_ = [], []
        with torch.inference_mode():
            for images, indices in loader:
                images = images.to(device)
                raw_tokens = model.forward_patch_tokens(images)
                pooled = model.pool_patch_tokens(raw_tokens)
                batch_paths = [shard_records[int(index)]["relative_path"] for index in indices]
                parity_values.append(verify_global_parity(batch_paths, pooled, global_lookup, args.parity_tolerance))
                tokens.append(raw_tokens.cpu().to(torch.float16))
                globals_.append(pooled.cpu())
        payload = {
            "schema_version": "1.0",
            "checkpoint_hash": model.checkpoint_hash,
            "representation": "final_patch_prepool",
            "preprocessing_version": PREPROCESSING_VERSION,
            "grid_rows": GRID_ROWS,
            "grid_columns": GRID_COLS,
            "relative_paths": [record["relative_path"] for record in shard_records],
            "patient_ids": [record["patient_id"] for record in shard_records],
            "patch_tokens": torch.cat(tokens, dim=0),
            "pooled_global_embeddings": torch.cat(globals_, dim=0),
        }
        if payload["patch_tokens"].shape != (len(shard_records), TOKENS, TOKEN_DIM):
            raise RuntimeError("patch-token shard shape mismatch")
        torch.save(payload, shard_path)
        shard_evidence.append({"path": str(shard_path), "images": len(shard_records), "bytes": shard_path.stat().st_size, "sha256": sha256_path(shard_path)})
        print(json.dumps({"event": "patch_shard_complete", "processed_images": end + 1, "total_images": len(records), "shard": str(shard_path)}), flush=True)
    record = {
        "schema_version": "1.0",
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "images": len(records),
        "patients": len(selected),
        "failures": 0,
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(device),
        "checkpoint": str(checkpoint),
        "checkpoint_hash": model.checkpoint_hash,
        "checkpoint_hash_provenance": {"source": str(source_evidence_path), "source_sha256": sha256_path(source_evidence_path), "mode": "prior_verified_global_extraction"},
        "manifest": str(manifest),
        "manifest_sha256": sha256_path(manifest),
        "selected_patients_sha256": sha256_path(selected_path),
        "global_run_evidence_sha256": sha256_path(global_root / "extraction_evidence.json"),
        "representation": "final_patch_prepool",
        "grid": [GRID_ROWS, GRID_COLS],
        "patch_token_dtype": "float16",
        "parity": {"max_abs_difference": max(item["max_abs_difference"] for item in parity_values), "mean_abs_difference": sum(item["mean_abs_difference"] for item in parity_values) / len(parity_values), "tolerance": args.parity_tolerance},
        "shards": shard_evidence,
        "platform": platform.platform(),
        "process_max_rss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    evidence_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(evidence_path), "parity": record["parity"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
