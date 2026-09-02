from __future__ import annotations

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
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from retina_lanse.data.manifest import sha256_path
from retina_lanse.data.preprocessing import preprocess_retfound_cfp
from retina_lanse.encoders import load_retfound_cfp
from retina_lanse.utils import verify_file_hash_bundle


PREPROCESSING_VERSION = "retinal-field-crop-v1+retfound-cfp-channel-zscore-v1"


class ManifestImages(Dataset):
    def __init__(self, root: Path, records: list[dict]):
        self.root, self.records = root, records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        path = (self.root / record["relative_path"]).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise RuntimeError(f"manifest path escapes dataset root: {path}") from exc
        with Image.open(path) as image:
            tensor = preprocess_retfound_cfp(image)
        return tensor, index


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def assert_under_external(path: Path, external_root: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(external_root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"{label} must be under external root {external_root}: {resolved}") from exc
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract frozen official RETFound CFP embeddings into resumable shards.")
    parser.add_argument("--external-root", default="/Volumes/Elements/Retina-LanSE")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--upstream-repo", default="/Volumes/Elements/Retina-LanSE/runtime/RETFound")
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("train", "validation", "internal_test", "all"), default="all")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=10000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--representation", choices=("final_mean_patch", "final_cls"), default="final_mean_patch")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--test-freeze-manifest")
    args = parser.parse_args()

    external_root = Path(args.external_root).resolve()
    if not external_root.is_dir():
        parser.error(f"external root is not mounted: {external_root}")
    dataset_root = assert_under_external(Path(args.dataset_root), external_root, "dataset root")
    manifest_path = Path(args.manifest).resolve()
    checkpoint = assert_under_external(Path(args.checkpoint), external_root, "checkpoint")
    upstream_repo = assert_under_external(Path(args.upstream_repo), external_root, "upstream repository")
    output = assert_under_external(Path(args.output), external_root, "output")
    output.mkdir(parents=True, exist_ok=True)

    if args.split == "internal_test":
        if not args.test_freeze_manifest:
            parser.error("internal_test extraction requires --test-freeze-manifest")
        freeze_path = assert_under_external(
            Path(args.test_freeze_manifest), external_root, "test freeze manifest"
        )
        if not freeze_path.is_file():
            parser.error(f"test freeze manifest does not exist: {freeze_path}")
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        if freeze.get("status") != "test_access_authorized":
            parser.error(f"test freeze manifest is not authorized: {freeze_path}")
        if freeze.get("dataset_manifest_sha256") != sha256_path(manifest_path):
            parser.error("test freeze manifest does not match the requested dataset manifest")
        for key in ("mean_patch_selection_sha256", "cls_selection_sha256", "analysis_bundle_sha256"):
            if not freeze.get(key):
                parser.error(f"test freeze manifest is missing {key}")
        analysis_hashes = freeze.get("analysis_files")
        if not isinstance(analysis_hashes, dict) or not analysis_hashes:
            parser.error("test freeze manifest has no frozen analysis file set")
        verified, reason = verify_file_hash_bundle(
            analysis_hashes, freeze["analysis_bundle_sha256"]
        )
        if not verified:
            parser.error(reason)
        selection_verified, selection_reason = verify_file_hash_bundle(
            freeze.get("selection_evidence_files", {}),
            freeze.get("selection_evidence_bundle_sha256", ""),
        )
        if not selection_verified:
            parser.error(selection_reason)

    records = read_jsonl(manifest_path)
    if args.split != "all":
        records = [record for record in records if record["split"] == args.split]
    if args.max_images is not None:
        if args.max_images < 1:
            parser.error("--max-images must be positive")
        records = records[: args.max_images]
    if not records:
        parser.error("selected manifest contains no images")

    device = choose_device(args.device)
    if args.amp and device.type not in {"cuda", "mps"}:
        parser.error("--amp requires CUDA or MPS")
    model = load_retfound_cfp(checkpoint, upstream_repo, representation=args.representation).to(device)
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    shard_evidence = []
    processed = 0
    reused_images = 0
    computed_images = 0
    for shard_start in range(0, len(records), args.shard_size):
        shard_records = records[shard_start : shard_start + args.shard_size]
        shard_end = shard_start + len(shard_records) - 1
        shard_path = output / f"embeddings_{shard_start:06d}_{shard_end:06d}.pt"
        if shard_path.exists():
            payload = torch.load(shard_path, map_location="cpu", weights_only=False)
            if payload.get("checkpoint_hash") != model.checkpoint_hash:
                raise RuntimeError(f"checkpoint mismatch in existing shard {shard_path}")
            if payload.get("representation") != model.representation:
                raise RuntimeError(f"representation mismatch in existing shard {shard_path}")
            if payload.get("relative_paths") != [record["relative_path"] for record in shard_records]:
                raise RuntimeError(f"manifest-order mismatch in existing shard {shard_path}")
            if tuple(payload["embeddings"].shape) != (len(shard_records), model.embedding_dim):
                raise RuntimeError(f"embedding-shape mismatch in existing shard {shard_path}")
            reused_images += len(shard_records)
            shard_action = "reused"
        else:
            loader = DataLoader(
                ManifestImages(dataset_root, shard_records),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=device.type == "cuda",
                persistent_workers=args.workers > 0,
            )
            batches = []
            with torch.inference_mode():
                for images, indices in loader:
                    images = images.to(device, non_blocking=device.type == "cuda")
                    with torch.autocast(device_type=device.type, enabled=args.amp):
                        embeddings = model(images)
                    batches.append(embeddings.float().cpu())
            payload = {
                "schema_version": "1.0",
                "checkpoint_hash": model.checkpoint_hash,
                "representation": model.representation,
                "preprocessing_version": PREPROCESSING_VERSION,
                "relative_paths": [record["relative_path"] for record in shard_records],
                "patient_ids": [record["patient_id"] for record in shard_records],
                "splits": [record["split"] for record in shard_records],
                "embeddings": torch.cat(batches, dim=0),
            }
            torch.save(payload, shard_path)
            computed_images += len(shard_records)
            shard_action = "computed"
        processed += len(shard_records)
        shard_evidence.append({
            "path": str(shard_path),
            "images": len(shard_records),
            "bytes": shard_path.stat().st_size,
            "sha256": sha256_path(shard_path),
        })
        print(
            json.dumps(
                {
                    "event": "shard_complete",
                    "processed_images": processed,
                    "total_images": len(records),
                    "shard": str(shard_path),
                    "action": shard_action,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    elapsed = time.perf_counter() - started
    peak_cuda = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    evidence = {
        "schema_version": "1.1",
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "images": processed,
        "failures": 0,
        "elapsed_seconds_this_invocation": elapsed,
        "images_reused_this_invocation": reused_images,
        "images_computed_this_invocation": computed_images,
        "images_per_second_this_invocation": computed_images / elapsed if computed_images else None,
        "throughput_excludes_reused_shards": True,
        "device": str(device),
        "amp": args.amp,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "peak_cuda_bytes": peak_cuda,
        "process_max_rss_platform_units": max_rss,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "checkpoint": str(checkpoint),
        "checkpoint_hash": model.checkpoint_hash,
        "checkpoint_load_report": model.load_report,
        "representation": model.representation,
        "manifest": str(manifest_path),
        "manifest_hash": sha256_path(manifest_path),
        "preprocessing_version": PREPROCESSING_VERSION,
        "shards": shard_evidence,
    }
    evidence_path = output / "extraction_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
