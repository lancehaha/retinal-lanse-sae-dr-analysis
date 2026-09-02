from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from retina_lanse.data.embedding_shards import load_embedding_run
from retina_lanse.data.manifest import sha256_path


EXTERNAL_ROOT = Path("/Volumes/Elements/Retina-LanSE")


def require_external(path: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(EXTERNAL_ROOT)
    except ValueError as exc:
        raise RuntimeError(f"{label} must stay under {EXTERNAL_ROOT}: {resolved}") from exc
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize one split from a completed embedding run.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--split", choices=("train", "validation", "internal_test"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shard-size", type=int, default=1000)
    args = parser.parse_args()

    source = require_external(Path(args.source), "source run")
    output = require_external(Path(args.output), "split output")
    if args.shard_size < 1:
        parser.error("--shard-size must be positive")
    output.mkdir(parents=True, exist_ok=True)
    run = load_embedding_run(source, verify_sha256=True)
    indices = [index for index, split in enumerate(run.splits) if split == args.split]
    if not indices:
        parser.error(f"source run contains no {args.split} embeddings")

    selected = run.embeddings[indices]
    selected_paths = [run.relative_paths[index] for index in indices]
    selected_patients = [run.patient_ids[index] for index in indices]
    shards: list[dict] = []
    for start in range(0, len(indices), args.shard_size):
        end = min(start + args.shard_size, len(indices))
        shard = output / f"embeddings_{start:06d}_{end - 1:06d}.pt"
        payload = {
            "schema_version": "1.0",
            "checkpoint_hash": run.evidence["checkpoint_hash"],
            "representation": run.evidence["representation"],
            "preprocessing_version": run.evidence["preprocessing_version"],
            "relative_paths": selected_paths[start:end],
            "patient_ids": selected_patients[start:end],
            "splits": [args.split] * (end - start),
            "embeddings": selected[start:end],
        }
        torch.save(payload, shard)
        shards.append(
            {
                "path": str(shard),
                "images": end - start,
                "bytes": shard.stat().st_size,
                "sha256": sha256_path(shard),
            }
        )

    evidence = {
        "schema_version": "1.0",
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "images": len(indices),
        "failures": 0,
        "device": "derived_without_reencoding",
        "amp": run.evidence["amp"],
        "batch_size": run.evidence["batch_size"],
        "workers": run.evidence["workers"],
        "checkpoint": run.evidence["checkpoint"],
        "checkpoint_hash": run.evidence["checkpoint_hash"],
        "checkpoint_load_report": run.evidence["checkpoint_load_report"],
        "representation": run.evidence["representation"],
        "manifest": run.evidence["manifest"],
        "manifest_hash": run.evidence["manifest_hash"],
        "preprocessing_version": run.evidence["preprocessing_version"],
        "derived_from": str(source / "extraction_evidence.json"),
        "derived_split": args.split,
        "shards": shards,
    }
    evidence_path = output / "extraction_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "split": args.split, "images": len(indices)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
