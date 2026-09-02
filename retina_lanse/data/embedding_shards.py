from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from retina_lanse.data.manifest import sha256_path


@dataclass(frozen=True)
class EmbeddingRun:
    embeddings: torch.Tensor
    relative_paths: list[str]
    patient_ids: list[str]
    splits: list[str]
    evidence: dict


def load_embedding_run(
    directory: str | Path,
    *,
    expected_split: str | None = None,
    verify_sha256: bool = True,
) -> EmbeddingRun:
    root = Path(directory).resolve()
    evidence_path = root / "extraction_evidence.json"
    if not evidence_path.is_file():
        raise RuntimeError(f"embedding run is incomplete; evidence is absent: {evidence_path}")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("status") != "complete" or evidence.get("failures") != 0:
        raise RuntimeError(f"embedding evidence is not a zero-failure completed run: {evidence_path}")

    tensors: list[torch.Tensor] = []
    paths: list[str] = []
    patient_ids: list[str] = []
    splits: list[str] = []
    evidence_images = 0
    for item in evidence["shards"]:
        shard = Path(item["path"]).resolve()
        try:
            shard.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(f"evidence shard escapes run directory: {shard}") from exc
        if not shard.is_file() or shard.stat().st_size != int(item["bytes"]):
            raise RuntimeError(f"missing or size-mismatched embedding shard: {shard}")
        if verify_sha256 and sha256_path(shard) != item["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch for embedding shard: {shard}")
        payload = torch.load(shard, map_location="cpu", weights_only=False)
        if payload["checkpoint_hash"] != evidence["checkpoint_hash"]:
            raise RuntimeError(f"checkpoint mismatch in embedding shard: {shard}")
        if payload["representation"] != evidence["representation"]:
            raise RuntimeError(f"representation mismatch in embedding shard: {shard}")
        shard_embeddings = payload["embeddings"].float()
        shard_paths = payload["relative_paths"]
        shard_patients = payload["patient_ids"]
        shard_splits = payload["splits"]
        shard_count = len(shard_embeddings)
        if shard_embeddings.ndim != 2 or not torch.isfinite(shard_embeddings).all():
            raise RuntimeError(f"non-finite or non-matrix embeddings in shard: {shard}")
        if not (
            shard_count
            == len(shard_paths)
            == len(shard_patients)
            == len(shard_splits)
            == int(item["images"])
        ):
            raise RuntimeError(f"embedding shard data/evidence lengths differ: {shard}")
        if any(not str(patient).strip() for patient in shard_patients):
            raise RuntimeError(f"empty patient identifier in shard: {shard}")
        tensors.append(shard_embeddings)
        paths.extend(shard_paths)
        patient_ids.extend(shard_patients)
        splits.extend(shard_splits)
        evidence_images += int(item["images"])

    embeddings = torch.cat(tensors, dim=0)
    if len(embeddings) != int(evidence["images"]):
        raise RuntimeError(
            f"embedding count differs from evidence: {len(embeddings)} versus {evidence['images']}"
        )
    if evidence_images != int(evidence["images"]):
        raise RuntimeError("embedding shard evidence counts do not sum to run evidence")
    if not (len(paths) == len(patient_ids) == len(splits) == len(embeddings)):
        raise RuntimeError("embedding shard metadata lengths differ")
    if len(set(paths)) != len(paths):
        raise RuntimeError("embedding run contains duplicate relative paths")
    if expected_split is not None and set(splits) != {expected_split}:
        raise RuntimeError(f"expected only split {expected_split}, observed {sorted(set(splits))}")
    return EmbeddingRun(embeddings, paths, patient_ids, splits, evidence)
