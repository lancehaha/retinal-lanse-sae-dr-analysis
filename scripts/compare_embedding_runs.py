from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def load_run(directory: Path) -> tuple[list[str], torch.Tensor, dict]:
    evidence = json.loads((directory / "extraction_evidence.json").read_text(encoding="utf-8"))
    paths, embeddings = [], []
    for shard in evidence["shards"]:
        payload = torch.load(shard["path"], map_location="cpu", weights_only=False)
        paths.extend(payload["relative_paths"])
        embeddings.append(payload["embeddings"].float())
    return paths, torch.cat(embeddings, dim=0), evidence


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare deterministic or FP32/AMP RETFound extraction runs.")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-absolute-tolerance", type=float, default=1e-5)
    parser.add_argument("--minimum-cosine", type=float, default=0.99999)
    args = parser.parse_args()

    ref_paths, reference, ref_evidence = load_run(Path(args.reference))
    candidate_paths, candidate, candidate_evidence = load_run(Path(args.candidate))
    if ref_paths != candidate_paths:
        raise RuntimeError("run manifests or image ordering differ")
    if reference.shape != candidate.shape:
        raise RuntimeError(f"embedding shapes differ: {reference.shape} versus {candidate.shape}")
    if ref_evidence["checkpoint_hash"] != candidate_evidence["checkpoint_hash"]:
        raise RuntimeError("checkpoint hashes differ")

    delta = (reference - candidate).abs()
    cosine = F.cosine_similarity(reference, candidate, dim=1)
    report = {
        "schema_version": "1.0",
        "images": len(ref_paths),
        "shape": list(reference.shape),
        "checkpoint_hash": ref_evidence["checkpoint_hash"],
        "reference_amp": ref_evidence["amp"],
        "candidate_amp": candidate_evidence["amp"],
        "exactly_equal": torch.equal(reference, candidate),
        "max_absolute_error": delta.max().item(),
        "mean_absolute_error": delta.mean().item(),
        "minimum_cosine_similarity": cosine.min().item(),
        "mean_cosine_similarity": cosine.mean().item(),
        "thresholds": {
            "max_absolute_tolerance": args.max_absolute_tolerance,
            "minimum_cosine": args.minimum_cosine,
        },
    }
    report["pass"] = (
        report["max_absolute_error"] <= args.max_absolute_tolerance
        and report["minimum_cosine_similarity"] >= args.minimum_cosine
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
