from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from retina_lanse.data.manifest import sha256_path
from scripts.run_global_sae_search import require_external


def equal_nested(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) and torch.is_tensor(right):
        return left.shape == right.shape and left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(equal_nested(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(equal_nested(a, b) for a, b in zip(left, right))
    return left == right


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare independent same-seed SAE training runs.")
    parser.add_argument("--left-metrics", required=True)
    parser.add_argument("--right-metrics", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    left_path = require_external(Path(args.left_metrics), "left metrics")
    right_path = require_external(Path(args.right_metrics), "right metrics")
    output = require_external(Path(args.output), "repeatability report")
    left = json.loads(left_path.read_text(encoding="utf-8"))
    right = json.loads(right_path.read_text(encoding="utf-8"))
    left_checkpoint_path = Path(left["checkpoint"]).resolve()
    right_checkpoint_path = Path(right["checkpoint"]).resolve()
    left_checkpoint = torch.load(left_checkpoint_path, map_location="cpu", weights_only=False)
    right_checkpoint = torch.load(right_checkpoint_path, map_location="cpu", weights_only=False)

    deterministic_metric_fields = (
        "configuration",
        "config_hash",
        "run_signature",
        "initial_validation",
        "train",
        "validation",
        "reload_validation",
        "gates",
    )
    metric_fields_equal = {
        field: equal_nested(left.get(field), right.get(field))
        for field in deterministic_metric_fields
    }
    curves_left = left_path.parent / "curves.csv"
    curves_right = right_path.parent / "curves.csv"
    project_root = Path(__file__).resolve().parents[1]
    analysis_paths = [
        project_root / "scripts/run_global_sae_search.py",
        project_root / "retina_lanse/sae/model.py",
        project_root / "retina_lanse/sae/trainer.py",
        project_root / "retina_lanse/evaluation/metrics.py",
        project_root / "configs/paper_global_sae.yaml",
    ]
    report = {
        "schema_version": "1.0",
        "status": "pass",
        "scope": "real_embedding_sae_training_repeatability",
        "left_metrics": str(left_path),
        "left_metrics_sha256": sha256_path(left_path),
        "right_metrics": str(right_path),
        "right_metrics_sha256": sha256_path(right_path),
        "config_hash": left.get("config_hash"),
        "run_signature_equal": left.get("run_signature") == right.get("run_signature"),
        "metric_fields_equal": metric_fields_equal,
        "model_state_exact": equal_nested(
            left_checkpoint["model_state"], right_checkpoint["model_state"]
        ),
        "optimizer_state_exact": equal_nested(
            left_checkpoint["optimizer_state"], right_checkpoint["optimizer_state"]
        ),
        "checkpoint_metadata_exact": equal_nested(
            {
                key: value
                for key, value in left_checkpoint.items()
                if key not in {"model_state", "optimizer_state", "normalization_path"}
            },
            {
                key: value
                for key, value in right_checkpoint.items()
                if key not in {"model_state", "optimizer_state", "normalization_path"}
            },
        ),
        "normalization_artifacts_byte_identical": (
            Path(left_checkpoint["normalization_path"]).read_bytes()
            == Path(right_checkpoint["normalization_path"]).read_bytes()
        ),
        "curves_byte_identical": curves_left.read_bytes() == curves_right.read_bytes(),
        "analysis_files": {str(path): sha256_path(path) for path in analysis_paths},
    }
    report["pass"] = all(
        [
            report["run_signature_equal"],
            *metric_fields_equal.values(),
            report["model_state_exact"],
            report["optimizer_state_exact"],
            report["checkpoint_metadata_exact"],
            report["normalization_artifacts_byte_identical"],
            report["curves_byte_identical"],
        ]
    )
    report["status"] = "pass" if report["pass"] else "fail"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
