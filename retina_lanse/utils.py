from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(records: Iterable[Any]) -> str:
    payload = json.dumps(list(records), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_file_hash_bundle(
    file_hashes: dict[str, str], expected_bundle_hash: str
) -> tuple[bool, str]:
    """Fail closed when a frozen analysis file is missing, changed, or inconsistently bundled."""
    if not file_hashes:
        return False, "frozen analysis file set is empty"
    observed_hashes: dict[str, str] = {}
    for frozen_path, expected_hash in file_hashes.items():
        path = Path(frozen_path).resolve()
        if not path.is_file():
            return False, f"frozen analysis file is missing: {path}"
        observed = sha256_file(path)
        if observed != expected_hash:
            return False, f"frozen analysis file changed after authorization: {path}"
        observed_hashes[str(path)] = observed
    observed_bundle = stable_hash(
        [
            {"path": path, "sha256": digest}
            for path, digest in sorted(observed_hashes.items())
        ]
    )
    if observed_bundle != expected_bundle_hash:
        return False, "frozen analysis bundle hash is inconsistent"
    return True, "verified"


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
