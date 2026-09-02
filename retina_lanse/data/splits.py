from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Iterable


def patient_level_splits(
    patient_ids: Iterable[str],
    fractions: tuple[float, float, float],
    seed: int,
) -> dict[str, str]:
    """Assign every patient to exactly one deterministic split."""
    if abs(sum(fractions) - 1.0) > 1e-8:
        raise ValueError("split fractions must sum to one")
    names = ("train", "validation", "test")
    cutoffs = (fractions[0], fractions[0] + fractions[1], 1.0)
    result: dict[str, str] = {}
    for patient_id in sorted(set(patient_ids)):
        token = f"{seed}:{patient_id}".encode("utf-8")
        value = int.from_bytes(hashlib.sha256(token).digest()[:8], "big") / 2**64
        result[patient_id] = next(name for name, cutoff in zip(names, cutoffs) if value < cutoff)
    return result


def assert_no_patient_leakage(records: list[dict]) -> None:
    by_patient: dict[str, set[str]] = defaultdict(set)
    for record in records:
        by_patient[record["patient_id"]].add(record["split"])
    leaking = {key: sorted(value) for key, value in by_patient.items() if len(value) != 1}
    if leaking:
        raise ValueError(f"patient leakage detected: {leaking}")

