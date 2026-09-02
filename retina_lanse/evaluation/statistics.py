from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np


def paired_patient_bootstrap_difference(
    left: Sequence[float] | np.ndarray,
    right: Sequence[float] | np.ndarray,
    patient_ids: Sequence[str],
    *,
    resamples: int = 2000,
    seed: int = 4106,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Paired cluster bootstrap of a per-image metric difference, resampling patients."""
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.ndim != 1 or right_array.shape != left_array.shape:
        raise ValueError("left and right must be same-length one-dimensional arrays")
    if len(left_array) != len(patient_ids) or len(left_array) == 0:
        raise ValueError("patient_ids must match a non-empty metric array")
    if resamples < 1 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid bootstrap settings")

    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for difference, patient in zip(left_array - right_array, patient_ids):
        key = str(patient)
        sums[key] += float(difference)
        counts[key] += 1
    patients = sorted(sums)
    cluster_sums = np.asarray([sums[patient] for patient in patients], dtype=np.float64)
    cluster_counts = np.asarray([counts[patient] for patient in patients], dtype=np.int64)
    rng = np.random.default_rng(seed)
    replicates = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled = rng.integers(0, len(patients), size=len(patients))
        replicates[index] = cluster_sums[sampled].sum() / cluster_counts[sampled].sum()
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(replicates, [alpha, 1.0 - alpha])
    return {
        "difference": float((left_array - right_array).mean()),
        "confidence": float(confidence),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "resamples": int(resamples),
        "patients": len(patients),
        "images": len(left_array),
        "seed": int(seed),
    }
