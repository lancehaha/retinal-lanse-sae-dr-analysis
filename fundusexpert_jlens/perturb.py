"""Deterministic, size-matched lesion and background image perturbations."""

from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image, ImageFilter


def paired_blurs(image_path: str, mask_path: str, image_id: str) -> tuple[Image.Image, Image.Image]:
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L").resize(image.size, Image.Resampling.NEAREST)
    expanded = np.asarray(mask.filter(ImageFilter.MaxFilter(15))) > 0
    if not expanded.any():
        raise ValueError(f"Empty positive mask: {image_id}")
    original = np.asarray(image)
    blurred = np.asarray(image.filter(ImageFilter.GaussianBlur(8)))
    field = original.max(axis=2) > 30
    candidates = np.flatnonzero((field & ~expanded).ravel())
    count = int(expanded.sum())
    if count > len(candidates):
        raise ValueError(f"Insufficient matched background area: {image_id}")
    seed = int.from_bytes(hashlib.sha256(image_id.encode()).digest()[:8], "big")
    selected = np.random.default_rng(seed).choice(candidates, count, replace=False)
    background = np.zeros(expanded.shape, dtype=bool)
    background.ravel()[selected] = True
    lesion_img, background_img = original.copy(), original.copy()
    lesion_img[expanded] = blurred[expanded]
    background_img[background] = blurred[background]
    return Image.fromarray(lesion_img), Image.fromarray(background_img)
