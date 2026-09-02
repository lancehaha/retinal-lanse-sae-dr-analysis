from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def crop_retinal_field(image: Image.Image, threshold: int = 8) -> Image.Image:
    """Crop dark borders while retaining all non-black retinal pixels."""
    rgb = np.asarray(image.convert("RGB"))
    valid = rgb.max(axis=2) > threshold
    if not valid.any():
        return image.convert("RGB")
    ys, xs = np.where(valid)
    return image.convert("RGB").crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))


def preprocess_global(image: Image.Image, size: int) -> torch.Tensor:
    cropped = crop_retinal_field(image)
    resized = cropped.resize((size, size), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def normalize_retfound_cfp(image: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Apply the official RETFound CFP notebook's per-image, per-channel z-score.

    Accepts ``CHW`` or ``NCHW`` tensors in the [0, 1] range. The epsilon is an
    explicit stability addition for constant channels, which otherwise produce
    NaNs in the upstream notebook.
    """
    if image.ndim not in (3, 4):
        raise ValueError(f"Expected CHW or NCHW input, received shape {tuple(image.shape)}")
    batched = image.unsqueeze(0) if image.ndim == 3 else image
    if batched.shape[1] != 3:
        raise ValueError(f"Expected three RGB channels, received {batched.shape[1]}")
    means = batched.mean(dim=(-2, -1), keepdim=True)
    stds = batched.std(dim=(-2, -1), correction=0, keepdim=True)
    normalized = (batched - means) / stds.clamp_min(eps)
    return normalized.squeeze(0) if image.ndim == 3 else normalized


def preprocess_retfound_cfp(image: Image.Image, size: int = 224) -> torch.Tensor:
    """Versioned retinal-field crop plus official RETFound CFP normalization."""
    return normalize_retfound_cfp(preprocess_global(image, size))
