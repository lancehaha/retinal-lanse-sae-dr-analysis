from __future__ import annotations

import hashlib

import torch
import torch.nn.functional as F

from retina_lanse.encoders.base import FrozenRetinalEncoder


class DeterministicSmokeEncoder(FrozenRetinalEncoder):
    """Non-clinical fixed projection used only to validate pipeline mechanics."""

    def __init__(self, embedding_dim: int = 64, pool_size: int = 8, seed: int = 173):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        input_dim = 3 * pool_size * pool_size
        projection = torch.randn(input_dim, embedding_dim, generator=generator) / input_dim**0.5
        self.register_buffer("projection", projection)
        self.pool_size = pool_size
        self.embedding_dim = embedding_dim
        self.seed = seed
        self.freeze()

    @property
    def checkpoint_hash(self) -> str:
        payload = self.projection.detach().cpu().numpy().tobytes()
        return hashlib.sha256(payload).hexdigest()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(images, (self.pool_size, self.pool_size)).flatten(1)
        embeddings = pooled @ self.projection
        return F.layer_norm(embeddings, (self.embedding_dim,))

