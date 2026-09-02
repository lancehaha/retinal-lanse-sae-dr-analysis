from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class TopKSAE(nn.Module):
    def __init__(self, input_dim: int, dictionary_size: int, k: int):
        super().__init__()
        if not 0 < k <= dictionary_size:
            raise ValueError("k must be between one and dictionary_size")
        self.input_dim = input_dim
        self.dictionary_size = dictionary_size
        self.k = k
        self.encoder = nn.Linear(input_dim, dictionary_size)
        self.decoder = nn.Linear(dictionary_size, input_dim)
        self.normalize_decoder_columns()

    def encode(self, embeddings: torch.Tensor) -> torch.Tensor:
        preactivations = F.relu(self.encoder(embeddings))
        values, indices = torch.topk(preactivations, self.k, dim=-1)
        sparse = torch.zeros_like(preactivations)
        return sparse.scatter(-1, indices, values)

    def forward(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sparse = self.encode(embeddings)
        return self.decoder(sparse), sparse

    @torch.no_grad()
    def normalize_decoder_columns(self) -> None:
        norms = self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-8)
        self.decoder.weight.div_(norms)

