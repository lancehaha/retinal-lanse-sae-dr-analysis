from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class FrozenRetinalEncoder(nn.Module, ABC):
    @property
    @abstractmethod
    def checkpoint_hash(self) -> str: ...

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def freeze(self) -> None:
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @abstractmethod
    def forward(self, images: torch.Tensor) -> torch.Tensor: ...

