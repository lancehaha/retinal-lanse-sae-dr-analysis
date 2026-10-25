"""Shared objectives for matched RetinaVLM B/C/D adapter training.

This module deliberately contains no dataset selection or model loading. It
keeps the supervised answer objective identical across arms and makes the
extra trainable probe parameters in C explicit. A fixed J-Lens direction and
calibration are buffers, never optimized with D.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from retinavlm_binary_answer import answer_loss
from retinavlm_projected_jacobian import projected_score


class ProbeHead(nn.Module):
    def __init__(self, hidden_width: int):
        super().__init__()
        if hidden_width < 1:
            raise ValueError("Probe hidden width must be positive")
        self.linear = nn.Linear(hidden_width, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 2:
            raise ValueError("Probe expects one pre-answer state per image")
        return self.linear(hidden.float()).squeeze(-1)


class FixedLensHead(nn.Module):
    def __init__(self, direction: torch.Tensor, scale: float, intercept: float):
        super().__init__()
        if direction.ndim != 1 or not direction.numel() or not torch.isfinite(direction).all():
            raise ValueError("Lens direction must be a finite nonempty vector")
        if not math.isfinite(scale) or not math.isfinite(intercept) or scale == 0:
            raise ValueError("Lens calibration must be finite with nonzero scale")
        self.register_buffer("direction", direction.detach().float().clone())
        self.register_buffer("scale", torch.tensor(float(scale)))
        self.register_buffer("intercept", torch.tensor(float(intercept)))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 2:
            raise ValueError("Lens expects one pre-answer state per image")
        return self.scale * projected_score(hidden, self.direction) + self.intercept


def adapter_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Freeze all RetinaVLM components except its visual-to-language adapter."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    adapter = getattr(model, "llama_proj", None)
    if not isinstance(adapter, nn.Module):
        raise ValueError("Expected RetinaVLM MiniGPT4.llama_proj adapter")
    parameters = list(adapter.parameters())
    if not parameters:
        raise ValueError("Visual adapter has no parameters")
    for parameter in parameters:
        parameter.requires_grad_(True)
    return parameters


def arm_loss(
    arm: str,
    next_token_logits: torch.Tensor,
    labels: torch.Tensor,
    absent_id: int,
    present_id: int,
    *,
    pre_answer_hidden: torch.Tensor | None = None,
    probe: ProbeHead | None = None,
    lens: FixedLensHead | None = None,
    auxiliary_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if arm not in {"B", "C", "D"}:
        raise ValueError("Only trained arms B/C/D have a loss")
    if not math.isfinite(auxiliary_weight) or auxiliary_weight < 0:
        raise ValueError("Auxiliary weight must be finite and nonnegative")
    base = answer_loss(next_token_logits, labels, absent_id, present_id)
    terms = {"answer": base}
    if arm == "B":
        if probe is not None or lens is not None:
            raise ValueError("Arm B must not receive an auxiliary head")
        return base, terms
    if pre_answer_hidden is None or pre_answer_hidden.ndim != 2:
        raise ValueError("Auxiliary arms require batched pre-answer hidden states")
    if pre_answer_hidden.shape[0] != labels.numel():
        raise ValueError("Hidden-state batch and labels differ")
    if arm == "C":
        if probe is None or lens is not None:
            raise ValueError("Arm C requires only a learned probe")
        auxiliary_logits = probe(pre_answer_hidden)
    else:
        if lens is None or probe is not None or any(p.requires_grad for p in lens.parameters()):
            raise ValueError("Arm D requires only a fixed lens")
        auxiliary_logits = lens(pre_answer_hidden)
    auxiliary = F.binary_cross_entropy_with_logits(auxiliary_logits, labels.float())
    terms["auxiliary"] = auxiliary
    return base + auxiliary_weight * auxiliary, terms
