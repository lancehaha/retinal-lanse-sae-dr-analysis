"""Targeted and norm-matched control edits of one decoder prompt state.

These edits test whether a chosen direction changes the next-token answer.
They do not identify lesion location or fully reconstruct model reasoning.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import nn

from retinavlm_pre_answer import decoder_blocks, pre_answer_forward


def orthogonal_control(direction: torch.Tensor, seed: int) -> torch.Tensor:
    if (direction.ndim != 1 or not direction.numel()
            or not torch.isfinite(direction).all() or direction.float().norm() == 0):
        raise ValueError("Expected a finite decoder direction")
    normalized = direction.float() / direction.float().norm()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(8):
        random_direction = torch.randn(normalized.shape, generator=generator)
        orthogonal = random_direction - (random_direction @ normalized.cpu()) * normalized.cpu()
        norm = orthogonal.norm()
        if norm > 1e-6:
            return orthogonal / norm
    raise ValueError("Could not construct an orthogonal control direction")


def edited_pre_answer_forward(mini_gpt4: nn.Module, samples: dict,
                              layer_index: int, delta: torch.Tensor,
                              target_index: int = 0) -> torch.Tensor:
    blocks = decoder_blocks(mini_gpt4)
    if not 0 <= layer_index < len(blocks) or delta.ndim != 1 or not torch.isfinite(delta).all():
        raise ValueError("Invalid decoder layer or state edit")
    fired = 0

    def hook(_module, _inputs, output):
        nonlocal fired
        fired += 1
        hidden = output[0] if isinstance(output, tuple) else output
        if (hidden.ndim != 3 or hidden.shape[-1] != delta.numel()
                or not 0 <= target_index < hidden.shape[0]):
            raise ValueError("Edit direction does not match decoder state")
        updated = hidden.clone()
        updated[target_index, -1, :] = (updated[target_index, -1, :]
                                        + delta.to(updated.device, updated.dtype))
        return (updated, *output[1:]) if isinstance(output, tuple) else updated

    handle = blocks[layer_index].register_forward_hook(hook)
    try:
        embeddings, _, attention_mask = mini_gpt4.form_input(
            samples, answer_preamble=[""] * len(samples["Question"]),
            query=True, pad_side="left"
        )
        outputs = mini_gpt4.llama_model(inputs_embeds=embeddings,
                                         attention_mask=attention_mask,
                                         use_cache=False, return_dict=True)
        if fired != 1:
            raise ValueError("Selected decoder layer did not execute exactly once")
        return outputs.logits[:, -1, :]
    finally:
        handle.remove()


def compare_state_edits(mini_gpt4: nn.Module, make_samples: Callable[[], dict],
                        layer_index: int, direction: torch.Tensor,
                        absent_id: int, present_id: int, *,
                        strength_fraction: float = 0.05, control_seed: int = 4106,
                        control_seeds: Sequence[int] | None = None,
                        orientation: float = 1.0,
                        target_index: int = 0) -> dict[str, float | list[float]]:
    if not 0 < strength_fraction <= 1 or orientation not in {-1.0, 1.0}:
        raise ValueError("Edit strength and orientation must be predeclared")
    if direction.ndim != 1 or not torch.isfinite(direction).all() or direction.float().norm() == 0:
        raise ValueError("Lens direction is invalid")
    with torch.inference_mode():
        base_logits, hidden = pre_answer_forward(mini_gpt4, make_samples(), layer_index)
        if hidden is None or not 0 <= target_index < hidden.shape[0]:
            raise ValueError("State-edit target is outside the image batch")
        hidden_norm = float(hidden[target_index].float().norm())
        if not torch.isfinite(torch.tensor(hidden_norm)) or hidden_norm == 0:
            raise ValueError("Pre-answer state has invalid norm")
        edit_norm = strength_fraction * hidden_norm
        normalized = orientation * direction.float() / direction.float().norm()
        target_delta = edit_norm * normalized
        target_logits = edited_pre_answer_forward(
            mini_gpt4, make_samples(), layer_index, target_delta, target_index)
        seeds = list(control_seeds) if control_seeds is not None else [control_seed]
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("Control seeds must be nonempty and distinct")
        controls = [orthogonal_control(normalized.cpu(), seed) for seed in seeds]
        control_logits = [edited_pre_answer_forward(
            mini_gpt4, make_samples(), layer_index, edit_norm * control, target_index
        ) for control in controls]
        def margin(logits: torch.Tensor) -> float:
            return float((logits[target_index, present_id]
                          - logits[target_index, absent_id]).float())
        base_margin = margin(base_logits)
        target_margin = margin(target_logits)
        control_margins = [margin(logits) for logits in control_logits]
        control_changes = [value - base_margin for value in control_margins]
    return {"base_present_minus_absent": base_margin,
            "target_present_minus_absent": target_margin,
            "control_present_minus_absent": control_margins[0],
            "control_present_minus_absent_all": control_margins,
            "target_margin_change": target_margin - base_margin,
            "control_margin_change": control_changes[0],
            "control_margin_changes": control_changes,
            "hidden_norm": hidden_norm, "edit_norm": edit_norm,
            "strength_fraction": strength_fraction,
            "target_control_max_absolute_cosine": max(
                abs(float(normalized.cpu() @ control)) for control in controls
            )}
