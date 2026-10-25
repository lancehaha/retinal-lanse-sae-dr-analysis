"""One image-conditioned RetinaVLM forward at the pre-answer position.

The authors' ``form_input`` inserts image embeddings into the chat prompt.
With left padding, the final sequence position is the last prompt token. A
temporary decoder-layer hook captures that same position for C/D without
retaining every layer's hidden states in the output.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def decoder_blocks(mini_gpt4: nn.Module) -> nn.ModuleList:
    language_model = getattr(mini_gpt4, "llama_model", None)
    if not isinstance(language_model, nn.Module):
        raise ValueError("Expected a Llama language model")
    # PEFT wraps the causal model under base_model.model; the released model
    # exposes the decoder directly under llama_model.model. Resolve either
    # path without assuming a particular PEFT version's attribute proxying.
    queue = [language_model]
    visited: set[int] = set()
    while queue:
        module = queue.pop(0)
        if id(module) in visited:
            continue
        visited.add(id(module))
        blocks = getattr(module, "layers", None)
        if isinstance(blocks, (nn.ModuleList, list, tuple)) and blocks:
            return blocks
        for attribute in ("model", "base_model"):
            child = getattr(module, attribute, None)
            if isinstance(child, nn.Module):
                queue.append(child)
    raise ValueError("Expected Llama decoder blocks within llama_model")


def pre_answer_forward(mini_gpt4: nn.Module, samples: dict[str, Any],
                       layer_index: int | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return full-vocabulary next-token logits and an optional layer state.

    ``samples`` must include Image, Question, and Answer lists/tensors in the
    format expected by the released authors' ``MiniGPT4.form_input``. The
    returned tensors retain gradients to ``llama_proj`` for adapter training.
    """
    blocks = decoder_blocks(mini_gpt4) if layer_index is not None else None
    if blocks is not None and not 0 <= layer_index < len(blocks):
        raise ValueError("Selected decoder layer is outside the model")
    captured: list[torch.Tensor] = []

    def hook(_module: nn.Module, _inputs: tuple, output: torch.Tensor | tuple) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(hidden) or hidden.ndim != 3:
            raise ValueError("Decoder layer did not return batched token states")
        captured.append(hidden[:, -1, :])

    handle = blocks[layer_index].register_forward_hook(hook) if blocks is not None else None
    try:
        embeddings, _, attention_mask = mini_gpt4.form_input(
            samples, answer_preamble=[""] * len(samples["Question"]),
            query=True, pad_side="left"
        )
        outputs = mini_gpt4.llama_model(
            inputs_embeds=embeddings, attention_mask=attention_mask,
            use_cache=False, return_dict=True
        )
        logits = outputs.logits[:, -1, :]
        if logits.ndim != 2 or logits.shape[0] != len(samples["Question"]):
            raise ValueError("Language model returned unexpected next-token logits")
        if layer_index is not None and len(captured) != 1:
            raise ValueError("Selected decoder block did not execute exactly once")
        return logits, captured[0] if captured else None
    finally:
        if handle is not None:
            handle.remove()


def pre_answer_forward_layers(mini_gpt4: nn.Module, samples: dict[str, Any],
                              layer_indices: list[int] | tuple[int, ...]
                              ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """Return answer logits and several pre-answer states in one forward pass."""
    blocks = decoder_blocks(mini_gpt4)
    indices = sorted(set(layer_indices))
    if len(indices) != len(layer_indices) or any(i < 0 or i >= len(blocks) for i in indices):
        raise ValueError("Decoder layer indices must be distinct and valid")
    captured = {}
    handles = []

    def make_hook(index: int):
        def hook(_module: nn.Module, _inputs: tuple, output: torch.Tensor | tuple) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(hidden) or hidden.ndim != 3:
                raise ValueError("Decoder layer did not return batched token states")
            captured[index] = hidden[:, -1, :]
        return hook

    try:
        for index in indices:
            handles.append(blocks[index].register_forward_hook(make_hook(index)))
        embeddings, _, attention_mask = mini_gpt4.form_input(
            samples, answer_preamble=[""] * len(samples["Question"]),
            query=True, pad_side="left"
        )
        outputs = mini_gpt4.llama_model(
            inputs_embeds=embeddings, attention_mask=attention_mask,
            use_cache=False, return_dict=True
        )
        logits = outputs.logits[:, -1, :]
        if logits.ndim != 2 or logits.shape[0] != len(samples["Question"]):
            raise ValueError("Language model returned unexpected next-token logits")
        if set(captured) != set(indices):
            raise ValueError("Selected decoder blocks did not execute exactly once")
        return logits, captured
    finally:
        for handle in handles:
            handle.remove()
