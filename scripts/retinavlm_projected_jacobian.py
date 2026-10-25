"""Fit a single-token, image-conditioned residual-Jacobian direction.

This computes ``u.T @ E[J_l]`` at the final prompt position, where ``u`` is a
fixed unembedding row and ``J_l`` maps a chosen decoder-layer output to a later
decoder-layer output. It is the token projection of a Jacobian transport, not
the full published J-Lens: its readout does not apply final RMSNorm, and the
fit averages image/question contexts at one position rather than text-corpus
positions. Verify the full-model behavior before calling it a useful lens.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import torch
from torch import nn


def _tensor(output: torch.Tensor | tuple) -> torch.Tensor:
    return output if torch.is_tensor(output) else output[0]


def projected_directions_for_batch(
    forward: Callable[[], object],
    blocks: Sequence[nn.Module],
    source_layers: Sequence[int],
    target_layer: int,
    unembedding_row: torch.Tensor,
    *,
    position: int = -1,
) -> dict[int, torch.Tensor]:
    """Return per-example ``u.T @ J_l`` rows of shape ``[batch, width]``.

    ``forward`` must execute the image-conditioned decoder with gradients
    enabled, without generation or detaching its input embeddings. The first
    requested block output is made a graph root if all parameters are frozen.
    The caller supplies one fixed token's unembedding row.
    """
    sources = sorted(set(source_layers))
    if not sources or sources[0] < 0 or sources[-1] >= target_layer or target_layer >= len(blocks):
        raise ValueError("Source layers must be nonempty and precede the target layer")
    if unembedding_row.ndim != 1:
        raise ValueError("Expected one unembedding row")
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(index: int):
        def hook(_module, _inputs, output):
            activation = _tensor(output)
            if index == sources[0] and not activation.requires_grad:
                activation.requires_grad_(True)
            captured[index] = activation
        return hook

    try:
        for index in [*sources, target_layer]:
            handles.append(blocks[index].register_forward_hook(make_hook(index)))
        with torch.enable_grad():
            forward()
            target = captured[target_layer]
            if target.ndim != 3 or target.shape[-1] != unembedding_row.numel():
                raise ValueError("Decoder target shape does not match unembedding width")
            position_index = position % target.shape[1]
            direction = unembedding_row.to(device=target.device, dtype=target.dtype)
            scalar = (target[:, position_index, :] * direction).sum()
            gradients = torch.autograd.grad(
                scalar, [captured[index] for index in sources], allow_unused=False
            )
        return {index: gradient[:, position_index, :].float().detach().cpu()
                for index, gradient in zip(sources, gradients, strict=True)}
    finally:
        for handle in handles:
            handle.remove()


def mean_projected_directions(
    forwards: Iterable[Callable[[], object]],
    blocks: Sequence[nn.Module],
    source_layers: Sequence[int],
    target_layer: int,
    unembedding_row: torch.Tensor,
    *,
    position: int = -1,
) -> tuple[dict[int, torch.Tensor], int]:
    """Average the per-example directions over supplied training contexts."""
    totals: dict[int, torch.Tensor] = {}
    count = 0
    for forward in forwards:
        batch = projected_directions_for_batch(
            forward, blocks, source_layers, target_layer,
            unembedding_row, position=position
        )
        batch_count = next(iter(batch.values())).shape[0]
        for index, vectors in batch.items():
            totals[index] = totals.get(index, torch.zeros_like(vectors[0])) + vectors.sum(0)
        count += batch_count
    if not count:
        raise ValueError("No training contexts supplied")
    return {index: total / count for index, total in totals.items()}, count


def projected_score(hidden: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Score a pre-answer residual; retains gradients through ``hidden``."""
    if hidden.shape[-1] != direction.numel():
        raise ValueError("Hidden state and direction width differ")
    return hidden.float() @ direction.to(hidden.device, dtype=torch.float32)


def answer_margin_directions_for_batch(
    forward: Callable[[], torch.Tensor],
    blocks: Sequence[nn.Module],
    source_layers: Sequence[int],
    absent_id: int,
    present_id: int,
    *,
    position: int = -1,
) -> dict[int, torch.Tensor]:
    """Exact VJP of the final answer margin through the decoder, per image.

    Unlike a bare unembedding-row projection, this includes the final norm.
    The caller supplies next-token logits from the image-conditioned forward.
    Returned vectors are detached CPU float32 tensors of shape [batch, width].
    """
    sources = sorted(set(source_layers))
    if not sources or sources[0] < 0 or sources[-1] >= len(blocks):
        raise ValueError("Invalid source layers")
    if min(absent_id, present_id) < 0 or absent_id == present_id:
        raise ValueError("Invalid answer token IDs")
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(index: int):
        def hook(_module, _inputs, output):
            activation = _tensor(output)
            if index == sources[0] and not activation.requires_grad:
                activation.requires_grad_(True)
            captured[index] = activation
        return hook

    try:
        for index in sources:
            handles.append(blocks[index].register_forward_hook(make_hook(index)))
        with torch.enable_grad():
            logits = forward()
            if logits.ndim != 2 or max(absent_id, present_id) >= logits.shape[1]:
                raise ValueError("Expected full-vocabulary next-token logits")
            if set(captured) != set(sources):
                raise ValueError("Not all requested decoder layers executed")
            if any(captured[index].ndim != 3 or captured[index].shape[0] != logits.shape[0]
                   for index in sources):
                raise ValueError("Expected batched decoder token states")
            margin = logits[:, present_id] - logits[:, absent_id]
            gradients = torch.autograd.grad(margin.sum(), [captured[index] for index in sources])
        return {index: gradient[:, position, :].float().detach().cpu()
                for index, gradient in zip(sources, gradients, strict=True)}
    finally:
        for handle in handles:
            handle.remove()


def explicit_submatrix_check(
    forward: Callable[[], object],
    blocks: Sequence[nn.Module],
    source_layer: int,
    target_layer: int,
    unembedding_row: torch.Tensor,
    input_indices: Sequence[int],
    output_indices: Sequence[int],
    *,
    position: int = -1,
) -> dict[str, float | int]:
    """Compare a projected VJP with explicit rows of a small real Jacobian.

    Only the specified output coordinates receive nonzero token-row weights.
    This checks a feasible Jacobian submatrix, not the full decoder Jacobian.
    Use one deterministic image-conditioned example with the model in eval mode.
    """
    if (not input_indices or not output_indices or source_layer < 0
            or source_layer >= target_layer or target_layer >= len(blocks)):
        raise ValueError("Invalid Jacobian submatrix layers or coordinates")
    width = unembedding_row.numel()
    if (unembedding_row.ndim != 1 or any(i < 0 or i >= width for i in input_indices)
            or any(i < 0 or i >= width for i in output_indices)):
        raise ValueError("Jacobian coordinate outside decoder width")
    if len(set(input_indices)) != len(input_indices) or len(set(output_indices)) != len(output_indices):
        raise ValueError("Jacobian coordinates must be unique")

    captured: dict[int, torch.Tensor] = {}
    handles = []

    def capture(index: int):
        def hook(_module, _inputs, output):
            activation = _tensor(output)
            if index == source_layer and not activation.requires_grad:
                activation.requires_grad_(True)
            captured[index] = activation
        return hook

    try:
        for index in (source_layer, target_layer):
            handles.append(blocks[index].register_forward_hook(capture(index)))
        with torch.enable_grad():
            forward()
            source, target = captured[source_layer], captured[target_layer]
            if source.ndim != 3 or target.ndim != 3 or source.shape[0] != 1 or target.shape[0] != 1:
                raise ValueError("Explicit submatrix check requires one batched sequence")
            if source.shape[-1] != width or target.shape[-1] != width:
                raise ValueError("Jacobian state width differs from token row")
            source_pos = position % source.shape[1]
            target_pos = position % target.shape[1]
            weights = unembedding_row.to(target.device, target.dtype)
            explicit = torch.zeros(len(input_indices), device=target.device, dtype=torch.float32)
            for output_index in output_indices:
                gradient = torch.autograd.grad(
                    target[0, target_pos, output_index], source, retain_graph=True
                )[0]
                explicit += (weights[output_index].float()
                             * gradient[0, source_pos, list(input_indices)].float())
    finally:
        for handle in handles:
            handle.remove()

    sparse_row = torch.zeros_like(unembedding_row)
    sparse_row[list(output_indices)] = unembedding_row[list(output_indices)]
    projected = projected_directions_for_batch(
        forward, blocks, [source_layer], target_layer, sparse_row, position=position
    )[source_layer][0, list(input_indices)]
    difference = (projected - explicit.cpu()).abs()
    return {"input_coordinates": len(input_indices),
            "output_coordinates": len(output_indices),
            "max_absolute_error": float(difference.max()),
            "max_relative_error": float((difference / explicit.cpu().abs().clamp_min(1e-6)).max())}
