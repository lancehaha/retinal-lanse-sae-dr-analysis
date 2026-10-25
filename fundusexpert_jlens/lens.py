"""Position-conditioned Jacobian lens for a pre-answer lesion concept.

The source is the final prompt position. A fixed, label-free JSON prefix
places the target at the future hard-exudate status token. This is a declared
task-conditioned variant of the published average Jacobian lens.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def concept_score(hidden: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Continuous cosine score; vocabulary rank is never used as a loss."""
    return F.cosine_similarity(hidden.float(), direction.float().to(hidden.device), dim=-1)


def estimate_direction(model, prepared_examples: list[dict], layer: int, token_id: int) -> torch.Tensor:
    """Average d(future concept logit)/d(pre-answer state) over training prompts.

    Each example contains the image, question, and fixed, label-free readout
    prefix. Ground-truth labels and status values never enter this computation.
    """
    if not prepared_examples:
        raise ValueError("At least one training prompt is required")
    was_training = model.training
    model.eval()
    directions = []
    with torch.enable_grad():
        for example in prepared_examples:
            kwargs = {k: example[k] for k in ("pixel_values", "input_ids", "attention_mask", "image_flags")}
            outputs = model(**kwargs, output_hidden_states=True, return_dict=True, use_cache=False)
            position = int(example["pre_answer_position"])
            target = int(example.get("readout_position", position))
            hidden = outputs.hidden_states[layer]
            if target != int(example["attention_mask"].sum().item()) - 1 or target < position:
                raise ValueError("readout_position must be at or after the pre-answer state")
            logit = outputs.logits[0, target, token_id]
            gradient = torch.autograd.grad(logit, hidden, retain_graph=False)[0][0, position]
            directions.append(F.normalize(gradient.detach().float().cpu(), dim=0))
    model.train(was_training)
    return F.normalize(torch.stack(directions).mean(dim=0), dim=0)


def fit_calibrator(scores: torch.Tensor, labels: torch.Tensor, steps: int = 300) -> tuple[float, float]:
    """Fit scalar slope/intercept on training labels and freeze them thereafter."""
    scores = scores.detach().float().reshape(-1)
    labels = labels.detach().float().reshape(-1)
    if scores.numel() != labels.numel() or set(labels.tolist()) != {0.0, 1.0}:
        raise ValueError("Calibration needs aligned verified positives and negatives")
    slope = torch.nn.Parameter(torch.tensor(1.0))
    intercept = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.Adam([slope, intercept], lr=0.05)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(slope * scores + intercept, labels)
        loss.backward()
        optimizer.step()
    return float(slope.detach()), float(intercept.detach())


def auxiliary_loss(hidden: torch.Tensor, direction: torch.Tensor, labels: torch.Tensor,
                   slope: float, intercept: float) -> torch.Tensor:
    score = concept_score(hidden, direction)
    known = labels >= 0
    if not bool(known.any()):
        return score.sum() * 0
    logits = slope * score[known] + intercept
    return F.binary_cross_entropy_with_logits(logits, labels[known].float())
