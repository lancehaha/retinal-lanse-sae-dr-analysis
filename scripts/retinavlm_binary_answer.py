"""Constrained present/absent readout for RetinaVLM's structured OCT task.

Train with ordinary full-vocabulary next-token loss, then use the two actual
next-token logits at the pre-answer position for forced-choice evaluation.
The authors tokenize the prompt and answer separately. The two candidate
answers therefore use their separately tokenized leading-space forms, each
verified to be one Llama 3 token. This avoids dropping unparsable text.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def post_image_segment(author_prompt: str) -> str:
    """Return the exact suffix tokenized after image embeddings by prompt_wrap."""
    parts = author_prompt.split("<ImageHere>")
    if len(parts) != 2:
        raise ValueError("Author prompt must contain exactly one image marker")
    return parts[1]


def continuation_ids(tokenizer, prefix: str, continuation: str) -> list[int]:
    before = tokenizer.encode(prefix, add_special_tokens=False)
    after = tokenizer.encode(prefix + continuation, add_special_tokens=False)
    if after[:len(before)] != before:
        raise ValueError(f"Tokenization boundary changed for {continuation!r}")
    return after[len(before):]


ANSWER_FORMS = (" absent", " present")


def answer_token_ids(tokenizer, full_prompt: str) -> tuple[int, int]:
    """Return separately tokenized (absent ID, present ID)."""
    if not full_prompt:
        raise ValueError("Author prompt suffix is empty")
    answers = [tokenizer.encode(word, add_special_tokens=False)
               for word in ANSWER_FORMS]
    if any(len(ids) != 1 for ids in answers) or answers[0][0] == answers[1][0]:
        raise ValueError("Leading-space present/absent are not distinct single tokens")
    return answers[0][0], answers[1][0]


def two_choice_logits(next_token_logits: torch.Tensor,
                      absent_id: int, present_id: int) -> torch.Tensor:
    """Return shape ``[..., 2]`` ordered absent then present."""
    if next_token_logits.ndim < 1 or min(absent_id, present_id) < 0:
        raise ValueError("Invalid next-token logits or token IDs")
    if max(absent_id, present_id) >= next_token_logits.shape[-1]:
        raise ValueError("Answer token ID is outside the model vocabulary")
    return next_token_logits[..., [absent_id, present_id]]


def answer_loss(next_token_logits: torch.Tensor, labels: torch.Tensor,
                absent_id: int, present_id: int) -> torch.Tensor:
    """Full-vocabulary next-token cross-entropy; labels are 0/1."""
    if labels.dtype != torch.long or not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("Expected integer binary labels")
    if next_token_logits.ndim != 2 or next_token_logits.shape[0] != labels.numel():
        raise ValueError("Expected one pre-answer logit row per label")
    if max(absent_id, present_id) >= next_token_logits.shape[-1]:
        raise ValueError("Answer token ID is outside the model vocabulary")
    target_ids = torch.where(labels == 1, present_id, absent_id)
    return F.cross_entropy(next_token_logits, target_ids)


def predict_answer(next_token_logits: torch.Tensor,
                   absent_id: int, present_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return binary predictions and conditional present probabilities."""
    choice = two_choice_logits(next_token_logits, absent_id, present_id)
    probability = (choice[..., 1] - choice[..., 0]).sigmoid()
    return (probability >= 0.5).long(), probability
