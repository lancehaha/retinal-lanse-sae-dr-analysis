from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from retina_lanse.sae.model import TopKSAE


def save_checkpoint(path: str | Path, model: TopKSAE, optimizer, metadata: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config": {
            "input_dim": model.input_dim,
            "dictionary_size": model.dictionary_size,
            "k": model.k,
        },
        "metadata": metadata,
    }, target)


def load_checkpoint(path: str | Path) -> tuple[TopKSAE, dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    model = TopKSAE(**payload["model_config"])
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload

