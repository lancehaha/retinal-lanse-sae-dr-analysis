from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path

import torch

from retina_lanse.encoders.base import FrozenRetinalEncoder


class RETFoundUnavailableError(RuntimeError):
    pass


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


class RETFoundCFPEncoder(FrozenRetinalEncoder):
    """Frozen adapter for the official RETFound MAE CFP ViT-L checkpoint."""

    embedding_dim = 1024
    input_size = 224

    def __init__(
        self,
        model: torch.nn.Module,
        checkpoint_hash: str,
        representation: str,
        load_report: dict[str, list[str]] | None = None,
    ):
        super().__init__()
        self.model = model
        self._checkpoint_hash = checkpoint_hash
        self.representation = representation
        self.load_report = load_report or {"missing_keys": [], "unexpected_keys": []}
        self.freeze()

    @property
    def checkpoint_hash(self) -> str:
        return self._checkpoint_hash

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1:] != (3, self.input_size, self.input_size):
            raise ValueError(
                f"RETFound CFP expects NCHW tensors shaped (N, 3, 224, 224); got {tuple(images.shape)}"
            )
        features = self.model.forward_features(images.float())
        if features.ndim == 3 and features.shape[1] == 1:
            features = features[:, 0]
        if features.ndim != 2 or features.shape[1] != self.embedding_dim:
            raise RuntimeError(f"Unexpected RETFound feature shape: {tuple(features.shape)}")
        return features


def load_retfound_cfp(
    checkpoint: str | Path,
    upstream_repo: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    representation: str = "final_mean_patch",
) -> RETFoundCFPEncoder:
    """Load official architecture and base CFP weights, failing closed on drift."""
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    repo_path = Path(upstream_repo).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise RETFoundUnavailableError(
            "RETFound-CFP checkpoint is absent. Obtain it through the official gated route and record its terms."
        )
    if not (repo_path / "models_vit.py").is_file():
        raise RETFoundUnavailableError(f"Official RETFound source not found at {repo_path}")

    repo_string = str(repo_path)
    if repo_string not in sys.path:
        sys.path.insert(0, repo_string)
    models_vit = importlib.import_module("models_vit")
    if representation not in {"final_mean_patch", "final_cls"}:
        raise ValueError(f"Unsupported RETFound representation: {representation}")
    global_pool = representation == "final_mean_patch"
    model = models_vit.RETFound_mae(
        img_size=224,
        num_classes=5,
        drop_path_rate=0,
        global_pool=global_pool,
    )
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise RETFoundUnavailableError("Checkpoint does not contain the official RETFound `model` state dictionary")
    incompatible = model.load_state_dict(payload["model"], strict=False)
    # The official feature-vector notebook loads with global_pool=True and
    # strict=False. MAE checkpoints may therefore exchange the CLS `norm` for
    # the pooling `fc_norm`; both LayerNorms have deterministic identity
    # initialization. Preserve and report exactly this documented mismatch.
    allowed_missing_keys = {"head.weight", "head.bias"}
    allowed_unexpected_keys: set[str] = set()
    if global_pool:
        allowed_missing_keys |= {"fc_norm.weight", "fc_norm.bias"}
        allowed_unexpected_keys |= {"norm.weight", "norm.bias"}
    else:
        allowed_missing_keys |= {"norm.weight", "norm.bias"}
        allowed_unexpected_keys |= {"fc_norm.weight", "fc_norm.bias"}
    unexpected = set(incompatible.unexpected_keys)
    disallowed_missing = set(incompatible.missing_keys) - allowed_missing_keys
    # The gated Nature CFP artifact is a full MAE pretraining checkpoint. The
    # official feature notebook loads only its encoder into models_vit with
    # strict=False, so decoder_* tensors and mask_token are expected unused
    # keys. Keep this allow-list prefix-bounded instead of accepting arbitrary
    # unexpected tensors.
    expected_mae_decoder = {
        key for key in unexpected if key.startswith("decoder_") or key == "mask_token"
    }
    disallowed_unexpected = unexpected - allowed_unexpected_keys - expected_mae_decoder
    if disallowed_unexpected or disallowed_missing:
        raise RETFoundUnavailableError(
            f"Checkpoint/architecture mismatch: missing={sorted(disallowed_missing)}, "
            f"unexpected={sorted(disallowed_unexpected)}"
        )
    load_report = {
        "missing_keys": sorted(incompatible.missing_keys),
        "unexpected_keys": sorted(incompatible.unexpected_keys),
    }
    return RETFoundCFPEncoder(model, sha256_file(checkpoint_path), representation, load_report)
