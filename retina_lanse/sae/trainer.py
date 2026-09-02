from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from retina_lanse.evaluation.metrics import evaluate_embeddings, evaluate_embeddings_batched
from retina_lanse.sae.losses import reconstruction_loss


def train_sae(model, train_embeddings: torch.Tensor, validation_embeddings: torch.Tensor, *,
              epochs: int, batch_size: int, learning_rate: float, weight_decay: float, seed: int):
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(train_embeddings), batch_size=batch_size, shuffle=True, generator=generator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for (batch,) in loader:
            optimizer.zero_grad(set_to_none=True)
            reconstruction, _ = model(batch)
            loss = reconstruction_loss(reconstruction, batch)
            loss.backward()
            optimizer.step()
            model.normalize_decoder_columns()
            total_loss += float(loss.item()) * len(batch)
            total_count += len(batch)
        model.eval()
        val = evaluate_embeddings(model, validation_embeddings)
        history.append({"epoch": epoch, "train_mse": total_loss / total_count, "validation_mse": val["mse"]})
    return history, optimizer


def write_curves(path: str | Path, history: list[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_mse", "validation_mse"])
        writer.writeheader()
        writer.writerows(history)


def _clear_resampled_optimizer_state(
    optimizer: torch.optim.Optimizer,
    model,
    dead_indices: torch.Tensor,
) -> None:
    parameter_slices = {
        model.encoder.weight: (dead_indices, slice(None)),
        model.encoder.bias: (dead_indices,),
        model.decoder.weight: (slice(None), dead_indices),
    }
    for parameter, index in parameter_slices.items():
        state = optimizer.state.get(parameter, {})
        for value in state.values():
            if torch.is_tensor(value) and value.shape == parameter.shape:
                value[index] = 0


@torch.no_grad()
def resample_dead_features(
    model,
    optimizer: torch.optim.Optimizer,
    calibration_embeddings: torch.Tensor,
    dead_indices: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device | str,
) -> int:
    """Reinitialize dead atoms from the highest-error calibration residuals."""
    if dead_indices.numel() == 0:
        return 0
    target_device = torch.device(device)
    model.eval()
    residuals: list[torch.Tensor] = []
    residual_norms: list[torch.Tensor] = []
    loader = DataLoader(TensorDataset(calibration_embeddings), batch_size=batch_size, shuffle=False)
    for (cpu_batch,) in loader:
        batch = cpu_batch.to(target_device)
        reconstruction, _ = model(batch)
        residual = batch - reconstruction
        residuals.append(residual.cpu())
        residual_norms.append(residual.norm(dim=1).cpu())
    residual_bank = torch.cat(residuals, dim=0)
    norm_bank = torch.cat(residual_norms, dim=0)
    ranked = torch.argsort(norm_bank, descending=True)
    selected = ranked[torch.arange(len(dead_indices)) % len(ranked)]
    directions = residual_bank[selected]
    directions = directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-8)
    dead_on_device = dead_indices.to(target_device)
    directions = directions.to(target_device)
    model.encoder.weight[dead_on_device] = directions
    model.encoder.bias[dead_on_device] = 0
    model.decoder.weight[:, dead_on_device] = directions.T
    model.normalize_decoder_columns()
    _clear_resampled_optimizer_state(optimizer, model, dead_on_device)
    return len(dead_indices)


def train_sae_batched(
    model,
    train_embeddings: torch.Tensor,
    validation_embeddings: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    evaluation_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device | str,
    resample_dead: bool,
    resample_warmup_epochs: int = 5,
    resample_interval_epochs: int = 5,
    resample_calibration_size: int = 8192,
) -> tuple[list[dict], torch.optim.Optimizer]:
    """Publication-scale fixed-epoch trainer with optional deterministic dead-atom resampling."""
    if epochs < 1 or batch_size < 1 or evaluation_batch_size < 1:
        raise ValueError("epochs and batch sizes must be positive")
    target_device = torch.device(device)
    model.to(target_device)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_embeddings),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    calibration_generator = torch.Generator().manual_seed(seed + 1_000_003)
    calibration_count = min(resample_calibration_size, len(train_embeddings))
    calibration_indices = torch.randperm(len(train_embeddings), generator=calibration_generator)[
        :calibration_count
    ]
    calibration = train_embeddings[calibration_indices]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_squared_error = 0.0
        total_elements = 0
        feature_counts = torch.zeros(model.dictionary_size, dtype=torch.int64)
        for (cpu_batch,) in loader:
            batch = cpu_batch.to(target_device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction, activations = model(batch)
            loss = reconstruction_loss(reconstruction, batch)
            loss.backward()
            optimizer.step()
            model.normalize_decoder_columns()
            total_squared_error += float(((reconstruction.detach() - batch).float() ** 2).sum().item())
            total_elements += batch.numel()
            feature_counts += (activations.detach() > 0).sum(dim=0).cpu()

        resampled = 0
        should_resample = (
            resample_dead
            and epoch >= resample_warmup_epochs
            and epoch < epochs
            and (epoch - resample_warmup_epochs) % resample_interval_epochs == 0
        )
        if should_resample:
            dead_indices = torch.nonzero(feature_counts == 0, as_tuple=False).flatten()
            resampled = resample_dead_features(
                model,
                optimizer,
                calibration,
                dead_indices,
                batch_size=evaluation_batch_size,
                device=target_device,
            )

        validation = evaluate_embeddings_batched(
            model,
            validation_embeddings,
            batch_size=evaluation_batch_size,
            device=target_device,
        )
        history.append(
            {
                "epoch": epoch,
                "train_mse": total_squared_error / total_elements,
                "resampled_features": resampled,
                **{f"validation_{key}": value for key, value in validation.items()},
            }
        )
    return history, optimizer
