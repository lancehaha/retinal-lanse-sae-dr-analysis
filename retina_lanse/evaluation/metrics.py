from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


@torch.no_grad()
def evaluate_reconstruction(
    embeddings: torch.Tensor, reconstruction: torch.Tensor
) -> dict[str, float]:
    if embeddings.shape != reconstruction.shape or embeddings.ndim != 2:
        raise ValueError("embeddings and reconstruction must be same-shape rank-2 tensors")
    error = ((embeddings.double() - reconstruction.double()) ** 2).sum()
    centered = (
        (embeddings.double() - embeddings.double().mean(dim=0, keepdim=True)) ** 2
    ).sum().clamp_min(1e-12)
    return {
        "mse": float((error / embeddings.numel()).item()),
        "explained_variance": float((1.0 - error / centered).item()),
        "cosine_similarity": float(
            F.cosine_similarity(reconstruction, embeddings, dim=1).double().mean().item()
        ),
    }


@torch.no_grad()
def reconstruct_embeddings_batched(
    model,
    embeddings: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Materialize reconstructions without retaining wide sparse activations."""
    if embeddings.ndim != 2 or len(embeddings) == 0:
        raise ValueError("embeddings must be a non-empty rank-2 tensor")
    target_device = torch.device(device)
    model.eval()
    outputs = []
    loader = DataLoader(TensorDataset(embeddings), batch_size=batch_size, shuffle=False)
    for (cpu_batch,) in loader:
        reconstruction, _ = model(cpu_batch.to(target_device))
        outputs.append(reconstruction.float().cpu())
    return torch.cat(outputs, dim=0)


@torch.no_grad()
def evaluate_embeddings(model, embeddings: torch.Tensor) -> dict[str, float | int]:
    reconstruction, activations = model(embeddings)
    residual = ((embeddings - reconstruction) ** 2).sum()
    centered = ((embeddings - embeddings.mean(dim=0, keepdim=True)) ** 2).sum().clamp_min(1e-12)
    active_counts = (activations > 0).sum(dim=1)
    feature_frequency = (activations > 0).float().mean(dim=0)
    return {
        "mse": float(F.mse_loss(reconstruction, embeddings).item()),
        "explained_variance": float((1.0 - residual / centered).item()),
        "cosine_similarity": float(F.cosine_similarity(reconstruction, embeddings, dim=1).mean().item()),
        "mean_active_features": float(active_counts.float().mean().item()),
        "max_active_features": int(active_counts.max().item()),
        "dead_feature_fraction": float((feature_frequency == 0).float().mean().item()),
        "saturated_feature_fraction": float((feature_frequency > 0.95).float().mean().item()),
        "mean_activation": float(activations[activations > 0].mean().item()) if (activations > 0).any() else 0.0,
    }


@torch.no_grad()
def evaluate_embeddings_batched(
    model,
    embeddings: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device | str,
) -> dict[str, float | int]:
    """Evaluate a wide SAE without materializing all validation activations."""
    if embeddings.ndim != 2 or len(embeddings) == 0:
        raise ValueError("embeddings must be a non-empty rank-2 tensor")
    target_device = torch.device(device)
    model.eval()
    mean = embeddings.double().mean(dim=0)
    centered_sum = float(((embeddings.double() - mean) ** 2).sum().item())
    loader = DataLoader(TensorDataset(embeddings), batch_size=batch_size, shuffle=False)
    squared_error = 0.0
    cosine_sum = 0.0
    active_sum = 0
    max_active = 0
    positive_activation_sum = 0.0
    positive_activation_count = 0
    feature_counts = torch.zeros(model.dictionary_size, dtype=torch.int64)
    total = 0
    for (cpu_batch,) in loader:
        batch = cpu_batch.to(target_device)
        reconstruction, activations = model(batch)
        squared_error += float(((batch - reconstruction).float() ** 2).sum().item())
        cosine_sum += float(F.cosine_similarity(reconstruction, batch, dim=1).float().sum().item())
        active = activations > 0
        active_counts = active.sum(dim=1)
        active_sum += int(active_counts.sum().item())
        max_active = max(max_active, int(active_counts.max().item()))
        feature_counts += active.sum(dim=0).cpu()
        positive_activation_sum += float(activations[active].float().sum().item())
        positive_activation_count += int(active.sum().item())
        total += len(batch)
    feature_frequency = feature_counts.float() / total
    return {
        "mse": squared_error / (total * embeddings.shape[1]),
        "explained_variance": 1.0 - squared_error / max(centered_sum, 1e-12),
        "cosine_similarity": cosine_sum / total,
        "mean_active_features": active_sum / total,
        "max_active_features": max_active,
        "dead_feature_fraction": float((feature_counts == 0).float().mean().item()),
        "saturated_feature_fraction": float((feature_frequency > 0.95).float().mean().item()),
        "mean_activation": positive_activation_sum / max(positive_activation_count, 1),
    }


@torch.no_grad()
def sparse_coordinate_baseline(
    embeddings: torch.Tensor, reference_mean: torch.Tensor, k: int
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Reconstruct with only the k largest centered input coordinates."""
    residual = embeddings - reference_mean
    indices = residual.abs().topk(k, dim=1).indices
    sparse_residual = torch.zeros_like(residual).scatter(1, indices, residual.gather(1, indices))
    reconstruction = reference_mean + sparse_residual
    error = ((embeddings - reconstruction) ** 2).sum()
    centered = ((embeddings - embeddings.mean(dim=0, keepdim=True)) ** 2).sum().clamp_min(1e-12)
    metrics = {
        "mse": float(F.mse_loss(reconstruction, embeddings).item()),
        "explained_variance": float((1.0 - error / centered).item()),
        "cosine_similarity": float(F.cosine_similarity(reconstruction, embeddings, dim=1).mean().item()),
        "mean_active_features": float(k),
        "max_active_features": int(k),
    }
    return reconstruction, metrics
