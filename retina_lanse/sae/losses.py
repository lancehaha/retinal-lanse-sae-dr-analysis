import torch
import torch.nn.functional as F


def reconstruction_loss(reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(reconstruction, target)

