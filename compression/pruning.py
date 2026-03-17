from __future__ import annotations

import torch
from torch import nn

from .linear import CompressedLinear


def apply_magnitude_pruning(model: nn.Module, ratio: float) -> None:
    """Set binary masks using global magnitude threshold across compressed layers."""
    weights = []
    layers = []
    for module in model.modules():
        if isinstance(module, CompressedLinear):
            weights.append(module.weight.detach().abs().flatten())
            layers.append(module)

    if not weights:
        return

    all_weights = torch.cat(weights)
    k = int(ratio * all_weights.numel())
    if k <= 0:
        return
    threshold = torch.kthvalue(all_weights, k).values

    for layer in layers:
        new_mask = (layer.weight.detach().abs() > threshold).float()
        layer.mask.copy_(new_mask)


def global_sparsity(model: nn.Module) -> float:
    total = 0
    zero = 0
    for module in model.modules():
        if isinstance(module, CompressedLinear):
            total += module.mask.numel()
            zero += (module.mask == 0).sum().item()
    return zero / max(total, 1)
