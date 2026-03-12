from __future__ import annotations

import torch
from torch import nn


class FeatureMLP(nn.Module):
    """MLP classifier that consumes precomputed feature vectors."""

    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], num_classes: int):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev, dim), nn.ReLU()])
            prev = dim
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
