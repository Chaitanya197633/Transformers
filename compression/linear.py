from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CompressedLinear(nn.Module):
    """Linear layer with explicit mask support for pruning-aware forward pass."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.register_buffer("mask", torch.ones(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            bound = 1 / in_features**0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.mask, self.bias)

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "CompressedLinear":
        layer = cls(linear.in_features, linear.out_features, linear.bias is not None)
        with torch.no_grad():
            layer.weight.copy_(linear.weight)
            if layer.bias is not None and linear.bias is not None:
                layer.bias.copy_(linear.bias)
        return layer


def replace_linear_layers(module: nn.Module) -> nn.Module:
    """Recursively replace nn.Linear with CompressedLinear layers."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, CompressedLinear.from_linear(child))
        else:
            replace_linear_layers(child)
    return module
