from __future__ import annotations

import torch
from torch import nn

from .linear import CompressedLinear


def _uniform_quantize(x: torch.Tensor, bits: int) -> torch.Tensor:
    levels = 2**bits - 1
    min_val = x.min()
    max_val = x.max()
    if torch.isclose(min_val, max_val):
        return x
    scale = (max_val - min_val) / levels
    q = torch.round((x - min_val) / scale)
    return q * scale + min_val


def quantize_model_weights(model: nn.Module, bits: int = 8) -> None:
    """In-place uniform quantization for compressed linear layer weights."""
    for module in model.modules():
        if isinstance(module, CompressedLinear):
            with torch.no_grad():
                active = module.mask.bool()
                w = module.weight
                quantized = w.clone()
                quantized[active] = _uniform_quantize(w[active], bits)
                quantized[~active] = 0.0
                module.weight.copy_(quantized)
