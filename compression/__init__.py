from .linear import CompressedLinear, replace_linear_layers
from .pruning import apply_magnitude_pruning, global_sparsity
from .quantization import quantize_model_weights

__all__ = [
    "CompressedLinear",
    "replace_linear_layers",
    "apply_magnitude_pruning",
    "global_sparsity",
    "quantize_model_weights",
]
