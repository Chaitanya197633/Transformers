from dataclasses import dataclass


@dataclass
class Config:
    """Project-wide hyperparameters for feature-based MLP compression."""

    seed: int = 42
    num_samples: int = 2000
    image_size: int = 32
    num_channels: int = 3
    feature_dim: int = 128
    num_classes: int = 10

    batch_size: int = 64
    epochs: int = 12
    finetune_epochs: int = 6
    lr: float = 1e-3

    hidden_dims: tuple[int, ...] = (256, 128)
    pruning_ratio: float = 0.6
    quantization_bits: int = 8

    npz_output_path: str = "compressed_models/compressed_mlp.npz"
    huffman_output_path: str = "compressed_models/compressed_mlp.huff"


DEFAULT_CONFIG = Config()
