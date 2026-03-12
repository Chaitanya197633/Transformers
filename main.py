from __future__ import annotations

import random
from importlib.util import find_spec

from config import DEFAULT_CONFIG


def _missing_dependencies() -> list[str]:
    required = ["numpy", "torch"]
    return [name for name in required if find_spec(name) is None]


def _ensure_runtime_dependencies() -> None:
    missing = _missing_dependencies()
    if not missing:
        return

    joined = ", ".join(missing)
    raise RuntimeError(
        "Missing required Python packages: "
        f"{joined}.\n"
        "Install dependencies in your active VS Code/Jupyter environment with:\n"
        "  python -m pip install -r requirements.txt\n"
        "If using a notebook, run:\n"
        "  %pip install -r requirements.txt\n"
        "Then restart the kernel/interpreter and run again."
    )


def set_seed(seed: int, np_module, torch_module) -> None:
    random.seed(seed)
    np_module.random.seed(seed)
    torch_module.manual_seed(seed)


def run_pipeline() -> dict[str, float | int]:
    _ensure_runtime_dependencies()

    import numpy as np
    import torch

    from compression import (
        apply_magnitude_pruning,
        global_sparsity,
        quantize_model_weights,
        replace_linear_layers,
    )
    from data import build_feature_dataloaders
    from models import FeatureMLP
    from utils import evaluate, filesize_bytes, huffman_encode_file, save_compressed_npz, train_model

    cfg = DEFAULT_CONFIG
    set_seed(cfg.seed, np, torch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = build_feature_dataloaders(
        num_samples=cfg.num_samples,
        image_size=cfg.image_size,
        num_channels=cfg.num_channels,
        feature_dim=cfg.feature_dim,
        num_classes=cfg.num_classes,
        batch_size=cfg.batch_size,
    )

    model = FeatureMLP(cfg.feature_dim, cfg.hidden_dims, cfg.num_classes)
    baseline_params = model.parameter_count()
    train_model(model, train_loader, cfg.epochs, cfg.lr, device)
    baseline_acc = evaluate(model, test_loader, device)

    model = replace_linear_layers(model)
    apply_magnitude_pruning(model, cfg.pruning_ratio)
    train_model(model, train_loader, cfg.finetune_epochs, cfg.lr, device)
    pruned_acc = evaluate(model, test_loader, device)

    quantize_model_weights(model, cfg.quantization_bits)
    quantized_acc = evaluate(model, test_loader, device)

    metadata = {
        "feature_dim": cfg.feature_dim,
        "num_classes": cfg.num_classes,
        "pruning_ratio": cfg.pruning_ratio,
        "quantization_bits": cfg.quantization_bits,
    }
    save_compressed_npz(model, cfg.npz_output_path, metadata)
    npz_size = filesize_bytes(cfg.npz_output_path)
    huffman_ratio = huffman_encode_file(cfg.npz_output_path, cfg.huffman_output_path)
    huff_size = filesize_bytes(cfg.huffman_output_path)

    return {
        "baseline_accuracy": baseline_acc,
        "pruned_accuracy": pruned_acc,
        "quantized_accuracy": quantized_acc,
        "baseline_parameters": baseline_params,
        "global_sparsity": global_sparsity(model),
        "npz_size_bytes": npz_size,
        "huffman_size_bytes": huff_size,
        "huffman_ratio": huffman_ratio,
    }


if __name__ == "__main__":
    metrics = run_pipeline()
    for key, value in metrics.items():
        print(f"{key}: {value}")
