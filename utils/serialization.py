from __future__ import annotations

import os

import numpy as np
import torch


def save_compressed_npz(model: torch.nn.Module, output_path: str, metadata: dict) -> None:
    payload: dict[str, np.ndarray] = {}
    for name, tensor in model.state_dict().items():
        payload[name] = tensor.detach().cpu().numpy()

    for key, value in metadata.items():
        payload[f"meta__{key}"] = np.array(value)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, **payload)


def filesize_bytes(path: str) -> int:
    return os.path.getsize(path)
