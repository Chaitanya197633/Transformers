from __future__ import annotations

import heapq
import os
import random
from collections import Counter

import numpy as np


class _Node:
    def __init__(self, freq: int, symbol: int | None = None, left=None, right=None):
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq


def _build_codes(node, prefix: str, table: dict[int, str]) -> None:
    if node.symbol is not None:
        table[node.symbol] = prefix or "0"
        return
    _build_codes(node.left, prefix + "0", table)
    _build_codes(node.right, prefix + "1", table)


def huffman_encode_file(input_path: str, output_path: str) -> float:
    with open(input_path, "rb") as f:
        raw = f.read()

    freq = Counter(raw)
    heap = [_Node(fr, symbol=sym) for sym, fr in freq.items()]
    heapq.heapify(heap)

    if len(heap) == 1:
        codes = {heap[0].symbol: "0"}
    else:
        while len(heap) > 1:
            a = heapq.heappop(heap)
            b = heapq.heappop(heap)
            heapq.heappush(heap, _Node(a.freq + b.freq, left=a, right=b))
        codes: dict[int, str] = {}
        _build_codes(heap[0], "", codes)

    bitstring = "".join(codes[b] for b in raw)
    padding = (8 - len(bitstring) % 8) % 8
    bitstring += "0" * padding

    out = bytearray([padding])
    for i in range(0, len(bitstring), 8):
        out.append(int(bitstring[i : i + 8], 2))

    with open(output_path, "wb") as f:
        f.write(out)

    return len(out) / max(len(raw), 1)


class NumpyMLP:
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], num_classes: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        dims = (input_dim,) + hidden_dims + (num_classes,)
        self.weights = []
        self.biases = []
        for i in range(len(dims) - 1):
            fan_in = dims[i]
            w = rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, dims[i + 1])).astype(np.float32)
            b = np.zeros((dims[i + 1],), dtype=np.float32)
            self.weights.append(w)
            self.biases.append(b)

    def parameter_count(self) -> int:
        return int(sum(w.size + b.size for w, b in zip(self.weights, self.biases)))

    def forward(self, x: np.ndarray):
        activations = [x]
        preacts = []
        a = x
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ w + b
            preacts.append(z)
            a = np.maximum(z, 0) if i < len(self.weights) - 1 else z
            activations.append(a)
        return activations, preacts

    def predict(self, x: np.ndarray) -> np.ndarray:
        a = x
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ w + b
            a = np.maximum(z, 0) if i < len(self.weights) - 1 else z
        return np.argmax(a, axis=1)


def _softmax_cross_entropy(logits: np.ndarray, y: np.ndarray):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    n = logits.shape[0]
    grad = probs
    grad[np.arange(n), y] -= 1
    grad /= n
    return grad


def _batch_iter(x: np.ndarray, y: np.ndarray, batch_size: int, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(x))
    rng.shuffle(idx)
    for i in range(0, len(x), batch_size):
        sel = idx[i : i + batch_size]
        yield x[sel], y[sel]


def train_numpy_mlp(model: NumpyMLP, x: np.ndarray, y: np.ndarray, epochs: int, lr: float, batch_size: int, seed: int) -> None:
    for ep in range(epochs):
        for xb, yb in _batch_iter(x, y, batch_size, seed + ep):
            activations, preacts = model.forward(xb)
            dz = _softmax_cross_entropy(activations[-1], yb)

            grad_w = []
            grad_b = []
            for i in range(len(model.weights) - 1, -1, -1):
                a_prev = activations[i]
                dw = a_prev.T @ dz
                db = dz.sum(axis=0)
                grad_w.append(dw)
                grad_b.append(db)
                if i > 0:
                    dz = (dz @ model.weights[i].T) * (preacts[i - 1] > 0)

            grad_w.reverse()
            grad_b.reverse()
            for i in range(len(model.weights)):
                model.weights[i] -= lr * grad_w[i].astype(np.float32)
                model.biases[i] -= lr * grad_b[i].astype(np.float32)


def evaluate_numpy(model: NumpyMLP, x: np.ndarray, y: np.ndarray) -> float:
    return float((model.predict(x) == y).mean())


def extract_features_numpy(images: np.ndarray, feature_dim: int, seed: int = 7) -> np.ndarray:
    n = images.shape[0]
    flattened = images.reshape(n, -1)
    pooled_mean = images.mean(axis=(2, 3))
    pooled_std = images.std(axis=(2, 3))
    raw = np.concatenate([flattened, pooled_mean, pooled_std], axis=1)
    rng = np.random.default_rng(seed)
    projection = rng.standard_normal((raw.shape[1], feature_dim)).astype(np.float32)
    return (raw @ projection) / np.sqrt(raw.shape[1])


def build_feature_arrays(num_samples: int, image_size: int, num_channels: int, feature_dim: int, num_classes: int, seed: int):
    rng = np.random.default_rng(seed)
    images = rng.random((num_samples, num_channels, image_size, image_size), dtype=np.float32)
    labels = rng.integers(0, num_classes, size=(num_samples,), dtype=np.int64)
    features = extract_features_numpy(images, feature_dim)
    idx = np.arange(num_samples)
    rng.shuffle(idx)
    split = int(0.8 * num_samples)
    tr, te = idx[:split], idx[split:]
    return features[tr].astype(np.float32), labels[tr], features[te].astype(np.float32), labels[te]


def apply_magnitude_pruning_numpy(weights: list[np.ndarray], ratio: float) -> list[np.ndarray]:
    flat = np.concatenate([np.abs(w).ravel() for w in weights])
    k = int(ratio * flat.size)
    if k <= 0:
        return [np.ones_like(w, dtype=np.float32) for w in weights]
    threshold = np.partition(flat, k - 1)[k - 1]
    return [(np.abs(w) > threshold).astype(np.float32) for w in weights]


def global_sparsity_numpy(masks: list[np.ndarray]) -> float:
    total = sum(m.size for m in masks)
    zero = sum((m == 0).sum() for m in masks)
    return float(zero / max(total, 1))


def quantize_active_weights_numpy(weights: list[np.ndarray], masks: list[np.ndarray], bits: int) -> None:
    levels = (2**bits) - 1
    for i, w in enumerate(weights):
        active = masks[i] > 0
        out = np.zeros_like(w)
        if np.any(active):
            vals = w[active]
            min_val = vals.min()
            max_val = vals.max()
            if np.isclose(min_val, max_val):
                out[active] = vals
            else:
                scale = (max_val - min_val) / levels
                out[active] = np.round((vals - min_val) / scale) * scale + min_val
        weights[i] = out.astype(np.float32)


def save_numpy_compressed_npz(weights, biases, masks, output_path: str, metadata: dict) -> None:
    payload = {}
    for i, w in enumerate(weights):
        payload[f"layer_{i}.weight"] = w
        payload[f"layer_{i}.mask"] = masks[i]
        payload[f"layer_{i}.bias"] = biases[i]
    for k, v in metadata.items():
        payload[f"meta__{k}"] = np.array(v)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, **payload)


def filesize_bytes(path: str) -> int:
    return os.path.getsize(path)


def run_numpy_pipeline(cfg) -> dict[str, float | int | str]:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    x_train, y_train, x_test, y_test = build_feature_arrays(
        cfg.num_samples, cfg.image_size, cfg.num_channels, cfg.feature_dim, cfg.num_classes, cfg.seed
    )

    model = NumpyMLP(cfg.feature_dim, cfg.hidden_dims, cfg.num_classes, seed=cfg.seed)
    baseline_params = model.parameter_count()
    train_numpy_mlp(model, x_train, y_train, cfg.epochs, cfg.lr, cfg.batch_size, cfg.seed)
    baseline_acc = evaluate_numpy(model, x_test, y_test)

    masks = apply_magnitude_pruning_numpy(model.weights, cfg.pruning_ratio)
    for i in range(len(model.weights)):
        model.weights[i] *= masks[i]

    train_numpy_mlp(model, x_train, y_train, cfg.finetune_epochs, cfg.lr, cfg.batch_size, cfg.seed + 101)
    for i in range(len(model.weights)):
        model.weights[i] *= masks[i]
    pruned_acc = evaluate_numpy(model, x_test, y_test)

    quantize_active_weights_numpy(model.weights, masks, cfg.quantization_bits)
    quantized_acc = evaluate_numpy(model, x_test, y_test)

    metadata = {
        "backend": "numpy-fallback",
        "feature_dim": cfg.feature_dim,
        "num_classes": cfg.num_classes,
        "pruning_ratio": cfg.pruning_ratio,
        "quantization_bits": cfg.quantization_bits,
    }
    save_numpy_compressed_npz(model.weights, model.biases, masks, cfg.npz_output_path, metadata)
    npz_size = filesize_bytes(cfg.npz_output_path)
    huffman_ratio = huffman_encode_file(cfg.npz_output_path, cfg.huffman_output_path)
    huff_size = filesize_bytes(cfg.huffman_output_path)

    return {
        "backend": "numpy-fallback",
        "baseline_accuracy": baseline_acc,
        "pruned_accuracy": pruned_acc,
        "quantized_accuracy": quantized_acc,
        "baseline_parameters": baseline_params,
        "global_sparsity": global_sparsity_numpy(masks),
        "npz_size_bytes": npz_size,
        "huffman_size_bytes": huff_size,
        "huffman_ratio": huffman_ratio,
    }
