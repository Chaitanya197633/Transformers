from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


def extract_features(images: torch.Tensor, feature_dim: int) -> torch.Tensor:
    """Extract fixed-dim feature vectors from image tensors without training a CNN.

    Uses deterministic random projections over pooled image statistics to mimic
    image-derived features in notebook/kaggle workflows.
    """
    batch = images.shape[0]
    flattened = images.view(batch, -1)
    pooled_mean = images.mean(dim=(2, 3))
    pooled_std = images.std(dim=(2, 3))
    raw = torch.cat([flattened, pooled_mean, pooled_std], dim=1)

    generator = torch.Generator().manual_seed(7)
    projection = torch.randn(raw.shape[1], feature_dim, generator=generator, device=raw.device)
    return raw @ projection / raw.shape[1] ** 0.5


def build_feature_dataloaders(
    num_samples: int,
    image_size: int,
    num_channels: int,
    feature_dim: int,
    num_classes: int,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    """Create train/test loaders from synthetic image data and extracted features."""
    g = torch.Generator().manual_seed(11)
    images = torch.rand(num_samples, num_channels, image_size, image_size, generator=g)
    labels = torch.randint(0, num_classes, (num_samples,), generator=g)
    features = extract_features(images, feature_dim)

    dataset = TensorDataset(features.float(), labels.long())
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size], generator=g)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
