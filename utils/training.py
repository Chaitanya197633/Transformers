from __future__ import annotations

import torch
from torch import nn


def train_model(model: nn.Module, loader, epochs: int, lr: float, device: torch.device) -> None:
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()


def evaluate(model: nn.Module, loader, device: torch.device) -> float:
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)
