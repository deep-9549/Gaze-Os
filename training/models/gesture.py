"""Gesture classifiers over 30 x 17 feature windows (PRD 13.2)."""

from __future__ import annotations

import torch
from torch import nn

NUM_CLASSES = 9
NUM_FEATURES = 17


class TemporalCNN(nn.Module):
    """B1: 3 conv layers (kernel 5, channels 32-64-64), global pool, FC to 9 classes (~30k params)."""

    def __init__(self, num_features: int = NUM_FEATURES, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        c = num_features
        for out in (32, 64, 64):
            layers += [nn.Conv1d(c, out, 5, padding=2), nn.BatchNorm1d(out), nn.ReLU(inplace=True)]
            c = out
        self.net = nn.Sequential(*layers)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, F)
        h = self.net(x.transpose(1, 2))
        return self.fc(h.mean(dim=2))

    def head(self) -> nn.Module:
        return self.fc


class GRUClassifier(nn.Module):
    """B2: 2-layer GRU, 64 hidden units."""

    def __init__(self, num_features: int = NUM_FEATURES, num_classes: int = NUM_CLASSES, hidden: int = 64) -> None:
        super().__init__()
        self.gru = nn.GRU(num_features, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.fc(out[:, -1])

    def head(self) -> nn.Module:
        return self.fc


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
