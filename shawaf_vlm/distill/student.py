"""Project a small vision backbone into the teacher embedding space."""

from __future__ import annotations

import torch
from torch import nn


class StudentWithProjection(nn.Module):
    def __init__(self, backbone: nn.Module, in_dim: int, out_dim: int) -> None:
        super().__init__()
        if in_dim < 1 or out_dim < 1:
            raise ValueError("in_dim and out_dim must be >= 1")
        self.backbone = backbone
        self.projection = nn.Linear(in_dim, out_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.backbone(inputs)
        if isinstance(features, tuple):
            features = features[0]
        return self.projection(features)
