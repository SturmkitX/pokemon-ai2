from __future__ import annotations

import torch
from torch import nn
from torchvision.models import VGG16_Weights, vgg16


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((pred - target) ** 2 + self.eps**2).mean()


class PerceptualLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        weights = VGG16_Weights.IMAGENET1K_V1
        features = vgg16(weights=weights).features[:16].eval()
        for param in features.parameters():
            param.requires_grad_(False)
        self.features = features
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.loss = nn.L1Loss()

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        x = (x + 1.0) * 0.5
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_features = self.features(self._prep(pred))
        with torch.no_grad():
            target_features = self.features(self._prep(target))
        return self.loss(pred_features, target_features)
