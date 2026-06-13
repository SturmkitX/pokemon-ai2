from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import save_image


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def denormalize(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1.0) * 0.5


def save_sample_grid(
    path: str | Path,
    source: torch.Tensor,
    fake: torch.Tensor,
    target: torch.Tensor,
    max_samples: int = 8,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = min(source.size(0), max_samples)
    grid = torch.cat([source[:n], fake[:n], target[:n]], dim=0)
    save_image(denormalize(grid), path, nrow=n)
