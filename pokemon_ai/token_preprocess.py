from __future__ import annotations

import torch

from .staged_preprocess import blur_color, sobel_edges


def token_source_condition(source: torch.Tensor, blur_factor: int = 16) -> torch.Tensor:
    """Condition for token models: color layout plus edges, without raw RGB."""

    return torch.cat([blur_color(source, blur_factor), sobel_edges(source)], dim=-3)
