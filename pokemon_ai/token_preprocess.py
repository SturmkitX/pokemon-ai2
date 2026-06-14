from __future__ import annotations

import torch

from .staged_preprocess import blur_color, sobel_edges


def token_source_condition(source: torch.Tensor, blur_factor: int = 16, mode: str = "safe") -> torch.Tensor:
    if mode == "safe":
        return torch.cat([blur_color(source, blur_factor), sobel_edges(source)], dim=-3)
    if mode == "rgb":
        return torch.cat([source, blur_color(source, blur_factor), sobel_edges(source)], dim=-3)
    raise ValueError(f"Unsupported token source condition mode: {mode}")


def token_condition_channels(mode: str) -> int:
    if mode == "safe":
        return 4
    if mode == "rgb":
        return 7
    raise ValueError(f"Unsupported token source condition mode: {mode}")
