from __future__ import annotations

import torch
from torch.nn import functional as F


def to_01(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1.0) * 0.5


def to_m11(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(0, 1) * 2.0 - 1.0


def blur_color(x: torch.Tensor, factor: int = 16) -> torch.Tensor:
    batched = x.unsqueeze(0) if x.dim() == 3 else x
    small = F.avg_pool2d(batched, kernel_size=factor, stride=factor)
    blurred = F.interpolate(small, size=batched.shape[-2:], mode="bilinear", align_corners=False)
    return blurred.squeeze(0) if x.dim() == 3 else blurred


def sobel_edges(x: torch.Tensor) -> torch.Tensor:
    batched = x.unsqueeze(0) if x.dim() == 3 else x
    gray = to_01(batched).mean(dim=1, keepdim=True)
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    edge = torch.sqrt(gx.square() + gy.square() + 1e-6).clamp(0, 1)
    edge = to_m11(edge)
    return edge.squeeze(0) if x.dim() == 3 else edge


def source_condition(source: torch.Tensor, blur_factor: int = 16) -> torch.Tensor:
    return torch.cat([source, blur_color(source, blur_factor), sobel_edges(source)], dim=-3)


def make_stage_tensors(
    source: torch.Tensor,
    target: torch.Tensor,
    stage: str,
    blur_factor: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    cond = source_condition(source, blur_factor)
    target_layout = blur_color(target, blur_factor)
    target_edges = sobel_edges(target)
    if stage == "layout":
        return cond, target_layout
    if stage == "edge":
        return torch.cat([cond, target_layout], dim=-3), target_edges
    if stage == "refine":
        return torch.cat([cond, target_layout, target_edges], dim=-3), target
    raise ValueError(f"Unsupported stage: {stage}")
