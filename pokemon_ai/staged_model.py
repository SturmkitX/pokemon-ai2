from __future__ import annotations

import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StageUNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, base_channels: int = 64) -> None:
        super().__init__()
        c = base_channels
        self.enc0 = Block(in_channels, c)
        self.down1 = nn.Sequential(nn.Conv2d(c, c * 2, 4, stride=2, padding=1), nn.SiLU(inplace=True))
        self.enc1 = Block(c * 2, c * 2)
        self.down2 = nn.Sequential(nn.Conv2d(c * 2, c * 4, 4, stride=2, padding=1), nn.SiLU(inplace=True))
        self.mid = nn.Sequential(Block(c * 4, c * 4), Block(c * 4, c * 4))
        self.up1 = nn.Conv2d(c * 4 + c * 2, c * 2, 3, padding=1)
        self.dec1 = Block(c * 2, c * 2)
        self.up0 = nn.Conv2d(c * 2 + c, c, 3, padding=1)
        self.dec0 = Block(c, c)
        self.out = nn.Sequential(nn.Conv2d(c, out_channels, 3, padding=1), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.enc0(x)
        s1 = self.enc1(self.down1(s0))
        x = self.mid(self.down2(s1))
        x = nn.functional.interpolate(x, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec1(self.up1(torch.cat([x, s1], dim=1)))
        x = nn.functional.interpolate(x, size=s0.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec0(self.up0(torch.cat([x, s0], dim=1)))
        return self.out(x)


def stage_channels(stage: str) -> tuple[int, int]:
    if stage == "layout":
        return 7, 3
    if stage == "edge":
        return 10, 1
    if stage == "refine":
        return 11, 3
    raise ValueError(f"Unsupported stage: {stage}")
