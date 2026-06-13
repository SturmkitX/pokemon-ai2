from __future__ import annotations

import math

import torch
from torch import nn


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, device=timesteps.device, dtype=torch.float32) / half
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimeResBlock(nn.Module):
    def __init__(self, channels: int, time_dim: int) -> None:
        super().__init__()
        self.time = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, channels))
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h = h + self.time(time)[:, :, None, None]
        h = self.block2(h)
        return x + h


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        self.block = TimeResBlock(out_channels, time_dim)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return self.block(self.proj(x), time)


class Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1)
        self.block = TimeResBlock(out_channels, time_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(self.proj(x), time)


class ConditionalLatentUNet(nn.Module):
    """Small latent noise predictor conditioned on the source image latent."""

    def __init__(self, latent_channels: int = 4, base_channels: int = 128, time_dim: int = 256) -> None:
        super().__init__()
        c = base_channels
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(inplace=True),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.stem = nn.Conv2d(latent_channels * 2, c, 3, padding=1)
        self.block0 = TimeResBlock(c, time_dim)
        self.down1 = Down(c, c * 2, time_dim)
        self.down2 = Down(c * 2, c * 4, time_dim)
        self.mid = nn.Sequential(
            TimeResBlock(c * 4, time_dim),
            TimeResBlock(c * 4, time_dim),
        )
        self.up2 = Up(c * 4, c * 2, c * 2, time_dim)
        self.up1 = Up(c * 2, c, c, time_dim)
        self.out = nn.Sequential(
            nn.GroupNorm(8, c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, latent_channels, 3, padding=1),
        )
        self.time_dim = time_dim

    def forward(self, noisy_target: torch.Tensor, source_latent: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        time = self.time_mlp(timestep_embedding(timesteps, self.time_dim))
        x = self.stem(torch.cat([noisy_target, source_latent], dim=1))
        s0 = self.block0(x, time)
        s1 = self.down1(s0, time)
        x = self.down2(s1, time)
        for block in self.mid:
            x = block(x, time)
        x = self.up2(x, s1, time)
        x = self.up1(x, s0, time)
        return self.out(x)
