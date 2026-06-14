from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


class TokenResBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.dropout(h)
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class TokenUNetCore(nn.Module):
    def __init__(self, channels: int, layers: int, dropout: float = 0.0) -> None:
        super().__init__()
        stem_layers = max(1, layers // 4)
        mid_layers = max(2, layers - stem_layers * 3)
        self.enc1 = nn.Sequential(*[TokenResBlock(channels, dropout) for _ in range(stem_layers)])
        self.down1 = nn.Conv2d(channels, channels, 4, stride=2, padding=1)
        self.enc2 = nn.Sequential(*[TokenResBlock(channels, dropout) for _ in range(stem_layers)])
        self.down2 = nn.Conv2d(channels, channels, 4, stride=2, padding=1)
        self.mid = nn.Sequential(*[TokenResBlock(channels, dropout) for _ in range(mid_layers)])
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            *[TokenResBlock(channels, dropout) for _ in range(stem_layers)],
        )
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            *[TokenResBlock(channels, dropout) for _ in range(stem_layers)],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip1 = self.enc1(x)
        skip2 = self.enc2(self.down1(skip1))
        h = self.mid(self.down2(skip2))
        h = self.up2(h)
        h = self.dec2(torch.cat([h, skip2], dim=1))
        h = self.up1(h)
        return self.dec1(torch.cat([h, skip1], dim=1))


class ConditionEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample_factor: int = 16) -> None:
        super().__init__()
        levels = int(math.log2(downsample_factor))
        if 2**levels != downsample_factor:
            raise ValueError("--downsample-factor must be a power of two")
        hidden = max(64, out_channels // 2)
        layers: list[nn.Module] = [nn.Conv2d(in_channels, hidden, 3, padding=1), nn.SiLU()]
        channels = hidden
        for _ in range(levels):
            next_channels = min(out_channels, channels * 2)
            layers += [nn.Conv2d(channels, next_channels, 4, stride=2, padding=1), nn.SiLU()]
            channels = next_channels
        layers += [nn.Conv2d(channels, out_channels, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        return self.net(condition)


@dataclass
class TokenPredictorConfig:
    codebook_size: int = 1024
    token_dim: int = 256
    model_dim: int = 384
    layers: int = 10
    downsample_factor: int = 16
    condition_channels: int = 4
    grid_size: int = 32
    stage: str = "rough"
    dropout: float = 0.05
    architecture: str = "resnet"


class MaskedTokenPredictor(nn.Module):
    def __init__(self, config: TokenPredictorConfig) -> None:
        super().__init__()
        self.config = config
        self.mask_token_id = config.codebook_size
        self.token_embed = nn.Embedding(config.codebook_size + 1, config.token_dim)
        self.rough_embed = nn.Embedding(config.codebook_size, config.token_dim)
        self.condition = ConditionEncoder(config.condition_channels, config.model_dim, config.downsample_factor)
        in_channels = config.model_dim + config.token_dim
        if config.stage == "refine":
            in_channels += config.token_dim
        self.input = nn.Conv2d(in_channels, config.model_dim, 1)
        self.position = nn.Parameter(torch.zeros(1, config.model_dim, config.grid_size, config.grid_size))
        if config.architecture == "unet":
            self.blocks = TokenUNetCore(config.model_dim, config.layers, config.dropout)
        elif config.architecture == "resnet":
            self.blocks = nn.Sequential(*[TokenResBlock(config.model_dim, config.dropout) for _ in range(config.layers)])
        else:
            raise ValueError(f"Unsupported token predictor architecture: {config.architecture}")
        self.out = nn.Sequential(
            nn.GroupNorm(8, config.model_dim),
            nn.SiLU(),
            nn.Conv2d(config.model_dim, config.codebook_size, 1),
        )

    def _embed_grid(self, embedding: nn.Embedding, tokens: torch.Tensor) -> torch.Tensor:
        return embedding(tokens.long()).permute(0, 3, 1, 2).contiguous()

    def forward(
        self,
        masked_tokens: torch.Tensor,
        condition: torch.Tensor,
        rough_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cond = self.condition(condition)
        token_features = self._embed_grid(self.token_embed, masked_tokens)
        features = [cond, token_features]
        if self.config.stage == "refine":
            if rough_tokens is None:
                raise ValueError("refine token predictor requires rough_tokens")
            features.append(self._embed_grid(self.rough_embed, rough_tokens.clamp_min(0).clamp_max(self.config.codebook_size - 1)))
        h = self.input(torch.cat(features, dim=1))
        if h.shape[-2:] == self.position.shape[-2:]:
            h = h + self.position
        else:
            h = h + F.interpolate(self.position, size=h.shape[-2:], mode="bilinear", align_corners=False)
        return self.out(self.blocks(h))


def build_token_predictor_from_state(state: dict, device: torch.device | str = "cpu") -> MaskedTokenPredictor:
    raw_config = dict(state["config"])
    raw_config.setdefault("architecture", "resnet")
    config = TokenPredictorConfig(**raw_config)
    model = MaskedTokenPredictor(config).to(device)
    model.load_state_dict(state["model"])
    return model
