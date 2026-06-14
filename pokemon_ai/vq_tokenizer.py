from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class Encoder(nn.Module):
    def __init__(self, embedding_dim: int, downsample_factor: int = 16, base_channels: int = 96) -> None:
        super().__init__()
        levels = int(math.log2(downsample_factor))
        if 2**levels != downsample_factor:
            raise ValueError("--downsample-factor must be a power of two")

        layers: list[nn.Module] = [nn.Conv2d(3, base_channels, 3, padding=1)]
        channels = base_channels
        for level in range(levels):
            next_channels = min(base_channels * 2 ** (level + 1), 384)
            layers += [
                ResBlock(channels),
                nn.Conv2d(channels, next_channels, 4, stride=2, padding=1),
            ]
            channels = next_channels
        layers += [ResBlock(channels), ResBlock(channels), nn.GroupNorm(8, channels), nn.SiLU(), nn.Conv2d(channels, embedding_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, embedding_dim: int, downsample_factor: int = 16, base_channels: int = 96) -> None:
        super().__init__()
        levels = int(math.log2(downsample_factor))
        channels = min(base_channels * 2**levels, 384)
        layers: list[nn.Module] = [nn.Conv2d(embedding_dim, channels, 3, padding=1), ResBlock(channels), ResBlock(channels)]
        for level in reversed(range(levels)):
            next_channels = min(base_channels * 2**level, 384)
            layers += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(channels, next_channels, 3, padding=1),
                ResBlock(next_channels),
            ]
            channels = next_channels
        layers += [nn.GroupNorm(8, channels), nn.SiLU(), nn.Conv2d(channels, 3, 3, padding=1), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int = 1024, embedding_dim: int = 256, commitment_cost: float = 0.25) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(codebook_size, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_hw = z.permute(0, 2, 3, 1).contiguous()
        z_lookup = z_hw.float()
        flat = z_lookup.view(-1, self.embedding_dim)
        codebook = self.embedding.weight.float()
        distances = flat.square().sum(dim=1, keepdim=True) - 2 * flat @ codebook.t() + codebook.square().sum(dim=1)
        indices = distances.argmin(dim=1)
        quantized = self.embedding(indices).view_as(z_lookup)
        codebook_loss = F.mse_loss(quantized, z_lookup.detach())
        commit_loss = F.mse_loss(z_lookup, quantized.detach())
        vq_loss = codebook_loss + self.commitment_cost * commit_loss
        quantized = z_lookup + (quantized - z_lookup).detach()
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        indices = indices.view(z.shape[0], z.shape[2], z.shape[3])
        return quantized, indices, vq_loss

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        z = self.embedding(indices.long())
        return z.permute(0, 3, 1, 2).contiguous()


@dataclass
class VQTokenizerConfig:
    codebook_size: int = 1024
    embedding_dim: int = 256
    downsample_factor: int = 16
    base_channels: int = 96
    commitment_cost: float = 0.25


class VQTokenizer(nn.Module):
    def __init__(self, config: VQTokenizerConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = Encoder(config.embedding_dim, config.downsample_factor, config.base_channels)
        self.quantizer = VectorQuantizer(config.codebook_size, config.embedding_dim, config.commitment_cost)
        self.decoder = Decoder(config.embedding_dim, config.downsample_factor, config.base_channels)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(image)
        z_q, indices, vq_loss = self.quantizer(z)
        recon = self.decoder(z_q)
        return {"recon": recon, "indices": indices, "vq_loss": vq_loss}

    @torch.no_grad()
    def encode_indices(self, image: torch.Tensor) -> torch.Tensor:
        z = self.encoder(image)
        _, indices, _ = self.quantizer(z)
        return indices

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.quantizer.indices_to_codes(indices))


def build_vq_tokenizer_from_state(state: dict, device: torch.device | str = "cpu") -> VQTokenizer:
    config = VQTokenizerConfig(**state["config"])
    model = VQTokenizer(config).to(device)
    model.load_state_dict(state["model"])
    return model
