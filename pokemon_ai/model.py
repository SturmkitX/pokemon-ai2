from __future__ import annotations

import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels, affine=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels, affine=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.SiLU(inplace=True),
            ResBlock(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.SiLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.SiLU(inplace=True),
            ResBlock(out_channels),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([x, skip], dim=1))


class StylizerUNet(nn.Module):
    def __init__(self, base_channels: int = 48) -> None:
        super().__init__()
        c = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, c, 7, padding=3),
            nn.InstanceNorm2d(c, affine=True),
            nn.SiLU(inplace=True),
        )
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)
        self.mid = nn.Sequential(ResBlock(c * 8), ResBlock(c * 8), ResBlock(c * 8))
        self.up3 = Up(c * 8, c * 4, c * 4)
        self.up2 = Up(c * 4, c * 2, c * 2)
        self.up1 = Up(c * 2, c, c)
        self.out = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, 3, 7, padding=3),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.stem(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        x = self.down3(s2)
        x = self.mid(x)
        x = self.up3(x, s2)
        x = self.up2(x, s1)
        x = self.up1(x, s0)
        return self.out(x)


class PatchDiscriminator(nn.Module):
    def __init__(self, base_channels: int = 48) -> None:
        super().__init__()
        c = base_channels

        def block(in_channels: int, out_channels: int, norm: bool = True) -> nn.Sequential:
            layers: list[nn.Module] = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)]
            if norm:
                layers.append(nn.InstanceNorm2d(out_channels, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.net = nn.Sequential(
            block(6, c, norm=False),
            block(c, c * 2),
            block(c * 2, c * 4),
            nn.Conv2d(c * 4, c * 8, 4, stride=1, padding=1),
            nn.InstanceNorm2d(c * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 8, 1, 4, stride=1, padding=1),
        )

    def forward(self, source: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([source, image], dim=1))
