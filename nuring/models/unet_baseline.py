from __future__ import annotations

import torch
from torch import nn

from .blocks import ConvBlock, DownBlock, UpBlock


class UNetBaseline(nn.Module):
    """Small Cartesian U-Net baseline returning [B,1,H,W] logits."""

    def __init__(self, in_channels: int, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels
        self.enc1 = ConvBlock(in_channels, c)
        self.enc2 = DownBlock(c, c * 2)
        self.enc3 = DownBlock(c * 2, c * 4)
        self.up2 = UpBlock(c * 4, c * 2, c * 2)
        self.up1 = UpBlock(c * 2, c, c)
        self.head = nn.Conv2d(c, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        d2 = self.up2(e3, e2)
        d1 = self.up1(d2, e1)
        return self.head(d1)
