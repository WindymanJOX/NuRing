from __future__ import annotations

import torch
from torch import nn

from .blocks import ConvBlock, DownBlock, UpBlock


class PolarRingNet(nn.Module):
    """Lightweight polar U-Net with radial, confidence, and mask heads.

    Input:
        x: [B, Cin, R, A]
    Outputs:
        radius: [B, A] in pixel radius units
        confidence: [B, A] in [0,1]
        mask_logits: [B, 1, R, A]
    """

    def __init__(self, in_channels: int, base_channels: int = 32, num_radial_bins: int = 64, max_radius: float = 64.0) -> None:
        super().__init__()
        self.num_radial_bins = int(num_radial_bins)
        self.max_radius = float(max_radius)

        c = base_channels
        self.enc1 = ConvBlock(in_channels, c)
        self.enc2 = DownBlock(c, c * 2)
        self.enc3 = DownBlock(c * 2, c * 4)
        self.bottleneck = DownBlock(c * 4, c * 8)
        self.up3 = UpBlock(c * 8, c * 4, c * 4)
        self.up2 = UpBlock(c * 4, c * 2, c * 2)
        self.up1 = UpBlock(c * 2, c, c)

        self.mask_head = nn.Conv2d(c, 1, 1)
        self.radius_logits_head = nn.Conv2d(c, 1, 1)
        self.conf_head = nn.Sequential(
            nn.Conv1d(c, c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(c, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        assert x.ndim == 4, f"x must be [B,C,R,A], got {x.shape}"
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bottleneck(e3)
        d3 = self.up3(b, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        mask_logits = self.mask_head(d1)
        radial_logits = self.radius_logits_head(d1).squeeze(1)
        assert radial_logits.ndim == 3, f"radial logits must be [B,R,A], got {radial_logits.shape}"
        probs = torch.softmax(radial_logits, dim=1)
        radii = torch.linspace(0.0, self.max_radius, radial_logits.shape[1], device=x.device, dtype=x.dtype)
        radius = (probs * radii.view(1, -1, 1)).sum(dim=1)

        angle_features = d1.mean(dim=2)
        confidence = torch.sigmoid(self.conf_head(angle_features).squeeze(1))
        return {"radius": radius, "confidence": confidence, "mask_logits": mask_logits, "radial_logits": radial_logits}
