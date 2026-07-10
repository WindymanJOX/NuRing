from __future__ import annotations

import numpy as np
import sys
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nuring.baselines import dilation_baseline, voronoi_baseline
from nuring.data.label_utils import radial_boundary_from_polar_mask
from nuring.data.polar_transform import cartesian_to_polar_tensor, fuse_polar_prediction, polar_to_cartesian_mask, radius_to_soft_polar_mask
from nuring.losses import NuRingLoss
from nuring.models import PolarRingNet


def main() -> None:
    x = torch.rand(2, 8, 128, 128)
    polar = cartesian_to_polar_tensor(x, 64, 128, 64)
    assert polar.shape == (2, 8, 64, 128), polar.shape

    model = PolarRingNet(8, base_channels=8, num_radial_bins=64, max_radius=64)
    out = model(polar)
    assert out["radius"].shape == (2, 128)
    assert out["confidence"].shape == (2, 128)
    assert out["mask_logits"].shape == (2, 1, 64, 128)

    cart = polar_to_cartesian_mask(torch.sigmoid(out["mask_logits"]), 128, 64)
    assert cart.shape == (2, 1, 128, 128)
    radius_prior = radius_to_soft_polar_mask(out["radius"], 64, 64)
    assert radius_prior.shape == (2, 1, 64, 128)
    assert torch.all(radius_prior[:, :, 0] >= radius_prior[:, :, -1])
    fused = fuse_polar_prediction(torch.sigmoid(out["mask_logits"]), out["radius"], out["confidence"], 64, 64)
    assert fused.shape == (2, 1, 64, 128)

    batch = {
        "y_mask_polar": torch.rand(2, 1, 64, 128).round(),
        "y_radius": torch.rand(2, 128) * 64,
        "y_conf": torch.ones(2, 128),
        "target_nucleus_polar": torch.zeros(2, 1, 64, 128),
        "neighbor_nuclei_polar": torch.zeros(2, 1, 64, 128),
    }
    batch["target_nucleus_polar"][:, :, 0:5, :] = 1
    losses = NuRingLoss()(out, batch)
    assert torch.isfinite(losses["loss"]), losses
    assert "conf" in losses

    polar_gt = np.zeros((16, 8), np.float32)
    polar_gt[:8, :] = 1
    polar_gt[14, 3] = 1
    radius = radial_boundary_from_polar_mask(polar_gt, max_radius=16, gap_tolerance=1, median_filter_size=3, min_radius=1)
    assert radius[3] < 12, radius

    nuclei = np.zeros((64, 64), np.int32)
    nuclei[20:25, 20:25] = 1
    nuclei[40:45, 40:45] = 2
    assert dilation_baseline(nuclei, 5).shape == nuclei.shape
    assert voronoi_baseline(nuclei, 12).shape == nuclei.shape

    print(
        "smoke ok",
        tuple(polar.shape),
        round(float(out["radius"].detach().min()), 3),
        round(float(out["radius"].detach().max()), 3),
        round(float(losses["loss"].detach()), 3),
    )


if __name__ == "__main__":
    main()
