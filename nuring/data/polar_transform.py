from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def sample_polar_grid(
    batch_size: int,
    num_radial_bins: int,
    num_angles: int,
    max_radius: float,
    image_size: Tuple[int, int],
    center: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a grid_sample grid for polar sampling.

    Returns shape [B, R, A, 2], where the last dimension stores normalized
    Cartesian coordinates [x, y] in [-1, 1].
    """

    h, w = image_size
    device = device or torch.device("cpu")
    if center is None:
        center = torch.tensor([(w - 1) * 0.5, (h - 1) * 0.5], device=device, dtype=dtype)
        center = center.view(1, 2).repeat(batch_size, 1)
    else:
        center = center.to(device=device, dtype=dtype)
        if center.ndim == 1:
            center = center.view(1, 2).repeat(batch_size, 1)
    assert center.shape == (batch_size, 2), f"center must be [B,2], got {center.shape}"

    radii = torch.linspace(0.0, float(max_radius), num_radial_bins, device=device, dtype=dtype)
    angles = torch.linspace(0.0, 2.0 * math.pi, num_angles + 1, device=device, dtype=dtype)[:-1]
    rr, aa = torch.meshgrid(radii, angles, indexing="ij")
    x = center[:, 0].view(batch_size, 1, 1) + rr.view(1, num_radial_bins, num_angles) * torch.cos(aa).view(1, num_radial_bins, num_angles)
    y = center[:, 1].view(batch_size, 1, 1) + rr.view(1, num_radial_bins, num_angles) * torch.sin(aa).view(1, num_radial_bins, num_angles)

    x_norm = 2.0 * x / max(w - 1, 1) - 1.0
    y_norm = 2.0 * y / max(h - 1, 1) - 1.0
    return torch.stack([x_norm, y_norm], dim=-1)


def cartesian_to_polar_tensor(
    image: torch.Tensor,
    num_radial_bins: int,
    num_angles: int,
    max_radius: float,
    center: Optional[torch.Tensor] = None,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
) -> torch.Tensor:
    """Sample Cartesian image/crop into polar space.

    Args:
        image: Tensor [B, C, H, W].
    Returns:
        Tensor [B, C, R, A].
    """

    assert image.ndim == 4, f"image must be [B,C,H,W], got {image.shape}"
    b, _, h, w = image.shape
    grid = sample_polar_grid(
        b,
        num_radial_bins,
        num_angles,
        max_radius,
        (h, w),
        center=center,
        device=image.device,
        dtype=image.dtype,
    )
    return F.grid_sample(image, grid, mode=mode, padding_mode=padding_mode, align_corners=True)


def polar_to_cartesian_mask(
    polar: torch.Tensor,
    crop_size: int,
    max_radius: float,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Warp a polar probability map back to a centered Cartesian crop.

    Args:
        polar: Tensor [B, C, R, A].
    Returns:
        Tensor [B, C, crop_size, crop_size].
    """

    assert polar.ndim == 4, f"polar must be [B,C,R,A], got {polar.shape}"
    b, _, r_bins, a_bins = polar.shape
    y, x = torch.meshgrid(
        torch.arange(crop_size, device=polar.device, dtype=polar.dtype),
        torch.arange(crop_size, device=polar.device, dtype=polar.dtype),
        indexing="ij",
    )
    cx = cy = (crop_size - 1) * 0.5
    dx = x - cx
    dy = y - cy
    radius = torch.sqrt(dx * dx + dy * dy)
    angle = torch.remainder(torch.atan2(dy, dx), 2.0 * math.pi)

    r_norm = 2.0 * (radius / max(float(max_radius), 1e-6)) - 1.0
    a_norm = 2.0 * (angle / (2.0 * math.pi)) - 1.0
    grid = torch.stack([a_norm, r_norm], dim=-1).view(1, crop_size, crop_size, 2).repeat(b, 1, 1, 1)
    cart = F.grid_sample(polar, grid, mode=mode, padding_mode="zeros", align_corners=True)
    inside = (radius <= float(max_radius)).view(1, 1, crop_size, crop_size)
    return cart * inside.to(dtype=cart.dtype)


def radius_to_polar_mask(radius: torch.Tensor, num_radial_bins: int, max_radius: float) -> torch.Tensor:
    """Convert angle-wise radius [B,A] to a filled polar mask [B,1,R,A]."""

    assert radius.ndim == 2, f"radius must be [B,A], got {radius.shape}"
    b, a = radius.shape
    radii = torch.linspace(0.0, float(max_radius), num_radial_bins, device=radius.device, dtype=radius.dtype)
    mask = radii.view(1, num_radial_bins, 1) <= radius.view(b, 1, a)
    return mask.unsqueeze(1).to(dtype=radius.dtype)
