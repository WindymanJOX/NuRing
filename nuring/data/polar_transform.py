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


def radius_to_soft_polar_mask(
    radius: torch.Tensor,
    num_radial_bins: int,
    max_radius: float,
    sigma: float = 2.0,
) -> torch.Tensor:
    """Convert predicted radius [B,A] into a soft filled polar mask [B,1,R,A].

    Values are near 1 inside the boundary and near 0 outside. The transition
    width is controlled by sigma in pixel-radius units.
    """

    assert radius.ndim == 2, f"radius must be [B,A], got {radius.shape}"
    radii = torch.linspace(0.0, float(max_radius), num_radial_bins, device=radius.device, dtype=radius.dtype)
    dist = radii.view(1, num_radial_bins, 1) - radius.view(radius.shape[0], 1, radius.shape[1])
    soft = torch.sigmoid(-dist / max(float(sigma), 1e-6))
    return soft.unsqueeze(1)


def fuse_polar_prediction(
    mask_prob: torch.Tensor,
    radius: torch.Tensor,
    confidence: torch.Tensor,
    num_radial_bins: int,
    max_radius: float,
    mode: str = "confidence_fusion",
    radius_sigma: float = 2.0,
    radius_prior_weight: float = 0.5,
) -> torch.Tensor:
    """Fuse mask and radius/confidence branches during inference.

    mode:
        mask_only: use mask branch only.
        mask_radius_average: weighted average of mask branch and radius prior.
        confidence_fusion: angle-wise confidence chooses radius prior vs mask.
    """

    assert mask_prob.ndim == 4, f"mask_prob must be [B,1,R,A], got {mask_prob.shape}"
    assert radius.ndim == 2, f"radius must be [B,A], got {radius.shape}"
    assert confidence.shape == radius.shape, f"confidence shape {confidence.shape} must match radius {radius.shape}"
    if mode == "mask_only":
        return mask_prob
    radius_prior = radius_to_soft_polar_mask(radius, num_radial_bins, max_radius, sigma=radius_sigma)
    assert radius_prior.shape == mask_prob.shape, f"radius prior {radius_prior.shape} vs mask {mask_prob.shape}"
    if mode == "mask_radius_average":
        w = float(radius_prior_weight)
        return (1.0 - w) * mask_prob + w * radius_prior
    if mode == "confidence_fusion":
        conf = confidence.clamp(0, 1).unsqueeze(1).unsqueeze(2)
        return conf * radius_prior + (1.0 - conf) * mask_prob
    raise ValueError(f"unknown fusion mode: {mode}")
