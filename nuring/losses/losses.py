from __future__ import annotations

from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    inter = (probs * target).sum(dim=dims)
    denom = probs.sum(dim=dims) + target.sum(dim=dims)
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


def weighted_smooth_l1(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    return (loss * weight).sum() / weight.sum().clamp_min(1e-6)


def angle_smoothness_loss(radius_norm: torch.Tensor) -> torch.Tensor:
    diff = torch.abs(radius_norm - torch.roll(radius_norm, shifts=1, dims=-1))
    return diff.mean()


class NuRingLoss(nn.Module):
    """Combined first-stage NuRing loss in polar space."""

    def __init__(
        self,
        lambda_mask: float = 1.0,
        lambda_radius: float = 0.05,
        lambda_conf: float = 0.1,
        lambda_smooth: float = 0.005,
        lambda_contain: float = 0.05,
        lambda_neighbor: float = 0.05,
        max_radius: float = 64.0,
    ) -> None:
        super().__init__()
        self.lambda_mask = lambda_mask
        self.lambda_radius = lambda_radius
        self.lambda_conf = lambda_conf
        self.lambda_smooth = lambda_smooth
        self.lambda_contain = lambda_contain
        self.lambda_neighbor = lambda_neighbor
        self.max_radius = float(max_radius)

    def forward(self, pred: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        mask_logits = pred["mask_logits"]
        radius = pred["radius"]
        confidence = pred["confidence"]
        target_mask = batch["y_mask_polar"]
        y_radius = batch["y_radius"]
        y_conf = batch["y_conf"]
        target_nucleus = batch["target_nucleus_polar"]
        neighbor = batch["neighbor_nuclei_polar"]

        assert mask_logits.shape == target_mask.shape, f"mask shape mismatch {mask_logits.shape} vs {target_mask.shape}"
        assert radius.shape == y_radius.shape, f"radius shape mismatch {radius.shape} vs {y_radius.shape}"
        assert confidence.shape == y_conf.shape, f"confidence shape mismatch {confidence.shape} vs {y_conf.shape}"

        bce = F.binary_cross_entropy_with_logits(mask_logits, target_mask)
        dsc = dice_loss(mask_logits, target_mask)
        mask_loss = bce + dsc
        radius_norm = radius / max(self.max_radius, 1e-6)
        y_radius_norm = y_radius / max(self.max_radius, 1e-6)
        radius_loss = weighted_smooth_l1(radius_norm, y_radius_norm, y_conf)
        conf_loss = F.binary_cross_entropy(confidence.clamp(1e-5, 1 - 1e-5), y_conf)
        smooth_loss = angle_smoothness_loss(radius_norm)
        probs = torch.sigmoid(mask_logits)
        contain_loss = F.binary_cross_entropy(probs.clamp(1e-5, 1 - 1e-5), torch.ones_like(probs), reduction="none")
        contain_loss = (contain_loss * target_nucleus).sum() / target_nucleus.sum().clamp_min(1.0)
        neighbor_loss = F.binary_cross_entropy(probs.clamp(1e-5, 1 - 1e-5), torch.zeros_like(probs), reduction="none")
        neighbor_loss = (neighbor_loss * neighbor).sum() / neighbor.sum().clamp_min(1.0)

        total = (
            self.lambda_mask * mask_loss
            + self.lambda_radius * radius_loss
            + self.lambda_conf * conf_loss
            + self.lambda_smooth * smooth_loss
            + self.lambda_contain * contain_loss
            + self.lambda_neighbor * neighbor_loss
        )
        return {
            "loss": total,
            "mask": mask_loss.detach(),
            "radius": radius_loss.detach(),
            "conf": conf_loss.detach(),
            "smooth": smooth_loss.detach(),
            "contain": contain_loss.detach(),
            "neighbor": neighbor_loss.detach(),
        }
