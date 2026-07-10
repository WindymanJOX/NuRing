from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from nuring.utils.morphology import binary_dilation, distance_transform_edt


def instance_ids(mask: np.ndarray) -> np.ndarray:
    ids = np.unique(mask)
    return ids[ids > 0]


def instance_centroid(mask: np.ndarray, inst_id: int) -> Tuple[float, float]:
    ys, xs = np.nonzero(mask == inst_id)
    if len(xs) == 0:
        raise ValueError(f"instance id {inst_id} not found")
    return float(xs.mean()), float(ys.mean())


def crop_with_pad(array: np.ndarray, center_xy: Tuple[float, float], crop_size: int, fill: float = 0) -> np.ndarray:
    """Crop around center from [H,W] or [C,H,W], padding outside image."""

    cx, cy = center_xy
    half = crop_size // 2
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    x1 = x0 + crop_size
    y1 = y0 + crop_size

    if array.ndim == 2:
        out = np.full((crop_size, crop_size), fill, dtype=array.dtype)
        h, w = array.shape
        src_y0, src_y1 = max(y0, 0), min(y1, h)
        src_x0, src_x1 = max(x0, 0), min(x1, w)
        dst_y0, dst_x0 = src_y0 - y0, src_x0 - x0
        out[dst_y0 : dst_y0 + src_y1 - src_y0, dst_x0 : dst_x0 + src_x1 - src_x0] = array[src_y0:src_y1, src_x0:src_x1]
        return out

    assert array.ndim == 3, f"expected [H,W] or [C,H,W], got {array.shape}"
    c, h, w = array.shape
    out = np.full((c, crop_size, crop_size), fill, dtype=array.dtype)
    src_y0, src_y1 = max(y0, 0), min(y1, h)
    src_x0, src_x1 = max(x0, 0), min(x1, w)
    dst_y0, dst_x0 = src_y0 - y0, src_x0 - x0
    out[:, dst_y0 : dst_y0 + src_y1 - src_y0, dst_x0 : dst_x0 + src_x1 - src_x0] = array[:, src_y0:src_y1, src_x0:src_x1]
    return out


def normalize_channels(image: np.ndarray) -> np.ndarray:
    """Robust per-channel normalization for image [C,H,W]."""

    image = image.astype(np.float32, copy=False)
    out = np.empty_like(image, dtype=np.float32)
    for c in range(image.shape[0]):
        x = image[c]
        lo, hi = np.percentile(x, (1, 99))
        if hi <= lo:
            out[c] = 0
        else:
            out[c] = np.clip((x - lo) / (hi - lo), 0, 1)
    return out


def build_feature_crop(image_crop: np.ndarray, nucleus_crop: np.ndarray, target_id: int, tissue_crop: Optional[np.ndarray] = None) -> np.ndarray:
    """Build Cartesian input channels before polar transform.

    Output shape is [Cin,H,W]: raw normalized channels, marker_sum, marker_max,
    target nucleus, neighbor nuclei, distance-to-target, distance-to-neighbor,
    and optional tissue mask.
    """

    assert image_crop.ndim == 3, f"image_crop must be [C,H,W], got {image_crop.shape}"
    image_crop = normalize_channels(image_crop)
    marker = image_crop[1:] if image_crop.shape[0] > 1 else image_crop
    marker_sum = marker.sum(axis=0, keepdims=True)
    if marker_sum.max() > 0:
        marker_sum = marker_sum / max(float(marker_sum.max()), 1e-6)
    marker_max = marker.max(axis=0, keepdims=True)

    target = (nucleus_crop == target_id).astype(np.float32)[None]
    all_nuclei = (nucleus_crop > 0).astype(np.float32)[None]
    neighbor = ((nucleus_crop > 0) & (nucleus_crop != target_id)).astype(np.float32)[None]

    dist_target = distance_transform_edt(1 - target[0]).astype(np.float32)
    dist_target = dist_target[None] / max(float(max(nucleus_crop.shape)), 1.0)
    if neighbor[0].sum() > 0:
        dist_neighbor = distance_transform_edt(1 - neighbor[0]).astype(np.float32)
        dist_neighbor = dist_neighbor[None] / max(float(max(nucleus_crop.shape)), 1.0)
    else:
        dist_neighbor = np.ones_like(target, dtype=np.float32)

    channels = [image_crop, marker_sum.astype(np.float32), marker_max.astype(np.float32), target, all_nuclei, neighbor, dist_target, dist_neighbor]
    if tissue_crop is not None:
        channels.append((tissue_crop > 0).astype(np.float32)[None])
    return np.concatenate(channels, axis=0).astype(np.float32)


def matched_cell_id(cell_mask: Optional[np.ndarray], target_nucleus: np.ndarray) -> int:
    if cell_mask is None:
        return 0
    vals, counts = np.unique(cell_mask[target_nucleus > 0], return_counts=True)
    keep = vals > 0
    if not np.any(keep):
        return 0
    vals, counts = vals[keep], counts[keep]
    return int(vals[np.argmax(counts)])


def pseudo_cell_mask(nucleus_crop: np.ndarray, target_id: int, radius: int = 10, mode: str = "dilation") -> np.ndarray:
    target = nucleus_crop == target_id
    if mode == "none":
        return target.astype(np.uint8)
    mask = binary_dilation(target, iterations=int(radius))
    neighbor = (nucleus_crop > 0) & (nucleus_crop != target_id)
    mask[neighbor] = False
    return mask.astype(np.uint8)


def radial_boundary_from_polar_mask(polar_mask: np.ndarray, max_radius: float) -> np.ndarray:
    """Get outer radius per angle from binary polar mask [R,A]."""

    assert polar_mask.ndim == 2, f"polar_mask must be [R,A], got {polar_mask.shape}"
    r_bins, angles = polar_mask.shape
    radius = np.zeros((angles,), dtype=np.float32)
    scale = float(max_radius) / max(r_bins - 1, 1)
    for a in range(angles):
        idx = np.flatnonzero(polar_mask[:, a] > 0.5)
        radius[a] = float(idx.max()) * scale if idx.size else 0.0
    return radius
