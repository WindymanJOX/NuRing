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


def select_feature_channels(features: np.ndarray, raw_channels: int, mode: str = "all") -> np.ndarray:
    """Select input channels for input ablations.

    Feature layout: raw channels, marker_sum, marker_max, target, all_nuclei,
    neighbor, distance_target, distance_neighbor, optional tissue.
    """

    mode = mode or "all"
    marker_sum = raw_channels
    marker_max = raw_channels + 1
    target = raw_channels + 2
    all_nuclei = raw_channels + 3
    neighbor = raw_channels + 4
    dist_target = raw_channels + 5
    dist_neighbor = raw_channels + 6
    tissue = raw_channels + 7
    base_geom = [target, all_nuclei, dist_target]
    if mode == "dapi_only":
        idx = [0, *base_geom]
    elif mode == "dapi_marker_sum":
        idx = [0, marker_sum, *base_geom]
    elif mode == "dapi_marker_max":
        idx = [0, marker_max, *base_geom]
    elif mode == "all_markers":
        idx = [*range(raw_channels), marker_sum, marker_max, *base_geom]
    elif mode == "all_plus_neighbor":
        idx = [*range(raw_channels), marker_sum, marker_max, target, all_nuclei, neighbor, dist_target, dist_neighbor]
    elif mode == "all":
        return features
    else:
        raise ValueError(f"unknown input mode: {mode}")
    if tissue < features.shape[0]:
        idx.append(tissue)
    idx = [i for i in idx if i < features.shape[0]]
    return features[idx].astype(np.float32)


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


def _median_filter_1d_circular(values: np.ndarray, size: int) -> np.ndarray:
    if size <= 1:
        return values.astype(np.float32)
    if size % 2 == 0:
        size += 1
    pad = size // 2
    padded = np.concatenate([values[-pad:], values, values[:pad]])
    out = np.empty_like(values, dtype=np.float32)
    for i in range(values.size):
        out[i] = np.median(padded[i : i + size])
    return out


def radial_boundary_from_polar_mask(
    polar_mask: np.ndarray,
    max_radius: float,
    gap_tolerance: int = 2,
    median_filter_size: int = 5,
    min_radius: float = 3.0,
) -> np.ndarray:
    """Estimate robust connected radial boundary from binary polar mask [R,A].

    The scan follows the central connected positive segment and ignores far
    isolated positives. Small radial gaps are tolerated before the segment is
    considered ended.
    """

    assert polar_mask.ndim == 2, f"polar_mask must be [R,A], got {polar_mask.shape}"
    r_bins, angles = polar_mask.shape
    radius = np.zeros((angles,), dtype=np.float32)
    scale = float(max_radius) / max(r_bins - 1, 1)
    min_bin = int(round(float(min_radius) / max(scale, 1e-6)))
    for a in range(angles):
        col = polar_mask[:, a] > 0.5
        positives = np.flatnonzero(col)
        if positives.size == 0 or positives[0] > max(min_bin, 1):
            radius[a] = 0.0
            continue
        end = int(positives[0])
        gaps = 0
        seen = False
        for r in range(r_bins):
            if col[r]:
                seen = True
                end = r
                gaps = 0
            elif seen:
                gaps += 1
                if gaps > int(gap_tolerance):
                    break
        radius[a] = float(end) * scale if end >= min_bin else 0.0
    valid = radius > 0
    if valid.any() and median_filter_size > 1:
        filtered = _median_filter_1d_circular(radius, median_filter_size)
        radius[valid] = filtered[valid]
    return radius.astype(np.float32)


def confidence_from_marker_ring(
    marker_polar: np.ndarray,
    cell_polar: np.ndarray,
    nucleus_polar: np.ndarray,
    cell_radius: np.ndarray,
    max_radius: float,
    marker_quantile: float = 75.0,
    margin_bins: int = 1,
) -> np.ndarray:
    """Build angle-wise confidence labels from marker evidence in cytoplasm ring.

    Returns [A] with 1.0 for strong marker evidence, 0.5 for label-supported
    weak-marker angles, and 0.0 for unlabeled/background angles.
    """

    assert marker_polar.ndim == 2, f"marker_polar must be [R,A], got {marker_polar.shape}"
    assert cell_polar.shape == marker_polar.shape
    assert nucleus_polar.shape == marker_polar.shape
    r_bins, angles = marker_polar.shape
    radii = np.linspace(0.0, float(max_radius), r_bins, dtype=np.float32)
    cell_radius = np.asarray(cell_radius, dtype=np.float32)
    positives = marker_polar[cell_polar > 0.5]
    threshold = np.percentile(positives, marker_quantile) if positives.size else np.percentile(marker_polar, marker_quantile)
    conf = np.zeros((angles,), dtype=np.float32)
    for a in range(angles):
        if cell_radius[a] <= 0:
            continue
        nuc_idx = np.flatnonzero(nucleus_polar[:, a] > 0.5)
        nuc_radius = radii[nuc_idx.max()] if nuc_idx.size else 0.0
        ring = (radii > nuc_radius + margin_bins * (radii[1] - radii[0] if r_bins > 1 else 1.0)) & (radii <= cell_radius[a]) & (cell_polar[:, a] > 0.5)
        if not np.any(ring):
            conf[a] = 0.5
            continue
        signal = float(marker_polar[:, a][ring].mean())
        conf[a] = 1.0 if signal >= threshold else 0.5
    return conf
