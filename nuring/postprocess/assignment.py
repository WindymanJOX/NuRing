from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np


def paste_crop_slices(center_xy: Tuple[float, float], crop_size: int, image_shape: Tuple[int, int]):
    h, w = image_shape
    cx, cy = center_xy
    half = crop_size // 2
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    x1, y1 = x0 + crop_size, y0 + crop_size
    src_x0, src_y0 = max(0, -x0), max(0, -y0)
    dst_x0, dst_y0 = max(0, x0), max(0, y0)
    width = min(x1, w) - dst_x0
    height = min(y1, h) - dst_y0
    if width <= 0 or height <= 0:
        return None
    return (slice(src_y0, src_y0 + height), slice(src_x0, src_x0 + width)), (slice(dst_y0, dst_y0 + height), slice(dst_x0, dst_x0 + width))


def assign_instances(
    prob_crops: Sequence[np.ndarray],
    instance_ids: Sequence[int],
    centers_xy: Sequence[Tuple[float, float]],
    image_shape: Tuple[int, int],
    crop_size: int,
    threshold: float = 0.5,
    distance_weight: float = 0.01,
) -> np.ndarray:
    """Resolve overlapping per-cell probability crops into one instance mask.

    score_i(p) = P_i(p) - distance_weight * distance_to_centroid_i(p).
    """

    label = np.zeros(image_shape, dtype=np.int32)
    best = np.full(image_shape, -np.inf, dtype=np.float32)
    yy_full, xx_full = np.indices(image_shape)
    for prob, inst_id, center in zip(prob_crops, instance_ids, centers_xy):
        prob = np.asarray(prob, dtype=np.float32)
        if prob.ndim == 3:
            prob = prob[0]
        slices = paste_crop_slices(center, crop_size, image_shape)
        if slices is None:
            continue
        src, dst = slices
        local_prob = prob[src]
        cx, cy = center
        dist = np.sqrt((xx_full[dst] - cx) ** 2 + (yy_full[dst] - cy) ** 2)
        score = local_prob - float(distance_weight) * dist
        take = (local_prob >= threshold) & (score > best[dst])
        dst_y, dst_x = dst
        sub_best = best[dst]
        sub_label = label[dst]
        sub_best[take] = score[take]
        sub_label[take] = int(inst_id)
        best[dst] = sub_best
        label[dst] = sub_label
    return label
