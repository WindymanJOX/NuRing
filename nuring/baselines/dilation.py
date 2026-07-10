from __future__ import annotations

import numpy as np

from nuring.utils.morphology import binary_dilation, distance_transform_edt


def dilation_baseline(nucleus_mask: np.ndarray, radius: int = 10) -> np.ndarray:
    """Dilate each nucleus instance and resolve overlaps by nearest nucleus.

    Args:
        nucleus_mask: [H,W] integer instance mask.
    Returns:
        [H,W] whole-cell instance mask.
    """

    out = np.zeros_like(nucleus_mask, dtype=np.int32)
    scores = np.full(nucleus_mask.shape, np.inf, dtype=np.float32)
    for inst_id in np.unique(nucleus_mask):
        if inst_id == 0:
            continue
        nuc = nucleus_mask == inst_id
        dil = binary_dilation(nuc, iterations=int(radius))
        dist = distance_transform_edt(~nuc)
        take = dil & (dist < scores)
        out[take] = int(inst_id)
        scores[take] = dist[take]
    out[nucleus_mask > 0] = nucleus_mask[nucleus_mask > 0]
    return out
