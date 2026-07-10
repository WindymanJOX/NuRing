from __future__ import annotations

import numpy as np

from nuring.utils.morphology import distance_transform_edt


def voronoi_baseline(nucleus_mask: np.ndarray, max_radius: int = 32) -> np.ndarray:
    """Assign pixels to the nearest nucleus pixel with a max-radius cutoff."""

    labels = nucleus_mask.astype(np.int32)
    foreground = labels > 0
    if not np.any(foreground):
        return labels
    dist, indices = distance_transform_edt(~foreground, return_indices=True)
    nearest = labels[indices[0], indices[1]]
    out = np.where(dist <= float(max_radius), nearest, 0).astype(np.int32)
    out[foreground] = labels[foreground]
    return out
