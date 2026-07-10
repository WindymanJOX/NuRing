from __future__ import annotations

import numpy as np

from nuring.utils.morphology import binary_dilation, distance_transform_edt


def watershed_baseline(nucleus_mask: np.ndarray, marker_image: np.ndarray | None = None, max_radius: int = 32) -> np.ndarray:
    """Watershed baseline using nucleus labels as seeds.

    If scikit-image is unavailable, falls back to Voronoi assignment.
    """

    try:
        from skimage.segmentation import watershed
    except Exception:
        from .voronoi import voronoi_baseline

        return voronoi_baseline(nucleus_mask, max_radius=max_radius)

    seeds = nucleus_mask.astype(np.int32)
    if marker_image is None:
        distance = distance_transform_edt(seeds == 0)
        elevation = -distance
    else:
        marker_image = marker_image.astype(np.float32)
        lo, hi = np.percentile(marker_image, (1, 99))
        elevation = 1.0 - np.clip((marker_image - lo) / max(hi - lo, 1e-6), 0, 1)
    mask = binary_dilation(seeds > 0, iterations=int(max_radius))
    out = watershed(elevation, markers=seeds, mask=mask)
    out[seeds > 0] = seeds[seeds > 0]
    return out.astype(np.int32)
