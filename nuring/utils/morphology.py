from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    from scipy import ndimage as _ndi
except Exception:
    _ndi = None


def binary_dilation(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    mask = mask.astype(bool)
    if _ndi is not None:
        return _ndi.binary_dilation(mask, iterations=int(iterations))
    out = mask.copy()
    for _ in range(int(iterations)):
        p = np.pad(out, 1, mode="constant")
        out = (
            p[1:-1, 1:-1]
            | p[:-2, 1:-1]
            | p[2:, 1:-1]
            | p[1:-1, :-2]
            | p[1:-1, 2:]
            | p[:-2, :-2]
            | p[:-2, 2:]
            | p[2:, :-2]
            | p[2:, 2:]
        )
    return out


def binary_erosion(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    mask = mask.astype(bool)
    if _ndi is not None:
        return _ndi.binary_erosion(mask, iterations=int(iterations))
    out = mask.copy()
    for _ in range(int(iterations)):
        p = np.pad(out, 1, mode="constant")
        out = (
            p[1:-1, 1:-1]
            & p[:-2, 1:-1]
            & p[2:, 1:-1]
            & p[1:-1, :-2]
            & p[1:-1, 2:]
            & p[:-2, :-2]
            & p[:-2, 2:]
            & p[2:, :-2]
            & p[2:, 2:]
        )
    return out


def distance_transform_edt(mask: np.ndarray, return_indices: bool = False):
    """Distance to the nearest zero pixel, matching scipy.ndimage semantics.

    The fallback is chunked and intended for small crops or smoke tests.
    """

    mask = mask.astype(bool)
    if _ndi is not None:
        return _ndi.distance_transform_edt(mask, return_indices=return_indices)

    h, w = mask.shape
    zeros = np.column_stack(np.nonzero(~mask))
    if zeros.size == 0:
        dist = np.full((h, w), np.inf, dtype=np.float32)
        indices = np.zeros((2, h, w), dtype=np.int64)
        return (dist, indices) if return_indices else dist
    pts = np.indices((h, w)).reshape(2, -1).T
    dist_flat = np.empty((h * w,), dtype=np.float32)
    nearest_flat = np.empty((h * w, 2), dtype=np.int64)
    for start in range(0, h * w, 8192):
        chunk = pts[start : start + 8192]
        d2 = ((chunk[:, None, :] - zeros[None, :, :]) ** 2).sum(axis=2)
        arg = np.argmin(d2, axis=1)
        dist_flat[start : start + len(chunk)] = np.sqrt(d2[np.arange(len(chunk)), arg])
        nearest_flat[start : start + len(chunk)] = zeros[arg]
    dist = dist_flat.reshape(h, w)
    indices = nearest_flat.T.reshape(2, h, w)
    return (dist, indices) if return_indices else dist
