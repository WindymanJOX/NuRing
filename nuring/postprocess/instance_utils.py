from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def relabel_sequential(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask, dtype=np.int32)
    for new_id, old_id in enumerate([i for i in np.unique(mask) if i > 0], start=1):
        out[mask == old_id] = new_id
    return out


def centroids(mask: np.ndarray) -> Dict[int, Tuple[float, float]]:
    result: Dict[int, Tuple[float, float]] = {}
    for inst_id in np.unique(mask):
        if inst_id == 0:
            continue
        ys, xs = np.nonzero(mask == inst_id)
        if len(xs):
            result[int(inst_id)] = (float(xs.mean()), float(ys.mean()))
    return result
