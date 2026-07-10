from __future__ import annotations

import numpy as np


def mean_absolute_radial_error(pred_radius: np.ndarray, gt_radius: np.ndarray, conf: np.ndarray | None = None) -> float:
    err = np.abs(np.asarray(pred_radius) - np.asarray(gt_radius))
    if conf is not None:
        w = np.asarray(conf).astype(np.float32)
        return float((err * w).sum() / max(w.sum(), 1e-6))
    return float(err.mean())
