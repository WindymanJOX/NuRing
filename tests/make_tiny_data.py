from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def disk_mask(h: int, w: int, cy: int, cx: int, r: int) -> np.ndarray:
    yy, xx = np.indices((h, w))
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/nuring_tiny")
    args = parser.parse_args()
    root = Path(args.out)
    for sub in ["registered", "nucleiseg", "cells"]:
        (root / sub).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(7)
    for i in range(2):
        h = w = 128
        image = rng.normal(0, 0.02, size=(3, h, w)).astype(np.float32)
        nuclei = np.zeros((h, w), dtype=np.int32)
        cells = np.zeros((h, w), dtype=np.int32)
        centers = [(40, 42), (82, 80), (50, 88)]
        for inst_id, (cy, cx) in enumerate(centers, start=1):
            nuc = disk_mask(h, w, cy, cx, 5)
            cell = disk_mask(h, w, cy, cx, 15)
            nuclei[nuc] = inst_id
            cells[cell] = inst_id
            image[0, nuc] += 1.0
            ring = cell & ~disk_mask(h, w, cy, cx, 8)
            image[1 + (inst_id % 2), ring] += 0.7
        np.save(root / "registered" / f"sample{i:02d}.npy", image)
        np.save(root / "nucleiseg" / f"sample{i:02d}.npy", nuclei)
        np.save(root / "cells" / f"sample{i:02d}.npy", cells)
    print(root)


if __name__ == "__main__":
    main()
