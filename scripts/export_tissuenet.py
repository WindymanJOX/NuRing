from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from tqdm import tqdm


def export_split(npz_path: Path, out_root: Path, split: str, limit: int | None = None) -> Path:
    data = np.load(npz_path, allow_pickle=True)
    x = data["X"]
    y = data["y"]
    n = x.shape[0] if limit is None else min(int(limit), x.shape[0])
    split_root = out_root / split
    image_dir = split_root / "images"
    nucleus_dir = split_root / "nuclei"
    cell_dir = split_root / "cells"
    for d in [image_dir, nucleus_dir, cell_dir]:
        d.mkdir(parents=True, exist_ok=True)
    sample_list = split_root / "samples.txt"
    names = []
    for i in tqdm(range(n), desc=f"export {split}"):
        name = f"{split}_{i:06d}"
        img = x[i].astype(np.float32)
        if img.ndim == 3 and img.shape[-1] <= 16:
            img = np.moveaxis(img, -1, 0)
        # TissueNet y[...,0] is whole-cell, y[...,1] is nuclear.
        cell = y[i, ..., 0].astype(np.int32)
        nucleus = y[i, ..., 1].astype(np.int32)
        np.save(image_dir / f"{name}.npy", img)
        np.save(nucleus_dir / f"{name}.npy", nucleus)
        np.save(cell_dir / f"{name}.npy", cell)
        names.append(name)
    sample_list.write_text("\n".join(names) + "\n", encoding="utf-8")
    return split_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tissuenet-dir", default="/data/wzx/tissuenet_v1.1")
    parser.add_argument("--out-dir", default="/data/wzx/tissuenet_v1.1/nuring_export")
    parser.add_argument("--train-limit", type=int, default=128)
    parser.add_argument("--val-limit", type=int, default=32)
    parser.add_argument("--test-limit", type=int, default=32)
    args = parser.parse_args()

    root = Path(args.tissuenet_dir)
    out = Path(args.out_dir)
    export_split(root / "tissuenet_v1.1_train.npz", out, "train", args.train_limit)
    export_split(root / "tissuenet_v1.1_val.npz", out, "val", args.val_limit)
    export_split(root / "tissuenet_v1.1_test.npz", out, "test", args.test_limit)
    print(out)


if __name__ == "__main__":
    main()
