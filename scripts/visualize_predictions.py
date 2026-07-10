from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from nuring.data.dataset import ensure_chw, load_array


def normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, (1, 99))
    return np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)


def label_overlay(gray: np.ndarray, labels: np.ndarray) -> np.ndarray:
    base = np.stack([gray, gray, gray], axis=-1)
    edges = (labels > 0) ^ np.pad((labels[1:-1, 1:-1] > 0), 1, mode="constant")
    base[edges] = [1, 0.1, 0.1]
    return base


def to_rgb(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = normalize(arr)
        arr = np.stack([arr, arr, arr], axis=-1)
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


def save_panel_image(panels: list[np.ndarray], out: Path) -> Path:
    rgb = [to_rgb(p) for p in panels]
    h = max(p.shape[0] for p in rgb)
    padded = []
    for p in rgb:
        if p.shape[0] < h:
            pad = np.full((h - p.shape[0], p.shape[1], 3), 255, dtype=np.uint8)
            p = np.concatenate([p, pad], axis=0)
        padded.append(p)
        padded.append(np.full((h, 6, 3), 255, dtype=np.uint8))
    canvas = np.concatenate(padded[:-1], axis=1)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image

        Image.fromarray(canvas).save(out)
        return out
    except Exception:
        ppm = out.with_suffix(".ppm")
        with ppm.open("wb") as f:
            f.write(f"P6\n{canvas.shape[1]} {canvas.shape[0]}\n255\n".encode("ascii"))
            f.write(canvas.tobytes())
        return ppm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--nucleus", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--baseline")
    parser.add_argument("--out", default="outputs/vis.png")
    args = parser.parse_args()

    image = ensure_chw(load_array(args.image))
    nuclei = load_array(args.nucleus)
    pred = load_array(args.pred)
    baseline = load_array(args.baseline) if args.baseline else None
    dapi = normalize(image[0])
    marker = normalize(image[1:].sum(axis=0) if image.shape[0] > 1 else image[0])

    panels = [dapi, marker, nuclei > 0, label_overlay(dapi, pred)]
    if baseline is not None:
        panels.append(label_overlay(dapi, baseline))
    out = Path(args.out)
    written = save_panel_image(panels, out)
    print(f"wrote {written}")


if __name__ == "__main__":
    main()
