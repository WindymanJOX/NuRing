from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from tqdm import tqdm

from nuring.baselines.dilation import dilation_baseline
from nuring.baselines.voronoi import voronoi_baseline
from nuring.baselines.watershed import watershed_baseline
from nuring.data.dataset import ensure_chw, load_array


def discover_names(image_dir: Path, nucleus_dir: Path, image_suffix: str) -> list[str]:
    names = sorted(p.name[: -len(image_suffix)] for p in image_dir.glob(f"*{image_suffix}")) if image_suffix else []
    if names:
        return names
    dir_names = sorted(p.name for p in image_dir.iterdir() if p.is_dir())
    nucleus_names = {p.stem for p in nucleus_dir.glob("*")}
    return [name for name in dir_names if not nucleus_names or name in nucleus_names]


def resolve_image_path(image_dir: Path, name: str, image_suffix: str) -> Path:
    file_path = image_dir / f"{name}{image_suffix}"
    return file_path if image_suffix and file_path.exists() else image_dir / name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", default="/data/wzx/TLS/mIF50/registered")
    parser.add_argument("--nucleus-dir", default="/data/wzx/TLS/mIF50/nucleiseg")
    parser.add_argument("--output-dir", default="outputs/baselines")
    parser.add_argument("--image-suffix", default=".npy")
    parser.add_argument("--nucleus-suffix", default=".npy")
    parser.add_argument("--sample-list")
    parser.add_argument("--method", choices=["dilation", "voronoi", "watershed"], default="dilation")
    parser.add_argument("--radius", type=int, default=10)
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    nucleus_dir = Path(args.nucleus_dir)
    out_dir = Path(args.output_dir) / args.method
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.sample_list:
        names = [line.strip() for line in Path(args.sample_list).read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        names = discover_names(image_dir, nucleus_dir, args.image_suffix)
    for name in tqdm(names):
        nuclei = load_array(nucleus_dir / f"{name}{args.nucleus_suffix}").astype(np.int32)
        marker = None
        if args.method == "watershed":
            img = ensure_chw(load_array(resolve_image_path(image_dir, name, args.image_suffix)))
            marker = img[1:].sum(axis=0) if img.shape[0] > 1 else img[0]
        if args.method == "dilation":
            pred = dilation_baseline(nuclei, radius=args.radius)
        elif args.method == "voronoi":
            pred = voronoi_baseline(nuclei, max_radius=args.radius)
        else:
            pred = watershed_baseline(nuclei, marker_image=marker, max_radius=args.radius)
        np.save(out_dir / f"{name}_pred.npy", pred.astype(np.int32))


if __name__ == "__main__":
    main()
