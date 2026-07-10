from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nuring.data.dataset import NuRingDataset


def main() -> None:
    print("tifffile", importlib.util.find_spec("tifffile") is not None)
    nucleus_dir = Path("/data/wzx/TLS/mIF50/nucleiseg")
    first = sorted(p.stem for p in nucleus_dir.glob("*.npy"))[0]
    sample_list = Path("/tmp/nuring_one_sample.txt")
    sample_list.write_text(first + "\n", encoding="utf-8")
    ds = NuRingDataset(
        image_dir="/data/wzx/TLS/mIF50/registered",
        nucleus_dir=str(nucleus_dir),
        sample_list=str(sample_list),
        image_suffix=".npy",
        nucleus_suffix=".npy",
        max_instances_per_image=1,
    )
    print("records", len(ds.records), "instances", len(ds), "first_image", ds.records[0].image_path)


if __name__ == "__main__":
    main()
