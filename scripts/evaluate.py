from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from tqdm import tqdm

from nuring.data.dataset import load_array
from nuring.metrics.segmentation_metrics import compute_instance_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--nucleus-dir")
    parser.add_argument("--output-csv", default="outputs/eval_metrics.csv")
    parser.add_argument("--pred-suffix", default="_pred.npy")
    parser.add_argument("--gt-suffix", default=".npy")
    parser.add_argument("--nucleus-suffix", default=".npy")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    nucleus_dir = Path(args.nucleus_dir) if args.nucleus_dir else None
    rows = []
    for pred_path in tqdm(sorted(pred_dir.glob(f"*{args.pred_suffix}"))):
        name = pred_path.name[: -len(args.pred_suffix)]
        gt_path = gt_dir / f"{name}{args.gt_suffix}"
        if not gt_path.exists():
            continue
        pred = load_array(pred_path).astype(np.int32)
        gt = load_array(gt_path).astype(np.int32)
        nuclei = load_array(nucleus_dir / f"{name}{args.nucleus_suffix}").astype(np.int32) if nucleus_dir else None
        row = {"sample": name}
        row.update(compute_instance_metrics(pred, gt, nuclei))
        rows.append(row)
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"wrote {out} with {len(rows)} rows")


if __name__ == "__main__":
    main()
