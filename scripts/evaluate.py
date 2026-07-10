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


def median_nearest_nucleus_distance(nuclei: np.ndarray) -> float:
    centers = []
    for nid in np.unique(nuclei):
        if nid == 0:
            continue
        ys, xs = np.nonzero(nuclei == nid)
        if len(xs):
            centers.append((float(xs.mean()), float(ys.mean())))
    if len(centers) < 2:
        return float("inf")
    pts = np.asarray(centers, dtype=np.float32)
    dists = []
    for i in range(len(pts)):
        diff = pts - pts[i]
        dist = np.sqrt((diff * diff).sum(axis=1))
        dist[i] = np.inf
        dists.append(float(dist.min()))
    return float(np.median(dists))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--nucleus-dir")
    parser.add_argument("--output-csv", default="outputs/eval_metrics.csv")
    parser.add_argument("--pred-suffix", default="_pred.npy")
    parser.add_argument("--gt-suffix", default=".npy")
    parser.add_argument("--nucleus-suffix", default=".npy")
    parser.add_argument("--crowded-analysis", action="store_true")
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
        if args.crowded_analysis and nuclei is not None:
            row["median_nn_distance"] = median_nearest_nucleus_distance(nuclei)
        rows.append(row)
    if args.crowded_analysis and rows:
        finite = np.array([r["median_nn_distance"] for r in rows if np.isfinite(r["median_nn_distance"])], dtype=np.float32)
        if finite.size:
            q1, q2 = np.percentile(finite, [33.3, 66.6])
            for row in rows:
                d = row["median_nn_distance"]
                row["crowd_group"] = "sparse" if d >= q2 else ("medium" if d >= q1 else "crowded")
        else:
            for row in rows:
                row["crowd_group"] = "unknown"
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        if args.crowded_analysis and "crowd_group" in rows[0]:
            group_out = out.with_name(out.stem + "_crowded_summary.csv")
            metric_keys = [k for k in rows[0] if k not in {"sample", "median_nn_distance", "crowd_group"}]
            groups = sorted({r["crowd_group"] for r in rows})
            with group_out.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["crowd_group", "n", *metric_keys])
                writer.writeheader()
                for group in groups:
                    subset = [r for r in rows if r["crowd_group"] == group]
                    out_row = {"crowd_group": group, "n": len(subset)}
                    for key in metric_keys:
                        out_row[key] = float(np.mean([float(r[key]) for r in subset]))
                    writer.writerow(out_row)
    print(f"wrote {out} with {len(rows)} rows")


if __name__ == "__main__":
    main()
