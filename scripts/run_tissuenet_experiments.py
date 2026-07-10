from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np


def run(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("RUN", " ".join(cmd))
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}); see {log_path}")


def summarize_csv(csv_path: Path) -> dict[str, float]:
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    summary: dict[str, float] = {"n": float(len(rows))}
    if not rows:
        return summary
    keys = [k for k in rows[0].keys() if k != "sample"]
    for key in keys:
        vals = [float(r[key]) for r in rows if r.get(key) not in {"", None}]
        if vals:
            summary[key] = float(np.mean(vals))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", default="/data/wzx/tissuenet_v1.1/nuring_export")
    parser.add_argument("--work-dir", default="runs/tissuenet_experiments")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--max-instances-per-image", type=int, default=96)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    export = Path(args.export_dir)
    work = Path(args.work_dir)
    logs = work / "logs"
    metrics_dir = work / "metrics"
    pred_root = work / "predictions"
    summary: dict[str, dict[str, float]] = {}

    train = export / "train"
    test = export / "test"
    train_args = [
        "--config", args.config,
        "--image-dir", str(train / "images"),
        "--nucleus-dir", str(train / "nuclei"),
        "--cell-dir", str(train / "cells"),
        "--sample-list", str(train / "samples.txt"),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--device", args.device,
        "--base-channels", str(args.base_channels),
        "--max-instances-per-image", str(args.max_instances_per_image),
    ]
    variants = {
        "nuring_full": {},
        "ablate_mask_only": {"--lambda-radius": "0", "--lambda-smooth": "0", "--lambda-contain": "0", "--lambda-neighbor": "0"},
        "ablate_no_shape": {"--lambda-smooth": "0", "--lambda-contain": "0", "--lambda-neighbor": "0"},
        "ablate_no_neighbor": {"--lambda-neighbor": "0"},
    }
    for name, overrides in variants.items():
        run_dir = work / "models" / name
        if not args.skip_train:
            cmd = [sys.executable, "scripts/train_nuring.py", *train_args, "--save-dir", str(run_dir)]
            for k, v in overrides.items():
                cmd.extend([k, v])
            run(cmd, logs / f"train_{name}.log")
        pred_dir = pred_root / name
        run(
            [
                sys.executable,
                "scripts/infer_nuring.py",
                "--checkpoint",
                str(run_dir / "best.pt"),
                "--image-dir",
                str(test / "images"),
                "--nucleus-dir",
                str(test / "nuclei"),
                "--sample-list",
                str(test / "samples.txt"),
                "--output-dir",
                str(pred_dir),
                "--device",
                args.device,
            ],
            logs / f"infer_{name}.log",
        )
        csv_path = metrics_dir / f"{name}.csv"
        run(
            [
                sys.executable,
                "scripts/evaluate.py",
                "--pred-dir",
                str(pred_dir),
                "--gt-dir",
                str(test / "cells"),
                "--nucleus-dir",
                str(test / "nuclei"),
                "--output-csv",
                str(csv_path),
            ],
            logs / f"eval_{name}.log",
        )
        summary[name] = summarize_csv(csv_path)

    for method, radius in [("dilation", "10"), ("voronoi", "32"), ("watershed", "32")]:
        out_dir = pred_root / "baselines" / method
        run(
            [
                sys.executable,
                "scripts/run_baselines.py",
                "--image-dir",
                str(test / "images"),
                "--nucleus-dir",
                str(test / "nuclei"),
                "--sample-list",
                str(test / "samples.txt"),
                "--output-dir",
                str(pred_root / "baselines"),
                "--method",
                method,
                "--radius",
                radius,
            ],
            logs / f"baseline_{method}.log",
        )
        csv_path = metrics_dir / f"baseline_{method}.csv"
        run(
            [
                sys.executable,
                "scripts/evaluate.py",
                "--pred-dir",
                str(out_dir),
                "--gt-dir",
                str(test / "cells"),
                "--nucleus-dir",
                str(test / "nuclei"),
                "--output-csv",
                str(csv_path),
            ],
            logs / f"eval_baseline_{method}.log",
        )
        summary[f"baseline_{method}"] = summarize_csv(csv_path)

    work.mkdir(parents=True, exist_ok=True)
    (work / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
