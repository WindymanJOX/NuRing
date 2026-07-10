from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def fmt(value: float) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--title", default="TissueNet NuRing Experiments")
    parser.add_argument("--train-count", type=int)
    parser.add_argument("--val-count", type=int)
    parser.add_argument("--test-count", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--experiment-kind", choices=["smoke", "formal"], default="smoke")
    parser.add_argument("--out")
    args = parser.parse_args()

    work = Path(args.work_dir)
    summary = json.loads((work / "summary.json").read_text(encoding="utf-8"))
    metrics = ["dice", "iou", "aji", "pq", "boundary_f1", "nucleus_containment", "neighbor_leakage"]
    lines = [f"# {args.title}", ""]
    if args.train_count is not None:
        lines.append(f"- TissueNet split: train={args.train_count}, val={args.val_count}, test={args.test_count}")
    if args.epochs is not None:
        lines.append(f"- NuRing epochs: {args.epochs}")
    if args.experiment_kind == "smoke":
        lines.append("- Experiment kind: smoke test; use this to validate code paths, not as a final method conclusion.")
    else:
        lines.append("- Experiment kind: formal experiment.")
    lines.extend(
        [
            f"- Work dir: `{work}`",
            f"- Metrics dir: `{work / 'metrics'}`",
            f"- Predictions dir: `{work / 'predictions'}`",
            "",
            "| method | n | dice | iou | aji | pq | boundary_f1 | containment | leakage |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method, row in summary.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    method,
                    fmt(row.get("n", 0.0)),
                    fmt(row.get("dice", 0.0)),
                    fmt(row.get("iou", 0.0)),
                    fmt(row.get("aji", 0.0)),
                    fmt(row.get("pq", 0.0)),
                    fmt(row.get("boundary_f1", 0.0)),
                    fmt(row.get("nucleus_containment", 0.0)),
                    fmt(row.get("neighbor_leakage", 0.0)),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `loss_mask_only` trains and infers with only the polar mask branch.",
            "- `loss_mask_radius` adds normalized radius supervision and uses mask/radius average at inference.",
            "- `loss_mask_radius_conf` adds confidence supervision and uses confidence fusion.",
            "- `loss_mask_radius_conf_smooth` adds normalized circular smoothness.",
            "- `nuring_full` adds containment and neighbor exclusion with reduced default weights.",
            "- `fusion_mask_only`, `fusion_mask_radius_average`, and `fusion_confidence_fusion` evaluate the same full checkpoint with different inference fusion modes.",
            "- Baselines use nucleus dilation, Voronoi nearest-nucleus assignment, and watershed.",
        ]
    )
    out = Path(args.out) if args.out else work / "report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
