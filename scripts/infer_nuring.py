from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml
from tqdm import tqdm

from nuring.data.dataset import ensure_chw, load_array
from nuring.data.label_utils import build_feature_crop, crop_with_pad, instance_centroid, instance_ids, select_feature_channels
from nuring.data.polar_transform import cartesian_to_polar_tensor, fuse_polar_prediction, polar_to_cartesian_mask
from nuring.models import PolarRingNet
from nuring.postprocess.assignment import assign_instances


def load_model(checkpoint: str, device: torch.device) -> tuple[PolarRingNet, dict]:
    ckpt = torch.load(checkpoint, map_location=device)
    cfg = ckpt["config"]
    model_cfg = cfg["model"]
    model = PolarRingNet(ckpt["in_channels"], model_cfg["base_channels"], model_cfg["num_radial_bins"], model_cfg["max_radius"])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, cfg


def infer_one(
    image_path: Path,
    nucleus_path: Path,
    model: PolarRingNet,
    cfg: dict,
    device: torch.device,
    threshold: float,
    fusion_mode: str,
    radius_sigma: float,
    radius_prior_weight: float,
) -> np.ndarray:
    model_cfg = cfg["model"]
    data_cfg = cfg.get("data", {})
    image = ensure_chw(load_array(image_path)).astype(np.float32)
    nuclei = load_array(nucleus_path).astype(np.int32)
    h, w = nuclei.shape
    prob_crops: List[np.ndarray] = []
    ids: List[int] = []
    centers: List[Tuple[float, float]] = []
    for inst_id in tqdm(instance_ids(nuclei), desc=f"infer {image_path.stem}", leave=False):
        center = instance_centroid(nuclei, int(inst_id))
        image_crop = crop_with_pad(image, center, model_cfg["crop_size"], fill=0)
        nucleus_crop = crop_with_pad(nuclei, center, model_cfg["crop_size"], fill=0)
        features_all = build_feature_crop(image_crop, nucleus_crop, int(inst_id))
        features = select_feature_channels(features_all, image_crop.shape[0], data_cfg.get("input_mode", "all"))
        x = torch.from_numpy(features).unsqueeze(0).float().to(device)
        x_polar = cartesian_to_polar_tensor(x, model_cfg["num_radial_bins"], model_cfg["num_angles"], model_cfg["max_radius"])
        with torch.no_grad():
            out = model(x_polar)
            mask_prob = torch.sigmoid(out["mask_logits"])
            polar_prob = fuse_polar_prediction(
                mask_prob,
                out["radius"],
                out["confidence"],
                model_cfg["num_radial_bins"],
                model_cfg["max_radius"],
                mode=fusion_mode,
                radius_sigma=radius_sigma,
                radius_prior_weight=radius_prior_weight,
            )
            cart_prob = polar_to_cartesian_mask(polar_prob, model_cfg["crop_size"], model_cfg["max_radius"])
        prob_crops.append(cart_prob.squeeze().detach().cpu().numpy())
        ids.append(int(inst_id))
        centers.append(center)
    return assign_instances(prob_crops, ids, centers, (h, w), model_cfg["crop_size"], threshold=threshold)


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
    parser.add_argument("--checkpoint", default="runs/nuring/best.pt")
    parser.add_argument("--image-dir")
    parser.add_argument("--nucleus-dir")
    parser.add_argument("--output-dir", default="outputs/nuring")
    parser.add_argument("--sample", help="single sample stem without suffix")
    parser.add_argument("--sample-list")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fusion-mode", choices=["mask_only", "mask_radius_average", "confidence_fusion"])
    parser.add_argument("--radius-sigma", type=float)
    parser.add_argument("--radius-prior-weight", type=float)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, cfg = load_model(args.checkpoint, device)
    data_cfg = cfg["data"]
    infer_cfg = cfg.get("infer", {})
    image_dir = Path(args.image_dir or data_cfg["image_dir"])
    nucleus_dir = Path(args.nucleus_dir or data_cfg["nucleus_dir"])
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = data_cfg.get("image_suffix", ".npy")
    if args.sample:
        names = [args.sample]
    elif args.sample_list:
        names = [line.strip() for line in Path(args.sample_list).read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        names = discover_names(image_dir, nucleus_dir, suffix)
    for name in tqdm(names, desc="samples"):
        pred = infer_one(
            resolve_image_path(image_dir, name, suffix),
            nucleus_dir / f"{name}{data_cfg.get('nucleus_suffix', '.npy')}",
            model,
            cfg,
            device,
            args.threshold,
            args.fusion_mode or infer_cfg.get("fusion_mode", "confidence_fusion"),
            args.radius_sigma if args.radius_sigma is not None else float(infer_cfg.get("radius_sigma", 2.0)),
            args.radius_prior_weight if args.radius_prior_weight is not None else float(infer_cfg.get("radius_prior_weight", 0.5)),
        )
        np.save(out_dir / f"{name}_pred.npy", pred.astype(np.int32))


if __name__ == "__main__":
    main()
