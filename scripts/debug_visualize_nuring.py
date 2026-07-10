from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from nuring.data.dataset import ensure_chw, load_array
from nuring.data.label_utils import build_feature_crop, crop_with_pad, instance_centroid, instance_ids, matched_cell_id, select_feature_channels
from nuring.data.polar_transform import cartesian_to_polar_tensor, fuse_polar_prediction, polar_to_cartesian_mask, radius_to_soft_polar_mask
from nuring.models import PolarRingNet


def normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, (1, 99))
    return np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)


def rgb_panel(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        x = normalize(x)
        x = np.stack([x, x, x], axis=-1)
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def save_grid(panels: list[np.ndarray], out: Path, cols: int = 4) -> None:
    imgs = [rgb_panel(p) for p in panels]
    h = max(i.shape[0] for i in imgs)
    w = max(i.shape[1] for i in imgs)
    padded = []
    for img in imgs:
        canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
        canvas[: img.shape[0], : img.shape[1]] = img
        padded.append(canvas)
    rows = []
    for start in range(0, len(padded), cols):
        row = padded[start : start + cols]
        while len(row) < cols:
            row.append(np.ones((h, w, 3), dtype=np.uint8) * 255)
        rows.append(np.concatenate(row, axis=1))
    grid = np.concatenate(rows, axis=0)
    out.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    Image.fromarray(grid).save(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--nucleus", required=True)
    parser.add_argument("--cell")
    parser.add_argument("--instance-id", type=int)
    parser.add_argument("--out", default="outputs/debug_nuring.png")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fusion-mode", choices=["mask_only", "mask_radius_average", "confidence_fusion"], default="confidence_fusion")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]
    model_cfg = cfg["model"]
    data_cfg = cfg.get("data", {})
    model = PolarRingNet(ckpt["in_channels"], model_cfg["base_channels"], model_cfg["num_radial_bins"], model_cfg["max_radius"])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    image = ensure_chw(load_array(args.image)).astype(np.float32)
    nuclei = load_array(args.nucleus).astype(np.int32)
    inst_id = args.instance_id or int(instance_ids(nuclei)[0])
    center = instance_centroid(nuclei, inst_id)
    image_crop = crop_with_pad(image, center, model_cfg["crop_size"], fill=0)
    nucleus_crop = crop_with_pad(nuclei, center, model_cfg["crop_size"], fill=0)
    features_all = build_feature_crop(image_crop, nucleus_crop, inst_id)
    features = select_feature_channels(features_all, image_crop.shape[0], data_cfg.get("input_mode", "all"))
    x = torch.from_numpy(features).unsqueeze(0).float().to(device)
    x_polar = cartesian_to_polar_tensor(x, model_cfg["num_radial_bins"], model_cfg["num_angles"], model_cfg["max_radius"])
    with torch.no_grad():
        pred = model(x_polar)
        mask_prob = torch.sigmoid(pred["mask_logits"])
        radius_prior = radius_to_soft_polar_mask(pred["radius"], model_cfg["num_radial_bins"], model_cfg["max_radius"])
        fused = fuse_polar_prediction(mask_prob, pred["radius"], pred["confidence"], model_cfg["num_radial_bins"], model_cfg["max_radius"], mode=args.fusion_mode)
        cart_pred = polar_to_cartesian_mask(fused, model_cfg["crop_size"], model_cfg["max_radius"])

    target = (nucleus_crop == inst_id).astype(np.float32)
    neighbor = ((nucleus_crop > 0) & (nucleus_crop != inst_id)).astype(np.float32)
    gt = np.zeros_like(target)
    if args.cell:
        cell_crop = crop_with_pad(load_array(args.cell).astype(np.int32), center, model_cfg["crop_size"], fill=0)
        cid = matched_cell_id(cell_crop, target)
        gt = (cell_crop == cid).astype(np.float32) if cid else gt
    dapi = image_crop[0]
    overlay = np.stack([normalize(dapi), normalize(dapi), normalize(dapi)], axis=-1)
    overlay[cart_pred.squeeze().detach().cpu().numpy() > 0.5] = [1.0, 0.2, 0.1]
    panels = [
        target,
        neighbor,
        mask_prob.squeeze().detach().cpu().numpy(),
        radius_prior.squeeze().detach().cpu().numpy(),
        fused.squeeze().detach().cpu().numpy(),
        cart_pred.squeeze().detach().cpu().numpy(),
        gt,
        overlay,
    ]
    save_grid(panels, Path(args.out))
    print(args.out)


if __name__ == "__main__":
    main()
