from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import yaml

from nuring.data.dataset import NuRingDataset
from nuring.losses import NuRingLoss
from nuring.models import PolarRingNet


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--image-dir")
    parser.add_argument("--nucleus-dir")
    parser.add_argument("--cell-dir")
    parser.add_argument("--sample-list")
    parser.add_argument("--save-dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--max-instances-per-image", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--base-channels", type=int)
    parser.add_argument("--lambda-mask", type=float)
    parser.add_argument("--lambda-radius", type=float)
    parser.add_argument("--lambda-smooth", type=float)
    parser.add_argument("--lambda-contain", type=float)
    parser.add_argument("--lambda-neighbor", type=float)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    loss_cfg = cfg["loss"]
    for key in ["image_dir", "nucleus_dir", "cell_dir"]:
        val = getattr(args, key.replace("_", "-"), None)
        if val:
            data_cfg[key] = val
    if args.image_dir:
        data_cfg["image_dir"] = args.image_dir
    if args.nucleus_dir:
        data_cfg["nucleus_dir"] = args.nucleus_dir
    if args.cell_dir:
        data_cfg["cell_dir"] = args.cell_dir
    if args.sample_list:
        data_cfg["sample_list"] = args.sample_list
    if args.save_dir:
        train_cfg["save_dir"] = args.save_dir
    if args.epochs:
        train_cfg["epochs"] = args.epochs
    if args.batch_size:
        train_cfg["batch_size"] = args.batch_size
    if args.device:
        train_cfg["device"] = args.device
    if args.num_workers is not None:
        train_cfg["num_workers"] = args.num_workers
    if args.base_channels:
        model_cfg["base_channels"] = args.base_channels
    for key in ["lambda_mask", "lambda_radius", "lambda_smooth", "lambda_contain", "lambda_neighbor"]:
        val = getattr(args, key)
        if val is not None:
            loss_cfg[key] = val

    seed_all(int(train_cfg.get("seed", 7)))
    device = torch.device(train_cfg["device"] if torch.cuda.is_available() or train_cfg["device"] == "cpu" else "cpu")
    save_dir = Path(train_cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset = NuRingDataset(
        image_dir=data_cfg["image_dir"],
        nucleus_dir=data_cfg["nucleus_dir"],
        cell_dir=data_cfg.get("cell_dir"),
        tissue_dir=data_cfg.get("tissue_dir"),
        sample_list=data_cfg.get("sample_list"),
        image_suffix=data_cfg.get("image_suffix", ".npy"),
        nucleus_suffix=data_cfg.get("nucleus_suffix", ".npy"),
        crop_size=model_cfg["crop_size"],
        max_radius=model_cfg["max_radius"],
        num_radial_bins=model_cfg["num_radial_bins"],
        num_angles=model_cfg["num_angles"],
        pseudo_label=data_cfg.get("pseudo_label", "dilation"),
        max_instances_per_image=args.max_instances_per_image,
    )
    val_len = max(1, int(0.1 * len(dataset))) if len(dataset) > 10 else 0
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(dataset, [train_len, val_len]) if val_len else (dataset, None)
    train_loader = DataLoader(train_set, batch_size=train_cfg["batch_size"], shuffle=True, num_workers=train_cfg["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=train_cfg["batch_size"], shuffle=False, num_workers=train_cfg["num_workers"], pin_memory=True) if val_set else None

    first = dataset[0]["x_polar"]
    in_channels = int(model_cfg.get("input_channels") or first.shape[0])
    model = PolarRingNet(in_channels, model_cfg["base_channels"], model_cfg["num_radial_bins"], model_cfg["max_radius"]).to(device)
    criterion = NuRingLoss(**loss_cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])

    best_val = float("inf")
    history = []
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{train_cfg['epochs']} train")
        for batch in pbar:
            batch = move_batch(batch, device)
            pred = model(batch["x_polar"])
            losses = criterion(pred, batch)
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            optimizer.step()
            train_loss += float(losses["loss"].item())
            pbar.set_postfix(loss=f"{losses['loss'].item():.4f}", mask=f"{losses['mask'].item():.4f}", rad=f"{losses['radius'].item():.3f}")
        train_loss /= max(len(train_loader), 1)

        val_loss = train_loss
        if val_loader is not None:
            model.eval()
            vals = []
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"epoch {epoch}/{train_cfg['epochs']} val"):
                    batch = move_batch(batch, device)
                    vals.append(float(criterion(model(batch["x_polar"]), batch)["loss"].item()))
            val_loss = float(np.mean(vals)) if vals else train_loss

        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(row)
        (save_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        ckpt = {
            "model": model.state_dict(),
            "config": cfg,
            "in_channels": in_channels,
            "epoch": epoch,
            "val_loss": val_loss,
        }
        torch.save(ckpt, save_dir / "last.pt")
        if val_loss <= best_val:
            best_val = val_loss
            torch.save(ckpt, save_dir / "best.pt")
        print(row)


if __name__ == "__main__":
    main()
