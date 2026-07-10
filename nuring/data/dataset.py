from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .label_utils import (
    build_feature_crop,
    crop_with_pad,
    instance_centroid,
    instance_ids,
    matched_cell_id,
    pseudo_cell_mask,
    radial_boundary_from_polar_mask,
)
from .polar_transform import cartesian_to_polar_tensor


def load_array(path: str | Path, key: Optional[str] = None) -> np.ndarray:
    """Load npy/npz/h5/tif arrays. NPY is the primary supported format."""

    path = Path(path)
    if path.is_dir():
        slide_dir = path / "registered_slides"
        files = list(slide_dir.glob("*.tif*")) if slide_dir.exists() else list(path.glob("*.tif*"))
        if not files:
            raise FileNotFoundError(f"no tif files found in {path}")
        priority = ["DAPI", "CD3", "CD4", "CD8", "CD20", "CD23", "CCL19", "CCR7", "PANCK", "col1a1"]

        def channel_key(p: Path) -> tuple[int, str]:
            stem = p.name.upper()
            for i, name in enumerate(priority):
                if stem.startswith(name.upper()):
                    return i, p.name
            return len(priority), p.name

        arrays = []
        for f in sorted(files, key=channel_key):
            arr = read_tiff_channel(f)
            arr = np.squeeze(arr)
            if arr.ndim != 2:
                raise ValueError(f"expected 2D channel image from {f}, got {arr.shape}")
            arrays.append(arr.astype(np.float32))
        return np.stack(arrays, axis=0)

    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path, allow_pickle=True)
        return unpack_object_array(arr, path)
    if suffix == ".npz":
        data = np.load(path)
        if key is None:
            key = data.files[0]
        return data[key]
    if suffix in {".h5", ".hdf5"}:
        import h5py

        with h5py.File(path, "r") as f:
            if key is None:
                key = next(iter(f.keys()))
            return f[key][()]
    if suffix in {".tif", ".tiff"}:
        import tifffile

        arr = tifffile.imread(path)
        if arr.ndim == 3 and arr.shape[-1] < arr.shape[0]:
            arr = np.moveaxis(arr, -1, 0)
        return arr
    raise ValueError(f"unsupported file type: {path}")


def unpack_object_array(arr: np.ndarray, path: Path) -> np.ndarray:
    if arr.dtype != object:
        return arr
    obj = arr.item() if arr.shape == () else arr
    if isinstance(obj, dict):
        for key in ["mask", "masks", "nucleus", "nuclei", "label", "labels", "segmentation", "instances"]:
            if key in obj:
                return np.asarray(obj[key])
        if "coords" in obj and "label_id" in obj:
            return coords_dict_to_mask(obj)
        if len(obj) == 1:
            return np.asarray(next(iter(obj.values())))
        raise ValueError(f"object npy {path} is a dict; cannot choose mask key from {list(obj.keys())}")
    return np.asarray(obj)


def coords_dict_to_mask(obj: dict) -> np.ndarray:
    coords = obj["coords"]
    label_ids = np.asarray(obj["label_id"]).reshape(-1)
    shape = None
    metadata = obj.get("metadata", {})
    if isinstance(metadata, dict):
        for key in ["shape", "image_shape", "mask_shape", "spatial_shape"]:
            if key in metadata:
                shape = tuple(int(v) for v in metadata[key][-2:])
                break
        if shape is None and {"height", "width"}.issubset(metadata):
            shape = (int(metadata["height"]), int(metadata["width"]))
    if shape is None and "bbox" in obj:
        bbox = np.asarray(obj["bbox"])
        if bbox.size:
            shape = (int(np.max(bbox[:, [0, 2]])) + 1, int(np.max(bbox[:, [1, 3]])) + 1)
    if shape is None:
        max_y = max_x = 0
        for c in coords:
            arr = np.asarray(c)
            if arr.size == 0:
                continue
            if arr.ndim == 1:
                if arr.size % 2 != 0:
                    continue
                arr = arr.reshape(-1, 2)
            if arr.ndim == 2 and arr.shape[0] == 2 and arr.shape[1] != 2:
                arr = arr.T
            max_y = max(max_y, int(arr[:, 0].max()))
            max_x = max(max_x, int(arr[:, 1].max()))
        shape = (max_y + 1, max_x + 1)
    mask = np.zeros(shape, dtype=np.int32)
    for i, c in enumerate(coords):
        arr = np.asarray(c)
        if arr.size == 0:
            continue
        if arr.ndim == 1:
            if arr.size % 2 != 0:
                continue
            arr = arr.reshape(-1, 2)
        if arr.ndim == 2 and arr.shape[0] == 2 and arr.shape[1] != 2:
            arr = arr.T
        arr = arr.astype(np.int64)
        ys = np.clip(arr[:, 0], 0, shape[0] - 1)
        xs = np.clip(arr[:, 1], 0, shape[1] - 1)
        label = int(label_ids[i]) if i < len(label_ids) else i + 1
        mask[ys, xs] = label
    return mask


def read_tiff_channel(path: Path) -> np.ndarray:
    try:
        import tifffile

        return tifffile.imread(path)
    except ModuleNotFoundError:
        try:
            from PIL import Image

            return np.asarray(Image.open(path))
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("reading OME-TIFF requires tifffile or Pillow; install with `pip install tifffile pillow`") from exc


def ensure_chw(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image[None]
    if image.ndim != 3:
        raise ValueError(f"image must be [C,H,W] or [H,W,C], got {image.shape}")
    if image.shape[0] <= 64:
        return image
    return np.moveaxis(image, -1, 0)


def load_instance_ids_fast(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path, allow_pickle=True)
        if arr.dtype == object:
            obj = arr.item() if arr.shape == () else arr
            if isinstance(obj, dict) and "label_id" in obj:
                ids = np.asarray(obj["label_id"]).reshape(-1)
                return ids[ids > 0].astype(np.int32)
    return instance_ids(load_array(path))


@dataclass
class SampleRecord:
    name: str
    image_path: Path
    nucleus_path: Path
    cell_path: Optional[Path] = None
    tissue_path: Optional[Path] = None


class NuRingDataset(Dataset):
    """One item is one nucleus-centered crop converted to polar space.

    Returned tensors:
        x_polar: [Cin, R, A]
        y_mask_polar: [1, R, A]
        y_radius: [A]
        y_conf: [A]
        target_nucleus_polar: [1, R, A]
        neighbor_nuclei_polar: [1, R, A]
    """

    def __init__(
        self,
        image_dir: str,
        nucleus_dir: str,
        cell_dir: Optional[str] = None,
        tissue_dir: Optional[str] = None,
        sample_list: Optional[str] = None,
        image_suffix: str = ".npy",
        nucleus_suffix: str = ".npy",
        crop_size: int = 128,
        max_radius: int = 64,
        num_radial_bins: int = 64,
        num_angles: int = 128,
        pseudo_label: str = "dilation",
        max_instances_per_image: Optional[int] = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.nucleus_dir = Path(nucleus_dir)
        self.cell_dir = Path(cell_dir) if cell_dir else None
        self.tissue_dir = Path(tissue_dir) if tissue_dir else None
        self.crop_size = int(crop_size)
        self.max_radius = int(max_radius)
        self.num_radial_bins = int(num_radial_bins)
        self.num_angles = int(num_angles)
        self.pseudo_label = pseudo_label

        names = self._discover_names(sample_list, image_suffix)
        self.records = [
            SampleRecord(
                name=name,
                image_path=self._resolve_image_path(name, image_suffix),
                nucleus_path=self.nucleus_dir / f"{name}{nucleus_suffix}",
                cell_path=(self.cell_dir / f"{name}{nucleus_suffix}") if self.cell_dir else None,
                tissue_path=(self.tissue_dir / f"{name}{nucleus_suffix}") if self.tissue_dir else None,
            )
            for name in names
        ]
        self.index: List[Tuple[int, int]] = []
        for rec_i, rec in enumerate(self.records):
            if not rec.nucleus_path.exists():
                continue
            ids = load_instance_ids_fast(rec.nucleus_path)
            if max_instances_per_image:
                ids = ids[:max_instances_per_image]
            self.index.extend((rec_i, int(inst_id)) for inst_id in ids)
        if not self.index:
            raise RuntimeError(f"no nucleus instances found under {self.nucleus_dir}")

    def _discover_names(self, sample_list: Optional[str], image_suffix: str) -> List[str]:
        if sample_list:
            with open(sample_list, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        file_names = sorted(p.name[: -len(image_suffix)] for p in self.image_dir.glob(f"*{image_suffix}")) if image_suffix else []
        if file_names:
            return file_names
        dir_names = sorted(p.name for p in self.image_dir.iterdir() if p.is_dir())
        nucleus_names = {p.stem for p in self.nucleus_dir.glob("*")}
        return [name for name in dir_names if not nucleus_names or name in nucleus_names]

    def _resolve_image_path(self, name: str, image_suffix: str) -> Path:
        file_path = self.image_dir / f"{name}{image_suffix}"
        if image_suffix and file_path.exists():
            return file_path
        return self.image_dir / name

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str | int]:
        rec_i, inst_id = self.index[idx]
        rec = self.records[rec_i]
        image = ensure_chw(load_array(rec.image_path)).astype(np.float32)
        nuclei = load_array(rec.nucleus_path).astype(np.int32)
        cell = load_array(rec.cell_path).astype(np.int32) if rec.cell_path and rec.cell_path.exists() else None
        tissue = load_array(rec.tissue_path).astype(np.uint8) if rec.tissue_path and rec.tissue_path.exists() else None

        center = instance_centroid(nuclei, inst_id)
        image_crop = crop_with_pad(image, center, self.crop_size, fill=0)
        nucleus_crop = crop_with_pad(nuclei, center, self.crop_size, fill=0)
        tissue_crop = crop_with_pad(tissue, center, self.crop_size, fill=0) if tissue is not None else None
        features = build_feature_crop(image_crop, nucleus_crop, inst_id, tissue_crop)

        target_nucleus = (nucleus_crop == inst_id).astype(np.uint8)
        if cell is not None:
            cell_crop_full = crop_with_pad(cell, center, self.crop_size, fill=0)
            cid = matched_cell_id(cell_crop_full, target_nucleus)
            cell_crop = (cell_crop_full == cid).astype(np.uint8) if cid else pseudo_cell_mask(nucleus_crop, inst_id, mode=self.pseudo_label)
        else:
            cell_crop = pseudo_cell_mask(nucleus_crop, inst_id, radius=max(4, self.crop_size // 12), mode=self.pseudo_label)

        x_cart = torch.from_numpy(features).unsqueeze(0).float()
        y_cart = torch.from_numpy(cell_crop[None, None].astype(np.float32))
        target_cart = torch.from_numpy(target_nucleus[None, None].astype(np.float32))
        neighbor_cart = torch.from_numpy((((nucleus_crop > 0) & (nucleus_crop != inst_id))[None, None]).astype(np.float32))

        x_polar = cartesian_to_polar_tensor(x_cart, self.num_radial_bins, self.num_angles, self.max_radius).squeeze(0)
        y_polar = cartesian_to_polar_tensor(y_cart, self.num_radial_bins, self.num_angles, self.max_radius, mode="nearest").squeeze(0)
        target_polar = cartesian_to_polar_tensor(target_cart, self.num_radial_bins, self.num_angles, self.max_radius, mode="nearest").squeeze(0)
        neighbor_polar = cartesian_to_polar_tensor(neighbor_cart, self.num_radial_bins, self.num_angles, self.max_radius, mode="nearest").squeeze(0)
        y_radius = torch.from_numpy(radial_boundary_from_polar_mask(y_polar[0].numpy(), self.max_radius))
        y_conf = (y_radius > 0).float()

        return {
            "x_polar": x_polar.float(),
            "y_mask_polar": y_polar.float(),
            "y_radius": y_radius.float(),
            "y_conf": y_conf.float(),
            "target_nucleus_polar": target_polar.float(),
            "neighbor_nuclei_polar": neighbor_polar.float(),
            "sample_name": rec.name,
            "instance_id": int(inst_id),
            "center_xy": torch.tensor(center, dtype=torch.float32),
        }


class TissueNetNPZDataset(Dataset):
    """Minimal TissueNet v1.1 npz adapter.

    Expects arrays shaped like X=[N,H,W,C] or [N,C,H,W] and y=[N,H,W,2],
    where y[...,0] is nuclear labels and y[...,1] is whole-cell labels.
    """

    def __init__(self, npz_path: str, **kwargs) -> None:
        data = np.load(npz_path)
        self.x = data["X"]
        self.y = data["y"]
        self.kwargs = kwargs
        raise NotImplementedError("Use export_tissuenet_to_npy() or add a sample adapter before direct training.")


def export_tissuenet_to_npy(npz_path: str, out_dir: str) -> None:
    """Export TissueNet X/y npz into image/nucleus/cell npy folders."""

    data = np.load(npz_path)
    x = data["X"]
    y = data["y"]
    out = Path(out_dir)
    for sub in ["images", "nuclei", "cells"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    for i in range(x.shape[0]):
        img = x[i]
        if img.ndim == 3 and img.shape[-1] <= 64:
            img = np.moveaxis(img, -1, 0)
        np.save(out / "images" / f"{i:06d}.npy", img.astype(np.float32))
        np.save(out / "nuclei" / f"{i:06d}.npy", y[i, ..., 0].astype(np.int32))
        np.save(out / "cells" / f"{i:06d}.npy", y[i, ..., 1].astype(np.int32))
