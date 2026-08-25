# data_loader.py
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from degradations import ProgressiveDegradation, build_degradation
from srf_utils import (
    WV2_VISIBLE_5_BANDS,
    WV2_VISIBLE_6_BANDS,
    WV2_ALL_8_BANDS,
    load_hsi_wavelengths,
    build_srf_weights,
    hsi_to_msi_numpy,
    print_srf_summary,
)

try:
    import scipy.io as scio
except ImportError:
    scio = None

try:
    import hdf5storage
except ImportError:
    hdf5storage = None

try:
    import h5py
except ImportError:
    h5py = None


def read_hsi_mat(file_path: str, candidate_keys: List[str]) -> np.ndarray:
    """读取 .mat 高光谱数据并统一返回 H×W×C。"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find data file: {file_path}")

    mat_data = None
    if hdf5storage is not None:
        try:
            mat_data = hdf5storage.loadmat(file_path)
        except Exception:
            mat_data = None

    if mat_data is None and scio is not None:
        try:
            mat_data = scio.loadmat(file_path)
        except Exception:
            mat_data = None

    if mat_data is not None:
        for key in candidate_keys:
            if key in mat_data and isinstance(mat_data[key], np.ndarray):
                return fix_hsi_shape(mat_data[key])
        for key, value in mat_data.items():
            if key.startswith("__"):
                continue
            if isinstance(value, np.ndarray) and value.ndim == 3:
                return fix_hsi_shape(value)

    if h5py is not None:
        with h5py.File(file_path, "r") as f:
            for key in candidate_keys:
                if key in f:
                    return fix_hsi_shape(np.array(f[key]))
            for key in f.keys():
                value = np.array(f[key])
                if value.ndim == 3:
                    return fix_hsi_shape(value)

    raise RuntimeError(f"No valid 3D HSI array found in {file_path}")


def fix_hsi_shape(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img).squeeze()
    if img.ndim != 3:
        raise ValueError(f"HSI data must be 3D, but got shape: {img.shape}")

    if img.shape[0] <= 256 and img.shape[1] > 256 and img.shape[2] > 256:
        img = np.transpose(img, (1, 2, 0))
    elif img.shape[1] <= 256 and img.shape[0] > 256 and img.shape[2] > 256:
        img = np.transpose(img, (0, 2, 1))

    return img.astype(np.float32)


def normalize_hsi(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    min_value = float(np.min(img))
    max_value = float(np.max(img))
    if max_value - min_value < 1e-8:
        return np.zeros_like(img, dtype=np.float32)
    return ((img - min_value) / (max_value - min_value)).astype(np.float32)


def crop_to_scale(img: np.ndarray, scale_ratio: int) -> np.ndarray:
    h, w, _ = img.shape
    new_h = h // scale_ratio * scale_ratio
    new_w = w // scale_ratio * scale_ratio
    return img[:new_h, :new_w, :]


def hsi_to_tensor(img: np.ndarray) -> torch.Tensor:
    """H×W×C -> C×H×W."""
    return torch.from_numpy(img).permute(2, 0, 1).contiguous().float()


def tensor_to_hsi(x: torch.Tensor) -> np.ndarray:
    """C×H×W -> H×W×C."""
    if x.ndim != 3:
        raise ValueError(f"Expected C×H×W tensor, got {tuple(x.shape)}")
    return x.detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)


def build_hsi_degradation(cfg):
    """Build the exact operator used by the dataset to generate LR-HSI."""
    mode = getattr(cfg, "degradation_mode", "physical")
    kwargs = {}
    if mode == "gaussian_bicubic":
        kwargs.update(
            sigma=getattr(cfg, "degradation_sigma", 2.0),
            kernel_size=getattr(cfg, "degradation_kernel_size", 5),
        )
    elif mode == "physical":
        kwargs.update(
            mtf_nyquist=getattr(cfg, "mtf_nyquist", 0.2),
            truncate=getattr(cfg, "psf_truncate", 3.0),
        )
    return build_degradation(mode, scale_ratio=cfg.scale_ratio, **kwargs)


def build_progressive_degradation(cfg, degradation_operator=None):
    operator = degradation_operator or build_hsi_degradation(cfg)
    lift_mode = getattr(cfg, "progressive_lift", None)
    if lift_mode is None:
        lift_mode = (
            "normalized_adjoint" if operator.mode == "physical" else "bilinear"
        )
    return ProgressiveDegradation(
        operator=operator,
        total_steps=getattr(cfg, "progressive_steps", 12),
        default_lift_mode=lift_mode,
    )


def make_lr_hsi(
    hr_hsi: np.ndarray,
    scale_ratio: int,
    degradation_operator=None,
) -> np.ndarray:
    """Generate LR-HSI.

    Backward compatibility:
    when no operator is supplied, reproduce the legacy conceptual baseline
    (Gaussian sigma=2, kernel=5, then bicubic downsampling).

    New experiments should build one operator from config and pass it into the
    dataset so that dataset degradation and progressive terminal degradation
    are the same object.
    """
    if degradation_operator is None:
        degradation_operator = build_degradation(
            "gaussian_bicubic",
            scale_ratio=scale_ratio,
            sigma=2.0,
            kernel_size=5,
        )

    x = hsi_to_tensor(hr_hsi).unsqueeze(0)
    with torch.no_grad():
        y = degradation_operator.degrade(x).squeeze(0)
    return tensor_to_hsi(y)


def make_hr_msi(hr_hsi: np.ndarray, n_select_bands: int) -> np.ndarray:
    """Fallback MSI generator used only when SRF mode is disabled."""
    n_bands = hr_hsi.shape[2]
    if n_select_bands > n_bands:
        raise ValueError(
            f"n_select_bands={n_select_bands} is larger than HSI bands={n_bands}"
        )
    band_indices = np.linspace(0, n_bands - 1, n_select_bands).round().astype(np.int64)
    return hr_hsi[:, :, band_indices].astype(np.float32)


def get_center_test_rect(h: int, w: int, test_size: int) -> Tuple[int, int, int, int]:
    top = max((h - test_size) // 2, 0)
    left = max((w - test_size) // 2, 0)
    bottom = min(top + test_size, h)
    right = min(left + test_size, w)
    return top, left, bottom, right


def intersects(rect1: Tuple[int, int, int, int], rect2: Tuple[int, int, int, int]) -> bool:
    t1, l1, b1, r1 = rect1
    t2, l2, b2, r2 = rect2
    return not (r1 <= l2 or r2 <= l1 or b1 <= t2 or b2 <= t1)


def build_patch_coords(
    h: int,
    w: int,
    patch_size: int,
    stride: int,
    test_rect: Tuple[int, int, int, int],
    split: str,
) -> List[Tuple[int, int]]:
    coords = []

    if split == "test":
        top, left, bottom, right = test_rect
        if bottom - top < patch_size or right - left < patch_size:
            top = max((h - patch_size) // 2, 0)
            left = max((w - patch_size) // 2, 0)
        return [(top, left)]

    for top in range(0, h - patch_size + 1, stride):
        for left in range(0, w - patch_size + 1, stride):
            patch_rect = (top, left, top + patch_size, left + patch_size)
            if not intersects(patch_rect, test_rect):
                coords.append((top, left))

    if len(coords) == 0:
        for top in range(0, h - patch_size + 1, stride):
            for left in range(0, w - patch_size + 1, stride):
                coords.append((top, left))

    return coords


class HSIHSRDataset(Dataset):
    """HSI-MSI fusion super-resolution dataset.

    Returns:
        lr_hsi: C×h×w
        hr_msi: M×H×W
        gt: C×H×W
    """

    def __init__(
        self,
        img: np.ndarray,
        dataset_name: str,
        patch_size: int,
        stride: int,
        scale_ratio: int,
        n_select_bands: int,
        split: str = "train",
        test_size: int = 128,
        augment: bool = True,
        srf_weights=None,
        degradation_operator=None,
    ):
        super().__init__()
        self.img = img
        self.dataset_name = dataset_name
        self.patch_size = patch_size
        self.stride = stride
        self.scale_ratio = scale_ratio
        self.n_select_bands = n_select_bands
        self.split = split
        self.augment = augment and split == "train"
        self.srf_weights = srf_weights
        self.degradation_operator = degradation_operator or build_degradation(
            "gaussian_bicubic",
            scale_ratio=scale_ratio,
            sigma=2.0,
            kernel_size=5,
        )

        h, w, _ = img.shape
        self.test_rect = get_center_test_rect(h, w, test_size)
        self.coords = build_patch_coords(
            h=h,
            w=w,
            patch_size=patch_size,
            stride=stride,
            test_rect=self.test_rect,
            split=split,
        )

    def __len__(self):
        return len(self.coords)

    def random_augment(self, patch: np.ndarray) -> np.ndarray:
        if random.random() < 0.5:
            patch = np.flip(patch, axis=0)
        if random.random() < 0.5:
            patch = np.flip(patch, axis=1)
        if random.random() < 0.5:
            patch = np.rot90(patch, k=random.randint(1, 3), axes=(0, 1))
        return np.ascontiguousarray(patch)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        top, left = self.coords[index]
        gt = self.img[
            top:top + self.patch_size,
            left:left + self.patch_size,
            :,
        ].copy()

        if self.augment:
            gt = self.random_augment(gt)

        gt_tensor = hsi_to_tensor(gt)
        with torch.no_grad():
            lr_hsi = self.degradation_operator.degrade(
                gt_tensor.unsqueeze(0)
            ).squeeze(0)

        if self.srf_weights is not None:
            hr_msi = hsi_to_msi_numpy(gt, self.srf_weights)
        else:
            hr_msi = make_hr_msi(gt, self.n_select_bands)

        return {
            "lr_hsi": lr_hsi.contiguous().float(),
            "hr_msi": hsi_to_tensor(hr_msi),
            "gt": gt_tensor,
            "dataset_id": torch.tensor(0, dtype=torch.long),
            "n_bands": torch.tensor(gt.shape[2], dtype=torch.long),
        }


def _build_srf(cfg, n_bands: int):
    srf_weights = None
    srf_band_names = None
    hsi_wavelengths = None

    if getattr(cfg, "msi_mode", "uniform") != "srf":
        return srf_weights, srf_band_names, hsi_wavelengths, cfg.n_select_bands

    wavelength_path = (
        cfg.wavelength_path
        if cfg.wavelength_path
        else os.path.join(cfg.wavelength_root, f"{cfg.dataset}.txt")
    )
    hsi_wavelengths = load_hsi_wavelengths(
        wavelength_path=wavelength_path,
        n_bands=n_bands,
    )

    if cfg.srf_band_set == "wv2_visible5":
        selected_bands = WV2_VISIBLE_5_BANDS
    elif cfg.srf_band_set == "wv2_visible6":
        selected_bands = WV2_VISIBLE_6_BANDS
    elif cfg.srf_band_set == "wv2_all8":
        selected_bands = WV2_ALL_8_BANDS
    else:
        raise ValueError(f"Unsupported srf_band_set: {cfg.srf_band_set}")

    srf_weights, srf_band_names = build_srf_weights(
        srf_path=cfg.srf_path,
        hsi_wavelengths=hsi_wavelengths,
        selected_bands=selected_bands,
        interp_kind=cfg.srf_interp,
        normalize=True,
    )
    print_srf_summary(
        srf_weights=srf_weights,
        band_names=srf_band_names,
        hsi_wavelengths=hsi_wavelengths,
    )
    return srf_weights, srf_band_names, hsi_wavelengths, srf_weights.shape[0]


def build_datasets(cfg):
    dataset_cfg = cfg.datasets[cfg.dataset]
    file_path = os.path.join(cfg.data_root, dataset_cfg.file_name)

    img = read_hsi_mat(file_path, dataset_cfg.mat_keys)
    img = normalize_hsi(img)
    img = crop_to_scale(img, cfg.scale_ratio)

    n_bands = img.shape[2]
    print(f"Loaded {cfg.dataset}: shape={img.shape}, bands={n_bands}")

    srf_weights, srf_band_names, hsi_wavelengths, n_select_bands = _build_srf(
        cfg, n_bands
    )

    degradation_operator = build_hsi_degradation(cfg)
    progressive_degradation = build_progressive_degradation(
        cfg, degradation_operator=degradation_operator
    )

    train_set = HSIHSRDataset(
        img=img,
        dataset_name=cfg.dataset,
        patch_size=cfg.patch_size,
        stride=cfg.stride,
        scale_ratio=cfg.scale_ratio,
        n_select_bands=n_select_bands,
        srf_weights=srf_weights,
        degradation_operator=degradation_operator,
        split="train",
        test_size=cfg.image_size,
        augment=True,
    )

    test_set = HSIHSRDataset(
        img=img,
        dataset_name=cfg.dataset,
        patch_size=cfg.image_size,
        stride=cfg.image_size,
        scale_ratio=cfg.scale_ratio,
        n_select_bands=n_select_bands,
        srf_weights=srf_weights,
        degradation_operator=degradation_operator,
        split="test",
        test_size=cfg.image_size,
        augment=False,
    )

    info = {
        "dataset": cfg.dataset,
        "n_bands": n_bands,
        "n_select_bands": n_select_bands,
        "scale_ratio": cfg.scale_ratio,
        "train_samples": len(train_set),
        "test_samples": len(test_set),
        "degradation_mode": degradation_operator.mode,
        "progressive_steps": progressive_degradation.total_steps,
        "progressive_lift": progressive_degradation.default_lift_mode,
        "degradation_operator": degradation_operator,
        "progressive_degradation": progressive_degradation,
        "msi_mode": getattr(cfg, "msi_mode", "uniform"),
        "srf_weights": srf_weights,
        "srf_band_names": srf_band_names,
        "hsi_wavelengths": hsi_wavelengths,
    }

    return train_set, test_set, info


def build_loaders(cfg):
    train_set, test_set, info = build_datasets(cfg)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, test_loader, info
