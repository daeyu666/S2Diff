# data_loader.py
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from srf_utils import (
    WV2_VISIBLE_5_BANDS,
    WV2_VISIBLE_6_BANDS,
    WV2_ALL_8_BANDS,
    load_hsi_wavelengths,
    build_srf_weights,
    hsi_to_msi_numpy,
    print_srf_summary,
)


IKONOS_4_BANDS = [
    "IKONOS Blue",
    "IKONOS Green",
    "IKONOS Red",
    "IKONOS NIR",
]
WV2_SRF_PATH = "./data/srf/wv2_relative_spectral_response_data_for_i.atcorr.csv"
IKONOS_SRF_PATH = "./data/srf/ikonos_relative_spectral_response.csv"
PAVIA_NOMINAL_WAVELENGTH_PATH = "./data/wavelengths/PaviaU_nominal_430_860.txt"

try:
    import cv2
except ImportError:
    cv2 = None

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


def _extract_3d_numeric_array(value):
    """Recursively unwrap common MATLAB containers and return a 3-D array."""
    if isinstance(value, dict):
        for nested in value.values():
            found = _extract_3d_numeric_array(nested)
            if found is not None:
                return found
        return None

    if isinstance(value, (list, tuple)):
        for nested in value:
            found = _extract_3d_numeric_array(nested)
            if found is not None:
                return found
        return None

    if not isinstance(value, np.ndarray):
        return None

    arr = value

    # MATLAB structs/cells are often represented as a singleton object array.
    visited = 0
    while isinstance(arr, np.ndarray) and arr.dtype == object and arr.size == 1 and visited < 8:
        arr = np.asarray(arr.reshape(-1)[0])
        visited += 1

    # Structured arrays may contain the actual cube in one field.
    if isinstance(arr, np.ndarray) and arr.dtype.names:
        for field in arr.dtype.names:
            found = _extract_3d_numeric_array(arr[field])
            if found is not None:
                return found
        return None

    if not isinstance(arr, np.ndarray):
        return None

    arr = np.squeeze(arr)
    if arr.ndim == 3 and np.issubdtype(arr.dtype, np.number):
        return arr

    if arr.dtype == object:
        for nested in arr.reshape(-1):
            found = _extract_3d_numeric_array(nested)
            if found is not None:
                return found
    return None


def _find_cube_in_mapping(mapping, candidate_keys: List[str]):
    for key in candidate_keys:
        if key in mapping:
            found = _extract_3d_numeric_array(mapping[key])
            if found is not None:
                return found, key

    for key, value in mapping.items():
        if str(key).startswith("__"):
            continue
        found = _extract_3d_numeric_array(value)
        if found is not None:
            return found, str(key)
    return None, None


def _mapping_summary(mapping) -> str:
    items = []
    for key, value in mapping.items():
        if str(key).startswith("__"):
            continue
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        items.append(f"{key}:shape={shape},dtype={dtype},type={type(value).__name__}")
    return "; ".join(items[:20]) or "<no non-metadata keys>"


def read_hsi_mat(file_path: str, candidate_keys: List[str]) -> np.ndarray:
    """Read a MATLAB HSI cube and return HxWxC.

    Each available backend is tried independently. A backend that can open the
    file but cannot expose a usable 3-D array no longer prevents later backends
    from being attempted. This matters for some MATLAB v7/v7.3 files whose
    representation differs between hdf5storage, scipy.io and h5py.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find data file: {file_path}")

    diagnostics = []

    if hdf5storage is not None:
        try:
            mat_data = hdf5storage.loadmat(file_path)
            img, key = _find_cube_in_mapping(mat_data, candidate_keys)
            if img is not None:
                print(f"MAT reader: hdf5storage key={key}, raw_shape={img.shape}")
                return fix_hsi_shape(img)
            diagnostics.append(
                "hdf5storage opened file but found no 3-D cube; " + _mapping_summary(mat_data)
            )
        except Exception as exc:
            diagnostics.append(f"hdf5storage failed: {type(exc).__name__}: {exc}")

    if scio is not None:
        try:
            mat_data = scio.loadmat(file_path)
            img, key = _find_cube_in_mapping(mat_data, candidate_keys)
            if img is not None:
                print(f"MAT reader: scipy.io key={key}, raw_shape={img.shape}")
                return fix_hsi_shape(img)
            diagnostics.append(
                "scipy.io opened file but found no 3-D cube; " + _mapping_summary(mat_data)
            )
        except Exception as exc:
            diagnostics.append(f"scipy.io failed: {type(exc).__name__}: {exc}")

    if h5py is not None:
        try:
            with h5py.File(file_path, "r") as f:
                # Prefer exact top-level candidate keys.
                for key in candidate_keys:
                    if key in f and isinstance(f[key], h5py.Dataset):
                        arr = np.array(f[key])
                        arr = np.squeeze(arr)
                        if arr.ndim == 3 and np.issubdtype(arr.dtype, np.number):
                            print(f"MAT reader: h5py key={key}, raw_shape={arr.shape}")
                            return fix_hsi_shape(arr)

                found = []

                def visitor(name, obj):
                    if found or not isinstance(obj, h5py.Dataset):
                        return
                    try:
                        shape = tuple(obj.shape)
                        if len(shape) == 3 and np.issubdtype(obj.dtype, np.number):
                            found.append((name, np.array(obj)))
                    except Exception:
                        return

                f.visititems(visitor)
                if found:
                    name, arr = found[0]
                    print(f"MAT reader: h5py dataset={name}, raw_shape={arr.shape}")
                    return fix_hsi_shape(arr)

                top = []
                for key in f.keys():
                    obj = f[key]
                    top.append(
                        f"{key}:type={type(obj).__name__},shape={getattr(obj, 'shape', None)}"
                    )
                diagnostics.append(
                    "h5py opened file but found no numeric 3-D dataset; " + "; ".join(top[:20])
                )
        except Exception as exc:
            diagnostics.append(f"h5py failed: {type(exc).__name__}: {exc}")

    detail = "\n  - ".join(diagnostics) if diagnostics else "no MAT backend is installed"
    raise RuntimeError(
        f"No valid 3D HSI array found in {file_path}. Backend diagnostics:\n  - {detail}"
    )


def fix_hsi_shape(img: np.ndarray) -> np.ndarray:
    """
    将输入统一为H×W×C。
    部分v7.3 mat文件读出后可能是C×W×H或C×H×W，需要做简单判断。
    """
    img = np.array(img)
    img = np.squeeze(img)

    if img.ndim != 3:
        raise ValueError(f"HSI data must be 3D, but got shape: {img.shape}")

    if img.shape[0] <= 256 and img.shape[1] > 256 and img.shape[2] > 256:
        img = np.transpose(img, (1, 2, 0))
    elif img.shape[1] <= 256 and img.shape[0] > 256 and img.shape[2] > 256:
        img = np.transpose(img, (0, 2, 1))

    img = img.astype(np.float32)
    return img


def normalize_hsi(img: np.ndarray) -> np.ndarray:
    """归一化到[0,1]。"""
    img = img.astype(np.float32)
    min_value = float(np.min(img))
    max_value = float(np.max(img))

    if max_value - min_value < 1e-8:
        return np.zeros_like(img, dtype=np.float32)

    img = (img - min_value) / (max_value - min_value)
    return img.astype(np.float32)


def crop_to_scale(img: np.ndarray, scale_ratio: int) -> np.ndarray:
    """裁掉不能被scale_ratio整除的边缘。"""
    h, w, c = img.shape
    new_h = h // scale_ratio * scale_ratio
    new_w = w // scale_ratio * scale_ratio
    return img[:new_h, :new_w, :]


def gaussian_blur_bandwise(img: np.ndarray, kernel_size: int = 5, sigma: float = 2.0) -> np.ndarray:
    """对每个光谱波段分别做高斯模糊。"""
    if cv2 is None:
        return img

    blurred = np.zeros_like(img, dtype=np.float32)
    for i in range(img.shape[2]):
        blurred[:, :, i] = cv2.GaussianBlur(img[:, :, i], (kernel_size, kernel_size), sigma)
    return blurred


def resize_hsi(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """对H×W×C格式HSI逐波段resize。"""
    if cv2 is None:
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        tensor = torch.nn.functional.interpolate(
            tensor,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )
        return tensor.squeeze(0).permute(1, 2, 0).numpy().astype(np.float32)

    out = np.zeros((target_h, target_w, img.shape[2]), dtype=np.float32)
    for i in range(img.shape[2]):
        out[:, :, i] = cv2.resize(
            img[:, :, i],
            (target_w, target_h),
            interpolation=cv2.INTER_CUBIC,
        )
    return out


def make_lr_hsi(hr_hsi: np.ndarray, scale_ratio: int) -> np.ndarray:
    """HR-HSI -> GaussianBlur -> bicubic resize，生成LR-HSI。"""
    h, w, _ = hr_hsi.shape
    blurred = gaussian_blur_bandwise(hr_hsi, kernel_size=5, sigma=2.0)
    lr_hsi = resize_hsi(blurred, h // scale_ratio, w // scale_ratio)
    return lr_hsi.astype(np.float32)


def make_hr_msi(hr_hsi: np.ndarray, n_select_bands: int) -> np.ndarray:
    """关闭SRF模式时，从HR-HSI均匀抽取波段生成HR-MSI。"""
    n_bands = hr_hsi.shape[2]

    if n_select_bands > n_bands:
        raise ValueError(
            f"n_select_bands={n_select_bands} is larger than HSI bands={n_bands}"
        )

    band_indices = np.linspace(0, n_bands - 1, n_select_bands).round().astype(np.int64)
    hr_msi = hr_hsi[:, :, band_indices]
    return hr_msi.astype(np.float32)


def hsi_to_tensor(img: np.ndarray) -> torch.Tensor:
    """H×W×C -> C×H×W。"""
    return torch.from_numpy(img).permute(2, 0, 1).contiguous().float()


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
    """HSI-MSI融合超分数据集。"""

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

        lr_hsi = make_lr_hsi(gt, self.scale_ratio)
        if self.srf_weights is not None:
            hr_msi = hsi_to_msi_numpy(gt, self.srf_weights)
        else:
            hr_msi = make_hr_msi(gt, self.n_select_bands)

        sample = {
            "lr_hsi": hsi_to_tensor(lr_hsi),
            "hr_msi": hsi_to_tensor(hr_msi),
            "gt": hsi_to_tensor(gt),
            "dataset_id": torch.tensor(0, dtype=torch.long),
            "n_bands": torch.tensor(gt.shape[2], dtype=torch.long),
        }
        return sample


def _resolve_srf_spec(cfg, n_bands: int):
    """解析与对比实验一致的固定传感器协议。"""
    requested = getattr(cfg, "srf_band_set", "auto")
    if requested == "auto":
        resolved = "ikonos4" if cfg.dataset == "PaviaU" else "wv2_all8"
    else:
        resolved = requested

    if resolved == "ikonos4":
        selected_bands = IKONOS_4_BANDS
        default_srf_path = IKONOS_SRF_PATH
    elif resolved == "wv2_visible5":
        selected_bands = WV2_VISIBLE_5_BANDS
        default_srf_path = WV2_SRF_PATH
    elif resolved == "wv2_visible6":
        selected_bands = WV2_VISIBLE_6_BANDS
        default_srf_path = WV2_SRF_PATH
    elif resolved == "wv2_all8":
        selected_bands = WV2_ALL_8_BANDS
        default_srf_path = WV2_SRF_PATH
    else:
        raise ValueError(f"Unsupported srf_band_set: {resolved}")

    srf_path = getattr(cfg, "srf_path", "") or default_srf_path

    if getattr(cfg, "wavelength_path", ""):
        wavelength_path = cfg.wavelength_path
        hsi_wavelengths = load_hsi_wavelengths(
            wavelength_path=wavelength_path,
            n_bands=n_bands,
        )
    elif cfg.dataset == "PaviaU" and resolved == "ikonos4":
        wavelength_path = PAVIA_NOMINAL_WAVELENGTH_PATH
        if n_bands == 103 and os.path.exists(wavelength_path):
            hsi_wavelengths = load_hsi_wavelengths(
                wavelength_path=wavelength_path,
                n_bands=n_bands,
            )
        else:
            hsi_wavelengths = np.linspace(430.0, 860.0, n_bands).astype(np.float32)
            wavelength_path = f"nominal:430-860nm/{n_bands}bands"
    else:
        wavelength_path = os.path.join(cfg.wavelength_root, f"{cfg.dataset}.txt")
        hsi_wavelengths = load_hsi_wavelengths(
            wavelength_path=wavelength_path,
            n_bands=n_bands,
        )

    cfg.resolved_srf_band_set = resolved
    cfg.resolved_srf_path = srf_path
    cfg.resolved_wavelength_path = wavelength_path
    return srf_path, selected_bands, hsi_wavelengths, wavelength_path, resolved


def build_datasets(cfg):
    dataset_cfg = cfg.datasets[cfg.dataset]
    file_path = os.path.join(cfg.data_root, dataset_cfg.file_name)

    img = read_hsi_mat(file_path, dataset_cfg.mat_keys)
    img = normalize_hsi(img)
    img = crop_to_scale(img, cfg.scale_ratio)

    n_bands = img.shape[2]
    print(f"Loaded {cfg.dataset}: shape={img.shape}, bands={n_bands}")

    srf_weights = None
    srf_band_names = None
    hsi_wavelengths = None
    resolved_band_set = None
    resolved_srf_path = None
    resolved_wavelength_path = None

    if getattr(cfg, "msi_mode", "srf") == "srf":
        (
            resolved_srf_path,
            selected_bands,
            hsi_wavelengths,
            resolved_wavelength_path,
            resolved_band_set,
        ) = _resolve_srf_spec(cfg, n_bands)

        srf_weights, srf_band_names = build_srf_weights(
            srf_path=resolved_srf_path,
            hsi_wavelengths=hsi_wavelengths,
            selected_bands=selected_bands,
            interp_kind=cfg.srf_interp,
            normalize=True,
        )

        print(
            f"Resolved SRF: dataset={cfg.dataset}, profile={resolved_band_set}, "
            f"path={resolved_srf_path}, wavelength_grid={resolved_wavelength_path}"
        )
        print_srf_summary(
            srf_weights=srf_weights,
            band_names=srf_band_names,
            hsi_wavelengths=hsi_wavelengths,
        )

        n_select_bands = srf_weights.shape[0]
    else:
        n_select_bands = cfg.n_select_bands

    train_set = HSIHSRDataset(
        img=img,
        dataset_name=cfg.dataset,
        patch_size=cfg.patch_size,
        stride=cfg.stride,
        scale_ratio=cfg.scale_ratio,
        n_select_bands=n_select_bands,
        srf_weights=srf_weights,
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
        "msi_mode": getattr(cfg, "msi_mode", "srf"),
        "srf_profile": resolved_band_set,
        "srf_path": resolved_srf_path,
        "wavelength_path": resolved_wavelength_path,
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
