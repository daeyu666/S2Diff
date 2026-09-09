"""S2Diff non-registration training with warp-before-crop HR-MSI context.

For train samples only, construct a larger HR parent around each 64x64 target,
generate HR-MSI on that parent, apply the existing radial translation warp, and
center-crop the warped MSI back to the 64x64 model input.  GT-HSI and the
progressive HSI trajectory remain unchanged.  Parent regions touching the held-
out center test region are excluded.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

import data_loader as shared_data
from srf_utils import hsi_to_msi_numpy


def _cli_float(name: str, default: float) -> float:
    flag = f"--{name}"
    argv = sys.argv[1:]
    for i, token in enumerate(argv):
        if token == flag and i + 1 < len(argv):
            return float(argv[i + 1])
        if token.startswith(flag + "="):
            return float(token.split("=", 1)[1])
    return float(default)


def _cli_int(name: str, default: int) -> int:
    return int(round(_cli_float(name, float(default))))


TRAIN_SHIFT = _cli_float("train_msi_translation_max_px", 0.0)
PATCH_SIZE = _cli_int("patch_size", 64)
CONTEXT_MARGIN = int(math.ceil(TRAIN_SHIFT)) + 2 if TRAIN_SHIFT > 0.0 else 0
OriginalDataset = shared_data.HSIHSRDataset


class ContextHSIHSRDataset(OriginalDataset):
    """Larger MSI parent for train split; test split is unchanged."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.misalignment_context_margin = CONTEXT_MARGIN if self.split == "train" else 0
        m = self.misalignment_context_margin
        if m <= 0:
            return

        h, w, _ = self.img.shape
        kept = []
        for top, left in self.coords:
            context_rect = (
                top - m,
                left - m,
                top + self.patch_size + m,
                left + self.patch_size + m,
            )
            if context_rect[0] < 0 or context_rect[1] < 0:
                continue
            if context_rect[2] > h or context_rect[3] > w:
                continue
            if shared_data.intersects(context_rect, self.test_rect):
                continue
            kept.append((top, left))

        if not kept:
            raise RuntimeError(
                f"No train patches remain after applying context margin={m}px"
            )
        print(
            f"Warp-before-crop train context: margin={m}px, "
            f"parent={self.patch_size + 2*m}x{self.patch_size + 2*m}, "
            f"target={self.patch_size}x{self.patch_size}, "
            f"patches={len(self.coords)}->{len(kept)}"
        )
        self.coords = kept

    def __getitem__(self, index: int):
        m = self.misalignment_context_margin
        if self.split != "train" or m <= 0:
            return super().__getitem__(index)

        top, left = self.coords[index]
        p = self.patch_size
        parent = self.img[
            top - m : top + p + m,
            left - m : left + p + m,
            :,
        ].copy()
        if self.augment:
            parent = self.random_augment(parent)

        gt = parent[m : m + p, m : m + p, :].copy()
        lr_hsi = shared_data.make_lr_hsi(gt, self.scale_ratio)
        if self.srf_weights is not None:
            hr_msi_parent = hsi_to_msi_numpy(parent, self.srf_weights)
        else:
            hr_msi_parent = shared_data.make_hr_msi(parent, self.n_select_bands)

        return {
            "lr_hsi": shared_data.hsi_to_tensor(lr_hsi),
            "hr_msi": shared_data.hsi_to_tensor(hr_msi_parent),
            "gt": shared_data.hsi_to_tensor(gt),
            "dataset_id": torch.tensor(0, dtype=torch.long),
            "n_bands": torch.tensor(gt.shape[2], dtype=torch.long),
        }


shared_data.HSIHSRDataset = ContextHSIHSRDataset

import innovation1 as innovation_engine  # noqa: E402

_original_augment = innovation_engine.augment_training_msi_translation


def _center_crop(x: torch.Tensor, size: int) -> torch.Tensor:
    h, w = x.shape[-2:]
    if h == size and w == size:
        return x
    if h < size or w < size:
        raise ValueError(f"Cannot center-crop {(h, w)} to {(size, size)}")
    top = (h - size) // 2
    left = (w - size) // 2
    return x[..., top : top + size, left : left + size]


def augment_warp_before_crop(hr_msi, *, max_shift_px, probability=1.0, generator=None):
    warped, magnitude = _original_augment(
        hr_msi,
        max_shift_px=max_shift_px,
        probability=probability,
        generator=generator,
    )
    expected_parent = PATCH_SIZE + 2 * CONTEXT_MARGIN
    if CONTEXT_MARGIN > 0 and tuple(hr_msi.shape[-2:]) == (expected_parent, expected_parent):
        warped = _center_crop(warped, PATCH_SIZE)
    return warped, magnitude


innovation_engine.augment_training_msi_translation = augment_warp_before_crop

import main as s2_main  # noqa: E402


if __name__ == "__main__":
    print(
        "S2Diff misalignment geometry: warp HR-MSI parent first, then center-crop "
        f"to {PATCH_SIZE}x{PATCH_SIZE}; context_margin={CONTEXT_MARGIN}px."
    )
    s2_main.main()
