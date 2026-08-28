"""Shared low-level operators for HSI degradation experiments."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def validate_hsi_tensor(x: torch.Tensor) -> None:
    if x.ndim != 4:
        raise ValueError(f"Expected BxCxHxW tensor, got shape={tuple(x.shape)}")
    if not torch.is_floating_point(x):
        raise TypeError(f"Expected floating tensor, got dtype={x.dtype}")


def validate_scale(x: torch.Tensor, scale: int) -> None:
    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale}")
    h, w = x.shape[-2:]
    if h % scale != 0 or w % scale != 0:
        raise ValueError(
            f"Spatial size {(h, w)} must be divisible by scale={scale}"
        )


def gaussian_kernel2d(
    sigma: float,
    *,
    dtype: torch.dtype,
    device: torch.device,
    kernel_size: Optional[int] = None,
    truncate: float = 3.0,
) -> Optional[torch.Tensor]:
    """Build a normalized symmetric 2-D Gaussian kernel.

    Returns None when sigma is effectively zero, which represents identity PSF.
    """
    if sigma <= 1e-8:
        return None

    if kernel_size is None:
        radius = max(1, int(math.ceil(truncate * sigma)))
        kernel_size = 2 * radius + 1
    else:
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        radius = kernel_size // 2

    coords = torch.arange(
        -radius, radius + 1, dtype=dtype, device=device
    )
    g = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    g = g / g.sum().clamp_min(torch.finfo(dtype).eps)
    kernel = torch.outer(g, g)
    kernel = kernel / kernel.sum().clamp_min(torch.finfo(dtype).eps)
    return kernel


def depthwise_psf(
    x: torch.Tensor,
    sigma: float,
    *,
    kernel_size: Optional[int] = None,
    truncate: float = 3.0,
) -> torch.Tensor:
    """Apply a zero-padded symmetric Gaussian PSF band-wise.

    Zero padding is intentional: with a symmetric kernel this linear operator
    has a simple self-adjoint implementation. The normalized-adjoint lift
    corrects the resulting boundary attenuation.
    """
    validate_hsi_tensor(x)
    kernel = gaussian_kernel2d(
        sigma,
        dtype=x.dtype,
        device=x.device,
        kernel_size=kernel_size,
        truncate=truncate,
    )
    if kernel is None:
        return x

    c = x.shape[1]
    k = kernel.shape[-1]
    weight = kernel.view(1, 1, k, k).repeat(c, 1, 1, 1)
    return F.conv2d(x, weight, padding=k // 2, groups=c)


def area_average_downsample(x: torch.Tensor, scale: int) -> torch.Tensor:
    """Pixel-area averaging + stride sampling.

    This is the code-level realization of A_r followed by S_r.
    """
    validate_hsi_tensor(x)
    validate_scale(x, scale)
    if scale == 1:
        return x
    return F.avg_pool2d(x, kernel_size=scale, stride=scale)


def area_average_adjoint(y: torch.Tensor, scale: int) -> torch.Tensor:
    """Exact transpose of non-overlapping avg_pool2d(kernel=stride=scale).

    Each LR value is distributed over its corresponding HR footprint with
    coefficient 1 / scale^2.
    """
    validate_hsi_tensor(y)
    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale}")
    if scale == 1:
        return y

    c = y.shape[1]
    kernel = torch.ones(
        (c, 1, scale, scale), dtype=y.dtype, device=y.device
    ) / float(scale * scale)
    return F.conv_transpose2d(y, kernel, stride=scale, groups=c)


def resize_down(
    x: torch.Tensor,
    scale: int,
    *,
    mode: str = "bicubic",
    antialias: bool = True,
) -> torch.Tensor:
    validate_hsi_tensor(x)
    validate_scale(x, scale)
    if scale == 1:
        return x
    h, w = x.shape[-2:]
    kwargs = {}
    if mode in ("bilinear", "bicubic"):
        kwargs["align_corners"] = False
        kwargs["antialias"] = antialias
    return F.interpolate(
        x, size=(h // scale, w // scale), mode=mode, **kwargs
    )


def resize_up(
    y: torch.Tensor,
    scale: int,
    *,
    mode: str = "bilinear",
    target_size: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    validate_hsi_tensor(y)
    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale}")
    if target_size is None:
        target_size = (y.shape[-2] * scale, y.shape[-1] * scale)
    if tuple(y.shape[-2:]) == tuple(target_size):
        return y

    kwargs = {}
    if mode in ("bilinear", "bicubic"):
        kwargs["align_corners"] = False
    return F.interpolate(y, size=target_size, mode=mode, **kwargs)


def assert_spatial_size(
    x: torch.Tensor, expected_size: Tuple[int, int], name: str
) -> None:
    if tuple(x.shape[-2:]) != tuple(expected_size):
        raise RuntimeError(
            f"{name} produced spatial size {tuple(x.shape[-2:])}, "
            f"expected {tuple(expected_size)}"
        )
