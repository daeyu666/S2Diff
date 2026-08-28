"""Benchmark-compatible physical HSI degradation and backprojection."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

from .base import BaseDegradation
from .common import (
    area_average_adjoint,
    area_average_downsample,
    assert_spatial_size,
    depthwise_psf,
    resize_up,
    validate_hsi_tensor,
)


def sigma_from_mtf_nyquist(scale_ratio: int, mtf_nyquist: float) -> float:
    """Convert MTF at the LR Nyquist frequency to Gaussian PSF sigma.

    On the HR grid the LR Nyquist frequency is f = 1 / (2 * scale_ratio).
    For a Gaussian PSF:
        MTF(f) = exp(-2*pi^2*sigma^2*f^2)
    hence:
        sigma = scale_ratio/pi * sqrt(-2*ln(MTF_Nyq)).
    """
    if scale_ratio < 1:
        raise ValueError("scale_ratio must be >= 1")
    if not 0.0 < mtf_nyquist < 1.0:
        raise ValueError("mtf_nyquist must lie strictly between 0 and 1")
    return (
        float(scale_ratio)
        / math.pi
        * math.sqrt(-2.0 * math.log(float(mtf_nyquist)))
    )


class PhysicalDegradation(BaseDegradation):
    """Shared-PSF spatial degradation with detector area integration.

    Forward model at a given progressive state:
        D_t(x) = B_{r_t} P_{sigma_t} x
    where B combines pixel-area averaging and stride sampling.

    The strict adjoint is:
        D_t^*(y) = P_{sigma_t} B_{r_t}^T y

    The recommended lift is response-normalized adjoint:
        U_t(y) = D_t^*(y) / (D_t^* D_t(1_HR) + eps)

    Using D_t^* D_t(1_HR) rather than D_t^*(1_LR) makes the composed
    diffusion state U_t D_t preserve a constant HR field even with the
    zero-padded PSF boundary convention.
    """

    mode = "physical"

    def __init__(
        self,
        scale_ratio: int = 4,
        mtf_nyquist: float = 0.2,
        truncate: float = 3.0,
    ):
        super().__init__(scale_ratio=scale_ratio)
        if truncate <= 0:
            raise ValueError("truncate must be > 0")
        self.mtf_nyquist = float(mtf_nyquist)
        self.truncate = float(truncate)
        self.terminal_sigma = sigma_from_mtf_nyquist(
            scale_ratio, mtf_nyquist
        )

    def sigma_at_strength(self, strength: float) -> float:
        strength = float(min(max(strength, 0.0), 1.0))
        return self.terminal_sigma * strength

    def degrade_at(
        self, x: torch.Tensor, *, scale: int, strength: float
    ) -> torch.Tensor:
        validate_hsi_tensor(x)
        sigma_t = self.sigma_at_strength(strength)
        optical = depthwise_psf(
            x, sigma_t, truncate=self.truncate
        )
        return area_average_downsample(optical, scale)

    def adjoint_at(
        self, y: torch.Tensor, *, scale: int, strength: float
    ) -> torch.Tensor:
        """Apply D_t^* for the physical forward operator."""
        validate_hsi_tensor(y)
        sigma_t = self.sigma_at_strength(strength)
        backprojected = area_average_adjoint(y, scale)
        return depthwise_psf(
            backprojected, sigma_t, truncate=self.truncate
        )

    def lift(
        self,
        y: torch.Tensor,
        *,
        scale: int,
        strength: float,
        lift_mode: str,
        target_size: Optional[Tuple[int, int]] = None,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        validate_hsi_tensor(y)

        if target_size is None:
            target_size = (
                y.shape[-2] * scale,
                y.shape[-1] * scale,
            )

        if lift_mode in ("bilinear", "nearest"):
            out = resize_up(
                y, scale, mode=lift_mode, target_size=target_size
            )
            assert_spatial_size(out, target_size, "interpolation lift")
            return out

        if lift_mode == "adjoint":
            out = self.adjoint_at(
                y, scale=scale, strength=strength
            )
            assert_spatial_size(out, target_size, "adjoint lift")
            return out

        if lift_mode == "normalized_adjoint":
            numerator = self.adjoint_at(
                y, scale=scale, strength=strength
            )

            # Normalize by the full system response to a constant HR field:
            # D_t^* D_t(1_HR). This preserves constant fields under
            # U_t D_t even when the PSF uses zero padding at image borders.
            ones_hr = torch.ones(
                (y.shape[0], y.shape[1], target_size[0], target_size[1]),
                dtype=y.dtype,
                device=y.device,
            )
            ones_observation = self.degrade_at(
                ones_hr, scale=scale, strength=strength
            )
            denominator = self.adjoint_at(
                ones_observation, scale=scale, strength=strength
            )
            out = numerator / denominator.clamp_min(eps)
            assert_spatial_size(
                out, target_size, "normalized-adjoint lift"
            )
            return out

        raise ValueError(
            "lift_mode must be one of: bilinear, nearest, adjoint, "
            "normalized_adjoint"
        )

    def extra_repr(self) -> str:
        return (
            super().extra_repr()
            + f", mtf_nyquist={self.mtf_nyquist}, "
            f"terminal_sigma={self.terminal_sigma:.6f}, "
            f"truncate={self.truncate}"
        )
