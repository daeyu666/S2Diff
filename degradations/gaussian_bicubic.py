"""Legacy Gaussian-blur + bicubic-resize HSI degradation baseline."""

from __future__ import annotations

import torch

from .base import BaseDegradation
from .common import depthwise_psf, resize_down


class GaussianBicubicDegradation(BaseDegradation):
    mode = "gaussian_bicubic"

    def __init__(
        self,
        scale_ratio: int = 4,
        sigma: float = 2.0,
        kernel_size: int = 5,
    ):
        super().__init__(scale_ratio=scale_ratio)
        if sigma < 0:
            raise ValueError("sigma must be >= 0")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.sigma = float(sigma)
        self.kernel_size = int(kernel_size)

    def degrade_at(
        self, x: torch.Tensor, *, scale: int, strength: float
    ) -> torch.Tensor:
        strength = float(min(max(strength, 0.0), 1.0))
        sigma_t = self.sigma * strength
        blurred = depthwise_psf(
            x, sigma_t, kernel_size=self.kernel_size
        )
        return resize_down(
            blurred, scale, mode="bicubic", antialias=True
        )

    def extra_repr(self) -> str:
        return (
            super().extra_repr()
            + f", sigma={self.sigma}, kernel_size={self.kernel_size}"
        )
