"""Plain bicubic HSI degradation baseline."""

from __future__ import annotations

import torch

from .base import BaseDegradation
from .common import resize_down


class BicubicDegradation(BaseDegradation):
    mode = "bicubic"

    def degrade_at(
        self, x: torch.Tensor, *, scale: int, strength: float
    ) -> torch.Tensor:
        del strength
        return resize_down(x, scale, mode="bicubic", antialias=True)
