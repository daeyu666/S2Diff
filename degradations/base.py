"""Base interface for switchable HSI degradation operators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .common import resize_up, validate_hsi_tensor


class BaseDegradation(nn.Module, ABC):
    """Common interface used by ordinary and physical degradation baselines."""

    mode: str = "base"

    def __init__(self, scale_ratio: int = 4):
        super().__init__()
        if scale_ratio < 1:
            raise ValueError("scale_ratio must be >= 1")
        self.scale_ratio = int(scale_ratio)

    def degrade(self, x: torch.Tensor) -> torch.Tensor:
        """Terminal observation operator D_T."""
        return self.degrade_at(x, scale=self.scale_ratio, strength=1.0)

    @abstractmethod
    def degrade_at(
        self, x: torch.Tensor, *, scale: int, strength: float
    ) -> torch.Tensor:
        """Apply the operator at an intermediate scale/strength."""
        raise NotImplementedError

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
        """Lift an LR state back to the common HR computational grid.

        Ordinary degradation modes intentionally support interpolation only.
        Strict adjoint variants are implemented only by PhysicalDegradation.
        """
        del strength, eps
        validate_hsi_tensor(y)
        if lift_mode not in ("bilinear", "nearest"):
            raise ValueError(
                f"{self.__class__.__name__} does not define strict "
                f"{lift_mode!r} lifting; use bilinear/nearest or physical mode."
            )
        return resize_up(
            y, scale, mode=lift_mode, target_size=target_size
        )

    def extra_repr(self) -> str:
        return f"mode={self.mode}, scale_ratio={self.scale_ratio}"
