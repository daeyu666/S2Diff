"""Progressive sensor-degradation trajectory for HSI super-resolution."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

from .base import BaseDegradation


@dataclass(frozen=True)
class ProgressiveState:
    t: int
    strength: float
    scale: int


def _unique_scales(scale_ratio: int) -> List[int]:
    """Default staged scales. v1 is designed around 1 -> 2 -> 4."""
    if scale_ratio < 1:
        raise ValueError("scale_ratio must be >= 1")
    if scale_ratio == 1:
        return [1]

    stages = [1]
    current = 2
    while current < scale_ratio:
        if scale_ratio % current == 0:
            stages.append(current)
        current *= 2
    stages.append(scale_ratio)

    out = []
    for value in stages:
        if value not in out:
            out.append(value)
    return out


class ProgressiveDegradation:
    """Wrap a terminal degradation operator with staged D_t and HR-grid lift.

    t is integer in [0, T].
    strength follows t/T.
    scale is piecewise constant over the configured stages.

    For T=12 and stages=(1, 2, 4):
        t=0      -> scale 1, strength 0
        t=1..4   -> scale 1
        t=5..8   -> scale 2
        t=9..12  -> scale 4
    """

    def __init__(
        self,
        operator: BaseDegradation,
        total_steps: int = 12,
        stages: Optional[Sequence[int]] = None,
        default_lift_mode: Optional[str] = None,
    ):
        if total_steps < 1:
            raise ValueError("total_steps must be >= 1")
        self.operator = operator
        self.total_steps = int(total_steps)
        self.stages = list(stages) if stages is not None else _unique_scales(
            operator.scale_ratio
        )
        if not self.stages or self.stages[0] != 1:
            raise ValueError("stages must start from 1")
        if self.stages[-1] != operator.scale_ratio:
            raise ValueError(
                "last stage must equal operator.scale_ratio "
                f"({operator.scale_ratio})"
            )
        if any(s < 1 for s in self.stages):
            raise ValueError("all stages must be >= 1")
        if default_lift_mode is None:
            default_lift_mode = (
                "normalized_adjoint"
                if operator.mode == "physical"
                else "bilinear"
            )
        self.default_lift_mode = default_lift_mode

    def _validate_t(self, t: int) -> int:
        t = int(t)
        if t < 0 or t > self.total_steps:
            raise ValueError(
                f"t must lie in [0, {self.total_steps}], got {t}"
            )
        return t

    def state(self, t: int) -> ProgressiveState:
        t = self._validate_t(t)
        strength = float(t) / float(self.total_steps)
        if t == 0:
            scale = 1
        else:
            idx = min(
                ((t - 1) * len(self.stages)) // self.total_steps,
                len(self.stages) - 1,
            )
            scale = int(self.stages[idx])
        return ProgressiveState(t=t, strength=strength, scale=scale)

    def degrade_at(self, x: torch.Tensor, t: int) -> torch.Tensor:
        s = self.state(t)
        return self.operator.degrade_at(
            x, scale=s.scale, strength=s.strength
        )

    def lift_at(
        self,
        y: torch.Tensor,
        t: int,
        *,
        target_size: Tuple[int, int],
        lift_mode: Optional[str] = None,
    ) -> torch.Tensor:
        s = self.state(t)
        return self.operator.lift(
            y,
            scale=s.scale,
            strength=s.strength,
            lift_mode=lift_mode or self.default_lift_mode,
            target_size=target_size,
        )

    def state_at(
        self,
        x: torch.Tensor,
        t: int,
        *,
        lift_mode: Optional[str] = None,
    ) -> torch.Tensor:
        target_size = tuple(x.shape[-2:])
        y_t = self.degrade_at(x, t)
        return self.lift_at(
            y_t, t, target_size=target_size, lift_mode=lift_mode
        )

    def terminal_observation(self, x: torch.Tensor) -> torch.Tensor:
        """Dataset LR-HSI generator. Must equal degrade_at(x, T)."""
        return self.operator.degrade(x)

    def terminal_state(
        self,
        y_terminal: torch.Tensor,
        *,
        target_size: Tuple[int, int],
        lift_mode: Optional[str] = None,
    ) -> torch.Tensor:
        """Initialize inference from the actually observed LR-HSI."""
        return self.lift_at(
            y_terminal,
            self.total_steps,
            target_size=target_size,
            lift_mode=lift_mode,
        )

    def reverse_update(
        self,
        x_t: torch.Tensor,
        x0_hat: torch.Tensor,
        t: int,
        *,
        lift_mode: Optional[str] = None,
    ) -> torch.Tensor:
        """Physics-consistent deterministic reverse increment.

        x_{t-1} = x_t + D~_{t-1}(x0_hat) - D~_t(x0_hat)
        where D~_t = U_t o D_t and every term lives on the HR grid.
        """
        t = self._validate_t(t)
        if t == 0:
            return x_t
        if tuple(x_t.shape) != tuple(x0_hat.shape):
            raise ValueError(
                "x_t and x0_hat must have identical BxCxHxW shape"
            )
        previous = self.state_at(
            x0_hat, t - 1, lift_mode=lift_mode
        )
        current = self.state_at(
            x0_hat, t, lift_mode=lift_mode
        )
        return x_t + previous - current

    def transition_timesteps(self, radius: int = 1) -> List[int]:
        """Return t values around staged scale transitions."""
        if radius < 0:
            raise ValueError("radius must be >= 0")

        scales = [self.state(t).scale for t in range(self.total_steps + 1)]
        centers = []
        for t in range(1, self.total_steps + 1):
            if scales[t] != scales[t - 1]:
                centers.extend([t - 1, t])

        selected = set()
        for center in centers:
            for dt in range(-radius, radius + 1):
                value = center + dt
                if 1 <= value <= self.total_steps:
                    selected.add(value)
        return sorted(selected)

    def sample_timesteps(
        self,
        batch_size: int,
        *,
        boundary_probability: float = 0.2,
        boundary_radius: int = 1,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Sample t with mostly-uniform coverage plus transition emphasis."""
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if not 0.0 <= boundary_probability <= 1.0:
            raise ValueError("boundary_probability must lie in [0, 1]")

        boundary = self.transition_timesteps(radius=boundary_radius)
        values = []
        for _ in range(batch_size):
            if boundary and random.random() < boundary_probability:
                values.append(random.choice(boundary))
            else:
                values.append(random.randint(1, self.total_steps))
        return torch.tensor(values, dtype=torch.long, device=device)

    def assert_terminal_closure(
        self,
        x: torch.Tensor,
        *,
        atol: float = 1e-6,
        rtol: float = 1e-5,
    ) -> None:
        direct = self.terminal_observation(x)
        progressive = self.degrade_at(x, self.total_steps)
        if not torch.allclose(direct, progressive, atol=atol, rtol=rtol):
            max_error = (direct - progressive).abs().max().item()
            raise AssertionError(
                "Terminal degradation closure failed: "
                f"max_abs_error={max_error:.6e}"
            )
