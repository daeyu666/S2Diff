"""Time-conditioned clean-HSI predictor for Innovation 1.

The predictor receives only the HR-grid progressive degradation state x_t and
its timestep t.  MSI is intentionally excluded in this stage so that the
sensor-degradation-consistent diffusion mechanism can be evaluated in
isolation.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _num_groups(channels: int, max_groups: int = 8) -> int:
    """Choose the largest GroupNorm group count that divides channels."""
    upper = min(int(max_groups), int(channels))
    for groups in range(upper, 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    """Continuous sinusoidal embedding for normalized diffusion time."""

    def __init__(self, dim: int):
        super().__init__()
        if dim < 4:
            raise ValueError("time embedding dim must be >= 4")
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim != 1:
            raise ValueError(f"t must have shape [B], got {tuple(t.shape)}")

        half = self.dim // 2
        denom = max(half - 1, 1)
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / float(denom)
        )
        angles = t.float().unsqueeze(1) * frequencies.unsqueeze(0)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if emb.shape[1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[1]))
        return emb


class TimeConditionedResBlock(nn.Module):
    """Residual block with FiLM modulation from the timestep embedding."""

    def __init__(
        self,
        channels: int,
        time_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        groups = _num_groups(channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, channels * 2),
        )
        self.norm2 = nn.GroupNorm(groups, channels)
        self.dropout = nn.Dropout(float(dropout))
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        # Start each residual branch near identity.  Together with the global
        # x_t skip this makes the initial clean estimate equal to x_t.
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time_proj(time_emb).chunk(2, dim=1)
        h = self.norm2(h)
        h = h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(h)))
        return x + h


class CleanHSIPredictor(nn.Module):
    """Compact U-Net-like predictor F_theta(x_t, t) -> X_0.

    The network uses 2-D spatial convolutions with spectral bands as channels.
    A 1x1 spectral projection maps the dataset-specific band count into a fixed
    feature width.  Time is injected through FiLM residual blocks at all
    resolutions.

    By default the output is parameterized as x_t + residual.  This is still a
    direct clean-HSI prediction, while providing a stable identity starting
    point before learning the missing high-frequency content.
    """

    def __init__(
        self,
        n_bands: int,
        total_steps: int = 12,
        base_channels: int = 64,
        time_dim: int = 256,
        dropout: float = 0.0,
        residual_prediction: bool = True,
    ):
        super().__init__()
        if n_bands < 1:
            raise ValueError("n_bands must be >= 1")
        if total_steps < 1:
            raise ValueError("total_steps must be >= 1")
        if base_channels < 8:
            raise ValueError("base_channels must be >= 8")

        self.n_bands = int(n_bands)
        self.total_steps = int(total_steps)
        self.residual_prediction = bool(residual_prediction)

        c1 = int(base_channels)
        c2 = c1 * 2
        c3 = c1 * 4

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.in_proj = nn.Conv2d(self.n_bands, c1, kernel_size=3, padding=1)

        self.enc1 = nn.ModuleList([
            TimeConditionedResBlock(c1, time_dim, dropout),
            TimeConditionedResBlock(c1, time_dim, dropout),
        ])
        self.down1 = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1)

        self.enc2 = nn.ModuleList([
            TimeConditionedResBlock(c2, time_dim, dropout),
            TimeConditionedResBlock(c2, time_dim, dropout),
        ])
        self.down2 = nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1)

        self.mid = nn.ModuleList([
            TimeConditionedResBlock(c3, time_dim, dropout),
            TimeConditionedResBlock(c3, time_dim, dropout),
        ])

        self.up2_proj = nn.Conv2d(c3, c2, kernel_size=3, padding=1)
        self.up2_fuse = nn.Conv2d(c2 + c2, c2, kernel_size=1)
        self.dec2 = nn.ModuleList([
            TimeConditionedResBlock(c2, time_dim, dropout),
            TimeConditionedResBlock(c2, time_dim, dropout),
        ])

        self.up1_proj = nn.Conv2d(c2, c1, kernel_size=3, padding=1)
        self.up1_fuse = nn.Conv2d(c1 + c1, c1, kernel_size=1)
        self.dec1 = nn.ModuleList([
            TimeConditionedResBlock(c1, time_dim, dropout),
            TimeConditionedResBlock(c1, time_dim, dropout),
        ])

        self.out_norm = nn.GroupNorm(_num_groups(c1), c1)
        self.out_conv = nn.Conv2d(c1, self.n_bands, kernel_size=3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def _time_embedding(self, t: torch.Tensor, batch_size: int) -> torch.Tensor:
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=self.out_conv.weight.device)
        if t.ndim == 0:
            t = t.repeat(batch_size)
        if t.ndim != 1 or t.shape[0] != batch_size:
            raise ValueError(
                f"t must be scalar or shape [B={batch_size}], got {tuple(t.shape)}"
            )
        # Map t in [0, T] to a numerically useful continuous range [0, 1000].
        t_scaled = t.float() / float(self.total_steps) * 1000.0
        return self.time_embed(t_scaled)

    @staticmethod
    def _apply_blocks(
        x: torch.Tensor,
        blocks: nn.ModuleList,
        time_emb: torch.Tensor,
    ) -> torch.Tensor:
        for block in blocks:
            x = block(x, time_emb)
        return x

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x_t.ndim != 4:
            raise ValueError(f"x_t must be BxCxHxW, got {tuple(x_t.shape)}")
        if x_t.shape[1] != self.n_bands:
            raise ValueError(
                f"expected {self.n_bands} spectral bands, got {x_t.shape[1]}"
            )

        time_emb = self._time_embedding(t, x_t.shape[0])
        input_state = x_t

        x = self.in_proj(x_t)
        skip1 = self._apply_blocks(x, self.enc1, time_emb)

        x = self.down1(skip1)
        skip2 = self._apply_blocks(x, self.enc2, time_emb)

        x = self.down2(skip2)
        x = self._apply_blocks(x, self.mid, time_emb)

        x = F.interpolate(x, size=skip2.shape[-2:], mode="nearest")
        x = self.up2_proj(x)
        x = self.up2_fuse(torch.cat([x, skip2], dim=1))
        x = self._apply_blocks(x, self.dec2, time_emb)

        x = F.interpolate(x, size=skip1.shape[-2:], mode="nearest")
        x = self.up1_proj(x)
        x = self.up1_fuse(torch.cat([x, skip1], dim=1))
        x = self._apply_blocks(x, self.dec1, time_emb)

        residual = self.out_conv(F.silu(self.out_norm(x)))
        if self.residual_prediction:
            return input_state + residual
        return residual
