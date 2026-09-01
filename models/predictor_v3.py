"""Innovation 2 predictor: spectrally safer HR-MSI high-frequency guidance.

The HSI branch keeps the V2 spectral-spatial backbone. HR-MSI is not directly
concatenated with HSI. Instead:

1) a fixed Gaussian high-pass extracts candidate MSI spatial details;
2) the high-frequency MSI is encoded at the same three U-Net resolutions;
3) a state/time-conditioned transfer gate decides how much MSI detail is
   injected into each HSI feature level;
4) an explicit alpha_t=t/T schedule makes guidance strongest at the beginning
   of reverse inference and progressively weaker near t=0.

This is the first implementation of Innovation 2. It intentionally keeps the
training loss unchanged (L1 + SAM) so that the benefit of MSI spatial evidence
can be isolated before adding extra spectral-consistency losses.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .predictor_v2 import (
    EMRInspiredTimeBlock,
    LocalSpectralStem,
    SinusoidalTimeEmbedding,
    _num_groups,
)


class FixedGaussianHighPass(nn.Module):
    """Extract MSI high-frequency residual M_HF = M - G_sigma(M)."""

    def __init__(self, kernel_size: int = 5, sigma: float = 1.0):
        super().__init__()
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd and >= 3")
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
        radius = kernel_size // 2
        coord = torch.arange(-radius, radius + 1, dtype=torch.float32)
        kernel_1d = torch.exp(-(coord ** 2) / (2.0 * sigma * sigma))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel_2d = kernel_2d / kernel_2d.sum()
        self.kernel_size = int(kernel_size)
        self.register_buffer("kernel", kernel_2d[None, None], persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"MSI must be BxCxHxW, got {tuple(x.shape)}")
        c = x.shape[1]
        pad = self.kernel_size // 2
        weight = self.kernel.expand(c, 1, -1, -1).to(dtype=x.dtype)
        padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        low = F.conv2d(padded, weight, groups=c)
        return x - low


class MSITransferGate(nn.Module):
    """Condition MSI-HF transfer on the current HSI feature and diffusion time."""

    def __init__(self, channels: int, time_dim: int):
        super().__init__()
        groups = _num_groups(channels)
        self.hsi_norm = nn.GroupNorm(groups, channels)
        self.msi_norm = nn.GroupNorm(groups, channels)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, channels),
        )

        # Start conservatively: sigmoid(-2) ~= 0.119 before learning.
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -2.0)
        nn.init.zeros_(self.time_proj[-1].weight)
        nn.init.zeros_(self.time_proj[-1].bias)

    def forward(
        self,
        hsi: torch.Tensor,
        msi_hf: torch.Tensor,
        time_emb: torch.Tensor,
        alpha_t: torch.Tensor,
    ) -> torch.Tensor:
        if hsi.shape != msi_hf.shape:
            raise ValueError(
                f"HSI/MSI feature shapes must match, got {hsi.shape} and {msi_hf.shape}"
            )
        logits = self.gate(
            torch.cat([self.hsi_norm(hsi), self.msi_norm(msi_hf)], dim=1)
        )
        logits = logits + self.time_proj(time_emb)[:, :, None, None]
        gate = torch.sigmoid(logits)
        return hsi + alpha_t * gate * msi_hf


class MSIHighFrequencyGuidedPredictor(nn.Module):
    """V3 predictor F_theta(x_t, HR-MSI, t) -> clean HR-HSI."""

    requires_msi = True

    def __init__(
        self,
        n_bands: int,
        n_msi_bands: int,
        total_steps: int = 12,
        base_channels: int = 64,
        time_dim: int = 256,
        dropout: float = 0.0,
        residual_prediction: bool = True,
        spectral_hidden: int = 8,
        msi_highpass_kernel: int = 5,
        msi_highpass_sigma: float = 1.0,
    ):
        super().__init__()
        if n_bands < 1 or n_msi_bands < 1:
            raise ValueError("n_bands and n_msi_bands must be >= 1")
        if total_steps < 1:
            raise ValueError("total_steps must be >= 1")
        if base_channels < 8:
            raise ValueError("base_channels must be >= 8")

        self.n_bands = int(n_bands)
        self.n_msi_bands = int(n_msi_bands)
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

        # HSI branch: keep V2's explicit local spectral modeling.
        self.spectral_stem = LocalSpectralStem(hidden_channels=spectral_hidden)
        self.in_proj = nn.Conv2d(self.n_bands, c1, kernel_size=3, padding=1)

        # MSI branch: only candidate high-frequency spatial evidence enters.
        self.msi_highpass = FixedGaussianHighPass(
            kernel_size=msi_highpass_kernel,
            sigma=msi_highpass_sigma,
        )
        self.msi_in = nn.Sequential(
            nn.Conv2d(self.n_msi_bands, c1, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
        )
        self.msi_down1 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(c2, c2, kernel_size=3, padding=1),
        )
        self.msi_down2 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(c3, c3, kernel_size=3, padding=1),
        )

        self.gate1 = MSITransferGate(c1, time_dim)
        self.gate2 = MSITransferGate(c2, time_dim)
        self.gate3 = MSITransferGate(c3, time_dim)

        self.enc1 = nn.ModuleList([
            EMRInspiredTimeBlock(c1, time_dim, dropout),
            EMRInspiredTimeBlock(c1, time_dim, dropout),
        ])
        self.down1 = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1)

        self.enc2 = nn.ModuleList([
            EMRInspiredTimeBlock(c2, time_dim, dropout),
            EMRInspiredTimeBlock(c2, time_dim, dropout),
        ])
        self.down2 = nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1)

        self.mid = nn.ModuleList([
            EMRInspiredTimeBlock(c3, time_dim, dropout),
            EMRInspiredTimeBlock(c3, time_dim, dropout),
        ])

        self.up2_proj = nn.Conv2d(c3, c2, kernel_size=3, padding=1)
        self.up2_fuse = nn.Conv2d(c2 + c2, c2, kernel_size=1)
        self.dec2 = nn.ModuleList([
            EMRInspiredTimeBlock(c2, time_dim, dropout),
            EMRInspiredTimeBlock(c2, time_dim, dropout),
        ])

        self.up1_proj = nn.Conv2d(c2, c1, kernel_size=3, padding=1)
        self.up1_fuse = nn.Conv2d(c1 + c1, c1, kernel_size=1)
        self.dec1 = nn.ModuleList([
            EMRInspiredTimeBlock(c1, time_dim, dropout),
            EMRInspiredTimeBlock(c1, time_dim, dropout),
        ])

        self.out_norm = nn.GroupNorm(_num_groups(c1), c1)
        self.out_conv = nn.Conv2d(c1, self.n_bands, kernel_size=3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def _time_embedding(self, t: torch.Tensor, batch_size: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=device)
        if t.ndim == 0:
            t = t.repeat(batch_size)
        if t.ndim != 1 or t.shape[0] != batch_size:
            raise ValueError(
                f"t must be scalar or shape [B={batch_size}], got {tuple(t.shape)}"
            )
        t = t.to(device=device)
        t_scaled = t.float() / float(self.total_steps) * 1000.0
        time_emb = self.time_embed(t_scaled)
        # Reverse inference begins at large t and ends near zero.
        alpha = (t.float() / float(self.total_steps)).clamp(0.0, 1.0)
        alpha = alpha[:, None, None, None]
        return time_emb, alpha

    @staticmethod
    def _apply_blocks(x, blocks, time_emb):
        for block in blocks:
            x = block(x, time_emb)
        return x

    def forward(
        self,
        x_t: torch.Tensor,
        hr_msi: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        if x_t.ndim != 4 or hr_msi.ndim != 4:
            raise ValueError("x_t and hr_msi must both be BxCxHxW")
        if x_t.shape[1] != self.n_bands:
            raise ValueError(
                f"expected {self.n_bands} HSI bands, got {x_t.shape[1]}"
            )
        if hr_msi.shape[1] != self.n_msi_bands:
            raise ValueError(
                f"expected {self.n_msi_bands} MSI bands, got {hr_msi.shape[1]}"
            )
        if hr_msi.shape[-2:] != x_t.shape[-2:]:
            raise ValueError(
                f"HR-MSI and x_t spatial sizes must match, got "
                f"{hr_msi.shape[-2:]} vs {x_t.shape[-2:]}"
            )

        time_emb, alpha_t = self._time_embedding(t, x_t.shape[0], x_t.device)
        input_state = x_t

        msi_hf = self.msi_highpass(hr_msi)
        m1 = self.msi_in(msi_hf)
        m2 = self.msi_down1(m1)
        m3 = self.msi_down2(m2)

        x = self.in_proj(self.spectral_stem(x_t))
        x = self.gate1(x, m1, time_emb, alpha_t)
        skip1 = self._apply_blocks(x, self.enc1, time_emb)

        x = self.down1(skip1)
        x = self.gate2(x, m2, time_emb, alpha_t)
        skip2 = self._apply_blocks(x, self.enc2, time_emb)

        x = self.down2(skip2)
        x = self.gate3(x, m3, time_emb, alpha_t)
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
