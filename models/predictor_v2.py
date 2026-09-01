"""Spectral-spatial clean-HSI predictor V2 for Innovation 1.

V2 keeps the same diffusion objective F_theta(x_t, t) -> X_0 but replaces the
plain residual blocks with an HSI-oriented backbone:

1) a local spectral stem processes neighbouring wavelengths before any latent
   channel compression;
2) EMR-Diff-inspired time-conditioned blocks combine spatial attention, a
   full-channel spatial branch, a depthwise spatial branch, adaptive branch
   weighting, and residual channel mixing;
3) the original compact multi-scale U-Net topology and clean-HSI residual
   prediction are retained.

Only the block design idea is adapted from EMR-Diff's MSGAB/MSGABT.  Its
multimodal second branch and dataset-specific fixed grouping are intentionally
not copied into Innovation 1.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _num_groups(channels: int, max_groups: int = 8) -> int:
    upper = min(int(max_groups), int(channels))
    for groups in range(upper, 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class SinusoidalTimeEmbedding(nn.Module):
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


class LocalSpectralStem(nn.Module):
    """Model local wavelength continuity before C_HSI -> latent projection.

    For every spatial pixel the HSI spectrum is treated as a 1-D signal.  The
    same small convolutional filters are shared over all pixels, so this stem
    explicitly models neighbouring wavelengths while preserving the original
    number and ordering of spectral bands.
    """

    def __init__(self, hidden_channels: int = 8):
        super().__init__()
        if hidden_channels < 2:
            raise ValueError("hidden_channels must be >= 2")
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, 1, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"x must be BxCxHxW, got {tuple(x.shape)}")
        b, c, h, w = x.shape
        spectrum = x.permute(0, 2, 3, 1).reshape(b * h * w, 1, c)
        refined = self.net(spectrum)
        refined = refined.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return x + refined


class SpatialAttention(nn.Module):
    """EMR-style spatial attention from channel average and channel maximum."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError("kernel_size must be 3 or 7")
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        maximum = torch.amax(x, dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg, maximum], dim=1)))


class EMRInspiredTimeBlock(nn.Module):
    """Time-conditioned dual-path residual block inspired by EMR-Diff MSGABT.

    Unlike EMR-Diff's dataset-coupled groups=[1, band], this block works in a
    latent feature space and uses:

      branch A: regular 3x3 convolution (cross-channel + spatial mixing)
      branch B: depthwise 3x3 convolution (channel-preserving spatial detail)

    Global pooled descriptors produce a 2-way softmax that adaptively weights
    the two branches.  FiLM modulation injects the physical degradation time.
    """

    def __init__(
        self,
        channels: int,
        time_dim: int,
        dropout: float = 0.0,
        spatial_kernel: int = 7,
        mlp_ratio: float = 2.0,
    ):
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be >= 1")
        hidden = channels * 2
        mlp_hidden = max(int(channels * mlp_ratio), channels)
        groups = _num_groups(channels)
        hidden_groups = _num_groups(hidden)

        self.pre_norm = nn.GroupNorm(groups, channels)
        self.pre_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.spatial_attention = SpatialAttention(spatial_kernel)

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, channels * 2),
        )

        self.full_branch = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(hidden_groups, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.SiLU(),
        )

        self.depth_branch = nn.Sequential(
            nn.Conv2d(
                channels, channels, kernel_size=3, padding=1,
                groups=channels, bias=False,
            ),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.GroupNorm(hidden_groups, hidden),
            nn.SiLU(),
            nn.Conv2d(
                hidden, hidden, kernel_size=3, padding=1,
                groups=hidden, bias=False,
            ),
            nn.SiLU(),
        )

        self.branch_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden * 2, 2, kernel_size=1),
        )

        self.channel_mlp = nn.Sequential(
            nn.Conv2d(hidden, mlp_hidden, kernel_size=1),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Conv2d(mlp_hidden, channels, kernel_size=1),
        )
        # Preserve a stable identity initialization for each residual block.
        nn.init.zeros_(self.channel_mlp[-1].weight)
        nn.init.zeros_(self.channel_mlp[-1].bias)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.pre_conv(F.silu(self.pre_norm(x)))
        h = h * self.spatial_attention(h)

        scale, shift = self.time_proj(time_emb).chunk(2, dim=1)
        h = h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]

        full = self.full_branch(h)
        depth = self.depth_branch(h)
        logits = self.branch_attention(torch.cat([full, depth], dim=1))
        weights = torch.softmax(logits, dim=1)
        fused = weights[:, 0:1] * full + weights[:, 1:2] * depth
        return x + self.channel_mlp(fused)


class SpectralSpatialCleanHSIPredictor(nn.Module):
    """Predictor V2: local spectral stem + EMR-inspired multi-scale U-Net."""

    def __init__(
        self,
        n_bands: int,
        total_steps: int = 12,
        base_channels: int = 64,
        time_dim: int = 256,
        dropout: float = 0.0,
        residual_prediction: bool = True,
        spectral_hidden: int = 8,
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

        self.spectral_stem = LocalSpectralStem(hidden_channels=spectral_hidden)
        self.in_proj = nn.Conv2d(self.n_bands, c1, kernel_size=3, padding=1)

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

    def _time_embedding(self, t: torch.Tensor, batch_size: int) -> torch.Tensor:
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=self.out_conv.weight.device)
        if t.ndim == 0:
            t = t.repeat(batch_size)
        if t.ndim != 1 or t.shape[0] != batch_size:
            raise ValueError(
                f"t must be scalar or shape [B={batch_size}], got {tuple(t.shape)}"
            )
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

        spectral = self.spectral_stem(x_t)
        x = self.in_proj(spectral)
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
