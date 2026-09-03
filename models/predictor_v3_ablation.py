"""Ablation variants for Innovation 2 MSI guidance and translation.

All variants keep the same V3 HSI backbone, progressive physical degradation,
losses, and reverse process. Historical groups are retained for reproducibility,
followed by minimal representation-translation candidates.

Legacy group (already trained):
- no_msi: complete V3 modules, but no MSI feature is transferred.
- full: high-pass MSI + learned time-conditioned gate + alpha_t=t/T.
- raw_msi: raw MSI + learned time-conditioned gate + alpha_t=t/T.
- hf_nogate: high-pass MSI direct injection + alpha_t=t/T.
- hf_const: high-pass MSI + learned time-conditioned gate + alpha=1.

Time-free orthogonal group (already trained):
- raw_direct: raw MSI, direct injection, no MSI-path time dependence.
- raw_gate: raw MSI, state-only learned gate, no MSI-path time dependence.
- hf_direct: high-pass MSI, direct injection, no MSI-path time dependence.
- hf_gate: high-pass MSI, state-only learned gate, no MSI-path time dependence.

Representation-translation candidates:
- raw_translate: raw MSI, low-rank residual channel translation, then direct
  injection. Translation depends only on the MSI feature.
- raw_translate_ctx: raw MSI, low-rank residual translation conditioned on the
  current same-scale HSI restoration feature, then direct injection.

Both translation candidates use no high-pass filtering, no transfer gate, and
no MSI-path timestep dependence. Their final residual projections are zero-
initialized, so both start exactly from the raw_direct feature transfer and can
only learn a residual representation correction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .predictor_v2 import _num_groups
from .predictor_v3 import MSIHighFrequencyGuidedPredictor


VALID_MSI_ABLATIONS = (
    "no_msi",
    "full",
    "raw_msi",
    "hf_nogate",
    "hf_const",
    "raw_direct",
    "raw_gate",
    "hf_direct",
    "hf_gate",
    "raw_translate",
    "raw_translate_ctx",
)

TIME_FREE_ABLATIONS = (
    "raw_direct",
    "raw_gate",
    "hf_direct",
    "hf_gate",
    "raw_translate",
    "raw_translate_ctx",
)


class LowRankResidualTranslationAdapter(nn.Module):
    """Identity-initialized low-rank channel translation for MSI features.

    Only 1x1 projections are used, so this module cannot gain performance by
    introducing an additional spatial filtering path. For channels C, the
    default bottleneck rank is approximately C/16 (at least 4).
    """

    def __init__(self, channels: int, rank: int | None = None):
        super().__init__()
        channels = int(channels)
        if channels < 1:
            raise ValueError("channels must be >= 1")
        if rank is None:
            rank = max(4, channels // 16)
        rank = min(int(rank), channels)
        if rank < 1:
            raise ValueError("rank must be >= 1")

        self.channels = channels
        self.rank = rank
        self.down = nn.Conv2d(channels, rank, kernel_size=1)
        self.act = nn.SiLU()
        self.up = nn.Conv2d(rank, channels, kernel_size=1)

        # T(F_M)=F_M exactly at initialization.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, msi_feature: torch.Tensor) -> torch.Tensor:
        if msi_feature.ndim != 4 or msi_feature.shape[1] != self.channels:
            raise ValueError(
                f"expected Bx{self.channels}xHxW MSI feature, "
                f"got {tuple(msi_feature.shape)}"
            )
        return msi_feature + self.up(self.act(self.down(msi_feature)))


class HSIContextResidualTranslationAdapter(nn.Module):
    """Translate MSI representation using the current same-scale HSI context.

    The residual branch receives normalized MSI and HSI features concatenated
    along channels, then applies only low-rank 1x1 projections:

        T(F_M, F_H) = F_M + B(SiLU(A([N_M(F_M), N_H(F_H)]))).

    GroupNorm is non-affine to avoid adding another learnable modulation path.
    The output projection B is zero-initialized, so the complete adapter is an
    exact identity mapping on F_M at initialization regardless of F_H.
    """

    def __init__(self, channels: int, rank: int | None = None):
        super().__init__()
        channels = int(channels)
        if channels < 1:
            raise ValueError("channels must be >= 1")
        if rank is None:
            rank = max(4, channels // 16)
        rank = min(int(rank), channels)
        if rank < 1:
            raise ValueError("rank must be >= 1")

        self.channels = channels
        self.rank = rank
        groups = _num_groups(channels)
        self.msi_norm = nn.GroupNorm(groups, channels, affine=False)
        self.hsi_norm = nn.GroupNorm(groups, channels, affine=False)
        self.down = nn.Conv2d(channels * 2, rank, kernel_size=1)
        self.act = nn.SiLU()
        self.up = nn.Conv2d(rank, channels, kernel_size=1)

        # T(F_M, F_H)=F_M exactly at initialization.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(
        self,
        msi_feature: torch.Tensor,
        hsi_context: torch.Tensor,
    ) -> torch.Tensor:
        if msi_feature.shape != hsi_context.shape:
            raise ValueError(
                "MSI/HSI context shapes must match, got "
                f"{tuple(msi_feature.shape)} and {tuple(hsi_context.shape)}"
            )
        if msi_feature.ndim != 4 or msi_feature.shape[1] != self.channels:
            raise ValueError(
                f"expected Bx{self.channels}xHxW features, "
                f"got {tuple(msi_feature.shape)}"
            )
        joint = torch.cat(
            [self.msi_norm(msi_feature), self.hsi_norm(hsi_context)],
            dim=1,
        )
        delta = self.up(self.act(self.down(joint)))
        return msi_feature + delta


class MSIAblationGuidedPredictor(MSIHighFrequencyGuidedPredictor):
    """V3 predictor with controlled MSI-guidance/translation ablations."""

    def __init__(self, *args, msi_ablation: str = "full", **kwargs):
        super().__init__(*args, **kwargs)
        mode = str(msi_ablation).lower()
        if mode not in VALID_MSI_ABLATIONS:
            raise ValueError(
                f"Unsupported msi_ablation={mode!r}; "
                f"expected one of {VALID_MSI_ABLATIONS}"
            )
        self.msi_ablation = mode

        c1 = int(self.msi_in[-1].out_channels)
        c2 = int(self.msi_down1[-1].out_channels)
        c3 = int(self.msi_down2[-1].out_channels)

        # Translation modules exist only in their corresponding new modes.
        # Historical modes therefore keep their old state_dict unchanged.
        if mode == "raw_translate":
            self.translate1 = LowRankResidualTranslationAdapter(c1)
            self.translate2 = LowRankResidualTranslationAdapter(c2)
            self.translate3 = LowRankResidualTranslationAdapter(c3)
        elif mode == "raw_translate_ctx":
            self.translate1 = HSIContextResidualTranslationAdapter(c1)
            self.translate2 = HSIContextResidualTranslationAdapter(c2)
            self.translate3 = HSIContextResidualTranslationAdapter(c3)

    @staticmethod
    def _state_only_gate(hsi, msi_feature, gate):
        """Learn transfer from HSI/MSI state compatibility, without timestep."""
        if hsi.shape != msi_feature.shape:
            raise ValueError(
                f"HSI/MSI feature shapes must match, got {hsi.shape} "
                f"and {msi_feature.shape}"
            )
        logits = gate.gate(
            torch.cat(
                [gate.hsi_norm(hsi), gate.msi_norm(msi_feature)],
                dim=1,
            )
        )
        weight = torch.sigmoid(logits)
        return hsi + weight * msi_feature

    def _inject(self, hsi, msi_feature, gate, time_emb, alpha_t):
        mode = self.msi_ablation

        if mode == "no_msi":
            return hsi

        # Fully time-free direct/translation paths: no t/T multiplier and no
        # gate.time_proj contribution.
        if mode in (
            "raw_direct",
            "hf_direct",
            "raw_translate",
            "raw_translate_ctx",
        ):
            return hsi + msi_feature
        if mode in ("raw_gate", "hf_gate"):
            return self._state_only_gate(hsi, msi_feature, gate)

        # Legacy ablations retained so previous results/checkpoints remain
        # reproducible.
        if mode == "hf_nogate":
            return hsi + alpha_t * msi_feature
        if mode == "hf_const":
            alpha_t = torch.ones_like(alpha_t)
        return gate(hsi, msi_feature, time_emb, alpha_t)

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
                "HR-MSI and x_t spatial sizes must match, got "
                f"{hr_msi.shape[-2:]} vs {x_t.shape[-2:]}"
            )

        time_emb, alpha_t = self._time_embedding(t, x_t.shape[0], x_t.device)
        input_state = x_t

        # All translation candidates deliberately keep the complete raw MSI.
        if self.msi_ablation in (
            "raw_msi",
            "raw_direct",
            "raw_gate",
            "raw_translate",
            "raw_translate_ctx",
        ):
            msi_guidance = hr_msi
        else:
            msi_guidance = self.msi_highpass(hr_msi)

        m1 = self.msi_in(msi_guidance)
        m2 = self.msi_down1(m1)
        m3 = self.msi_down2(m2)

        # MSI-only translation can be computed before the HSI path.
        if self.msi_ablation == "raw_translate":
            m1 = self.translate1(m1)
            m2 = self.translate2(m2)
            m3 = self.translate3(m3)

        # Scale 1: x is the current HSI feature before any scale-1 MSI transfer.
        x = self.in_proj(self.spectral_stem(x_t))
        if self.msi_ablation == "raw_translate_ctx":
            m1 = self.translate1(m1, x)
        x = self._inject(x, m1, self.gate1, time_emb, alpha_t)
        skip1 = self._apply_blocks(x, self.enc1, time_emb)

        # Scale 2: condition translation on the restoration feature arriving at
        # this scale, before scale-2 MSI is injected.
        x = self.down1(skip1)
        if self.msi_ablation == "raw_translate_ctx":
            m2 = self.translate2(m2, x)
        x = self._inject(x, m2, self.gate2, time_emb, alpha_t)
        skip2 = self._apply_blocks(x, self.enc2, time_emb)

        # Scale 3 follows the same rule.
        x = self.down2(skip2)
        if self.msi_ablation == "raw_translate_ctx":
            m3 = self.translate3(m3, x)
        x = self._inject(x, m3, self.gate3, time_emb, alpha_t)
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
