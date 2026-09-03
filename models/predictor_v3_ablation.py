"""Ablation variants for Innovation 2 MSI guidance.

All variants keep the same V3 HSI backbone, progressive physical degradation,
losses, and reverse process. Two historical groups are retained for
reproducibility, plus a minimal representation-translation candidate.

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

Representation-translation candidate:
- raw_translate: raw MSI, low-rank residual channel translation, then direct
  injection. It uses no high-pass filtering, no transfer gate, and no MSI-path
  timestep dependence. The final projection in every translation adapter is
  zero-initialized, so raw_translate starts exactly from the raw_direct mapping
  and can only learn a residual representation correction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

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
)

TIME_FREE_ABLATIONS = (
    "raw_direct",
    "raw_gate",
    "hf_direct",
    "hf_gate",
    "raw_translate",
)


class LowRankResidualTranslationAdapter(nn.Module):
    """Identity-initialized low-rank channel translation for MSI features.

    The adapter deliberately uses only 1x1 projections. Therefore it cannot
    obtain gains by adding a new spatial filtering path; it only learns a small
    channel-representation correction before MSI features enter the HSI
    backbone. For channels C, the default bottleneck rank is approximately
    C/16 (at least 4), keeping the added capacity small.
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

        # Critical control: T(F)=F exactly at initialization. This makes the
        # new mode begin from the same transferred feature as raw_direct.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(
                f"expected Bx{self.channels}xHxW feature, got {tuple(x.shape)}"
            )
        return x + self.up(self.act(self.down(x)))


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

        # Keep all previously trained modes byte-for-byte compatible with their
        # old state_dict structure. Translation modules exist only in the new
        # raw_translate mode, so raw_direct remains the untouched baseline.
        if mode == "raw_translate":
            c1 = int(self.msi_in[-1].out_channels)
            c2 = int(self.msi_down1[-1].out_channels)
            c3 = int(self.msi_down2[-1].out_channels)
            self.translate1 = LowRankResidualTranslationAdapter(c1)
            self.translate2 = LowRankResidualTranslationAdapter(c2)
            self.translate3 = LowRankResidualTranslationAdapter(c3)

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
        if mode in ("raw_direct", "hf_direct", "raw_translate"):
            return hsi + msi_feature
        if mode in ("raw_gate", "hf_gate"):
            return self._state_only_gate(hsi, msi_feature, gate)

        # Legacy ablations retained so previous 200-epoch results/checkpoints
        # remain reproducible.
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
                f"HR-MSI and x_t spatial sizes must match, got "
                f"{hr_msi.shape[-2:]} vs {x_t.shape[-2:]}"
            )

        time_emb, alpha_t = self._time_embedding(t, x_t.shape[0], x_t.device)
        input_state = x_t

        # Raw/HF selection is orthogonal to Direct/Gate selection. The new
        # translation candidate intentionally keeps the complete raw MSI.
        if self.msi_ablation in (
            "raw_msi",
            "raw_direct",
            "raw_gate",
            "raw_translate",
        ):
            msi_guidance = hr_msi
        else:
            msi_guidance = self.msi_highpass(hr_msi)

        m1 = self.msi_in(msi_guidance)
        m2 = self.msi_down1(m1)
        m3 = self.msi_down2(m2)

        # Only the new candidate changes MSI representation. No HSI feature,
        # timestep, gate, or high-pass signal enters these adapters.
        if self.msi_ablation == "raw_translate":
            m1 = self.translate1(m1)
            m2 = self.translate2(m2)
            m3 = self.translate3(m3)

        x = self.in_proj(self.spectral_stem(x_t))
        x = self._inject(x, m1, self.gate1, time_emb, alpha_t)
        skip1 = self._apply_blocks(x, self.enc1, time_emb)

        x = self.down1(skip1)
        x = self._inject(x, m2, self.gate2, time_emb, alpha_t)
        skip2 = self._apply_blocks(x, self.enc2, time_emb)

        x = self.down2(skip2)
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
