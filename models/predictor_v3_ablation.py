"""Ablation variants for Innovation 2 MSI guidance.

All variants share exactly the same V3 backbone and parameterization. Only the
MSI guidance mechanism changes:

- full: fixed MSI high-pass + learned transfer gate + alpha_t=t/T.
- raw_msi: raw MSI replaces high-pass MSI; gate and alpha_t are retained.
- hf_nogate: high-pass MSI is injected directly without the learned gate;
  alpha_t=t/T is retained.
- hf_const: high-pass MSI and the learned gate are retained, but the explicit
  alpha_t=t/T schedule is replaced by alpha_t=1.

The inherited module set is unchanged, so a full-mode model has exactly the
same state_dict structure as MSIHighFrequencyGuidedPredictor.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .predictor_v3 import MSIHighFrequencyGuidedPredictor


VALID_MSI_ABLATIONS = ("full", "raw_msi", "hf_nogate", "hf_const")


class MSIAblationGuidedPredictor(MSIHighFrequencyGuidedPredictor):
    """V3 predictor with controlled MSI-guidance ablations."""

    def __init__(self, *args, msi_ablation: str = "full", **kwargs):
        super().__init__(*args, **kwargs)
        mode = str(msi_ablation).lower()
        if mode not in VALID_MSI_ABLATIONS:
            raise ValueError(
                f"Unsupported msi_ablation={mode!r}; "
                f"expected one of {VALID_MSI_ABLATIONS}"
            )
        self.msi_ablation = mode

    def _inject(self, hsi, msi_feature, gate, time_emb, alpha_t):
        if self.msi_ablation == "hf_nogate":
            return hsi + alpha_t * msi_feature
        if self.msi_ablation == "hf_const":
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

        # Only raw_msi removes the high-pass operation. The encoder, gates,
        # backbone and parameter count remain identical across all variants.
        if self.msi_ablation == "raw_msi":
            msi_guidance = hr_msi
        else:
            msi_guidance = self.msi_highpass(hr_msi)

        m1 = self.msi_in(msi_guidance)
        m2 = self.msi_down1(m1)
        m3 = self.msi_down2(m2)

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
