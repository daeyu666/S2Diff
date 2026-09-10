"""Cost-volume-conditioned sub-pixel refinement for Innovation-2 V4.

The existing confidence-gated sparse 4->2->1 search is retained.  Each physical
scale receives one lightweight 1x1 refinement head that reads the discrete
candidate probability volume plus coarse residual/margin/gate and predicts a
continuous residual in [-subpixel_max_px, +subpixel_max_px].

The last convolution of every head is zero-initialized.  Enabling this module on
an existing confidence/gate-presence checkpoint therefore starts from exactly
the previous geometric behavior (epsilon == 0) instead of perturbing a stable
model at warm start.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .predictor_v4_alignment import (
    SparseProgressiveLocalAligner,
    _pixel_coords_to_grid,
    _pixel_grid,
)
from .predictor_v4_gate_presence import PresenceSupervisedConfidenceLocalAligner


class CostVolumeSubpixelLocalAligner(PresenceSupervisedConfidenceLocalAligner):
    """Confidence-gated progressive local aligner with continuous residual heads."""

    def __init__(
        self,
        *args,
        subpixel_hidden_channels: int = 32,
        subpixel_max_px: float = 0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if subpixel_hidden_channels < 4:
            raise ValueError("subpixel_hidden_channels must be >= 4")
        if subpixel_max_px <= 0.0:
            raise ValueError("subpixel_max_px must be > 0")
        self.subpixel_hidden_channels = int(subpixel_hidden_channels)
        self.subpixel_max_px = float(subpixel_max_px)

        heads = {}
        for scale, radius in sorted(self.radius_by_scale.items()):
            k = int((2 * int(radius) + 1) ** 2)
            # probability volume K + coarse dx/dy + normalized margin + gate
            in_channels = k + 4
            head = nn.Sequential(
                nn.Conv2d(in_channels, self.subpixel_hidden_channels, kernel_size=1),
                nn.SiLU(),
                nn.Conv2d(self.subpixel_hidden_channels, 2, kernel_size=1),
            )
            # Exact legacy behavior at warm start: epsilon == 0.
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
            heads[str(int(scale))] = head
        self.subpixel_heads = nn.ModuleDict(heads)

        self.last_raw_residual_by_scale: Dict[int, torch.Tensor] = {}
        self.last_subpixel_by_scale: Dict[int, torch.Tensor] = {}
        self.last_combined_residual_by_scale: Dict[int, torch.Tensor] = {}
        self.last_gated_residual_by_scale: Dict[int, torch.Tensor] = {}
        self.last_accumulated_control_by_scale: Dict[int, torch.Tensor] = {}

    def _subpixel_head(self, scale: int) -> nn.Module:
        key = str(int(scale))
        if key not in self.subpixel_heads:
            nearest = min(
                self.radius_by_scale,
                key=lambda s: abs(int(s) - int(scale)),
            )
            key = str(int(nearest))
        return self.subpixel_heads[key]

    def subpixel_snapshot(self) -> Dict[int, float]:
        return {
            int(scale): float(
                torch.linalg.vector_norm(value.detach().float(), dim=1).mean().item()
            )
            for scale, value in self.last_subpixel_by_scale.items()
        }

    def forward(
        self,
        z_h: torch.Tensor,
        z_m: torch.Tensor,
        previous_dense_offset: Optional[torch.Tensor],
        *,
        scale: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if z_h.shape != z_m.shape:
            raise ValueError(
                "local matched states must share shape, got "
                f"{z_h.shape} and {z_m.shape}"
            )

        b, _, h, w = z_h.shape
        h_feat = self.descriptor(z_h)
        m_feat = self.descriptor(z_m)

        grid_h = max(2, int(math.ceil(h / float(self.control_stride))))
        grid_w = max(2, int(math.ceil(w / float(self.control_stride))))
        base_x, base_y = _pixel_grid(
            h,
            w,
            grid_h,
            grid_w,
            device=z_h.device,
            dtype=z_h.dtype,
        )
        base_x = base_x.expand(b, -1, -1)
        base_y = base_y.expand(b, -1, -1)
        base_grid = _pixel_coords_to_grid(base_x, base_y, h, w)

        h_control = F.grid_sample(
            h_feat,
            base_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        h_control = F.normalize(h_control, dim=1, eps=1e-6)

        if previous_dense_offset is None:
            previous_control = torch.zeros(
                b,
                2,
                grid_h,
                grid_w,
                device=z_h.device,
                dtype=z_h.dtype,
            )
        else:
            previous_control = F.grid_sample(
                previous_dense_offset,
                base_grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )

        radius = self._radius(int(scale))
        coord = torch.arange(
            -radius,
            radius + 1,
            device=z_h.device,
            dtype=z_h.dtype,
        )
        dy, dx = torch.meshgrid(coord, coord, indexing="ij")
        candidate_dx = dx.reshape(-1)
        candidate_dy = dy.reshape(-1)
        k = int(candidate_dx.numel())
        center = k // 2

        source_x = (
            base_x[:, None]
            + previous_control[:, 0:1]
            + candidate_dx[None, :, None, None]
        )
        source_y = (
            base_y[:, None]
            + previous_control[:, 1:2]
            + candidate_dy[None, :, None, None]
        )
        candidate_grid = _pixel_coords_to_grid(source_x, source_y, h, w)

        m_repeat = (
            m_feat[:, None]
            .expand(-1, k, -1, -1, -1)
            .reshape(b * k, m_feat.shape[1], h, w)
        )
        sampled = F.grid_sample(
            m_repeat,
            candidate_grid.reshape(b * k, grid_h, grid_w, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled = sampled.view(b, k, m_feat.shape[1], grid_h, grid_w)
        sampled = F.normalize(sampled, dim=2, eps=1e-6)

        scores = (h_control[:, None] * sampled).sum(dim=2)
        valid = (
            (source_x >= 0.0)
            & (source_x <= float(w - 1))
            & (source_y >= 0.0)
            & (source_y <= float(h - 1))
        )
        scores = scores.masked_fill(~valid, -1e4)

        any_valid = valid.any(dim=1, keepdim=True)
        if not bool(any_valid.all()):
            fallback = torch.full_like(scores, -1e4)
            fallback[:, center] = 0.0
            scores = torch.where(any_valid.expand_as(scores), scores, fallback)

        temperature = self._temperature(int(scale))
        probability = torch.softmax(scores / temperature, dim=1)
        residual_x = (
            probability * candidate_dx[None, :, None, None]
        ).sum(dim=1)
        residual_y = (
            probability * candidate_dy[None, :, None, None]
        ).sum(dim=1)
        raw_residual_control = torch.stack([residual_x, residual_y], dim=1)

        center_score = scores[:, center]
        if k > 1:
            noncenter_mask = torch.ones(k, device=scores.device, dtype=torch.bool)
            noncenter_mask[center] = False
            best_move_score = scores[:, noncenter_mask].amax(dim=1)
            margin = best_move_score - center_score
        else:
            margin = torch.full_like(center_score, -1.0)

        gain, bias = self._confidence_params(int(scale))
        normalized_margin = (margin / temperature).clamp(-10.0, 10.0)
        gate_logits = gain * normalized_margin + bias
        gate = torch.sigmoid(gate_logits)

        head_input = torch.cat(
            [
                probability,
                raw_residual_control,
                normalized_margin[:, None],
                gate[:, None],
            ],
            dim=1,
        )
        epsilon = self.subpixel_max_px * torch.tanh(
            self._subpixel_head(int(scale))(head_input)
        )

        combined_residual_control = raw_residual_control + epsilon
        gated_residual_control = combined_residual_control * gate[:, None]
        control_offset = previous_control + gated_residual_control

        dense = F.interpolate(
            control_offset,
            size=(h, w),
            mode="bicubic",
            align_corners=True,
        )

        # Graph-connected gate logits remain available to L_gate.
        self.current_gate_logits_by_scale[int(scale)] = gate_logits
        # Detached diagnostics.
        self.last_confidence_by_scale[int(scale)] = gate.detach()
        self.last_margin_by_scale[int(scale)] = margin.detach()
        self.last_raw_residual_by_scale[int(scale)] = raw_residual_control.detach()
        self.last_subpixel_by_scale[int(scale)] = epsilon.detach()
        self.last_combined_residual_by_scale[int(scale)] = combined_residual_control.detach()
        self.last_gated_residual_by_scale[int(scale)] = gated_residual_control.detach()
        self.last_accumulated_control_by_scale[int(scale)] = control_offset.detach()
        return dense, control_offset


def enable_cost_volume_subpixel_refiner(
    model: torch.nn.Module,
    *,
    confidence_gain_init: float = 4.0,
    confidence_bias_init: float = 1.5,
    subpixel_hidden_channels: int = 32,
    subpixel_max_px: float = 0.5,
) -> torch.nn.Module:
    """Install sub-pixel local refinement while preserving all compatible weights."""
    if not hasattr(model, "geometry_aligner"):
        raise ValueError("model has no geometry_aligner; expected predictor V4")
    old = model.geometry_aligner.local_aligner
    if isinstance(old, CostVolumeSubpixelLocalAligner):
        return model
    if not isinstance(old, SparseProgressiveLocalAligner):
        raise TypeError("unexpected V4 local aligner type")

    first_conv = old.descriptor.net[0]
    parameter = next(old.parameters())
    new = CostVolumeSubpixelLocalAligner(
        n_msi_bands=int(first_conv.in_channels),
        descriptor_channels=int(first_conv.out_channels),
        control_stride=int(old.control_stride),
        radius_by_scale=dict(old.radius_by_scale),
        confidence_gain_init=float(confidence_gain_init),
        confidence_bias_init=float(confidence_bias_init),
        subpixel_hidden_channels=int(subpixel_hidden_channels),
        subpixel_max_px=float(subpixel_max_px),
    ).to(device=parameter.device, dtype=parameter.dtype)

    missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
    unexpected = list(unexpected)
    if unexpected:
        raise RuntimeError(f"unexpected old local-aligner keys: {unexpected}")

    allowed_prefixes = (
        "confidence_log_gain.",
        "confidence_bias.",
        "subpixel_heads.",
    )
    extra_missing = [
        key for key in missing if not key.startswith(allowed_prefixes)
    ]
    if extra_missing:
        raise RuntimeError(
            f"failed to preserve compatible local weights: missing={extra_missing}"
        )

    model.geometry_aligner.local_aligner = new
    return model


__all__ = [
    "CostVolumeSubpixelLocalAligner",
    "enable_cost_volume_subpixel_refiner",
]
