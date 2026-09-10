"""Residual-flow telemetry for Innovation-2 V4 confidence local alignment.

This module adds no trainable parameters.  It mirrors the current
presence-supervised confidence local aligner and records, at physical scales
4 -> 2 -> 1, the quantities needed to diagnose displacement shrinkage:

    raw residual -> confidence gate -> gated residual -> accumulated offset

Because the persistent parameter layout is unchanged, existing confidence and
gate-presence checkpoints remain directly loadable.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .predictor_v4_alignment import (
    SparseProgressiveLocalAligner,
    _pixel_coords_to_grid,
    _pixel_grid,
)
from .predictor_v4_gate_presence import PresenceSupervisedConfidenceLocalAligner


class ResidualTelemetryConfidenceLocalAligner(
    PresenceSupervisedConfidenceLocalAligner
):
    """Confidence local aligner exposing detached per-scale residual telemetry."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_raw_residual_by_scale: Dict[int, torch.Tensor] = {}
        self.last_gated_residual_by_scale: Dict[int, torch.Tensor] = {}
        self.last_accumulated_control_by_scale: Dict[int, torch.Tensor] = {}
        self.last_accumulated_dense_by_scale: Dict[int, torch.Tensor] = {}

    def clear_residual_telemetry(self) -> None:
        self.last_raw_residual_by_scale = {}
        self.last_gated_residual_by_scale = {}
        self.last_accumulated_control_by_scale = {}
        self.last_accumulated_dense_by_scale = {}
        self.last_confidence_by_scale = {}
        self.last_margin_by_scale = {}
        self.current_gate_logits_by_scale = {}

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
        gate_logits = gain * (margin / temperature) + bias
        gate = torch.sigmoid(gate_logits)
        gated_residual_control = raw_residual_control * gate[:, None]
        control_offset = previous_control + gated_residual_control

        dense = F.interpolate(
            control_offset,
            size=(h, w),
            mode="bicubic",
            align_corners=True,
        )

        scale_i = int(scale)
        self.current_gate_logits_by_scale[scale_i] = gate_logits
        self.last_confidence_by_scale[scale_i] = gate.detach()
        self.last_margin_by_scale[scale_i] = margin.detach()
        self.last_raw_residual_by_scale[scale_i] = raw_residual_control.detach()
        self.last_gated_residual_by_scale[scale_i] = gated_residual_control.detach()
        self.last_accumulated_control_by_scale[scale_i] = control_offset.detach()
        self.last_accumulated_dense_by_scale[scale_i] = dense.detach()
        return dense, control_offset


def enable_residual_telemetry_aligner(
    model: torch.nn.Module,
    *,
    confidence_gain_init: float = 4.0,
    confidence_bias_init: float = 1.5,
) -> torch.nn.Module:
    """Install telemetry aligner without changing the persistent state layout."""
    if not hasattr(model, "geometry_aligner"):
        raise ValueError("model has no geometry_aligner; expected predictor V4")

    old = model.geometry_aligner.local_aligner
    if isinstance(old, ResidualTelemetryConfidenceLocalAligner):
        return model
    if not isinstance(old, SparseProgressiveLocalAligner):
        raise TypeError("unexpected V4 local aligner type")

    first_conv = old.descriptor.net[0]
    parameter = next(old.parameters())
    new = ResidualTelemetryConfidenceLocalAligner(
        n_msi_bands=int(first_conv.in_channels),
        descriptor_channels=int(first_conv.out_channels),
        control_stride=int(old.control_stride),
        radius_by_scale=dict(old.radius_by_scale),
        confidence_gain_init=float(confidence_gain_init),
        confidence_bias_init=float(confidence_bias_init),
    ).to(device=parameter.device, dtype=parameter.dtype)

    missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
    unexpected = list(unexpected)
    if unexpected:
        raise RuntimeError(f"unexpected local-aligner keys: {unexpected}")

    allowed_missing = {
        *(f"confidence_log_gain.{scale}" for scale in new.radius_by_scale),
        *(f"confidence_bias.{scale}" for scale in new.radius_by_scale),
    }
    extra_missing = [key for key in missing if key not in allowed_missing]
    if extra_missing:
        raise RuntimeError(f"failed to preserve old local weights: missing={extra_missing}")

    model.geometry_aligner.local_aligner = new
    return model


__all__ = [
    "ResidualTelemetryConfidenceLocalAligner",
    "enable_residual_telemetry_aligner",
]
