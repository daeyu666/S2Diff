"""Innovation-2 recurrent residual-flow local alignment.

This module keeps the already validated V4 global rigid correction and the
Innovation-1 physical matched domain, but replaces the local soft-expectation
flow estimator with a correlation-evidence recurrent residual updater.

For each physical scale s in 4 -> 2 -> 1:
    Z_H^s = R x_t
    Z_M^s = D~_t(Y_M^G)

Sparse control points query a local correlation volume around the *current*
flow.  The complete correlation vector is encoded as evidence; it is never
collapsed to sum P(c)c.  A lightweight ConvGRU repeatedly predicts residual
flow updates:
    delta^{k+1} = delta^k + Delta delta^k

The control flow is bicubically interpolated to a dense HR-grid displacement.
The degraded MSI is used only for correspondence.  The final dense field warps
the complete Raw MSI before the existing Raw-Direct fusion backbone.

There is deliberately no confidence gate and no sub-pixel MLP in this branch.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .predictor_v4_alignment import (
    AlignmentDescriptor,
    DegradationDomainCoarseAligner,
    StateMatchedCoarseAlignedPredictor,
    _pixel_coords_to_grid,
    _pixel_grid,
)


class ConvGRUCell(nn.Module):
    """Small spatial GRU used on the sparse control grid."""

    def __init__(self, hidden_channels: int, input_channels: int):
        super().__init__()
        total = int(hidden_channels) + int(input_channels)
        self.hidden_channels = int(hidden_channels)
        self.gates = nn.Conv2d(total, 2 * hidden_channels, 3, padding=1)
        self.candidate = nn.Conv2d(total, hidden_channels, 3, padding=1)

    def forward(self, hidden: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        hx = torch.cat([hidden, x], dim=1)
        reset, update = torch.sigmoid(self.gates(hx)).chunk(2, dim=1)
        candidate = torch.tanh(
            self.candidate(torch.cat([reset * hidden, x], dim=1))
        )
        return (1.0 - update) * hidden + update * candidate


class CorrelationEvidenceEncoder(nn.Module):
    """Encode the full local cost volume plus candidate-validity mask."""

    def __init__(self, candidate_count: int, out_channels: int):
        super().__init__()
        in_channels = 2 * int(candidate_count)
        groups = 4 if out_channels % 4 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
        )

    def forward(
        self,
        correlation: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(
            torch.cat([correlation, valid.to(correlation.dtype)], dim=1)
        )


class RecurrentResidualFlowAligner(nn.Module):
    """Correlation-driven recurrent local non-rigid flow on sparse controls."""

    def __init__(
        self,
        n_msi_bands: int,
        descriptor_channels: int = 32,
        control_stride: int = 4,
        radius_by_scale: Optional[Dict[int, int]] = None,
        iterations_by_scale: Optional[Dict[int, int]] = None,
        max_update_by_scale: Optional[Dict[int, float]] = None,
        hidden_channels: int = 64,
        correlation_channels: int = 32,
    ):
        super().__init__()
        if control_stride < 2:
            raise ValueError("control_stride must be >= 2")
        if hidden_channels < 8 or correlation_channels < 4:
            raise ValueError("recurrent channels are too small")

        self.descriptor = AlignmentDescriptor(n_msi_bands, descriptor_channels)
        self.control_stride = int(control_stride)
        self.radius_by_scale = dict(radius_by_scale or {1: 1, 2: 2, 4: 3})
        self.iterations_by_scale = dict(iterations_by_scale or {1: 2, 2: 2, 4: 3})
        self.max_update_by_scale = dict(
            max_update_by_scale or {1: 0.5, 2: 1.0, 4: 2.0}
        )
        self.hidden_channels = int(hidden_channels)
        self.correlation_channels = int(correlation_channels)

        scales = sorted(self.radius_by_scale)
        if set(scales) != set(self.iterations_by_scale):
            raise ValueError("radius/iteration scale keys must match")
        if set(scales) != set(self.max_update_by_scale):
            raise ValueError("radius/max-update scale keys must match")
        if any(int(self.radius_by_scale[s]) < 1 for s in scales):
            raise ValueError("all local radii must be >= 1")
        if any(int(self.iterations_by_scale[s]) < 1 for s in scales):
            raise ValueError("all recurrent iteration counts must be >= 1")
        if any(float(self.max_update_by_scale[s]) <= 0.0 for s in scales):
            raise ValueError("all max recurrent updates must be > 0")

        self.correlation_encoders = nn.ModuleDict()
        for scale in scales:
            radius = int(self.radius_by_scale[scale])
            candidate_count = (2 * radius + 1) ** 2
            self.correlation_encoders[str(scale)] = CorrelationEvidenceEncoder(
                candidate_count,
                self.correlation_channels,
            )

        motion_in = 3 * descriptor_channels + 2 + 3
        motion_channels = max(16, self.hidden_channels // 2)
        groups = 4 if motion_channels % 4 == 0 else 1
        self.motion_encoder = nn.Sequential(
            nn.Conv2d(motion_in, motion_channels, 3, padding=1),
            nn.GroupNorm(groups, motion_channels),
            nn.SiLU(),
            nn.Conv2d(motion_channels, motion_channels, 3, padding=1),
            nn.SiLU(),
        )
        self.context_init = nn.Conv2d(
            descriptor_channels,
            self.hidden_channels,
            3,
            padding=1,
        )
        recurrent_input = self.correlation_channels + motion_channels
        self.gru = ConvGRUCell(self.hidden_channels, recurrent_input)
        self.delta_head = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(self.hidden_channels, 2, 3, padding=1),
        )
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

    def _resolve_scale(self, scale: int) -> int:
        scale = int(scale)
        if scale in self.radius_by_scale:
            return scale
        return min(self.radius_by_scale, key=lambda s: abs(int(s) - scale))

    @staticmethod
    def _scale_one_hot(
        scale: int,
        batch_size: int,
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        order = {4: 0, 2: 1, 1: 2}
        out = torch.zeros(
            batch_size, 3, height, width, device=device, dtype=dtype
        )
        out[:, order.get(int(scale), 2)] = 1.0
        return out

    @staticmethod
    def _candidate_offsets(
        radius: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        coord = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        dy, dx = torch.meshgrid(coord, coord, indexing="ij")
        return dx.reshape(-1), dy.reshape(-1)

    def _sample_control_features(
        self,
        feature: torch.Tensor,
        x_px: torch.Tensor,
        y_px: torch.Tensor,
    ) -> torch.Tensor:
        grid = _pixel_coords_to_grid(
            x_px,
            y_px,
            feature.shape[-2],
            feature.shape[-1],
        )
        sampled = F.grid_sample(
            feature,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return F.normalize(sampled, dim=1, eps=1e-6)

    def _correlation_evidence(
        self,
        h_control: torch.Tensor,
        m_feature: torch.Tensor,
        base_x: torch.Tensor,
        base_y: torch.Tensor,
        flow_control: torch.Tensor,
        *,
        radius: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, descriptor_channels, _, _ = h_control.shape
        _, _, h, w = m_feature.shape
        dx, dy = self._candidate_offsets(
            radius,
            device=m_feature.device,
            dtype=m_feature.dtype,
        )
        k = int(dx.numel())

        source_x = (
            base_x[:, None]
            + flow_control[:, 0:1]
            + dx[None, :, None, None]
        )
        source_y = (
            base_y[:, None]
            + flow_control[:, 1:2]
            + dy[None, :, None, None]
        )
        grid = _pixel_coords_to_grid(source_x, source_y, h, w)

        m_repeat = (
            m_feature[:, None]
            .expand(-1, k, -1, -1, -1)
            .reshape(b * k, descriptor_channels, h, w)
        )
        grid_h, grid_w = h_control.shape[-2:]
        sampled = F.grid_sample(
            m_repeat,
            grid.reshape(b * k, grid_h, grid_w, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).view(b, k, descriptor_channels, grid_h, grid_w)
        sampled = F.normalize(sampled, dim=2, eps=1e-6)

        correlation = (h_control[:, None] * sampled).sum(dim=2)
        valid = (
            (source_x >= 0.0)
            & (source_x <= float(w - 1))
            & (source_y >= 0.0)
            & (source_y <= float(h - 1))
        )
        correlation = correlation.masked_fill(~valid, -1.0)

        current_m = self._sample_control_features(
            m_feature,
            base_x + flow_control[:, 0],
            base_y + flow_control[:, 1],
        )
        return correlation, valid, current_m

    def forward(
        self,
        z_h: torch.Tensor,
        z_m: torch.Tensor,
        previous_dense_offset: Optional[torch.Tensor],
        *,
        scale: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        if z_h.shape != z_m.shape:
            raise ValueError(
                f"local matched states must share shape, got {z_h.shape} and {z_m.shape}"
            )
        scale = self._resolve_scale(int(scale))
        radius = int(self.radius_by_scale[scale])
        iterations = int(self.iterations_by_scale[scale])
        max_update = float(self.max_update_by_scale[scale])

        b, _, h, w = z_h.shape
        h_feature = self.descriptor(z_h)
        m_feature = self.descriptor(z_m)

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

        h_control = self._sample_control_features(h_feature, base_x, base_y)
        if previous_dense_offset is None:
            flow_control = torch.zeros(
                b, 2, grid_h, grid_w, device=z_h.device, dtype=z_h.dtype
            )
        else:
            base_grid = _pixel_coords_to_grid(base_x, base_y, h, w)
            flow_control = F.grid_sample(
                previous_dense_offset,
                base_grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )

        hidden = torch.tanh(self.context_init(h_control))
        sequence: List[torch.Tensor] = []
        scale_one_hot = self._scale_one_hot(
            scale,
            b,
            grid_h,
            grid_w,
            device=z_h.device,
            dtype=z_h.dtype,
        )

        for _ in range(iterations):
            correlation, valid, current_m = self._correlation_evidence(
                h_control,
                m_feature,
                base_x,
                base_y,
                flow_control,
                radius=radius,
            )
            corr_encoded = self.correlation_encoders[str(scale)](
                correlation,
                valid,
            )
            motion = torch.cat(
                [
                    h_control,
                    current_m,
                    h_control - current_m,
                    flow_control,
                    scale_one_hot,
                ],
                dim=1,
            )
            motion_encoded = self.motion_encoder(motion)
            hidden = self.gru(hidden, torch.cat([corr_encoded, motion_encoded], dim=1))
            delta = torch.tanh(self.delta_head(hidden)) * max_update
            flow_control = flow_control + delta
            dense = F.interpolate(
                flow_control,
                size=(h, w),
                mode="bicubic",
                align_corners=True,
            )
            sequence.append(dense)

        return sequence[-1], flow_control, sequence


class RecurrentDegradationDomainAligner(DegradationDomainCoarseAligner):
    """V4 global rigid alignment + physical-domain recurrent local flow."""

    def __init__(
        self,
        *args,
        recurrent_hidden_channels: int = 64,
        recurrent_correlation_channels: int = 32,
        recurrent_iterations_scale1: int = 2,
        recurrent_iterations_scale2: int = 2,
        recurrent_iterations_scale4: int = 3,
        recurrent_max_update_scale1: float = 0.5,
        recurrent_max_update_scale2: float = 1.0,
        recurrent_max_update_scale4: float = 2.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        n_msi_bands = int(self.srf_weights.shape[0])
        descriptor_channels = int(self.local_aligner.descriptor.net[0].out_channels)
        control_stride = int(self.local_aligner.control_stride)
        radius_by_scale = dict(self.local_aligner.radius_by_scale)

        self.local_aligner = RecurrentResidualFlowAligner(
            n_msi_bands=n_msi_bands,
            descriptor_channels=descriptor_channels,
            control_stride=control_stride,
            radius_by_scale=radius_by_scale,
            iterations_by_scale={
                1: int(recurrent_iterations_scale1),
                2: int(recurrent_iterations_scale2),
                4: int(recurrent_iterations_scale4),
            },
            max_update_by_scale={
                1: float(recurrent_max_update_scale1),
                2: float(recurrent_max_update_scale2),
                4: float(recurrent_max_update_scale4),
            },
            hidden_channels=int(recurrent_hidden_channels),
            correlation_channels=int(recurrent_correlation_channels),
        )
        self.last_update_sequence: List[torch.Tensor] = []
        self.last_training_sequence: Dict[int, List[torch.Tensor]] = {}

    def update_local(
        self,
        x_state: torch.Tensor,
        globally_aligned_msi: torch.Tensor,
        *,
        t: int,
        previous_dense_offset: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = x_state.shape[0]
        timesteps = torch.full(
            (b,), int(t), dtype=torch.long, device=x_state.device
        )
        z_h, z_m = self.matched_states(
            x_state,
            globally_aligned_msi,
            timesteps,
        )
        scale = int(self.process.state(int(t)).scale)
        dense, _, sequence = self.local_aligner(
            z_h,
            z_m,
            previous_dense_offset,
            scale=scale,
        )
        self.last_update_sequence = sequence
        return dense, z_h, z_m

    def prepare_training_pyramid(
        self,
        gt_hsi: torch.Tensor,
        hr_msi: torch.Tensor,
    ):
        t_global = self.process.total_steps
        x_global = self.process.state_at(gt_hsi, t_global)
        global_msi, shift, rotation = self.global_coarse_correction(
            x_global,
            hr_msi,
            t_global,
        )

        local_cache: Dict[int, torch.Tensor] = {}
        matched_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        sequence_cache: Dict[int, List[torch.Tensor]] = {}
        previous: Optional[torch.Tensor] = None

        for scale in sorted(self.stage_t_by_scale.keys(), reverse=True):
            t_stage = int(self.stage_t_by_scale[scale])
            x_stage = self.process.state_at(gt_hsi, t_stage)
            previous, z_h, z_m = self.update_local(
                x_stage,
                global_msi,
                t=t_stage,
                previous_dense_offset=previous,
            )
            local_cache[int(scale)] = previous
            matched_cache[int(scale)] = (z_h, z_m)
            sequence_cache[int(scale)] = list(self.last_update_sequence)

        self.last_training_sequence = sequence_cache
        return (
            global_msi,
            shift,
            rotation,
            local_cache,
            matched_cache,
            sequence_cache,
        )


class StateMatchedRecurrentFlowPredictor(StateMatchedCoarseAlignedPredictor):
    """Raw-Direct V4 predictor using recurrent residual local flow."""

    supports_recurrent_local_flow = True

    def __init__(
        self,
        *args,
        recurrent_hidden_channels: int = 64,
        recurrent_correlation_channels: int = 32,
        recurrent_iterations_scale1: int = 2,
        recurrent_iterations_scale2: int = 2,
        recurrent_iterations_scale4: int = 3,
        recurrent_max_update_scale1: float = 0.5,
        recurrent_max_update_scale2: float = 1.0,
        recurrent_max_update_scale4: float = 2.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        old = self.geometry_aligner
        self.geometry_aligner = RecurrentDegradationDomainAligner(
            self.progressive_process,
            old.srf_weights.detach().clone(),
            n_msi_bands=int(self.n_msi_bands),
            descriptor_channels=int(old.local_aligner.descriptor.net[0].out_channels),
            global_search_radius=int(old.global_aligner.translation_radius_px),
            global_rotation_max_deg=float(old.global_aligner.rotation_max_deg),
            global_rotation_step_deg=float(old.global_aligner.rotation_step_deg),
            global_feature_downsample=int(old.global_aligner.feature_downsample),
            global_candidate_chunk=int(old.global_aligner.candidate_chunk),
            control_stride=int(old.local_aligner.control_stride),
            local_radius_scale1=int(old.local_aligner.radius_by_scale[1]),
            local_radius_scale2=int(old.local_aligner.radius_by_scale[2]),
            local_radius_scale4=int(old.local_aligner.radius_by_scale[4]),
            recurrent_hidden_channels=int(recurrent_hidden_channels),
            recurrent_correlation_channels=int(recurrent_correlation_channels),
            recurrent_iterations_scale1=int(recurrent_iterations_scale1),
            recurrent_iterations_scale2=int(recurrent_iterations_scale2),
            recurrent_iterations_scale4=int(recurrent_iterations_scale4),
            recurrent_max_update_scale1=float(recurrent_max_update_scale1),
            recurrent_max_update_scale2=float(recurrent_max_update_scale2),
            recurrent_max_update_scale4=float(recurrent_max_update_scale4),
        )
        self._training_local_cache = {}
        self._training_sequence_cache: Dict[int, List[torch.Tensor]] = {}

    def prepare_training_alignment(
        self,
        gt_hsi: torch.Tensor,
        hr_msi: torch.Tensor,
    ) -> torch.Tensor:
        (
            global_msi,
            shift,
            rotation,
            local_cache,
            _,
            sequence_cache,
        ) = self.geometry_aligner.prepare_training_pyramid(gt_hsi, hr_msi)
        self.last_global_shift_px = shift.detach()
        self.last_global_rotation_deg = rotation.detach()
        self._training_local_cache = local_cache
        self._training_sequence_cache = sequence_cache
        return global_msi


__all__ = [
    "ConvGRUCell",
    "CorrelationEvidenceEncoder",
    "RecurrentResidualFlowAligner",
    "RecurrentDegradationDomainAligner",
    "StateMatchedRecurrentFlowPredictor",
]
