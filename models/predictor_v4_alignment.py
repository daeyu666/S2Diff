"""Innovation 2: learnable degradation-domain progressive geometric alignment.

The alignment front end contains the three stages agreed for the first trainable
version:

1. Predict one global rigid correction (large translation + small rotation) per
   HSI-MSI pair.
2. Match modalities in the same physical domain:
       Z_H^t = R x_t
       Z_M^t = D~_t(Y_M^G)
   using the frozen Innovation-1 degradation trajectory.
3. Estimate local coarse displacement on sparse control points. The local field
   is updated only when reverse inference enters physical scale 4 -> 2 -> 1.
   Each update predicts a residual around the previous field and bicubically
   interpolates the sparse control offsets into a smooth dense field.

The degraded MSI is used only to locate correspondence. The complete Raw MSI is
sampled with the resulting displacement and is still the image supplied to the
existing Raw-Direct fusion backbone.

No confidence gate and no separate sub-pixel residual network are included in
this version.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from degradations.misalignment import build_global_grid
from degradations.progressive import ProgressiveDegradation
from .predictor_v3_ablation import MSIAblationGuidedPredictor


@dataclass
class CoarseAlignmentDiagnostics:
    """Detached geometry outputs from the latest alignment call."""

    global_shift_px: torch.Tensor
    global_rotation_deg: torch.Tensor
    local_offset_px: torch.Tensor
    local_scale: int
    matched_hsi: torch.Tensor
    matched_msi: torch.Tensor


def _batch_state_at(
    process: ProgressiveDegradation,
    x: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """Evaluate D~_t(x) for a batch with possibly different integer t."""
    if x.ndim != 4:
        raise ValueError(f"x must be BxCxHxW, got {tuple(x.shape)}")
    if timesteps.ndim != 1 or timesteps.shape[0] != x.shape[0]:
        raise ValueError("timesteps must have shape [B]")

    out = torch.empty_like(x)
    for t_value in torch.unique(timesteps, sorted=True):
        mask = timesteps == t_value
        out[mask] = process.state_at(x[mask], int(t_value.item()))
    return out


def spectral_project_hsi(
    x_hsi: torch.Tensor,
    srf_weights: torch.Tensor,
) -> torch.Tensor:
    """Apply fixed spectral response R: BxCxHxW -> BxMxHxW."""
    if x_hsi.ndim != 4:
        raise ValueError(f"x_hsi must be BxCxHxW, got {tuple(x_hsi.shape)}")
    if srf_weights.ndim != 2:
        raise ValueError("srf_weights must have shape [M,C]")
    if x_hsi.shape[1] != srf_weights.shape[1]:
        raise ValueError(
            f"HSI bands={x_hsi.shape[1]} do not match "
            f"SRF columns={srf_weights.shape[1]}"
        )
    weights = srf_weights.to(device=x_hsi.device, dtype=x_hsi.dtype)
    return torch.einsum("mc,bchw->bmhw", weights, x_hsi)


def _identity_grid(
    batch_size: int,
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    theta = torch.zeros(batch_size, 2, 3, device=device, dtype=dtype)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    return F.affine_grid(
        theta,
        size=(batch_size, 1, height, width),
        align_corners=False,
    )


def _sample_with_source_offset(
    x: torch.Tensor,
    offset_px: torch.Tensor,
) -> torch.Tensor:
    """Sample x at source coordinate p + offset(p), offset=(dx,dy) in pixels."""
    if offset_px.ndim != 4 or offset_px.shape[1] != 2:
        raise ValueError("offset_px must have shape Bx2xHxW")
    b, _, h, w = x.shape
    if offset_px.shape[0] != b or offset_px.shape[-2:] != (h, w):
        raise ValueError("offset spatial size must match input")

    grid = _identity_grid(
        b,
        h,
        w,
        device=x.device,
        dtype=x.dtype,
    )
    grid = grid.clone()
    grid[..., 0] = grid[..., 0] + 2.0 * offset_px[:, 0] / float(w)
    grid[..., 1] = grid[..., 1] + 2.0 * offset_px[:, 1] / float(h)
    return F.grid_sample(
        x,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )


def _pixel_grid(
    height: int,
    width: int,
    grid_h: int,
    grid_w: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return sparse control-point pixel coordinates shaped [1,Hc,Wc]."""
    ys = torch.linspace(0.0, float(height - 1), grid_h, device=device, dtype=dtype)
    xs = torch.linspace(0.0, float(width - 1), grid_w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return xx.unsqueeze(0), yy.unsqueeze(0)


def _pixel_coords_to_grid(
    x_px: torch.Tensor,
    y_px: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Convert pixel-center coordinates to align_corners=False grid coordinates."""
    x_norm = 2.0 * (x_px + 0.5) / float(width) - 1.0
    y_norm = 2.0 * (y_px + 0.5) / float(height) - 1.0
    return torch.stack([x_norm, y_norm], dim=-1)


class AlignmentDescriptor(nn.Module):
    """Small trainable spatial descriptor for already state-matched MSI bands."""

    def __init__(self, in_channels: int, channels: int = 32):
        super().__init__()
        if in_channels < 1 or channels < 4:
            raise ValueError("descriptor channels must be positive")
        groups = 4 if channels % 4 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=1, eps=1e-6)


class LearnedGlobalRigidAligner(nn.Module):
    """Learn one global correction from a descriptor-conditioned cost volume."""

    def __init__(
        self,
        n_msi_bands: int,
        descriptor_channels: int = 32,
        translation_radius_px: int = 6,
        rotation_max_deg: float = 3.0,
        rotation_step_deg: float = 1.0,
        feature_downsample: int = 4,
        candidate_chunk: int = 64,
    ):
        super().__init__()
        if translation_radius_px < 0:
            raise ValueError("translation_radius_px must be >= 0")
        if rotation_max_deg < 0:
            raise ValueError("rotation_max_deg must be >= 0")
        if rotation_step_deg <= 0:
            raise ValueError("rotation_step_deg must be > 0")
        if feature_downsample < 1:
            raise ValueError("feature_downsample must be >= 1")
        if candidate_chunk < 1:
            raise ValueError("candidate_chunk must be >= 1")

        self.descriptor = AlignmentDescriptor(n_msi_bands, descriptor_channels)
        self.translation_radius_px = int(translation_radius_px)
        self.rotation_max_deg = float(rotation_max_deg)
        self.rotation_step_deg = float(rotation_step_deg)
        self.feature_downsample = int(feature_downsample)
        self.candidate_chunk = int(candidate_chunk)

        dx = torch.arange(
            -self.translation_radius_px,
            self.translation_radius_px + 1,
            dtype=torch.float32,
        )
        dy = torch.arange(
            -self.translation_radius_px,
            self.translation_radius_px + 1,
            dtype=torch.float32,
        )
        if self.rotation_max_deg <= 1e-12:
            rot = torch.zeros(1, dtype=torch.float32)
        else:
            n_rot = int(
                math.floor(
                    2.0 * self.rotation_max_deg / self.rotation_step_deg
                )
            ) + 1
            rot = torch.linspace(
                -self.rotation_max_deg,
                self.rotation_max_deg,
                n_rot,
                dtype=torch.float32,
            )

        rr, yy, xx = torch.meshgrid(rot, dy, dx, indexing="ij")
        candidates = torch.stack(
            [xx.reshape(-1), yy.reshape(-1), rr.reshape(-1)],
            dim=1,
        )
        self.register_buffer("candidates", candidates, persistent=False)
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(0.10), dtype=torch.float32)
        )

    def _pooled_descriptor(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.descriptor(x)
        d = self.feature_downsample
        if d > 1 and min(feat.shape[-2:]) >= d:
            feat = F.avg_pool2d(feat, kernel_size=d, stride=d)
            feat = F.normalize(feat, dim=1, eps=1e-6)
        return feat

    def _candidate_scores(
        self,
        h_feat: torch.Tensor,
        m_feat: torch.Tensor,
    ) -> torch.Tensor:
        b, c, h, w = h_feat.shape
        ones = torch.ones(
            b, 1, h, w, device=h_feat.device, dtype=h_feat.dtype
        )
        scores = []
        candidates = self.candidates.to(
            device=h_feat.device,
            dtype=h_feat.dtype,
        )
        d = float(self.feature_downsample)

        for start in range(0, candidates.shape[0], self.candidate_chunk):
            cand = candidates[start : start + self.candidate_chunk]
            k = cand.shape[0]

            dx = cand[:, 0] / d
            dy = cand[:, 1] / d
            rot = cand[:, 2]

            dx_b = dx[None, :].expand(b, -1).reshape(-1)
            dy_b = dy[None, :].expand(b, -1).reshape(-1)
            rot_b = rot[None, :].expand(b, -1).reshape(-1)

            m_repeat = (
                m_feat[:, None]
                .expand(-1, k, -1, -1, -1)
                .reshape(b * k, c, h, w)
            )
            grid = build_global_grid(
                b * k,
                h,
                w,
                dx_b,
                dy_b,
                rot_b,
                device=m_feat.device,
                dtype=m_feat.dtype,
            )
            warped = F.grid_sample(
                m_repeat,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            warped = F.normalize(warped, dim=1, eps=1e-6)
            warped = warped.view(b, k, c, h, w)

            valid_repeat = (
                ones[:, None]
                .expand(-1, k, -1, -1, -1)
                .reshape(b * k, 1, h, w)
            )
            valid = F.grid_sample(
                valid_repeat,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            ).view(b, k, 1, h, w)

            cosine = (h_feat[:, None] * warped).sum(dim=2, keepdim=True)
            denom = valid.sum(dim=(2, 3, 4)).clamp_min(1.0)
            score = (cosine * valid).sum(dim=(2, 3, 4)) / denom
            scores.append(score)

        return torch.cat(scores, dim=1)

    def forward(
        self,
        z_h_reference: torch.Tensor,
        z_m_reference: torch.Tensor,
        raw_msi: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_feat = self._pooled_descriptor(z_h_reference)
        m_feat = self._pooled_descriptor(z_m_reference)
        scores = self._candidate_scores(h_feat, m_feat)

        temperature = self.log_temperature.exp().clamp(0.03, 1.0)
        probability = torch.softmax(scores / temperature, dim=1)
        candidates = self.candidates.to(
            device=probability.device,
            dtype=probability.dtype,
        )
        expected = probability @ candidates
        shift = expected[:, :2]
        rotation = expected[:, 2]

        b, _, h, w = raw_msi.shape
        grid = build_global_grid(
            b,
            h,
            w,
            shift[:, 0].to(raw_msi.dtype),
            shift[:, 1].to(raw_msi.dtype),
            rotation.to(raw_msi.dtype),
            device=raw_msi.device,
            dtype=raw_msi.dtype,
        )
        aligned = F.grid_sample(
            raw_msi,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return aligned, shift.to(raw_msi.dtype), rotation.to(raw_msi.dtype)


class SparseProgressiveLocalAligner(nn.Module):
    """Sparse control-point local residual search with smooth dense interpolation."""

    def __init__(
        self,
        n_msi_bands: int,
        descriptor_channels: int = 32,
        control_stride: int = 4,
        radius_by_scale: Optional[Dict[int, int]] = None,
    ):
        super().__init__()
        if control_stride < 2:
            raise ValueError("control_stride must be >= 2")
        self.descriptor = AlignmentDescriptor(n_msi_bands, descriptor_channels)
        self.control_stride = int(control_stride)
        self.radius_by_scale = dict(radius_by_scale or {1: 1, 2: 2, 4: 3})
        if any(int(v) < 0 for v in self.radius_by_scale.values()):
            raise ValueError("local radii must be >= 0")

        self.log_temperature = nn.ParameterDict(
            {
                str(scale): nn.Parameter(
                    torch.tensor(math.log(0.10), dtype=torch.float32)
                )
                for scale in sorted(self.radius_by_scale)
            }
        )

    def _radius(self, scale: int) -> int:
        if scale in self.radius_by_scale:
            return int(self.radius_by_scale[scale])
        nearest = min(
            self.radius_by_scale,
            key=lambda s: abs(int(s) - int(scale)),
        )
        return int(self.radius_by_scale[nearest])

    def _temperature(self, scale: int) -> torch.Tensor:
        key = str(scale)
        if key not in self.log_temperature:
            nearest = min(
                self.radius_by_scale,
                key=lambda s: abs(int(s) - int(scale)),
            )
            key = str(nearest)
        return self.log_temperature[key].exp().clamp(0.03, 1.0)

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
        k = candidate_dx.numel()

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
            center = k // 2
            fallback = torch.full_like(scores, -1e4)
            fallback[:, center] = 0.0
            scores = torch.where(any_valid.expand_as(scores), scores, fallback)

        probability = torch.softmax(
            scores / self._temperature(int(scale)),
            dim=1,
        )
        residual_x = (
            probability * candidate_dx[None, :, None, None]
        ).sum(dim=1)
        residual_y = (
            probability * candidate_dy[None, :, None, None]
        ).sum(dim=1)
        residual_control = torch.stack([residual_x, residual_y], dim=1)
        control_offset = previous_control + residual_control

        dense = F.interpolate(
            control_offset,
            size=(h, w),
            mode="bicubic",
            align_corners=True,
        )
        return dense, control_offset


class DegradationDomainCoarseAligner(nn.Module):
    """Learnable geometry front end coupled to the frozen physical trajectory."""

    def __init__(
        self,
        process: ProgressiveDegradation,
        srf_weights: torch.Tensor,
        *,
        n_msi_bands: int,
        descriptor_channels: int = 32,
        global_search_radius: int = 6,
        global_rotation_max_deg: float = 3.0,
        global_rotation_step_deg: float = 1.0,
        global_feature_downsample: int = 4,
        global_candidate_chunk: int = 64,
        control_stride: int = 4,
        local_radius_scale1: int = 1,
        local_radius_scale2: int = 2,
        local_radius_scale4: int = 3,
    ):
        super().__init__()
        self.process = process
        self.register_buffer(
            "srf_weights",
            torch.as_tensor(srf_weights, dtype=torch.float32),
            persistent=True,
        )

        self.global_aligner = LearnedGlobalRigidAligner(
            n_msi_bands=n_msi_bands,
            descriptor_channels=descriptor_channels,
            translation_radius_px=global_search_radius,
            rotation_max_deg=global_rotation_max_deg,
            rotation_step_deg=global_rotation_step_deg,
            feature_downsample=global_feature_downsample,
            candidate_chunk=global_candidate_chunk,
        )
        self.local_aligner = SparseProgressiveLocalAligner(
            n_msi_bands=n_msi_bands,
            descriptor_channels=descriptor_channels,
            control_stride=control_stride,
            radius_by_scale={
                1: int(local_radius_scale1),
                2: int(local_radius_scale2),
                4: int(local_radius_scale4),
            },
        )
        self.stage_t_by_scale = self._build_stage_representatives()

    def _build_stage_representatives(self) -> Dict[int, int]:
        result: Dict[int, int] = {}
        for t in range(1, self.process.total_steps + 1):
            result[int(self.process.state(t).scale)] = int(t)
        return result

    def spectral_project(self, x_hsi: torch.Tensor) -> torch.Tensor:
        return spectral_project_hsi(x_hsi, self.srf_weights)

    def matched_states(
        self,
        x_t: torch.Tensor,
        globally_aligned_msi: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z_h = self.spectral_project(x_t)
        z_m = _batch_state_at(self.process, globally_aligned_msi, timesteps)
        if z_h.shape != z_m.shape:
            raise ValueError(
                f"matched-domain shapes differ: HSI={z_h.shape}, MSI={z_m.shape}"
            )
        return z_h, z_m

    def global_coarse_correction(
        self,
        reference_hsi_state: torch.Tensor,
        hr_msi: torch.Tensor,
        reference_t: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = reference_hsi_state.shape[0]
        timesteps = torch.full(
            (b,),
            int(reference_t),
            dtype=torch.long,
            device=reference_hsi_state.device,
        )
        z_h = self.spectral_project(reference_hsi_state)
        z_m = _batch_state_at(self.process, hr_msi, timesteps)
        return self.global_aligner(z_h, z_m, hr_msi)

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
            (b,),
            int(t),
            dtype=torch.long,
            device=x_state.device,
        )
        z_h, z_m = self.matched_states(
            x_state,
            globally_aligned_msi,
            timesteps,
        )
        scale = int(self.process.state(int(t)).scale)
        dense, _ = self.local_aligner(
            z_h,
            z_m,
            previous_dense_offset,
            scale=scale,
        )
        return dense, z_h, z_m

    def prepare_training_pyramid(
        self,
        gt_hsi: torch.Tensor,
        hr_msi: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[int, torch.Tensor],
        Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    ]:
        """Build 4->2->1 local residual fields once for a training batch."""
        t_global = self.process.total_steps
        x_global = self.process.state_at(gt_hsi, t_global)
        global_msi, shift, rotation = self.global_coarse_correction(
            x_global,
            hr_msi,
            t_global,
        )

        local_cache: Dict[int, torch.Tensor] = {}
        matched_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        previous: Optional[torch.Tensor] = None

        scales = sorted(self.stage_t_by_scale.keys(), reverse=True)
        for scale in scales:
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

        return global_msi, shift, rotation, local_cache, matched_cache


class StateMatchedCoarseAlignedPredictor(MSIAblationGuidedPredictor):
    """Raw-Direct predictor with learnable global + progressive local alignment."""

    requires_msi = True
    requires_global_msi_preparation = True
    supports_training_alignment_pyramid = True

    def __init__(
        self,
        *args,
        progressive_process: ProgressiveDegradation,
        srf_weights: torch.Tensor,
        alignment_descriptor_channels: int = 32,
        alignment_global_search_radius: int = 6,
        alignment_global_rotation_max_deg: float = 3.0,
        alignment_global_rotation_step_deg: float = 1.0,
        alignment_global_feature_downsample: int = 4,
        alignment_global_candidate_chunk: int = 64,
        alignment_control_stride: int = 4,
        alignment_local_radius_scale1: int = 1,
        alignment_local_radius_scale2: int = 2,
        alignment_local_radius_scale4: int = 3,
        **kwargs,
    ):
        kwargs["msi_ablation"] = "raw_direct"
        super().__init__(*args, **kwargs)

        n_msi_bands = int(self.n_msi_bands)
        self.progressive_process = progressive_process
        self.geometry_aligner = DegradationDomainCoarseAligner(
            progressive_process,
            srf_weights,
            n_msi_bands=n_msi_bands,
            descriptor_channels=alignment_descriptor_channels,
            global_search_radius=alignment_global_search_radius,
            global_rotation_max_deg=alignment_global_rotation_max_deg,
            global_rotation_step_deg=alignment_global_rotation_step_deg,
            global_feature_downsample=alignment_global_feature_downsample,
            global_candidate_chunk=alignment_global_candidate_chunk,
            control_stride=alignment_control_stride,
            local_radius_scale1=alignment_local_radius_scale1,
            local_radius_scale2=alignment_local_radius_scale2,
            local_radius_scale4=alignment_local_radius_scale4,
        )

        self.last_alignment: Optional[CoarseAlignmentDiagnostics] = None
        self.last_global_shift_px: Optional[torch.Tensor] = None
        self.last_global_rotation_deg: Optional[torch.Tensor] = None

        self._training_local_cache: Dict[int, torch.Tensor] = {}
        self._inference_local_offset: Optional[torch.Tensor] = None
        self._inference_local_scale: Optional[int] = None

    def _reset_inference_local_state(self) -> None:
        self._inference_local_offset = None
        self._inference_local_scale = None

    def prepare_training_alignment(
        self,
        gt_hsi: torch.Tensor,
        hr_msi: torch.Tensor,
    ) -> torch.Tensor:
        """Prepare one global correction and the 4->2->1 local pyramid.

        This method is intentionally differentiable. Reconstruction loss can
        update the global descriptor, local descriptor and both temperatures.
        """
        (
            global_msi,
            shift,
            rotation,
            local_cache,
            _,
        ) = self.geometry_aligner.prepare_training_pyramid(gt_hsi, hr_msi)

        self.last_global_shift_px = shift.detach()
        self.last_global_rotation_deg = rotation.detach()
        self._training_local_cache = local_cache
        return global_msi

    def prepare_global_msi(
        self,
        reference_hsi_state: torch.Tensor,
        hr_msi: torch.Tensor,
        reference_t: int,
    ) -> torch.Tensor:
        """Inference: predict the global translation/rotation exactly once."""
        aligned, shift, rotation = self.geometry_aligner.global_coarse_correction(
            reference_hsi_state,
            hr_msi,
            reference_t,
        )
        self.last_global_shift_px = shift.detach()
        self.last_global_rotation_deg = rotation.detach()
        self._reset_inference_local_state()
        return aligned

    def _training_offset_for_timesteps(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        global_msi: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        pieces = []
        scales = []
        for i in range(x_t.shape[0]):
            scale = int(
                self.progressive_process.state(int(timesteps[i].item())).scale
            )
            scales.append(scale)
            if scale in self._training_local_cache:
                pieces.append(self._training_local_cache[scale][i : i + 1])
            else:
                selected, _, _ = self.geometry_aligner.update_local(
                    x_t[i : i + 1],
                    global_msi[i : i + 1],
                    t=int(timesteps[i].item()),
                    previous_dense_offset=None,
                )
                pieces.append(selected)
        dense = torch.cat(pieces, dim=0)
        z_h, z_m = self.geometry_aligner.matched_states(
            x_t,
            global_msi,
            timesteps,
        )
        diagnostic_scale = scales[0] if all(s == scales[0] for s in scales) else -1
        return dense, z_h, z_m, diagnostic_scale

    def _inference_offset_for_t(
        self,
        x_t: torch.Tensor,
        global_msi: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        if not torch.all(timesteps == timesteps[0]):
            raise ValueError(
                "V4 inference expects one common reverse timestep per batch"
            )
        t_value = int(timesteps[0].item())
        scale = int(self.progressive_process.state(t_value).scale)

        should_update = (
            self._inference_local_offset is None
            or self._inference_local_scale != scale
        )
        if should_update:
            dense, z_h, z_m = self.geometry_aligner.update_local(
                x_t,
                global_msi,
                t=t_value,
                previous_dense_offset=self._inference_local_offset,
            )
            self._inference_local_offset = dense
            self._inference_local_scale = scale
        else:
            z_h, z_m = self.geometry_aligner.matched_states(
                x_t,
                global_msi,
                timesteps,
            )

        return self._inference_local_offset, z_h, z_m, scale

    def forward(
        self,
        x_t: torch.Tensor,
        globally_aligned_msi: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        if t.ndim == 0:
            t = t.repeat(x_t.shape[0])
        t = t.to(device=x_t.device, dtype=torch.long)

        if self.training and self._training_local_cache:
            dense_offset, z_h, z_m, scale = self._training_offset_for_timesteps(
                x_t,
                t,
                globally_aligned_msi,
            )
        else:
            dense_offset, z_h, z_m, scale = self._inference_offset_for_t(
                x_t,
                globally_aligned_msi,
                t,
            )

        coarse_aligned_msi = _sample_with_source_offset(
            globally_aligned_msi,
            dense_offset,
        )

        global_shift = self.last_global_shift_px
        if global_shift is None:
            global_shift = torch.zeros(
                x_t.shape[0],
                2,
                device=x_t.device,
                dtype=x_t.dtype,
            )
        global_rotation = self.last_global_rotation_deg
        if global_rotation is None:
            global_rotation = torch.zeros(
                x_t.shape[0],
                device=x_t.device,
                dtype=x_t.dtype,
            )

        self.last_alignment = CoarseAlignmentDiagnostics(
            global_shift_px=global_shift.detach(),
            global_rotation_deg=global_rotation.detach(),
            local_offset_px=dense_offset.detach(),
            local_scale=int(scale),
            matched_hsi=z_h.detach(),
            matched_msi=z_m.detach(),
        )

        return super().forward(x_t, coarse_aligned_msi, t)


__all__ = [
    "AlignmentDescriptor",
    "CoarseAlignmentDiagnostics",
    "DegradationDomainCoarseAligner",
    "LearnedGlobalRigidAligner",
    "SparseProgressiveLocalAligner",
    "StateMatchedCoarseAlignedPredictor",
    "spectral_project_hsi",
]
