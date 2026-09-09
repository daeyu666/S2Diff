"""Innovation 2 v1: degradation-domain coarse geometric alignment.

This first alignment implementation intentionally contains only three steps:

1. Global coarse translation correction estimated once per HSI-MSI pair.
2. State matching: HSI is spectrally projected by the sensor SRF while the
   globally corrected MSI is spatially degraded with the exact Innovation-1
   progressive operator at the current t.
3. Local candidate search in the matched domain to obtain a coarse integer
   displacement field. The field is used to sample the complete Raw MSI before
   the existing Raw-Direct fusion backbone.

No confidence gate, no learned sub-pixel residual, and no local non-rigid
refinement network are included here. Those are intentionally left for later
ablations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from degradations.misalignment import build_global_grid
from degradations.progressive import ProgressiveDegradation
from .predictor_v3_ablation import MSIAblationGuidedPredictor


@dataclass
class CoarseAlignmentDiagnostics:
    """Detached geometry outputs from the latest alignment call."""

    global_shift_px: torch.Tensor
    local_offset_px: torch.Tensor
    matched_hsi: torch.Tensor
    matched_msi: torch.Tensor


def _batch_state_at(
    process: ProgressiveDegradation,
    x: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """Evaluate D~_t(x) for a batch with possibly different integer t."""
    if timesteps.ndim != 1 or timesteps.shape[0] != x.shape[0]:
        raise ValueError("timesteps must have shape [B]")
    out = torch.empty_like(x)
    for t_value in torch.unique(timesteps, sorted=True):
        mask = timesteps == t_value
        out[mask] = process.state_at(x[mask], int(t_value.item()))
    return out


def spectral_project_hsi(x_hsi: torch.Tensor, srf_weights: torch.Tensor) -> torch.Tensor:
    """Apply the fixed sensor spectral response R to HSI: BxCxHxW -> BxMxHxW."""
    if x_hsi.ndim != 4:
        raise ValueError(f"x_hsi must be BxCxHxW, got {tuple(x_hsi.shape)}")
    if srf_weights.ndim != 2:
        raise ValueError("srf_weights must have shape [M,C]")
    if x_hsi.shape[1] != srf_weights.shape[1]:
        raise ValueError(
            f"HSI bands={x_hsi.shape[1]} do not match SRF columns={srf_weights.shape[1]}"
        )
    weights = srf_weights.to(device=x_hsi.device, dtype=x_hsi.dtype)
    return torch.einsum("mc,bchw->bmhw", weights, x_hsi)


def _integer_translate(x: torch.Tensor, dx: int, dy: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Translate content by integer (dx,dy) with zero fill and return validity."""
    b, _, h, w = x.shape
    out = torch.zeros_like(x)
    valid = torch.zeros(b, 1, h, w, device=x.device, dtype=x.dtype)

    if abs(dx) >= w or abs(dy) >= h:
        return out, valid

    src_x0 = max(-dx, 0)
    src_x1 = min(w - dx, w)
    dst_x0 = max(dx, 0)
    dst_x1 = min(w + dx, w)
    src_y0 = max(-dy, 0)
    src_y1 = min(h - dy, h)
    dst_y0 = max(dy, 0)
    dst_y1 = min(h + dy, h)

    out[..., dst_y0:dst_y1, dst_x0:dst_x1] = x[..., src_y0:src_y1, src_x0:src_x1]
    valid[..., dst_y0:dst_y1, dst_x0:dst_x1] = 1.0
    return out, valid


def _pixelwise_channel_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    centered = x - x.mean(dim=1, keepdim=True)
    return F.normalize(centered, dim=1, eps=eps)


def _sample_with_source_offset(x: torch.Tensor, offset_px: torch.Tensor) -> torch.Tensor:
    """Sample x at source coordinate p + offset(p).

    offset_px is Bx2xHxW with channel order (dx,dy). Integer coarse offsets are
    represented on a continuous grid so later sub-pixel residuals can reuse the
    same sampler without changing conventions.
    """
    if offset_px.ndim != 4 or offset_px.shape[1] != 2:
        raise ValueError("offset_px must have shape Bx2xHxW")
    b, _, h, w = x.shape
    if offset_px.shape[0] != b or offset_px.shape[-2:] != (h, w):
        raise ValueError("offset spatial size must match input")

    theta = torch.zeros(b, 2, 3, device=x.device, dtype=x.dtype)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    base = F.affine_grid(theta, size=(b, 1, h, w), align_corners=False)
    grid = base.clone()
    grid[..., 0] = grid[..., 0] + 2.0 * offset_px[:, 0] / float(w)
    grid[..., 1] = grid[..., 1] + 2.0 * offset_px[:, 1] / float(h)
    return F.grid_sample(
        x,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )


class DegradationDomainCoarseAligner:
    """Non-parametric first-stage geometry aligner for Innovation 2.

    The aligner deliberately uses geometry matching rather than a learned dense
    deformation field. Global correction is one translation vector per pair;
    local correspondence is one integer candidate offset per HR pixel.
    """

    def __init__(
        self,
        process: ProgressiveDegradation,
        srf_weights: torch.Tensor,
        *,
        global_search_radius: int = 6,
        local_radius_scale1: int = 1,
        local_radius_scale2: int = 2,
        local_radius_scale4: int = 3,
    ):
        self.process = process
        self.srf_weights = srf_weights.detach().float().cpu()
        self.global_search_radius = int(global_search_radius)
        if self.global_search_radius < 0:
            raise ValueError("global_search_radius must be >= 0")
        self.local_radius_by_scale: Dict[int, int] = {
            1: int(local_radius_scale1),
            2: int(local_radius_scale2),
            4: int(local_radius_scale4),
        }
        if any(v < 0 for v in self.local_radius_by_scale.values()):
            raise ValueError("local search radii must be >= 0")

    def spectral_project(self, x_hsi: torch.Tensor) -> torch.Tensor:
        return spectral_project_hsi(x_hsi, self.srf_weights)

    def matched_states(
        self,
        x_t: torch.Tensor,
        globally_aligned_msi: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Construct Z_H^t=R x_t and Z_M^t=D~_t(Y_M^G)."""
        z_h = self.spectral_project(x_t)
        z_m = _batch_state_at(self.process, globally_aligned_msi, timesteps)
        if z_h.shape != z_m.shape:
            raise ValueError(
                f"matched-domain shapes differ: HSI={z_h.shape}, MSI={z_m.shape}"
            )
        return z_h, z_m

    @torch.no_grad()
    def global_coarse_correction(
        self,
        reference_hsi_state: torch.Tensor,
        hr_msi: torch.Tensor,
        reference_t: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimate one coarse integer translation per pair and warp Raw MSI.

        Matching itself is performed in a state-matched proxy domain at
        ``reference_t``. The selected shift is then applied to the complete Raw
        MSI, preserving all MSI spatial information for later fusion.
        """
        b = reference_hsi_state.shape[0]
        t = torch.full(
            (b,), int(reference_t), dtype=torch.long, device=reference_hsi_state.device
        )
        z_h = self.spectral_project(reference_hsi_state)
        z_m = _batch_state_at(self.process, hr_msi, t)
        h_feat = _pixelwise_channel_normalize(z_h)

        best_score = torch.full((b,), -float("inf"), device=z_h.device)
        best_dx = torch.zeros(b, device=z_h.device, dtype=torch.long)
        best_dy = torch.zeros(b, device=z_h.device, dtype=torch.long)

        radius = self.global_search_radius
        # Search zero displacement first and then increasing distance. Together
        # with the strict improvement test this makes exact/near ties prefer the
        # smaller displacement rather than an arbitrary window corner.
        candidates = [
            (dx, dy)
            for dy in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1)
        ]
        candidates.sort(key=lambda p: (p[0] * p[0] + p[1] * p[1], abs(p[0]) + abs(p[1]), p[1], p[0]))

        for dx, dy in candidates:
            shifted, valid = _integer_translate(z_m, dx, dy)
            m_feat = _pixelwise_channel_normalize(shifted)
            similarity = (h_feat * m_feat).sum(dim=1, keepdim=True)
            denom = valid.sum(dim=(1, 2, 3)).clamp_min(1.0)
            score = (similarity * valid).sum(dim=(1, 2, 3)) / denom
            better = score > (best_score + 1e-7)
            best_score = torch.where(better, score, best_score)
            best_dx = torch.where(better, torch.full_like(best_dx, dx), best_dx)
            best_dy = torch.where(better, torch.full_like(best_dy, dy), best_dy)

        shifts = torch.stack([best_dx, best_dy], dim=1).to(hr_msi.dtype)
        zeros = torch.zeros(b, device=hr_msi.device, dtype=hr_msi.dtype)
        grid = build_global_grid(
            b,
            hr_msi.shape[-2],
            hr_msi.shape[-1],
            shifts[:, 0].to(hr_msi.device),
            shifts[:, 1].to(hr_msi.device),
            zeros,
            device=hr_msi.device,
            dtype=hr_msi.dtype,
        )
        aligned = F.grid_sample(
            hr_msi,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return aligned, shifts.to(hr_msi.device)

    def _radius_for_t(self, t: int) -> int:
        scale = int(self.process.state(int(t)).scale)
        if scale in self.local_radius_by_scale:
            return self.local_radius_by_scale[scale]
        nearest = min(self.local_radius_by_scale, key=lambda s: abs(s - scale))
        return self.local_radius_by_scale[nearest]

    @staticmethod
    def _local_best_offset(
        z_h: torch.Tensor,
        z_m: torch.Tensor,
        radius: int,
    ) -> torch.Tensor:
        """Find q in N_r(p) maximizing same-domain spectral correlation."""
        b, c, h, w = z_h.shape
        if radius == 0:
            return torch.zeros(b, 2, h, w, device=z_h.device, dtype=z_h.dtype)

        kernel = 2 * radius + 1
        k = kernel * kernel
        h_feat = _pixelwise_channel_normalize(z_h)
        m_feat = _pixelwise_channel_normalize(z_m)

        patches = F.unfold(m_feat, kernel_size=kernel, padding=radius)
        patches = patches.view(b, c, k, h, w)
        score = (h_feat.unsqueeze(2) * patches).sum(dim=1)

        ones = torch.ones(b, 1, h, w, device=z_h.device, dtype=z_h.dtype)
        valid = F.unfold(ones, kernel_size=kernel, padding=radius)
        valid = valid.view(b, k, h, w) > 0.5
        score = score.masked_fill(~valid, -float("inf"))

        coord = torch.arange(-radius, radius + 1, device=z_h.device)
        dy_grid, dx_grid = torch.meshgrid(coord, coord, indexing="ij")
        dx_candidates = dx_grid.reshape(-1)
        dy_candidates = dy_grid.reshape(-1)
        distance2 = (dx_candidates.square() + dy_candidates.square()).to(score.dtype)
        # Only resolve numerical/flat-region ties. 1e-6 is tiny compared with
        # cosine-score differences but ensures an ambiguous match prefers the
        # smallest displacement, with zero selected for a fully flat window.
        score = score - 1e-6 * distance2[None, :, None, None]

        best = score.argmax(dim=1)
        dx = dx_candidates[best].to(z_h.dtype)
        dy = dy_candidates[best].to(z_h.dtype)
        return torch.stack([dx, dy], dim=1)

    def local_coarse_correspondence(
        self,
        z_h: torch.Tensor,
        z_m: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Return coarse source offsets delta_t^c(p) in HR pixel units."""
        b, _, h, w = z_h.shape
        offsets = torch.zeros(b, 2, h, w, device=z_h.device, dtype=z_h.dtype)
        for t_value in torch.unique(timesteps, sorted=True):
            mask = timesteps == t_value
            radius = self._radius_for_t(int(t_value.item()))
            offsets[mask] = self._local_best_offset(z_h[mask], z_m[mask], radius)
        return offsets

    def align_local_raw_msi(
        self,
        globally_aligned_msi: torch.Tensor,
        coarse_offset_px: torch.Tensor,
    ) -> torch.Tensor:
        """Sample complete Raw MSI using only the coarse local offset field."""
        return _sample_with_source_offset(globally_aligned_msi, coarse_offset_px)


class StateMatchedCoarseAlignedPredictor(MSIAblationGuidedPredictor):
    """Raw-Direct predictor preceded by the first three Innovation-2 steps."""

    requires_msi = True
    requires_global_msi_preparation = True

    def __init__(
        self,
        *args,
        progressive_process: ProgressiveDegradation,
        srf_weights: torch.Tensor,
        alignment_global_search_radius: int = 6,
        alignment_local_radius_scale1: int = 1,
        alignment_local_radius_scale2: int = 2,
        alignment_local_radius_scale4: int = 3,
        **kwargs,
    ):
        kwargs["msi_ablation"] = "raw_direct"
        super().__init__(*args, **kwargs)
        self.geometry_aligner = DegradationDomainCoarseAligner(
            progressive_process,
            srf_weights,
            global_search_radius=alignment_global_search_radius,
            local_radius_scale1=alignment_local_radius_scale1,
            local_radius_scale2=alignment_local_radius_scale2,
            local_radius_scale4=alignment_local_radius_scale4,
        )
        self.last_alignment: CoarseAlignmentDiagnostics | None = None
        self.last_global_shift_px: torch.Tensor | None = None

    @torch.no_grad()
    def prepare_global_msi(
        self,
        reference_hsi_state: torch.Tensor,
        hr_msi: torch.Tensor,
        reference_t: int,
    ) -> torch.Tensor:
        aligned, shift = self.geometry_aligner.global_coarse_correction(
            reference_hsi_state,
            hr_msi,
            reference_t,
        )
        self.last_global_shift_px = shift.detach()
        return aligned

    def forward(
        self,
        x_t: torch.Tensor,
        globally_aligned_msi: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        if t.ndim == 0:
            t = t.repeat(x_t.shape[0])
        t = t.to(device=x_t.device, dtype=torch.long)

        z_h, z_m = self.geometry_aligner.matched_states(
            x_t,
            globally_aligned_msi,
            t,
        )
        coarse_offset = self.geometry_aligner.local_coarse_correspondence(z_h, z_m, t)
        coarse_aligned_msi = self.geometry_aligner.align_local_raw_msi(
            globally_aligned_msi,
            coarse_offset,
        )

        global_shift = self.last_global_shift_px
        if global_shift is None:
            global_shift = torch.zeros(
                x_t.shape[0], 2, device=x_t.device, dtype=x_t.dtype
            )
        self.last_alignment = CoarseAlignmentDiagnostics(
            global_shift_px=global_shift.detach(),
            local_offset_px=coarse_offset.detach(),
            matched_hsi=z_h.detach(),
            matched_msi=z_m.detach(),
        )

        return super().forward(x_t, coarse_aligned_msi, t)


__all__ = [
    "CoarseAlignmentDiagnostics",
    "DegradationDomainCoarseAligner",
    "StateMatchedCoarseAlignedPredictor",
    "spectral_project_hsi",
]
