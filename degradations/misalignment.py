"""Reusable synthetic HSI-MSI misalignment degradation.

Only the HR-MSI observation is warped. HR-HSI ground truth and LR-HSI remain
unchanged. The module supports registered, global-only, local-only and
combined global+local protocols.

Global deformation contains translation and small-angle rotation. Local
non-rigid deformation is generated on a coarse control grid and bicubically
interpolated to a dense, smooth displacement field. The exact same warps are
applied to an all-one image to obtain a soft validity mask.

All spatial transforms use continuous coordinates and bilinear sampling, so
sub-pixel misregistration is represented explicitly.

IMPORTANT TRANSLATION-SEVERITY DEFINITION
----------------------------------------
``translation_max_px = d`` means the Euclidean magnitude of the global
translation is bounded by d pixels:

    r ~ U(0, d), theta ~ U(0, 2*pi)
    dx = r*cos(theta), dy = r*sin(theta)
    sqrt(dx^2 + dy^2) = r <= d

This replaces the older independent-axis definition dx,dy~U(-d,d), whose
actual 2-D displacement could reach sqrt(2)*d.  Keeping the random radius and
direction normalized also makes paired severity sweeps geometrically clean:
using the same seed at d=0.5/1/2/... reuses the same normalized radius and
direction and only scales the displacement magnitude.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class MisalignmentParameters:
    """One batch of sampled geometric perturbations."""

    dx_px: torch.Tensor
    dy_px: torch.Tensor
    rotation_deg: torch.Tensor
    local_displacement_px: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.dx_px.shape[0])


def _base_grid(
    batch_size: int,
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Identity grid compatible with grid_sample(..., align_corners=False)."""
    theta = torch.zeros(batch_size, 2, 3, device=device, dtype=dtype)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    return F.affine_grid(
        theta,
        size=(batch_size, 1, height, width),
        align_corners=False,
    )


def build_global_grid(
    batch_size: int,
    height: int,
    width: int,
    dx_px: torch.Tensor,
    dy_px: torch.Tensor,
    rotation_deg: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build inverse-sampling grid for forward translation + rotation.

    Positive dx moves image content right and positive dy moves it down. The
    rotation is around the image center. The affine matrix maps output
    coordinates back to source coordinates, as required by ``affine_grid``.
    """
    for name, value in (
        ("dx_px", dx_px),
        ("dy_px", dy_px),
        ("rotation_deg", rotation_deg),
    ):
        if value.ndim != 1 or value.shape[0] != batch_size:
            raise ValueError(f"{name} must have shape [B={batch_size}]")

    dx = dx_px.to(device=device, dtype=dtype)
    dy = dy_px.to(device=device, dtype=dtype)
    angle = rotation_deg.to(device=device, dtype=dtype) * torch.pi / 180.0

    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)

    tx = 2.0 * dx / float(width)
    ty = 2.0 * dy / float(height)

    theta = torch.zeros(batch_size, 2, 3, device=device, dtype=dtype)
    theta[:, 0, 0] = cos_a
    theta[:, 0, 1] = sin_a
    theta[:, 1, 0] = -sin_a
    theta[:, 1, 1] = cos_a
    theta[:, 0, 2] = -(cos_a * tx + sin_a * ty)
    theta[:, 1, 2] = sin_a * tx - cos_a * ty

    return F.affine_grid(
        theta,
        size=(batch_size, 1, height, width),
        align_corners=False,
    )


def generate_smooth_local_displacement(
    batch_size: int,
    height: int,
    width: int,
    *,
    max_displacement_px: float,
    control_grid_size: int = 5,
    generator: Optional[torch.Generator] = None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Generate a smooth Bx2xHxW local displacement field in pixel units.

    Random displacement vectors are sampled on a coarse control grid, then
    bicubically interpolated. The final Euclidean displacement magnitude is
    clipped to ``max_displacement_px`` so the requested severity has a precise
    geometric meaning.
    """
    if control_grid_size < 2:
        raise ValueError("control_grid_size must be >= 2")
    max_disp = float(max_displacement_px)
    if max_disp < 0.0:
        raise ValueError("max_displacement_px must be >= 0")
    if max_disp == 0.0:
        return torch.zeros(
            batch_size, 2, height, width, device=device, dtype=dtype
        )

    controls = torch.rand(
        batch_size,
        2,
        control_grid_size,
        control_grid_size,
        generator=generator,
        device="cpu",
        dtype=torch.float32,
    ) * 2.0 - 1.0

    norm = torch.linalg.vector_norm(controls, dim=1, keepdim=True).clamp_min(1e-8)
    controls = controls / torch.maximum(norm, torch.ones_like(norm))
    controls = controls * max_disp
    controls = controls.to(device=device, dtype=dtype)

    dense = F.interpolate(
        controls,
        size=(height, width),
        mode="bicubic",
        align_corners=True,
    )

    dense_norm = torch.linalg.vector_norm(dense, dim=1, keepdim=True).clamp_min(1e-8)
    scale = torch.clamp(max_disp / dense_norm, max=1.0)
    return dense * scale


def build_local_grid(local_displacement_px: torch.Tensor) -> torch.Tensor:
    """Convert a dense content-displacement field to an inverse sampling grid."""
    if local_displacement_px.ndim != 4 or local_displacement_px.shape[1] != 2:
        raise ValueError(
            "local_displacement_px must have shape Bx2xHxW, got "
            f"{tuple(local_displacement_px.shape)}"
        )
    batch_size, _, height, width = local_displacement_px.shape
    base = _base_grid(
        batch_size,
        height,
        width,
        device=local_displacement_px.device,
        dtype=local_displacement_px.dtype,
    )
    dx = local_displacement_px[:, 0]
    dy = local_displacement_px[:, 1]
    grid = base.clone()
    grid[..., 0] = grid[..., 0] - 2.0 * dx / float(width)
    grid[..., 1] = grid[..., 1] - 2.0 * dy / float(height)
    return grid


def _warp(x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    return F.grid_sample(
        x,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )


def sample_misalignment_parameters(
    batch_size: int,
    height: int,
    width: int,
    *,
    translation_max_px: float = 0.0,
    rotation_max_deg: float = 0.0,
    local_max_displacement_px: float = 0.0,
    control_grid_size: int = 5,
    generator: Optional[torch.Generator] = None,
    device: torch.device,
    dtype: torch.dtype,
) -> MisalignmentParameters:
    """Sample one batch of global and local perturbation parameters.

    ``translation_max_px=d`` is the *maximum Euclidean translation magnitude*,
    not an independent x/y bound:

      radius ~ U(0, d)
      theta  ~ U(0, 2*pi)
      dx = radius*cos(theta), dy = radius*sin(theta)

    Hence sqrt(dx^2+dy^2) <= d for every sample. Rotation remains
    U(-rotation_max_deg, rotation_max_deg).
    """
    tmax = float(translation_max_px)
    rmax = float(rotation_max_deg)
    if tmax < 0.0 or rmax < 0.0:
        raise ValueError("translation/rotation maxima must be >= 0")

    # Always consume three unit random variables, even when a maximum is zero.
    # This preserves deterministic paired severity sweeps.  Across d values,
    # the same seed reuses the same normalized radius, direction and rotation.
    unit = torch.rand(
        batch_size,
        3,
        generator=generator,
        device="cpu",
        dtype=torch.float32,
    )
    radius = unit[:, 0] * tmax
    theta = unit[:, 1] * (2.0 * torch.pi)
    dx = (radius * torch.cos(theta)).to(device=device, dtype=dtype)
    dy = (radius * torch.sin(theta)).to(device=device, dtype=dtype)
    angle = ((unit[:, 2] * 2.0 - 1.0) * rmax).to(device=device, dtype=dtype)

    local = generate_smooth_local_displacement(
        batch_size,
        height,
        width,
        max_displacement_px=float(local_max_displacement_px),
        control_grid_size=int(control_grid_size),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    return MisalignmentParameters(dx, dy, angle, local)


def apply_misalignment(
    hr_msi: torch.Tensor,
    params: MisalignmentParameters,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply global then local warp and return warped MSI + soft validity mask."""
    if hr_msi.ndim != 4:
        raise ValueError(f"hr_msi must be BxCxHxW, got {tuple(hr_msi.shape)}")
    batch_size, _, height, width = hr_msi.shape
    if params.batch_size != batch_size:
        raise ValueError(
            f"parameter batch={params.batch_size} does not match MSI batch={batch_size}"
        )
    if params.local_displacement_px.shape != (batch_size, 2, height, width):
        raise ValueError(
            "local displacement shape must match MSI spatial size; got "
            f"{tuple(params.local_displacement_px.shape)}"
        )

    ones = torch.ones(
        batch_size,
        1,
        height,
        width,
        device=hr_msi.device,
        dtype=hr_msi.dtype,
    )

    global_grid = build_global_grid(
        batch_size,
        height,
        width,
        params.dx_px,
        params.dy_px,
        params.rotation_deg,
        device=hr_msi.device,
        dtype=hr_msi.dtype,
    )
    warped = _warp(hr_msi, global_grid)
    valid = _warp(ones, global_grid)

    local_grid = build_local_grid(
        params.local_displacement_px.to(device=hr_msi.device, dtype=hr_msi.dtype)
    )
    warped = _warp(warped, local_grid)
    valid = _warp(valid, local_grid)
    return warped, valid.clamp(0.0, 1.0)


def make_misaligned_msi(
    hr_msi: torch.Tensor,
    *,
    translation_max_px: float = 0.0,
    rotation_max_deg: float = 0.0,
    local_max_displacement_px: float = 0.0,
    control_grid_size: int = 5,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, MisalignmentParameters]:
    """Sample perturbations, warp MSI and return mask + parameters."""
    if hr_msi.ndim != 4:
        raise ValueError(f"hr_msi must be BxCxHxW, got {tuple(hr_msi.shape)}")
    batch_size, _, height, width = hr_msi.shape
    params = sample_misalignment_parameters(
        batch_size,
        height,
        width,
        translation_max_px=translation_max_px,
        rotation_max_deg=rotation_max_deg,
        local_max_displacement_px=local_max_displacement_px,
        control_grid_size=control_grid_size,
        generator=generator,
        device=hr_msi.device,
        dtype=hr_msi.dtype,
    )
    warped, valid = apply_misalignment(hr_msi, params)
    return warped, valid, params


__all__ = [
    "MisalignmentParameters",
    "apply_misalignment",
    "build_global_grid",
    "build_local_grid",
    "generate_smooth_local_displacement",
    "make_misaligned_msi",
    "sample_misalignment_parameters",
]
