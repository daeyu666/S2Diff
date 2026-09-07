"""Reusable synthetic HSI-MSI misalignment degradation.

Only the HR-MSI observation is warped. HR-HSI ground truth and LR-HSI remain
unchanged. The module supports the four protocols used by the current study:
registered, global-only, local-only and global+local.

Global deformation contains translation and small-angle rotation. Local
non-rigid deformation is generated on a coarse control grid and bicubically
interpolated to a dense, smooth displacement field. The exact same warps are
applied to an all-one image to obtain a soft validity mask.

All spatial transforms use continuous coordinates and bilinear sampling, so
sub-pixel misregistration is represented explicitly.
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

    # Desired forward translation in normalized image coordinates.
    tx = 2.0 * dx / float(width)
    ty = 2.0 * dy / float(height)

    # Inverse of y = R(a) x + t is x = R(-a) y - R(-a) t.
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

    # Sample on CPU so a normal CPU torch.Generator can be reused regardless
    # of whether inference runs on CPU or CUDA.
    controls = torch.rand(
        batch_size,
        2,
        control_grid_size,
        control_grid_size,
        generator=generator,
        device="cpu",
        dtype=torch.float32,
    ) * 2.0 - 1.0

    # Bound each coarse displacement vector to the unit disk.
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

    # Bicubic interpolation may overshoot near control points. Re-project the
    # dense field onto the requested displacement ball.
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
    # Source = output - displacement, so positive displacement moves content
    # right/down in the warped image.
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

    Global parameters follow
      dx,dy ~ U(-translation_max_px, translation_max_px)
      angle ~ U(-rotation_max_deg, rotation_max_deg).
    """
    tmax = float(translation_max_px)
    rmax = float(rotation_max_deg)
    if tmax < 0.0 or rmax < 0.0:
        raise ValueError("translation/rotation maxima must be >= 0")

    # Always draw unit global variables, even when a maximum is zero. This
    # keeps RNG consumption identical across paired severity sweeps.
    unit = torch.rand(
        batch_size,
        3,
        generator=generator,
        device="cpu",
        dtype=torch.float32,
    ) * 2.0 - 1.0
    dx = (unit[:, 0] * tmax).to(device=device, dtype=dtype)
    dy = (unit[:, 1] * tmax).to(device=device, dtype=dtype)
    angle = (unit[:, 2] * rmax).to(device=device, dtype=dtype)

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
    """Apply global then local warp and return warped MSI + soft validity mask.

    The returned validity mask is produced by applying the exact same sequence
    of warps to an all-one image. It is therefore suitable for valid-overlap
    evaluation after thresholding, e.g. ``valid_soft >= 0.999``.
    """
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
    """Convenience wrapper: sample perturbations, warp MSI and return mask."""
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
