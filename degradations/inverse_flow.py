"""Convert synthetic forward local deformation into the inverse sampling field.

The synthetic local warp in ``misalignment.py`` is defined by

    Y_warp(q) = Y(q - d(q))

where ``d`` is the generated forward/content displacement field.  The local
aligner, however, restores Raw MSI with positive source-coordinate sampling:

    Y_align(p) = Y_warp(p + u(p)).

For exact geometric consistency the required field is therefore not generally
``u=d``.  It satisfies the implicit inverse relation

    u(p) = d(p + u(p)).

For the smooth, small non-rigid fields used by S2Diff this can be solved by
fixed-point iteration.  All fields remain in HR-pixel units.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


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


def positive_offset_grid(offset_px: torch.Tensor) -> torch.Tensor:
    """Build a grid that samples source coordinate ``p + offset(p)``.

    ``offset_px`` must be Bx2xHxW in HR-pixel units with channel order dx,dy.
    This convention matches ``_sample_with_source_offset`` used by the V4
    alignment predictor.
    """
    if offset_px.ndim != 4 or offset_px.shape[1] != 2:
        raise ValueError(
            "offset_px must have shape Bx2xHxW, got "
            f"{tuple(offset_px.shape)}"
        )
    b, _, h, w = offset_px.shape
    grid = _identity_grid(
        b,
        h,
        w,
        device=offset_px.device,
        dtype=offset_px.dtype,
    ).clone()
    grid[..., 0] = grid[..., 0] + 2.0 * offset_px[:, 0] / float(w)
    grid[..., 1] = grid[..., 1] + 2.0 * offset_px[:, 1] / float(h)
    return grid


def sample_with_positive_offset(
    x: torch.Tensor,
    offset_px: torch.Tensor,
    *,
    padding_mode: str = "zeros",
) -> torch.Tensor:
    """Sample ``x`` at ``p + offset(p)`` with bilinear interpolation."""
    if x.ndim != 4:
        raise ValueError(f"x must be BxCxHxW, got {tuple(x.shape)}")
    if offset_px.shape[0] != x.shape[0] or offset_px.shape[-2:] != x.shape[-2:]:
        raise ValueError("offset batch/spatial shape must match x")
    return F.grid_sample(
        x,
        positive_offset_grid(offset_px),
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=False,
    )


def forward_to_inverse_sampling_field(
    forward_displacement_px: torch.Tensor,
    *,
    iterations: int = 8,
    padding_mode: str = "border",
) -> torch.Tensor:
    """Solve ``u(p)=d(p+u(p))`` for the inverse sampling displacement.

    The input ``d`` is the forward/content field used by ``build_local_grid`` in
    ``misalignment.py``.  The returned ``u`` is directly compatible with the V4
    positive-offset Raw-MSI sampler.

    ``border`` padding is intentional while solving the field: it avoids
    artificial zero vectors at the image boundary.  Reconstruction metrics use
    valid-overlap masks, so boundary extrapolation does not define the reported
    geometric accuracy.
    """
    if forward_displacement_px.ndim != 4 or forward_displacement_px.shape[1] != 2:
        raise ValueError(
            "forward_displacement_px must have shape Bx2xHxW, got "
            f"{tuple(forward_displacement_px.shape)}"
        )
    if int(iterations) < 1:
        raise ValueError("iterations must be >= 1")
    if padding_mode not in {"zeros", "border", "reflection"}:
        raise ValueError("unsupported padding_mode")

    d = forward_displacement_px
    u = d.clone()
    for _ in range(int(iterations)):
        u = sample_with_positive_offset(
            d,
            u,
            padding_mode=padding_mode,
        )
    return u


def inverse_fixed_point_residual(
    forward_displacement_px: torch.Tensor,
    inverse_sampling_px: torch.Tensor,
    *,
    padding_mode: str = "border",
) -> torch.Tensor:
    """Return per-pixel residual ``u - d(p+u)`` in pixel units."""
    if forward_displacement_px.shape != inverse_sampling_px.shape:
        raise ValueError("forward and inverse fields must share shape")
    sampled_forward = sample_with_positive_offset(
        forward_displacement_px,
        inverse_sampling_px,
        padding_mode=padding_mode,
    )
    return inverse_sampling_px - sampled_forward


def inverse_field_diagnostics(
    forward_displacement_px: torch.Tensor,
    inverse_sampling_px: torch.Tensor,
) -> Tuple[float, float, float]:
    """Return forward mean magnitude, inverse mean magnitude and residual EPE."""
    with torch.no_grad():
        forward_mag = torch.linalg.vector_norm(
            forward_displacement_px.float(), dim=1
        ).mean()
        inverse_mag = torch.linalg.vector_norm(
            inverse_sampling_px.float(), dim=1
        ).mean()
        residual = inverse_fixed_point_residual(
            forward_displacement_px.float(),
            inverse_sampling_px.float(),
        )
        residual_epe = torch.linalg.vector_norm(residual, dim=1).mean()
    return (
        float(forward_mag.item()),
        float(inverse_mag.item()),
        float(residual_epe.item()),
    )


__all__ = [
    "forward_to_inverse_sampling_field",
    "inverse_field_diagnostics",
    "inverse_fixed_point_residual",
    "positive_offset_grid",
    "sample_with_positive_offset",
]
