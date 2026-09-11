import torch

from degradations.inverse_flow import (
    forward_to_inverse_sampling_field,
    inverse_fixed_point_residual,
    sample_with_positive_offset,
)
from degradations.misalignment import (
    build_local_grid,
    generate_smooth_local_displacement,
)


def _mean_epe(field: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(field.float(), dim=1).mean().item())


def test_constant_translation_inverse_equals_forward():
    d = torch.zeros(2, 2, 32, 40)
    d[:, 0] = 0.75
    d[:, 1] = -0.40
    u = forward_to_inverse_sampling_field(d, iterations=6)
    assert torch.allclose(u, d, atol=1e-6, rtol=0.0)


def test_fixed_point_inverse_reduces_nonrigid_composition_residual():
    generator = torch.Generator(device="cpu")
    generator.manual_seed(10)
    d = generate_smooth_local_displacement(
        2,
        64,
        64,
        max_displacement_px=2.0,
        control_grid_size=5,
        generator=generator,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    # Naively using forward d as the positive restoration offset leaves the
    # non-rigid composition residual d(p)-d(p+d(p)).
    naive_residual = inverse_fixed_point_residual(d, d)
    u = forward_to_inverse_sampling_field(d, iterations=10)
    inverse_residual = inverse_fixed_point_residual(d, u)

    naive_epe = _mean_epe(naive_residual)
    inverse_epe = _mean_epe(inverse_residual)
    assert naive_epe > 1e-4
    assert inverse_epe < naive_epe * 0.20
    assert inverse_epe < 0.02


def test_inverse_sampling_improves_local_warp_round_trip():
    generator = torch.Generator(device="cpu")
    generator.manual_seed(23)
    h = w = 64
    d = generate_smooth_local_displacement(
        1,
        h,
        w,
        max_displacement_px=2.0,
        control_grid_size=5,
        generator=generator,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    y = torch.linspace(0.0, 1.0, h)
    x = torch.linspace(0.0, 1.0, w)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    image = torch.stack(
        [
            xx,
            yy,
            0.5 + 0.25 * torch.sin(6.0 * torch.pi * xx) * torch.cos(4.0 * torch.pi * yy),
        ],
        dim=0,
    ).unsqueeze(0)

    warped = torch.nn.functional.grid_sample(
        image,
        build_local_grid(d),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    naive = sample_with_positive_offset(warped, d, padding_mode="zeros")
    u = forward_to_inverse_sampling_field(d, iterations=10)
    restored = sample_with_positive_offset(warped, u, padding_mode="zeros")

    # Ignore the boundary region where the original forward warp samples zeros.
    margin = 5
    target = image[:, :, margin:-margin, margin:-margin]
    naive = naive[:, :, margin:-margin, margin:-margin]
    restored = restored[:, :, margin:-margin, margin:-margin]
    naive_mae = (naive - target).abs().mean()
    inverse_mae = (restored - target).abs().mean()
    assert inverse_mae < naive_mae


def test_zero_field_stays_zero():
    d = torch.zeros(1, 2, 24, 24)
    u = forward_to_inverse_sampling_field(d, iterations=8)
    assert torch.count_nonzero(u) == 0
