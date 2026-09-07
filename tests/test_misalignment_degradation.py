import torch

from degradations.misalignment import (
    MisalignmentParameters,
    apply_misalignment,
    generate_smooth_local_displacement,
    make_misaligned_msi,
)


def test_registered_misalignment_is_identity():
    x = torch.rand(2, 4, 32, 32)
    generator = torch.Generator().manual_seed(10)
    warped, valid, params = make_misaligned_msi(
        x,
        translation_max_px=0.0,
        rotation_max_deg=0.0,
        local_max_displacement_px=0.0,
        generator=generator,
    )
    assert torch.allclose(warped, x, atol=1e-6, rtol=1e-6)
    assert torch.allclose(valid, torch.ones_like(valid), atol=1e-6, rtol=1e-6)
    assert torch.count_nonzero(params.local_displacement_px) == 0


def test_integer_translation_moves_impulse_and_reduces_valid_area():
    x = torch.zeros(1, 1, 17, 17)
    x[0, 0, 8, 8] = 1.0
    local = torch.zeros(1, 2, 17, 17)
    params = MisalignmentParameters(
        dx_px=torch.tensor([2.0]),
        dy_px=torch.tensor([1.0]),
        rotation_deg=torch.tensor([0.0]),
        local_displacement_px=local,
    )
    warped, valid = apply_misalignment(x, params)
    peak = torch.nonzero(warped[0, 0] == warped[0, 0].max(), as_tuple=False)[0]
    assert tuple(peak.tolist()) == (9, 10)
    assert float(valid.min()) < 1.0
    assert float(valid.mean()) < 1.0


def test_local_field_obeys_requested_euclidean_bound():
    generator = torch.Generator().manual_seed(123)
    field = generate_smooth_local_displacement(
        3,
        48,
        40,
        max_displacement_px=2.0,
        control_grid_size=5,
        generator=generator,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    magnitude = torch.linalg.vector_norm(field, dim=1)
    assert field.shape == (3, 2, 48, 40)
    assert float(magnitude.max()) <= 2.0 + 1e-5
    assert float(magnitude.mean()) > 0.0


def test_complete_warp_is_deterministic_for_same_seed():
    x = torch.rand(1, 4, 32, 32)
    g1 = torch.Generator().manual_seed(77)
    g2 = torch.Generator().manual_seed(77)
    out1, mask1, params1 = make_misaligned_msi(
        x,
        translation_max_px=2.0,
        rotation_max_deg=1.0,
        local_max_displacement_px=1.0,
        control_grid_size=5,
        generator=g1,
    )
    out2, mask2, params2 = make_misaligned_msi(
        x,
        translation_max_px=2.0,
        rotation_max_deg=1.0,
        local_max_displacement_px=1.0,
        control_grid_size=5,
        generator=g2,
    )
    assert torch.allclose(out1, out2)
    assert torch.allclose(mask1, mask2)
    assert torch.allclose(params1.dx_px, params2.dx_px)
    assert torch.allclose(params1.dy_px, params2.dy_px)
    assert torch.allclose(params1.rotation_deg, params2.rotation_deg)
    assert torch.allclose(
        params1.local_displacement_px,
        params2.local_displacement_px,
    )


def test_valid_mask_tracks_same_global_local_warp():
    x = torch.ones(1, 3, 32, 32)
    generator = torch.Generator().manual_seed(99)
    warped, valid, _ = make_misaligned_msi(
        x,
        translation_max_px=3.0,
        rotation_max_deg=2.0,
        local_max_displacement_px=1.5,
        control_grid_size=5,
        generator=generator,
    )
    # Warping an all-one image with zero padding should equal the validity
    # image channel-wise because the exact same grids are used.
    for channel in range(warped.shape[1]):
        assert torch.allclose(warped[:, channel : channel + 1], valid, atol=1e-6)
