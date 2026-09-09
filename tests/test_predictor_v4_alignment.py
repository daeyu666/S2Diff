import torch
import torch.nn.functional as F

from degradations import BicubicDegradation, ProgressiveDegradation
from degradations.misalignment import build_global_grid
from models.predictor_v4_alignment import (
    DegradationDomainCoarseAligner,
    StateMatchedCoarseAlignedPredictor,
    spectral_project_hsi,
)


def _scale1_process():
    return ProgressiveDegradation(
        BicubicDegradation(scale_ratio=1),
        total_steps=1,
        default_lift_mode="bilinear",
    )


def _forward_translate(x, dx, dy):
    b = x.shape[0]
    zeros = torch.zeros(b, dtype=x.dtype, device=x.device)
    grid = build_global_grid(
        b,
        x.shape[-2],
        x.shape[-1],
        torch.full((b,), float(dx), dtype=x.dtype, device=x.device),
        torch.full((b,), float(dy), dtype=x.dtype, device=x.device),
        zeros,
        device=x.device,
        dtype=x.dtype,
    )
    return F.grid_sample(
        x,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )


def test_spectral_projection_applies_fixed_srf():
    x = torch.tensor(
        [[[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]]]
    )
    srf = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.25, 0.75],
        ]
    )
    out = spectral_project_hsi(x, srf)
    expected = torch.tensor([[[[1.0, 2.0]], [[4.5, 5.5]]]])
    assert torch.allclose(out, expected)


def test_global_coarse_correction_recovers_integer_translation():
    torch.manual_seed(3)
    process = _scale1_process()
    srf = torch.eye(3)
    aligner = DegradationDomainCoarseAligner(
        process,
        srf,
        global_search_radius=3,
        local_radius_scale1=1,
        local_radius_scale2=1,
        local_radius_scale4=1,
    )

    reference = torch.rand(1, 3, 24, 24)
    misaligned = _forward_translate(reference, dx=2, dy=-1)
    corrected, shift = aligner.global_coarse_correction(
        reference,
        misaligned,
        reference_t=1,
    )

    # MSI content was moved right by +2 and up by -1. Correction must apply
    # the inverse content translation: left 2 and down 1.
    assert tuple(shift[0].round().to(torch.int64).tolist()) == (-2, 1)
    assert torch.mean(torch.abs(corrected[..., 3:-3, 3:-3] - reference[..., 3:-3, 3:-3])) < 1e-5


def test_matched_states_remove_spectral_and_spatial_operator_difference():
    torch.manual_seed(4)
    process = _scale1_process()
    srf = torch.tensor(
        [
            [0.5, 0.5, 0.0],
            [0.0, 0.25, 0.75],
        ],
        dtype=torch.float32,
    )
    aligner = DegradationDomainCoarseAligner(process, srf)
    hsi = torch.rand(2, 3, 16, 16)
    msi = spectral_project_hsi(hsi, srf)
    t = torch.ones(2, dtype=torch.long)

    z_h, z_m = aligner.matched_states(hsi, msi, t)
    assert z_h.shape == z_m.shape == (2, 2, 16, 16)
    assert torch.allclose(z_h, z_m, atol=1e-6, rtol=1e-6)


def test_local_candidate_search_outputs_source_offset_toward_best_msi_match():
    torch.manual_seed(5)
    process = _scale1_process()
    srf = torch.eye(3)
    aligner = DegradationDomainCoarseAligner(
        process,
        srf,
        global_search_radius=0,
        local_radius_scale1=2,
        local_radius_scale2=2,
        local_radius_scale4=2,
    )

    z_h = torch.rand(1, 3, 20, 20)
    # Move MSI content right by one pixel. For HSI location p, the matching MSI
    # source location is therefore q=p+(+1,0).
    z_m = _forward_translate(z_h, dx=1, dy=0)
    offset = aligner.local_coarse_correspondence(
        z_h,
        z_m,
        torch.ones(1, dtype=torch.long),
    )

    interior = offset[0, :, 3:-3, 3:-3]
    assert torch.mean((interior[0] == 1).float()) > 0.99
    assert torch.mean((interior[1] == 0).float()) > 0.99


def test_v4_forward_keeps_raw_direct_zero_head_initialization():
    torch.manual_seed(6)
    process = _scale1_process()
    model = StateMatchedCoarseAlignedPredictor(
        n_bands=3,
        n_msi_bands=3,
        total_steps=1,
        base_channels=8,
        time_dim=32,
        dropout=0.0,
        residual_prediction=True,
        spectral_hidden=4,
        progressive_process=process,
        srf_weights=torch.eye(3),
        alignment_global_search_radius=1,
        alignment_local_radius_scale1=1,
        alignment_local_radius_scale2=1,
        alignment_local_radius_scale4=1,
    )
    x_t = torch.rand(1, 3, 16, 16)
    msi = x_t.clone()
    prepared = model.prepare_global_msi(x_t, msi, reference_t=1)
    out = model(x_t, prepared, torch.ones(1, dtype=torch.long))

    assert out.shape == x_t.shape
    assert torch.allclose(out, x_t, atol=1e-6, rtol=1e-6)
    assert model.last_alignment is not None
    assert model.last_alignment.local_offset_px.shape == (1, 2, 16, 16)
