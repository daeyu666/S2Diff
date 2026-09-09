import torch

from degradations import BicubicDegradation, ProgressiveDegradation
from models.predictor_v4_alignment import (
    DegradationDomainCoarseAligner,
    SparseProgressiveLocalAligner,
    StateMatchedCoarseAlignedPredictor,
    spectral_project_hsi,
)


def _scale1_process():
    return ProgressiveDegradation(
        BicubicDegradation(scale_ratio=1),
        total_steps=1,
        default_lift_mode="bilinear",
    )


def _scale4_process():
    return ProgressiveDegradation(
        BicubicDegradation(scale_ratio=4),
        total_steps=12,
        default_lift_mode="bilinear",
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


def test_v4_alignment_frontend_has_trainable_geometry_parameters():
    process = _scale1_process()
    aligner = DegradationDomainCoarseAligner(
        process,
        torch.eye(3),
        n_msi_bands=3,
        descriptor_channels=8,
        global_search_radius=1,
        global_rotation_max_deg=1.0,
        global_rotation_step_deg=1.0,
        global_feature_downsample=2,
        global_candidate_chunk=8,
        control_stride=4,
        local_radius_scale1=1,
        local_radius_scale2=1,
        local_radius_scale4=1,
    )
    names = [name for name, p in aligner.named_parameters() if p.requires_grad]
    assert any("global_aligner.descriptor" in name for name in names)
    assert any("local_aligner.descriptor" in name for name in names)
    assert any("global_aligner.log_temperature" in name for name in names)
    assert any("local_aligner.log_temperature" in name for name in names)


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
    aligner = DegradationDomainCoarseAligner(
        process,
        srf,
        n_msi_bands=2,
        descriptor_channels=8,
        global_search_radius=0,
        global_rotation_max_deg=0.0,
    )
    hsi = torch.rand(2, 3, 16, 16)
    msi = spectral_project_hsi(hsi, srf)
    t = torch.ones(2, dtype=torch.long)

    z_h, z_m = aligner.matched_states(hsi, msi, t)
    assert z_h.shape == z_m.shape == (2, 2, 16, 16)
    assert torch.allclose(z_h, z_m, atol=1e-6, rtol=1e-6)


def test_sparse_local_search_is_differentiable_and_outputs_smooth_dense_field():
    torch.manual_seed(5)
    local = SparseProgressiveLocalAligner(
        n_msi_bands=3,
        descriptor_channels=8,
        control_stride=4,
        radius_by_scale={1: 1, 2: 1, 4: 1},
    )
    z_h = torch.rand(2, 3, 20, 20)
    z_m = torch.rand(2, 3, 20, 20)
    dense, control = local(z_h, z_m, None, scale=4)

    assert dense.shape == (2, 2, 20, 20)
    assert control.shape[-2] < dense.shape[-2]
    assert control.shape[-1] < dense.shape[-1]

    loss = dense.square().mean()
    loss.backward()
    grad = local.descriptor.net[0].weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert float(grad.abs().sum()) > 0.0


def test_v4_inference_updates_local_field_only_at_scale_boundaries():
    torch.manual_seed(6)
    process = _scale4_process()
    model = StateMatchedCoarseAlignedPredictor(
        n_bands=3,
        n_msi_bands=3,
        total_steps=12,
        base_channels=8,
        time_dim=32,
        dropout=0.0,
        residual_prediction=True,
        spectral_hidden=4,
        progressive_process=process,
        srf_weights=torch.eye(3),
        alignment_descriptor_channels=8,
        alignment_global_search_radius=0,
        alignment_global_rotation_max_deg=0.0,
        alignment_global_feature_downsample=2,
        alignment_control_stride=4,
        alignment_local_radius_scale1=1,
        alignment_local_radius_scale2=1,
        alignment_local_radius_scale4=1,
    )
    model.eval()
    gt = torch.rand(1, 3, 16, 16)
    msi = gt.clone()
    x12 = process.state_at(gt, 12)
    prepared = model.prepare_global_msi(x12, msi, reference_t=12)

    ids = []
    scales = []
    with torch.no_grad():
        for t in [12, 11, 8, 7, 4, 3]:
            x_t = process.state_at(gt, t)
            out = model(x_t, prepared, torch.tensor([t]))
            assert out.shape == x_t.shape
            ids.append(id(model._inference_local_offset))
            scales.append(model._inference_local_scale)

    assert scales == [4, 4, 2, 2, 1, 1]
    assert ids[0] == ids[1]
    assert ids[2] == ids[3]
    assert ids[4] == ids[5]
    assert ids[0] != ids[2]
    assert ids[2] != ids[4]
