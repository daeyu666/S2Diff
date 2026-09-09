import torch

from degradations import BicubicDegradation, ProgressiveDegradation
from models import MSIAblationGuidedPredictor, StateMatchedCoarseAlignedPredictor


def test_v3_raw_direct_warm_starts_v4_backbone_with_geometry_missing_only():
    process = ProgressiveDegradation(
        BicubicDegradation(scale_ratio=1),
        total_steps=1,
        default_lift_mode="bilinear",
    )
    common = dict(
        n_bands=3,
        n_msi_bands=3,
        total_steps=1,
        base_channels=8,
        time_dim=32,
        dropout=0.0,
        residual_prediction=True,
        spectral_hidden=4,
    )
    v3 = MSIAblationGuidedPredictor(**common, msi_ablation="raw_direct")
    v4 = StateMatchedCoarseAlignedPredictor(
        **common,
        progressive_process=process,
        srf_weights=torch.eye(3),
        alignment_descriptor_channels=8,
        alignment_global_search_radius=0,
        alignment_global_rotation_max_deg=0.0,
        alignment_control_stride=4,
    )

    result = v4.load_state_dict(v3.state_dict(), strict=False)
    assert not result.unexpected_keys
    assert result.missing_keys
    assert all(key.startswith("geometry_aligner.") for key in result.missing_keys)

    v4_state = v4.state_dict()
    for key, value in v3.state_dict().items():
        assert torch.equal(v4_state[key], value)

    geometry_params = [
        p
        for name, p in v4.named_parameters()
        if name.startswith("geometry_aligner.") and p.requires_grad
    ]
    assert geometry_params
