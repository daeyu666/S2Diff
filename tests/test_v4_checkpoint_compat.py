import torch

from degradations import BicubicDegradation, ProgressiveDegradation
from models import MSIAblationGuidedPredictor, StateMatchedCoarseAlignedPredictor


def test_v4_state_dict_matches_v3_raw_direct_exactly():
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
    )

    assert set(v3.state_dict().keys()) == set(v4.state_dict().keys())
    v4.load_state_dict(v3.state_dict(), strict=True)
