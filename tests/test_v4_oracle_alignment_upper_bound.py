import torch

from degradations.inverse_flow import forward_to_inverse_sampling_field
from degradations.misalignment import MisalignmentParameters, apply_misalignment
from diagnose_v4_oracle_alignment_upper_bound import _safe_recovery
from models.predictor_v4_alignment import _sample_with_source_offset


def test_safe_recovery_ratio():
    value = _safe_recovery(30.0, 36.0, 38.0)
    assert abs(value - 0.75) < 1e-8


def test_oracle_inverse_reduces_msi_error_on_common_overlap():
    torch.manual_seed(0)
    h = w = 48
    yy = torch.linspace(0.0, 1.0, h).view(1, 1, h, 1)
    xx = torch.linspace(0.0, 1.0, w).view(1, 1, 1, w)
    image = torch.cat(
        [
            torch.sin(9.0 * xx) * torch.cos(7.0 * yy),
            torch.sin(5.0 * xx + 3.0 * yy),
            torch.cos(11.0 * xx - 2.0 * yy),
        ],
        dim=1,
    )

    # Smooth spatially varying content displacement, well below folding regime.
    dx = 0.8 * torch.sin(2.0 * torch.pi * yy).expand(1, 1, h, w)
    dy = 0.7 * torch.cos(2.0 * torch.pi * xx).expand(1, 1, h, w)
    forward = torch.cat([dx, dy], dim=1)
    params = MisalignmentParameters(
        dx_px=torch.zeros(1),
        dy_px=torch.zeros(1),
        rotation_deg=torch.zeros(1),
        local_displacement_px=forward,
    )
    warped, valid = apply_misalignment(image, params)
    inverse = forward_to_inverse_sampling_field(
        forward,
        iterations=12,
        padding_mode="border",
    )
    oracle = _sample_with_source_offset(warped, inverse)
    oracle_valid = _sample_with_source_offset(valid, inverse).clamp(0.0, 1.0)
    common = torch.minimum(valid, oracle_valid) >= 0.999
    common3 = common.expand(-1, image.shape[1], -1, -1)

    warped_l1 = (warped - image).abs()[common3].mean()
    oracle_l1 = (oracle - image).abs()[common3].mean()
    assert oracle_l1 < warped_l1
