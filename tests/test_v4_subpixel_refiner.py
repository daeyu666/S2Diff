import torch

from models.predictor_v4_gate_presence import PresenceSupervisedConfidenceLocalAligner
from models.predictor_v4_subpixel import CostVolumeSubpixelLocalAligner


def _make_pair():
    torch.manual_seed(7)
    base = PresenceSupervisedConfidenceLocalAligner(
        n_msi_bands=4,
        descriptor_channels=8,
        control_stride=4,
        radius_by_scale={1: 1, 2: 2, 4: 3},
        confidence_gain_init=4.0,
        confidence_bias_init=1.5,
    )
    refined = CostVolumeSubpixelLocalAligner(
        n_msi_bands=4,
        descriptor_channels=8,
        control_stride=4,
        radius_by_scale={1: 1, 2: 2, 4: 3},
        confidence_gain_init=4.0,
        confidence_bias_init=1.5,
        subpixel_hidden_channels=8,
        subpixel_max_px=0.5,
    )
    missing, unexpected = refined.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected
    assert missing
    assert all(key.startswith("subpixel_heads.") for key in missing)
    return base, refined


def test_zero_initialized_subpixel_heads_preserve_legacy_output():
    base, refined = _make_pair()
    z_h = torch.rand(2, 4, 16, 16)
    z_m = torch.rand(2, 4, 16, 16)

    base_dense, base_control = base(z_h, z_m, None, scale=1)
    refined_dense, refined_control = refined(z_h, z_m, None, scale=1)

    assert torch.allclose(refined_control, base_control, atol=1e-6, rtol=1e-6)
    assert torch.allclose(refined_dense, base_dense, atol=1e-6, rtol=1e-6)
    assert 1 in refined.last_subpixel_by_scale
    assert torch.count_nonzero(refined.last_subpixel_by_scale[1]) == 0


def test_subpixel_prediction_is_bounded_by_configured_range():
    _, refined = _make_pair()
    head = refined.subpixel_heads["1"]
    # Make the final layer produce very large logits deliberately.
    with torch.no_grad():
        head[-1].bias.fill_(100.0)
    z_h = torch.rand(1, 4, 16, 16)
    z_m = torch.rand(1, 4, 16, 16)
    refined(z_h, z_m, None, scale=1)
    epsilon = refined.last_subpixel_by_scale[1]
    assert float(epsilon.abs().max().item()) <= 0.500001


def test_subpixel_final_layer_receives_gradient():
    _, refined = _make_pair()
    z_h = torch.rand(1, 4, 16, 16)
    z_m = torch.rand(1, 4, 16, 16)
    dense, _ = refined(z_h, z_m, None, scale=1)
    target = torch.full_like(dense, 0.25)
    loss = torch.nn.functional.smooth_l1_loss(dense, target)
    loss.backward()

    final = refined.subpixel_heads["1"][-1]
    assert final.weight.grad is not None
    assert final.bias.grad is not None
    assert float(final.weight.grad.abs().sum().item()) > 0.0
    assert float(final.bias.grad.abs().sum().item()) > 0.0
