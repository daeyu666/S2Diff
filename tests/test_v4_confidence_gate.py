import torch

from models.predictor_v4_alignment import SparseProgressiveLocalAligner
from models.predictor_v4_confidence import ConfidenceGatedSparseProgressiveLocalAligner


def test_old_local_weights_load_into_confidence_aligner_except_new_gate_params():
    old = SparseProgressiveLocalAligner(
        n_msi_bands=4,
        descriptor_channels=8,
        control_stride=4,
        radius_by_scale={1: 1, 2: 2, 4: 3},
    )
    new = ConfidenceGatedSparseProgressiveLocalAligner(
        n_msi_bands=4,
        descriptor_channels=8,
        control_stride=4,
        radius_by_scale={1: 1, 2: 2, 4: 3},
    )
    missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
    assert not unexpected
    assert set(missing) == {
        "confidence_log_gain.1",
        "confidence_log_gain.2",
        "confidence_log_gain.4",
        "confidence_bias.1",
        "confidence_bias.2",
        "confidence_bias.4",
    }
    for key, value in old.state_dict().items():
        assert torch.allclose(new.state_dict()[key], value)


def test_confidence_is_monotonic_in_move_advantage():
    aligner = ConfidenceGatedSparseProgressiveLocalAligner(
        n_msi_bands=4,
        descriptor_channels=8,
        confidence_gain_init=4.0,
        confidence_bias_init=1.5,
    )
    gain, bias = aligner._confidence_params(1)
    temperature = aligner._temperature(1).detach()
    margins = torch.tensor([-0.10, -0.05, 0.0, 0.05, 0.10])
    gates = torch.sigmoid(gain * (margins / temperature) + bias)
    assert torch.all(gates[1:] > gates[:-1])
    assert float(gates[0]) < float(gates[-1])


def test_forward_records_bounded_confidence_and_gate_gets_gradient():
    torch.manual_seed(7)
    aligner = ConfidenceGatedSparseProgressiveLocalAligner(
        n_msi_bands=4,
        descriptor_channels=8,
        control_stride=4,
        radius_by_scale={1: 1, 2: 2, 4: 3},
    )
    z_h = torch.rand(2, 4, 16, 16)
    z_m = torch.roll(z_h, shifts=1, dims=-1)
    dense, control = aligner(z_h, z_m, None, scale=1)
    assert dense.shape == (2, 2, 16, 16)
    assert control.shape[0] == 2 and control.shape[1] == 2
    gate = aligner.last_confidence_by_scale[1]
    assert torch.all(gate >= 0.0)
    assert torch.all(gate <= 1.0)

    loss = dense.square().mean()
    loss.backward()
    assert aligner.confidence_log_gain["1"].grad is not None
    assert aligner.confidence_bias["1"].grad is not None


def test_default_gate_preserves_most_residual_at_zero_margin_but_can_close():
    aligner = ConfidenceGatedSparseProgressiveLocalAligner(
        n_msi_bands=4,
        descriptor_channels=8,
        confidence_gain_init=4.0,
        confidence_bias_init=1.5,
    )
    gain, bias = aligner._confidence_params(4)
    temperature = aligner._temperature(4).detach()
    gate_zero_margin = torch.sigmoid(bias)
    gate_center_better = torch.sigmoid(gain * (torch.tensor(-0.10) / temperature) + bias)
    assert float(gate_zero_margin) > 0.75
    assert float(gate_center_better) < 0.20
