import torch

from models.predictor_v4_gate_presence import PresenceSupervisedConfidenceLocalAligner
from models.predictor_v4_residual_telemetry import ResidualTelemetryConfidenceLocalAligner


def _make(cls):
    return cls(
        n_msi_bands=4,
        descriptor_channels=8,
        control_stride=4,
        radius_by_scale={1: 1, 2: 2, 4: 3},
        confidence_gain_init=4.0,
        confidence_bias_init=1.5,
    )


def test_telemetry_adds_no_persistent_parameters():
    base = _make(PresenceSupervisedConfidenceLocalAligner)
    telemetry = _make(ResidualTelemetryConfidenceLocalAligner)
    assert set(base.state_dict().keys()) == set(telemetry.state_dict().keys())
    telemetry.load_state_dict(base.state_dict(), strict=True)


def test_telemetry_records_raw_gate_gated_and_accumulated_flow():
    model = _make(ResidualTelemetryConfidenceLocalAligner)
    z_h = torch.rand(2, 4, 16, 16)
    z_m = torch.rand(2, 4, 16, 16)

    dense, control = model(z_h, z_m, None, scale=1)

    assert 1 in model.last_raw_residual_by_scale
    assert 1 in model.last_gated_residual_by_scale
    assert 1 in model.last_accumulated_control_by_scale
    assert 1 in model.last_accumulated_dense_by_scale
    assert 1 in model.last_confidence_by_scale

    raw = model.last_raw_residual_by_scale[1]
    gated = model.last_gated_residual_by_scale[1]
    accumulated_control = model.last_accumulated_control_by_scale[1]
    gate = model.last_confidence_by_scale[1]

    assert raw.shape == gated.shape == accumulated_control.shape == control.shape
    assert dense.shape == (2, 2, 16, 16)
    assert gate.shape == raw[:, 0].shape
    assert torch.all(gate >= 0.0)
    assert torch.all(gate <= 1.0)

    raw_mag = torch.linalg.vector_norm(raw, dim=1)
    gated_mag = torch.linalg.vector_norm(gated, dim=1)
    assert torch.all(gated_mag <= raw_mag + 1e-6)
    # With no previous field, accumulated control is exactly the gated residual.
    assert torch.allclose(accumulated_control, gated, atol=1e-6, rtol=1e-6)


def test_clear_residual_telemetry_removes_runtime_records_only():
    model = _make(ResidualTelemetryConfidenceLocalAligner)
    state_before = set(model.state_dict().keys())
    z = torch.rand(1, 4, 16, 16)
    model(z, z, None, scale=1)
    assert model.last_raw_residual_by_scale
    model.clear_residual_telemetry()
    assert model.last_raw_residual_by_scale == {}
    assert model.last_gated_residual_by_scale == {}
    assert model.last_accumulated_dense_by_scale == {}
    assert set(model.state_dict().keys()) == state_before
