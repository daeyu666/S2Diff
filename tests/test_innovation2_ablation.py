"""Controlled smoke tests for Innovation 2 MSI-guidance ablations."""

import torch

from models import MSIAblationGuidedPredictor


MODES = ("full", "no_msi", "raw_msi", "hf_nogate", "hf_const")


def _model(mode):
    return MSIAblationGuidedPredictor(
        n_bands=12,
        n_msi_bands=4,
        total_steps=4,
        base_channels=8,
        time_dim=16,
        dropout=0.0,
        residual_prediction=True,
        spectral_hidden=4,
        msi_highpass_kernel=5,
        msi_highpass_sigma=1.0,
        msi_ablation=mode,
    )


def test_all_ablation_modes_have_identical_parameterization():
    reference = _model("full")
    ref_keys = list(reference.state_dict().keys())
    ref_shapes = {k: tuple(v.shape) for k, v in reference.state_dict().items()}
    ref_params = sum(p.numel() for p in reference.parameters())

    for mode in MODES[1:]:
        model = _model(mode)
        assert list(model.state_dict().keys()) == ref_keys
        assert {k: tuple(v.shape) for k, v in model.state_dict().items()} == ref_shapes
        assert sum(p.numel() for p in model.parameters()) == ref_params


def test_full_checkpoint_loads_into_every_ablation_mode():
    torch.manual_seed(0)
    state = _model("full").state_dict()
    for mode in MODES:
        model = _model(mode)
        result = model.load_state_dict(state, strict=True)
        assert not result.missing_keys
        assert not result.unexpected_keys


def test_all_ablation_modes_preserve_shape_and_finite_backward():
    torch.manual_seed(1)
    x_t = torch.rand(2, 12, 16, 16)
    msi = torch.rand(2, 4, 16, 16)
    target = torch.rand_like(x_t)
    t = torch.tensor([1, 4], dtype=torch.long)

    for mode in MODES:
        model = _model(mode)
        pred = model(x_t, msi, t)
        assert pred.shape == x_t.shape
        assert torch.isfinite(pred).all()

        loss = torch.mean(torch.abs(pred - target))
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads
        assert all(torch.isfinite(g).all() for g in grads)


def test_hf_const_removes_only_explicit_alpha_schedule():
    model = _model("hf_const")
    t = torch.tensor([1, 4], dtype=torch.long)
    _, native_alpha = model._time_embedding(t, batch_size=2, device=t.device)
    # The inherited time embedding remains t-aware; hf_const replaces alpha
    # with ones only at the transfer point, leaving the rest of V3 unchanged.
    assert torch.allclose(native_alpha.flatten(), torch.tensor([0.25, 1.0]))
    assert model.msi_ablation == "hf_const"


def test_no_msi_is_invariant_to_msi_content_at_initialization():
    torch.manual_seed(2)
    model = _model("no_msi")
    x_t = torch.rand(1, 12, 16, 16)
    t = torch.tensor([4], dtype=torch.long)
    msi_a = torch.zeros(1, 4, 16, 16)
    msi_b = torch.rand(1, 4, 16, 16)
    pred_a = model(x_t, msi_a, t)
    pred_b = model(x_t, msi_b, t)
    assert torch.allclose(pred_a, pred_b, atol=1e-7, rtol=1e-7)
