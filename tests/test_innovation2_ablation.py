"""Controlled smoke tests for Innovation 2 MSI-guidance ablations."""

import torch
import torch.nn as nn

from models import MSIAblationGuidedPredictor


LEGACY_MODES = ("full", "no_msi", "raw_msi", "hf_nogate", "hf_const")
TIME_FREE_BASELINE_MODES = ("raw_direct", "raw_gate", "hf_direct", "hf_gate")
BASELINE_MODES = LEGACY_MODES + TIME_FREE_BASELINE_MODES
TRANSLATION_MODES = ("raw_translate",)
ALL_MODES = BASELINE_MODES + TRANSLATION_MODES


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


def test_pretranslation_ablation_modes_keep_identical_parameterization():
    """Adding raw_translate must not change any historical mode/checkpoint."""
    reference = _model("full")
    ref_keys = list(reference.state_dict().keys())
    ref_shapes = {k: tuple(v.shape) for k, v in reference.state_dict().items()}
    ref_params = sum(p.numel() for p in reference.parameters())

    for mode in BASELINE_MODES[1:]:
        model = _model(mode)
        assert list(model.state_dict().keys()) == ref_keys
        assert {k: tuple(v.shape) for k, v in model.state_dict().items()} == ref_shapes
        assert sum(p.numel() for p in model.parameters()) == ref_params


def test_full_checkpoint_loads_into_every_pretranslation_mode():
    torch.manual_seed(0)
    state = _model("full").state_dict()
    for mode in BASELINE_MODES:
        model = _model(mode)
        result = model.load_state_dict(state, strict=True)
        assert not result.missing_keys
        assert not result.unexpected_keys


def test_all_modes_preserve_shape_and_finite_backward():
    torch.manual_seed(1)
    x_t = torch.rand(2, 12, 16, 16)
    msi = torch.rand(2, 4, 16, 16)
    target = torch.rand_like(x_t)
    t = torch.tensor([1, 4], dtype=torch.long)

    for mode in ALL_MODES:
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
    # Legacy hf_const still has timestep-conditioned gate behavior. It is kept
    # only for reproducibility and is not part of the new time-free grid.
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


class _RaiseIfCalled(nn.Module):
    def forward(self, *args, **kwargs):
        raise AssertionError("this module must not be called")


def test_time_free_gate_modes_do_not_use_gate_timestep_projection():
    torch.manual_seed(3)
    x_t = torch.rand(1, 12, 16, 16)
    msi = torch.rand(1, 4, 16, 16)
    t = torch.tensor([3], dtype=torch.long)

    for mode in ("raw_gate", "hf_gate"):
        model = _model(mode)
        model.gate1.time_proj = _RaiseIfCalled()
        model.gate2.time_proj = _RaiseIfCalled()
        model.gate3.time_proj = _RaiseIfCalled()
        pred = model(x_t, msi, t)
        assert pred.shape == x_t.shape
        assert torch.isfinite(pred).all()


def test_time_free_direct_and_translation_modes_bypass_transfer_gates():
    torch.manual_seed(4)
    x_t = torch.rand(1, 12, 16, 16)
    msi = torch.rand(1, 4, 16, 16)
    t = torch.tensor([2], dtype=torch.long)

    for mode in ("raw_direct", "hf_direct", "raw_translate"):
        model = _model(mode)
        model.gate1.gate = _RaiseIfCalled()
        model.gate2.gate = _RaiseIfCalled()
        model.gate3.gate = _RaiseIfCalled()
        model.gate1.time_proj = _RaiseIfCalled()
        model.gate2.time_proj = _RaiseIfCalled()
        model.gate3.time_proj = _RaiseIfCalled()
        pred = model(x_t, msi, t)
        assert pred.shape == x_t.shape
        assert torch.isfinite(pred).all()


def test_raw_translate_adapters_are_identity_initialized():
    torch.manual_seed(5)
    model = _model("raw_translate")

    for adapter, channels in (
        (model.translate1, 8),
        (model.translate2, 16),
        (model.translate3, 32),
    ):
        feature = torch.rand(2, channels, 7, 9)
        translated = adapter(feature)
        assert torch.equal(translated, feature)
        assert torch.count_nonzero(adapter.up.weight) == 0
        assert torch.count_nonzero(adapter.up.bias) == 0


def test_raw_translate_uses_low_rank_extra_capacity_only_in_new_mode():
    raw_direct = _model("raw_direct")
    raw_translate = _model("raw_translate")

    assert not hasattr(raw_direct, "translate1")
    assert hasattr(raw_translate, "translate1")
    assert raw_translate.translate1.rank < raw_translate.translate1.channels
    assert raw_translate.translate2.rank < raw_translate.translate2.channels
    assert raw_translate.translate3.rank < raw_translate.translate3.channels

    direct_params = sum(p.numel() for p in raw_direct.parameters())
    translate_params = sum(p.numel() for p in raw_translate.parameters())
    assert translate_params > direct_params
