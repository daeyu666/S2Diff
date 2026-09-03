"""Controlled smoke tests for Innovation 2 MSI-guidance ablations."""

import torch
import torch.nn as nn

from models import MSIAblationGuidedPredictor


LEGACY_MODES = ("full", "no_msi", "raw_msi", "hf_nogate", "hf_const")
TIME_FREE_BASELINE_MODES = ("raw_direct", "raw_gate", "hf_direct", "hf_gate")
BASELINE_MODES = LEGACY_MODES + TIME_FREE_BASELINE_MODES
TRANSLATION_MODES = ("raw_translate", "raw_translate_ctx")
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
    """Adding translation modes must not change historical checkpoints."""
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

    for mode in (
        "raw_direct",
        "hf_direct",
        "raw_translate",
        "raw_translate_ctx",
    ):
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


def test_raw_translate_ctx_adapters_are_identity_initialized_for_any_context():
    torch.manual_seed(6)
    model = _model("raw_translate_ctx")

    for adapter, channels in (
        (model.translate1, 8),
        (model.translate2, 16),
        (model.translate3, 32),
    ):
        msi_feature = torch.rand(2, channels, 7, 9)
        hsi_a = torch.rand_like(msi_feature)
        hsi_b = torch.rand_like(msi_feature)
        translated_a = adapter(msi_feature, hsi_a)
        translated_b = adapter(msi_feature, hsi_b)
        assert torch.equal(translated_a, msi_feature)
        assert torch.equal(translated_b, msi_feature)
        assert torch.count_nonzero(adapter.up.weight) == 0
        assert torch.count_nonzero(adapter.up.bias) == 0


def test_raw_translate_ctx_can_use_hsi_context_after_residual_path_opens():
    torch.manual_seed(7)
    model = _model("raw_translate_ctx")
    adapter = model.translate1
    msi_feature = torch.rand(1, 8, 8, 8)
    hsi_a = torch.rand_like(msi_feature)
    hsi_b = torch.rand_like(msi_feature)

    with torch.no_grad():
        adapter.up.weight.normal_(mean=0.0, std=0.05)
        adapter.up.bias.zero_()

    translated_a = adapter(msi_feature, hsi_a)
    translated_b = adapter(msi_feature, hsi_b)
    assert not torch.allclose(translated_a, translated_b)


def test_raw_translate_ctx_full_model_starts_from_raw_direct_mapping():
    x_t = torch.rand(1, 12, 16, 16)
    msi = torch.rand(1, 4, 16, 16)
    t = torch.tensor([3], dtype=torch.long)

    torch.manual_seed(8)
    raw_direct = _model("raw_direct")
    torch.manual_seed(8)
    raw_translate_ctx = _model("raw_translate_ctx")

    pred_direct = raw_direct(x_t, msi, t)
    pred_ctx = raw_translate_ctx(x_t, msi, t)
    assert torch.allclose(pred_direct, pred_ctx, atol=1e-7, rtol=1e-7)


def test_translation_modes_use_low_rank_extra_capacity_only_in_new_modes():
    raw_direct = _model("raw_direct")
    raw_translate = _model("raw_translate")
    raw_translate_ctx = _model("raw_translate_ctx")

    assert not hasattr(raw_direct, "translate1")
    for model in (raw_translate, raw_translate_ctx):
        assert hasattr(model, "translate1")
        assert model.translate1.rank < model.translate1.channels
        assert model.translate2.rank < model.translate2.channels
        assert model.translate3.rank < model.translate3.channels

    direct_params = sum(p.numel() for p in raw_direct.parameters())
    translate_params = sum(p.numel() for p in raw_translate.parameters())
    ctx_params = sum(p.numel() for p in raw_translate_ctx.parameters())
    assert translate_params > direct_params
    assert ctx_params > translate_params
