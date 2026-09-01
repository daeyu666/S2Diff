"""Smoke tests for Innovation 2 MSI high-frequency guided predictor."""

import torch
import torch.nn.functional as F

from degradations import PhysicalDegradation, ProgressiveDegradation
from innovation1 import batch_state_at, train_one_epoch
from models import MSIHighFrequencyGuidedPredictor
from models.predictor_v3 import FixedGaussianHighPass


def _model(n_bands=12, n_msi_bands=4, total_steps=4):
    return MSIHighFrequencyGuidedPredictor(
        n_bands=n_bands,
        n_msi_bands=n_msi_bands,
        total_steps=total_steps,
        base_channels=8,
        time_dim=16,
        dropout=0.0,
        residual_prediction=True,
        spectral_hidden=4,
        msi_highpass_kernel=5,
        msi_highpass_sigma=1.0,
    )


def _process(total_steps=4):
    return ProgressiveDegradation(
        PhysicalDegradation(scale_ratio=4, mtf_nyquist=0.2),
        total_steps=total_steps,
    )


def test_fixed_highpass_removes_constant_msi():
    highpass = FixedGaussianHighPass(kernel_size=5, sigma=1.0)
    x = torch.ones(2, 4, 16, 16) * 0.37
    hf = highpass(x)
    assert hf.shape == x.shape
    assert torch.max(torch.abs(hf)).item() < 1e-5


def test_v3_initial_prediction_is_identity_and_shape_preserved():
    torch.manual_seed(0)
    model = _model()
    x_t = torch.rand(2, 12, 16, 16)
    msi = torch.rand(2, 4, 16, 16)
    t = torch.tensor([1, 4], dtype=torch.long)
    pred = model(x_t, msi, t)
    assert pred.shape == x_t.shape
    assert torch.allclose(pred, x_t, atol=1e-7, rtol=1e-7)


def test_time_schedule_uses_stronger_early_reverse_weight():
    model = _model(total_steps=4)
    t = torch.tensor([1, 4], dtype=torch.long)
    _, alpha = model._time_embedding(t, batch_size=2, device=t.device)
    assert torch.allclose(alpha.flatten(), torch.tensor([0.25, 1.0]))


def test_one_v3_training_step_is_finite():
    torch.manual_seed(1)
    process = _process()
    model = _model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    gt = torch.rand(2, 12, 16, 16)
    # Synthetic MSI for the smoke test only; real experiments use fixed SRF data.
    hr_msi = gt[:, [0, 3, 7, 11], :, :].contiguous()
    loader = [{"gt": gt, "hr_msi": hr_msi}]

    stats = train_one_epoch(
        model,
        loader,
        optimizer,
        process,
        torch.device("cpu"),
        lambda_l1=1.0,
        lambda_sam=0.1,
        lambda_deg=0.0,
        boundary_probability=0.2,
        boundary_radius=1,
        grad_clip=1.0,
    )
    assert torch.isfinite(torch.tensor(stats.loss))
    assert torch.isfinite(torch.tensor(stats.l1))
    assert torch.isfinite(torch.tensor(stats.sam))

    # After one optimizer step at least the zero-initialized output head should move.
    assert model.out_conv.weight.detach().abs().sum().item() > 0.0
