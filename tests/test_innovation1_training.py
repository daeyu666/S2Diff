"""Smoke tests for Innovation 1 predictor and training path."""

import torch

from degradations import PhysicalDegradation, ProgressiveDegradation
from innovation1 import batch_state_at, reconstruct_from_terminal_lr, train_one_epoch
from models import CleanHSIPredictor


def _small_predictor(n_bands=6, total_steps=4):
    return CleanHSIPredictor(
        n_bands=n_bands,
        total_steps=total_steps,
        base_channels=8,
        time_dim=16,
        dropout=0.0,
        residual_prediction=True,
    )


def _small_process(total_steps=4):
    return ProgressiveDegradation(
        PhysicalDegradation(scale_ratio=4, mtf_nyquist=0.2),
        total_steps=total_steps,
    )


def test_predictor_shape_and_identity_initialization():
    torch.manual_seed(0)
    model = _small_predictor()
    x_t = torch.rand(2, 6, 16, 16)
    t = torch.tensor([1, 4], dtype=torch.long)

    pred = model(x_t, t)
    assert pred.shape == x_t.shape
    # Zero-initialized output head makes the first clean estimate exactly x_t.
    assert torch.allclose(pred, x_t, atol=1e-7, rtol=1e-7)


def test_batch_state_at_supports_mixed_timesteps():
    torch.manual_seed(1)
    process = _small_process()
    x = torch.rand(3, 6, 16, 16)
    t = torch.tensor([1, 4, 2], dtype=torch.long)

    state = batch_state_at(process, x, t)
    assert state.shape == x.shape
    assert torch.isfinite(state).all()


def test_one_training_step_runs_without_msi():
    torch.manual_seed(2)
    process = _small_process()
    model = _small_predictor()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    gt = torch.rand(2, 6, 16, 16)
    # Only gt is required by Innovation 1.  No hr_msi is supplied here.
    loader = [{"gt": gt}]

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
    assert stats.loss > 0.0
    assert stats.l1 > 0.0
    assert torch.isfinite(torch.tensor(stats.loss))


def test_reverse_inference_starts_from_terminal_lr_and_returns_hr_grid():
    torch.manual_seed(3)
    process = _small_process()
    model = _small_predictor()
    gt = torch.rand(1, 6, 16, 16)
    lr = process.terminal_observation(gt)

    pred = reconstruct_from_terminal_lr(
        model,
        process,
        lr,
        target_size=(16, 16),
    )
    assert pred.shape == gt.shape
    assert torch.isfinite(pred).all()
