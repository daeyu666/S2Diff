"""Small invariants used by the Innovation 1 diagnostic workflow."""

import torch

from degradations import PhysicalDegradation, ProgressiveDegradation
from metrics import calc_metrics
from models import CleanHSIPredictor


def _process():
    return ProgressiveDegradation(
        PhysicalDegradation(scale_ratio=4, mtf_nyquist=0.2),
        total_steps=4,
    )


def _model(n_bands=6):
    return CleanHSIPredictor(
        n_bands=n_bands,
        total_steps=4,
        base_channels=8,
        time_dim=16,
        dropout=0.0,
        residual_prediction=True,
    )


def test_terminal_state_equals_oracle_terminal_state():
    torch.manual_seed(0)
    process = _process()
    gt = torch.rand(1, 6, 16, 16)

    terminal_lr = process.terminal_observation(gt)
    from_observation = process.terminal_state(
        terminal_lr, target_size=tuple(gt.shape[-2:])
    )
    oracle_terminal = process.state_at(gt, process.total_steps)

    assert torch.allclose(
        from_observation, oracle_terminal, atol=1e-6, rtol=1e-5
    )


def test_terminal_one_shot_matches_oracle_t_prediction():
    torch.manual_seed(1)
    process = _process()
    model = _model()
    gt = torch.rand(1, 6, 16, 16)
    T = process.total_steps

    terminal_lr = process.terminal_observation(gt)
    observed_state = process.terminal_state(
        terminal_lr, target_size=tuple(gt.shape[-2:])
    )
    oracle_state = process.state_at(gt, T)
    timestep = torch.tensor([T], dtype=torch.long)

    pred_observed = model(observed_state, timestep)
    pred_oracle = model(oracle_state, timestep)

    assert torch.allclose(pred_observed, pred_oracle, atol=1e-6, rtol=1e-5)


def test_reverse_drift_is_zero_at_terminal_initialization():
    torch.manual_seed(2)
    process = _process()
    gt = torch.rand(1, 6, 16, 16)
    T = process.total_steps

    terminal_lr = process.terminal_observation(gt)
    reverse_state = process.terminal_state(
        terminal_lr, target_size=tuple(gt.shape[-2:])
    )
    oracle_state = process.state_at(gt, T)

    drift_l1 = torch.mean(torch.abs(reverse_state - oracle_state))
    assert drift_l1.item() < 1e-6

    metrics = calc_metrics(reverse_state, gt, scale_ratio=4)
    assert torch.isfinite(torch.tensor(metrics["PSNR"]))
    assert torch.isfinite(torch.tensor(metrics["SAM"]))
