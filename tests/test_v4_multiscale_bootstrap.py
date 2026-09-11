import torch

from degradations.physical import PhysicalDegradation
from degradations.progressive import ProgressiveDegradation
from train_v4_multiscale_bootstrap import (
    physical_flow_targets,
    stage_sequence_loss,
)


def _process():
    operator = PhysicalDegradation(scale_ratio=4, mtf_nyquist=0.2)
    return ProgressiveDegradation(operator, total_steps=12)


def _gradient_energy(x):
    crop = x[..., 6:-6, 6:-6]
    dx = (crop[..., :, 1:] - crop[..., :, :-1]).abs().mean()
    dy = (crop[..., 1:, :] - crop[..., :-1, :]).abs().mean()
    return dx + dy


def test_physical_flow_targets_keep_hr_units_for_constant_field():
    process = _process()
    flow = torch.zeros(1, 2, 32, 32)
    flow[:, 0] = 0.75
    flow[:, 1] = -0.40
    targets = physical_flow_targets(
        process,
        {1: 4, 2: 8, 4: 12},
        flow,
    )

    assert torch.equal(targets[1], flow)
    assert torch.allclose(targets[2], flow, atol=2e-4, rtol=2e-4)
    assert torch.allclose(targets[4], flow, atol=2e-4, rtol=2e-4)


def test_physical_flow_targets_remove_spatial_detail_progressively():
    process = _process()
    yy, xx = torch.meshgrid(
        torch.arange(40, dtype=torch.float32),
        torch.arange(40, dtype=torch.float32),
        indexing="ij",
    )
    checker = ((xx + yy) % 2.0) * 2.0 - 1.0
    flow = torch.stack([checker, -checker], dim=0).unsqueeze(0) * 0.5

    targets = physical_flow_targets(
        process,
        {1: 4, 2: 8, 4: 12},
        flow,
    )
    e1 = _gradient_energy(targets[1])
    e2 = _gradient_energy(targets[2])
    e4 = _gradient_energy(targets[4])

    assert float(e4) < float(e2)
    assert float(e2) < float(e1)


def test_stage_sequence_loss_is_zero_when_each_stage_matches_its_target():
    targets = {
        4: torch.randn(2, 2, 16, 16),
        2: torch.randn(2, 2, 16, 16),
        1: torch.randn(2, 2, 16, 16),
    }
    sequence = {
        4: [targets[4].clone(), targets[4].clone()],
        2: [targets[2].clone()],
        1: [targets[1].clone(), targets[1].clone()],
    }
    loss = stage_sequence_loss(
        sequence,
        targets,
        normalization_px=1.0,
        gamma=0.8,
        stage_weights={4: 0.5, 2: 0.7, 1: 1.0},
    )
    assert float(loss) == 0.0
