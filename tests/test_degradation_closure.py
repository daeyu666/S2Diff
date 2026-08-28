"""Core invariants for progressive degradation v1."""

import torch

from degradations import (
    PhysicalDegradation,
    ProgressiveDegradation,
    sigma_from_mtf_nyquist,
)


def test_terminal_closure_physical():
    torch.manual_seed(0)
    x = torch.rand(2, 8, 64, 64)
    operator = PhysicalDegradation(
        scale_ratio=4, mtf_nyquist=0.2
    )
    trajectory = ProgressiveDegradation(
        operator, total_steps=12
    )
    trajectory.assert_terminal_closure(x)


def test_all_states_live_on_hr_grid():
    torch.manual_seed(0)
    x = torch.rand(1, 5, 64, 64)
    trajectory = ProgressiveDegradation(
        PhysicalDegradation(scale_ratio=4, mtf_nyquist=0.2),
        total_steps=12,
    )
    for t in range(13):
        x_t = trajectory.state_at(x, t)
        assert x_t.shape == x.shape


def test_normalized_adjoint_preserves_constant_field():
    x = torch.ones(1, 3, 64, 64) * 0.37
    trajectory = ProgressiveDegradation(
        PhysicalDegradation(scale_ratio=4, mtf_nyquist=0.2),
        total_steps=12,
    )
    for t in range(13):
        x_t = trajectory.state_at(
            x, t, lift_mode="normalized_adjoint"
        )
        assert torch.allclose(
            x_t, x, atol=2e-5, rtol=2e-5
        )


def test_avg_pool_psf_adjoint_inner_product():
    torch.manual_seed(1)
    operator = PhysicalDegradation(
        scale_ratio=4, mtf_nyquist=0.2
    )
    x = torch.randn(1, 4, 32, 32)
    scale = 4
    strength = 0.75
    y_shape = operator.degrade_at(
        x, scale=scale, strength=strength
    ).shape
    y = torch.randn(y_shape)

    dx = operator.degrade_at(
        x, scale=scale, strength=strength
    )
    dty = operator.adjoint_at(
        y, scale=scale, strength=strength
    )
    left = torch.sum(dx * y)
    right = torch.sum(x * dty)
    assert torch.allclose(left, right, atol=2e-5, rtol=2e-5)


def test_sigma_from_mtf_matches_expected_formula():
    sigma = sigma_from_mtf_nyquist(4, 0.2)
    assert 2.2 < sigma < 2.4
