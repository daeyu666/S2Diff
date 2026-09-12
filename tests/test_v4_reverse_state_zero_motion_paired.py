import pytest
import torch

from train_v4_reverse_state_zero_motion_paired import (
    compose_paired_loss,
    paired_joint_score,
    validate_paired_weights,
)


def test_validate_paired_weights():
    assert validate_paired_weights(1.5, 0.25) == (1.5, 0.25)
    with pytest.raises(ValueError):
        validate_paired_weights(0.0, 0.25)
    with pytest.raises(ValueError):
        validate_paired_weights(1.5, -0.1)


def test_compose_paired_loss_keeps_local_task_and_adds_zero_constraints():
    local = torch.tensor(2.0)
    zero = torch.tensor(3.0)
    reg = torch.tensor(4.0)
    loss = compose_paired_loss(
        local,
        zero,
        reg,
        lambda_zero=1.5,
        lambda_registered_recon=0.25,
    )
    assert torch.isclose(loss, torch.tensor(7.5))


def test_compose_paired_loss_backpropagates_all_branches():
    local = torch.tensor(1.0, requires_grad=True)
    zero = torch.tensor(1.0, requires_grad=True)
    reg = torch.tensor(1.0, requires_grad=True)
    loss = compose_paired_loss(
        local,
        zero,
        reg,
        lambda_zero=1.5,
        lambda_registered_recon=0.25,
    )
    loss.backward()
    assert local.grad.item() == pytest.approx(1.0)
    assert zero.grad.item() == pytest.approx(1.5)
    assert reg.grad.item() == pytest.approx(0.25)


def test_joint_score_is_continuous_sum_without_hard_guard():
    assert paired_joint_score(43.16, 39.255) == pytest.approx(82.415)
    assert paired_joint_score(43.70, 39.40) > paired_joint_score(43.16, 39.255)
