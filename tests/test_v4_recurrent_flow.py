import torch

from models.predictor_v4_recurrent_flow import RecurrentResidualFlowAligner
from train_v4_recurrent_flow import sequence_flow_loss


def test_recurrent_local_flow_is_identity_safe_and_emits_sequence():
    module = RecurrentResidualFlowAligner(
        n_msi_bands=4,
        descriptor_channels=8,
        control_stride=4,
        radius_by_scale={1: 1, 2: 1, 4: 1},
        iterations_by_scale={1: 2, 2: 2, 4: 3},
        max_update_by_scale={1: 0.5, 2: 1.0, 4: 2.0},
        hidden_channels=16,
        correlation_channels=8,
    )
    z_h = torch.rand(2, 4, 24, 24)
    z_m = torch.rand(2, 4, 24, 24)
    dense, control, sequence = module(z_h, z_m, None, scale=4)

    assert dense.shape == (2, 2, 24, 24)
    assert control.shape[1] == 2
    assert len(sequence) == 3
    # Final residual head is zero-initialized, so a new recurrent branch does
    # not disturb a registered/warm-started model before it learns.
    assert torch.allclose(dense, torch.zeros_like(dense), atol=1e-7)


def test_sequence_flow_loss_pushes_underestimated_flow_toward_target():
    pred = torch.tensor([[[[0.20]], [[0.0]]]], requires_grad=True)
    target = torch.tensor([[[[1.00]], [[0.0]]]])
    loss = sequence_flow_loss(
        {4: [pred], 2: [pred], 1: [pred]},
        target,
        torch.tensor([True]),
        normalization_px=1.0,
        gamma=0.8,
    )
    loss.backward()
    # Gradient descent subtracts grad.  A negative x-gradient therefore
    # increases the predicted displacement toward +1 px.
    assert pred.grad is not None
    assert float(pred.grad[0, 0, 0, 0]) < 0.0


def test_sequence_flow_loss_uses_same_objective_for_zero_response():
    pred = torch.tensor([[[[0.25]], [[-0.10]]]], requires_grad=True)
    target = torch.zeros_like(pred)
    loss = sequence_flow_loss(
        {4: [pred]},
        target,
        torch.tensor([True]),
        normalization_px=1.0,
        gamma=0.8,
    )
    loss.backward()
    assert loss.item() > 0.0
    # Descent should move positive x down and negative y up, both toward zero.
    assert float(pred.grad[0, 0, 0, 0]) > 0.0
    assert float(pred.grad[0, 1, 0, 0]) < 0.0
