import torch

from train_v4_rotation_repair import (
    augment_global_with_targets,
    normalized_inverse_rotation_loss,
)


def test_inverse_rotation_loss_is_zero_for_exact_inverse():
    applied = torch.tensor([0.5, -1.0, 0.0], dtype=torch.float32)
    predicted = -applied.clone()
    loss = normalized_inverse_rotation_loss(predicted, applied, 1.0)
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-8)


def test_inverse_rotation_loss_pushes_undercorrection_toward_full_inverse():
    applied = torch.tensor([1.0], dtype=torch.float32)
    predicted = torch.tensor([-0.5], dtype=torch.float32, requires_grad=True)
    loss = normalized_inverse_rotation_loss(predicted, applied, 1.0)
    loss.backward()
    assert predicted.grad is not None
    # Gradient descent subtracts this positive gradient, making the correction
    # more negative: -0.5 -> -1.0, exactly the intended repair direction.
    assert float(predicted.grad.item()) > 0.0


def test_probability_zero_keeps_registered_sample_and_zero_rotation_target():
    x = torch.rand(2, 4, 16, 16)
    out, rotation, mean_shift, mean_rotation = augment_global_with_targets(
        x,
        max_shift_px=2.0,
        max_rotation_deg=1.0,
        probability=0.0,
        generator=torch.Generator().manual_seed(10),
    )
    assert out is x
    assert torch.allclose(rotation, torch.zeros_like(rotation))
    assert mean_shift == 0.0
    assert mean_rotation == 0.0
