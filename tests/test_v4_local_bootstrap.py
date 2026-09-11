import torch
import torch.nn as nn

from models.predictor_v4_recurrent_flow import RecurrentResidualFlowAligner


def _tiny_open_output_head(model, std=1e-3):
    head = model.delta_head[-1]
    assert isinstance(head, nn.Conv2d)
    nn.init.normal_(head.weight, mean=0.0, std=std)
    nn.init.zeros_(head.bias)


def test_tiny_nonzero_head_opens_gradient_to_recurrent_stack():
    torch.manual_seed(7)
    aligner = RecurrentResidualFlowAligner(
        n_msi_bands=4,
        descriptor_channels=8,
        control_stride=4,
        radius_by_scale={1: 1},
        iterations_by_scale={1: 3},
        max_update_by_scale={1: 0.5},
        hidden_channels=16,
        correlation_channels=8,
    )
    _tiny_open_output_head(aligner)

    z_h = torch.rand(2, 4, 16, 16)
    z_m = torch.rand(2, 4, 16, 16)
    target = torch.rand(2, 2, 16, 16) * 0.4 - 0.2

    dense, _, sequence = aligner(
        z_h,
        z_m,
        previous_dense_offset=None,
        scale=1,
    )
    assert len(sequence) == 3
    assert dense.shape == target.shape

    loss = torch.nn.functional.smooth_l1_loss(dense, target)
    loss.backward()

    grads = {
        "descriptor": aligner.descriptor.net[0].weight.grad,
        "correlation": aligner.correlation_encoders["1"].net[0].weight.grad,
        "gru": aligner.gru.gates.weight.grad,
        "delta": aligner.delta_head[-1].weight.grad,
    }
    for name, grad in grads.items():
        assert grad is not None, name
        assert torch.isfinite(grad).all(), name
        assert grad.abs().sum().item() > 0.0, name


def test_exact_zero_head_blocks_earlier_stack_at_first_backward():
    torch.manual_seed(7)
    aligner = RecurrentResidualFlowAligner(
        n_msi_bands=4,
        descriptor_channels=8,
        control_stride=4,
        radius_by_scale={1: 1},
        iterations_by_scale={1: 1},
        max_update_by_scale={1: 0.5},
        hidden_channels=16,
        correlation_channels=8,
    )
    # Constructor intentionally zero-initializes the final delta conv.
    assert torch.count_nonzero(aligner.delta_head[-1].weight).item() == 0

    z_h = torch.rand(2, 4, 16, 16)
    z_m = torch.rand(2, 4, 16, 16)
    target = torch.rand(2, 2, 16, 16) * 0.4 - 0.2

    dense, _, _ = aligner(z_h, z_m, None, scale=1)
    torch.nn.functional.smooth_l1_loss(dense, target).backward()

    # The output layer learns immediately, but its zero weight cuts the first
    # backward path into the upstream recurrent/correlation stack.
    assert aligner.delta_head[-1].weight.grad.abs().sum().item() > 0.0
    upstream = aligner.gru.gates.weight.grad
    assert upstream is None or upstream.abs().sum().item() == 0.0
