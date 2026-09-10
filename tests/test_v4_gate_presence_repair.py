import torch

from models.predictor_v4_confidence import ConfidenceGatedSparseProgressiveLocalAligner
from models.predictor_v4_gate_presence import PresenceSupervisedConfidenceLocalAligner
from train_v4_gate_presence_repair import gate_presence_loss
from train_v4_local_zero_repair import GLOBAL_LOCAL, GLOBAL_ONLY, LOCAL_ONLY, REGISTERED


def test_presence_aligner_keeps_confidence_checkpoint_parameter_layout():
    kwargs = dict(
        n_msi_bands=4,
        descriptor_channels=8,
        control_stride=4,
        radius_by_scale={1: 1, 2: 1, 4: 1},
        confidence_gain_init=4.0,
        confidence_bias_init=1.5,
    )
    old = ConfidenceGatedSparseProgressiveLocalAligner(**kwargs)
    new = PresenceSupervisedConfidenceLocalAligner(**kwargs)
    assert set(old.state_dict().keys()) == set(new.state_dict().keys())


def test_gate_presence_loss_pushes_off_categories_down_and_on_categories_up():
    # Four samples: off, off, on, on. One control value per sample is enough
    # to verify the BCEWithLogits gradient direction.
    logits = torch.zeros(4, 1, 1, requires_grad=True)
    category = torch.tensor(
        [REGISTERED, GLOBAL_ONLY, LOCAL_ONLY, GLOBAL_LOCAL], dtype=torch.long
    )
    loss, diagnostics = gate_presence_loss({4: logits}, category)
    loss.backward()

    grad = logits.grad[:, 0, 0]
    # Gradient descent subtracts grad: positive -> logit decreases for target 0.
    assert float(grad[0]) > 0.0
    assert float(grad[1]) > 0.0
    # Negative -> logit increases for target 1.
    assert float(grad[2]) < 0.0
    assert float(grad[3]) < 0.0
    assert diagnostics["gate_off_mean"] == 0.5
    assert diagnostics["gate_on_mean"] == 0.5


def test_gate_presence_loss_averages_physical_scales_equally():
    category = torch.tensor([REGISTERED, LOCAL_ONLY], dtype=torch.long)
    logits4 = torch.tensor([[[2.0]], [[-2.0]]], requires_grad=True)
    logits2 = torch.tensor([[[0.0]], [[0.0]]], requires_grad=True)
    logits1 = torch.tensor([[[-2.0]], [[2.0]]], requires_grad=True)

    loss, diagnostics = gate_presence_loss(
        {4: logits4, 2: logits2, 1: logits1}, category
    )
    assert torch.isfinite(loss)
    assert set(diagnostics) >= {
        "gate_off_mean",
        "gate_on_mean",
        "gate_scale4",
        "gate_scale2",
        "gate_scale1",
    }
    loss.backward()
    assert logits4.grad is not None
    assert logits2.grad is not None
    assert logits1.grad is not None
