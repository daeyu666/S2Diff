"""Smoke tests for the spectral-spatial Innovation 1 predictor V2."""

import torch
import torch.nn.functional as F

from losses import SAMLoss
from models import SpectralSpatialCleanHSIPredictor
from models.predictor_v2 import LocalSpectralStem


def _model(n_bands=103):
    return SpectralSpatialCleanHSIPredictor(
        n_bands=n_bands,
        total_steps=12,
        base_channels=8,
        time_dim=16,
        dropout=0.0,
        residual_prediction=True,
        spectral_hidden=4,
    )


def test_spectral_stem_preserves_hsi_shape():
    torch.manual_seed(0)
    stem = LocalSpectralStem(hidden_channels=4)
    x = torch.rand(2, 103, 8, 8)
    y = stem(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_v2_predictor_shape_and_identity_initialization():
    torch.manual_seed(1)
    model = _model()
    x_t = torch.rand(2, 103, 16, 16)
    t = torch.tensor([9, 12], dtype=torch.long)
    pred = model(x_t, t)
    assert pred.shape == x_t.shape
    # Zero-initialized HSI output head preserves the same stable identity start
    # used by predictor V1 despite the richer hidden spectral-spatial backbone.
    assert torch.allclose(pred, x_t, atol=1e-7, rtol=1e-7)


def test_v2_one_backward_step_has_finite_gradients():
    torch.manual_seed(2)
    model = _model(n_bands=16)
    x_t = torch.rand(2, 16, 16, 16)
    gt = torch.rand_like(x_t)
    t = torch.tensor([8, 12], dtype=torch.long)

    pred = model(x_t, t)
    loss = F.l1_loss(pred, gt) + 0.1 * SAMLoss()(pred, gt)
    loss.backward()

    assert torch.isfinite(loss)
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
