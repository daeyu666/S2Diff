import torch

from diagnose_misalignment_translation import translate_msi


def test_integer_translation_moves_content_right_and_down():
    x = torch.zeros(1, 1, 7, 7)
    x[0, 0, 3, 3] = 1.0

    shifted, valid = translate_msi(
        x,
        dx=torch.tensor([1.0]),
        dy=torch.tensor([1.0]),
    )

    assert torch.allclose(shifted[0, 0, 4, 4], torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(shifted.sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.all(valid[:, :, 0, :] < 0.999)
    assert torch.all(valid[:, :, :, 0] < 0.999)
    assert torch.all(valid[:, :, 1:, 1:] > 0.999)


def test_zero_translation_is_identity_and_fully_valid():
    x = torch.rand(2, 4, 8, 9)
    shifted, valid = translate_msi(
        x,
        dx=torch.zeros(2),
        dy=torch.zeros(2),
    )

    assert torch.allclose(shifted, x, atol=1e-6)
    assert torch.all(valid > 0.999)


def test_subpixel_translation_remains_finite():
    x = torch.rand(1, 4, 16, 16)
    shifted, valid = translate_msi(
        x,
        dx=torch.tensor([0.5]),
        dy=torch.tensor([-0.25]),
    )

    assert shifted.shape == x.shape
    assert valid.shape == (1, 1, 16, 16)
    assert torch.isfinite(shifted).all()
    assert torch.isfinite(valid).all()
    assert 0.0 <= float(valid.min()) <= float(valid.max()) <= 1.0
