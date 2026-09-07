from types import SimpleNamespace

import torch

from innovation1 import augment_training_msi_translation
from main import _predictor_tag


def test_training_translation_zero_is_exact_identity():
    x = torch.rand(3, 4, 24, 24)
    out, mean_shift = augment_training_msi_translation(
        x,
        max_shift_px=0.0,
        probability=1.0,
        generator=torch.Generator().manual_seed(10),
    )
    assert out is x
    assert mean_shift == 0.0


def test_training_translation_is_deterministic_for_same_seed():
    x = torch.rand(4, 4, 24, 24)
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    out1, shift1 = augment_training_msi_translation(
        x,
        max_shift_px=2.0,
        probability=1.0,
        generator=g1,
    )
    out2, shift2 = augment_training_msi_translation(
        x,
        max_shift_px=2.0,
        probability=1.0,
        generator=g2,
    )
    assert torch.allclose(out1, out2)
    assert abs(shift1 - shift2) < 1e-12
    assert shift1 > 0.0
    assert not torch.allclose(out1, x)


def test_training_translation_probability_zero_disables_warp():
    x = torch.rand(2, 4, 20, 20)
    out, mean_shift = augment_training_msi_translation(
        x,
        max_shift_px=2.0,
        probability=0.0,
        generator=torch.Generator().manual_seed(5),
    )
    assert out is x
    assert mean_shift == 0.0


def test_augmented_checkpoint_tag_does_not_collide_with_registered_baseline():
    registered = SimpleNamespace(
        predictor_version="v3",
        predictor_base_channels=64,
        msi_ablation="raw_direct",
        train_msi_translation_max_px=0.0,
        train_msi_translation_probability=1.0,
    )
    aug1 = SimpleNamespace(
        predictor_version="v3",
        predictor_base_channels=64,
        msi_ablation="raw_direct",
        train_msi_translation_max_px=1.0,
        train_msi_translation_probability=1.0,
    )
    aug2 = SimpleNamespace(
        predictor_version="v3",
        predictor_base_channels=64,
        msi_ablation="raw_direct",
        train_msi_translation_max_px=2.0,
        train_msi_translation_probability=1.0,
    )
    assert _predictor_tag(registered) == "_v3_raw_direct"
    assert _predictor_tag(aug1) == "_v3_raw_direct_traug1"
    assert _predictor_tag(aug2) == "_v3_raw_direct_traug2"
