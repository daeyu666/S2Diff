import numpy as np
import torch

from config import TrainConfig, resolve_config_defaults, validate_config
from data_loader import HSIHSRDataset, build_progressive_degradation
from degradations import build_degradation


def synthetic_hsi(h=64, w=64, c=8):
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :, None]
    bands = np.linspace(0.2, 1.0, c, dtype=np.float32)[None, None, :]
    img = (0.35 * yy + 0.45 * xx) * bands
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def test_dataset_lr_hsi_uses_exact_same_operator():
    img = synthetic_hsi()
    operator = build_degradation(
        "physical",
        scale_ratio=4,
        mtf_nyquist=0.2,
        truncate=3.0,
    )
    dataset = HSIHSRDataset(
        img=img,
        dataset_name="synthetic",
        patch_size=32,
        stride=32,
        scale_ratio=4,
        n_select_bands=4,
        split="test",
        test_size=32,
        augment=False,
        srf_weights=None,
        degradation_operator=operator,
    )

    sample = dataset[0]
    expected = operator.degrade(sample["gt"].unsqueeze(0)).squeeze(0)

    assert sample["lr_hsi"].shape == (8, 8, 8)
    assert torch.allclose(sample["lr_hsi"], expected, atol=1e-7, rtol=1e-6)


def test_progressive_terminal_closes_to_dataset_operator():
    cfg = TrainConfig(
        scale_ratio=4,
        degradation_mode="physical",
        mtf_nyquist=0.2,
        psf_truncate=3.0,
        progressive_steps=12,
        progressive_lift="auto",
    )
    resolve_config_defaults(cfg)
    validate_config(cfg)

    operator = build_degradation(
        "physical",
        scale_ratio=cfg.scale_ratio,
        mtf_nyquist=cfg.mtf_nyquist,
        truncate=cfg.psf_truncate,
    )
    trajectory = build_progressive_degradation(cfg, operator)

    x = torch.from_numpy(synthetic_hsi()).permute(2, 0, 1).unsqueeze(0)
    y_dataset = operator.degrade(x)
    y_terminal = trajectory.degrade_at(x, trajectory.total_steps)

    assert trajectory.operator is operator
    assert torch.allclose(y_dataset, y_terminal, atol=1e-7, rtol=1e-6)


def test_auto_lift_depends_on_degradation_mode():
    physical = TrainConfig(degradation_mode="physical", progressive_lift="auto")
    resolve_config_defaults(physical)
    validate_config(physical)
    assert physical.progressive_lift == "normalized_adjoint"

    ordinary = TrainConfig(degradation_mode="gaussian_bicubic", progressive_lift="auto")
    resolve_config_defaults(ordinary)
    validate_config(ordinary)
    assert ordinary.progressive_lift == "bilinear"
