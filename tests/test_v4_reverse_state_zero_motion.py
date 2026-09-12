import torch

from train_v4_reverse_state_zero_motion import (
    sample_registered_batch,
    validate_registered_probability,
    zero_or_local_forward_target,
)


def test_registered_probability_validation():
    assert validate_registered_probability(0.3) == 0.3
    for bad in (0.0, 1.0, -0.1, 1.1):
        try:
            validate_registered_probability(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for probability={bad}")


def test_registered_batch_returns_identity_msi_and_zero_field():
    msi = torch.rand(2, 4, 16, 16)
    out, forward = zero_or_local_forward_target(
        msi,
        registered=True,
        local_max_px=1.0,
        control_grid=5,
        generator=torch.Generator(device="cpu").manual_seed(10),
    )
    assert torch.equal(out, msi)
    assert forward.shape == (2, 2, 16, 16)
    assert torch.count_nonzero(forward).item() == 0


def test_batch_mode_sampling_is_reproducible():
    g1 = torch.Generator(device="cpu").manual_seed(123)
    g2 = torch.Generator(device="cpu").manual_seed(123)
    a = [sample_registered_batch(0.3, generator=g1) for _ in range(32)]
    b = [sample_registered_batch(0.3, generator=g2) for _ in range(32)]
    assert a == b
    assert any(a)
    assert not all(a)
