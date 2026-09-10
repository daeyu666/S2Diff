import torch

from train_v4_local_repair import normalized_local_field_loss
from train_v4_local_zero_repair import (
    GLOBAL_LOCAL,
    GLOBAL_ONLY,
    LOCAL_ONLY,
    REGISTERED,
    augment_zero_repair_batch,
    category_masks_from_unit,
)


def test_four_way_category_split_matches_20_20_30_30():
    unit = torch.tensor([0.10, 0.30, 0.55, 0.85], dtype=torch.float32)
    category, registered, global_only, local_only, global_local = category_masks_from_unit(
        unit,
        registered_probability=0.20,
        global_only_probability=0.20,
        local_only_probability=0.30,
        global_local_probability=0.30,
    )
    assert category.tolist() == [REGISTERED, GLOBAL_ONLY, LOCAL_ONLY, GLOBAL_LOCAL]
    assert registered.tolist() == [True, False, False, False]
    assert global_only.tolist() == [False, True, False, False]
    assert local_only.tolist() == [False, False, True, False]
    assert global_local.tolist() == [False, False, False, True]


def test_zero_loss_pushes_false_local_offset_toward_zero():
    pred = torch.full((1, 2, 4, 4), 0.10, dtype=torch.float32, requires_grad=True)
    target = torch.zeros_like(pred)
    mask = torch.tensor([True])

    loss = normalized_local_field_loss(
        pred,
        target,
        mask,
        normalization_px=0.5,
    )
    loss.backward()

    assert loss.item() > 0.0
    assert pred.grad is not None
    # Positive prediction against a zero target gives a positive gradient;
    # gradient descent subtracts it and therefore reduces the false offset.
    assert float(pred.grad.mean().item()) > 0.0


def test_global_only_has_zero_local_supervision_target_classification():
    hr_msi = torch.rand(2, 4, 16, 16)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(123)

    (
        warped,
        rotation,
        target_local,
        local_supervised,
        zero_supervised,
        category,
        stats,
    ) = augment_zero_repair_batch(
        hr_msi,
        translation_max_px=4.0,
        rotation_max_deg=2.0,
        local_max_displacement_px=0.5,
        control_grid_size=5,
        registered_probability=0.0,
        global_only_probability=1.0,
        local_only_probability=0.0,
        global_local_probability=0.0,
        generator=generator,
    )

    assert warped.shape == hr_msi.shape
    assert torch.all(category == GLOBAL_ONLY)
    assert not bool(local_supervised.any())
    assert bool(zero_supervised.all())
    assert stats["global_only_fraction"] == 1.0
    # The sampler still constructs a candidate local field, but it is not
    # applied to global-only samples and is never used as a non-zero target.
    assert float(torch.linalg.vector_norm(target_local, dim=1).mean().item()) > 0.0
    assert rotation.shape == (2,)


def test_global_only_warp_is_independent_of_sampled_local_severity():
    hr_msi = torch.rand(2, 4, 16, 16)

    gen_a = torch.Generator(device="cpu")
    gen_a.manual_seed(456)
    out_a = augment_zero_repair_batch(
        hr_msi,
        translation_max_px=4.0,
        rotation_max_deg=2.0,
        local_max_displacement_px=0.5,
        control_grid_size=5,
        registered_probability=0.0,
        global_only_probability=1.0,
        local_only_probability=0.0,
        global_local_probability=0.0,
        generator=gen_a,
    )[0]

    gen_b = torch.Generator(device="cpu")
    gen_b.manual_seed(456)
    out_b = augment_zero_repair_batch(
        hr_msi,
        translation_max_px=4.0,
        rotation_max_deg=2.0,
        local_max_displacement_px=2.0,
        control_grid_size=5,
        registered_probability=0.0,
        global_only_probability=1.0,
        local_only_probability=0.0,
        global_local_probability=0.0,
        generator=gen_b,
    )[0]

    assert torch.allclose(out_a, out_b, atol=1e-6, rtol=1e-6)


def test_registered_is_supervised_only_by_zero_local_loss():
    hr_msi = torch.rand(2, 4, 16, 16)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(789)

    (
        warped,
        rotation,
        _,
        local_supervised,
        zero_supervised,
        category,
        stats,
    ) = augment_zero_repair_batch(
        hr_msi,
        translation_max_px=4.0,
        rotation_max_deg=2.0,
        local_max_displacement_px=0.5,
        control_grid_size=5,
        registered_probability=1.0,
        global_only_probability=0.0,
        local_only_probability=0.0,
        global_local_probability=0.0,
        generator=generator,
    )

    assert torch.all(category == REGISTERED)
    assert not bool(local_supervised.any())
    assert bool(zero_supervised.all())
    assert torch.allclose(warped, hr_msi, atol=1e-6, rtol=1e-6)
    assert torch.allclose(rotation, torch.zeros_like(rotation))
    assert stats["registered_fraction"] == 1.0
