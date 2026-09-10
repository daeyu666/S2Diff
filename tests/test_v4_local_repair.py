import torch

from train_v4_local_repair import (
    GLOBAL_LOCAL,
    LOCAL_ONLY,
    REGISTERED,
    augment_local_repair_batch,
    category_masks_from_unit,
    normalized_local_field_loss,
)


def test_category_masks_split_registered_local_and_combined():
    unit = torch.tensor([0.10, 0.30, 0.60, 0.90], dtype=torch.float32)
    category, registered, local_only, global_local = category_masks_from_unit(
        unit,
        registered_probability=0.20,
        global_local_fraction=0.50,
    )
    assert category.tolist() == [REGISTERED, GLOBAL_LOCAL, LOCAL_ONLY, LOCAL_ONLY]
    assert registered.tolist() == [True, False, False, False]
    assert global_local.tolist() == [False, True, False, False]
    assert local_only.tolist() == [False, False, True, True]


def test_local_loss_pushes_underestimated_offset_toward_target():
    pred = torch.full((1, 2, 4, 4), 0.03, dtype=torch.float32, requires_grad=True)
    target = torch.full((1, 2, 4, 4), 0.30, dtype=torch.float32)
    mask = torch.tensor([True])
    loss = normalized_local_field_loss(pred, target, mask, normalization_px=0.5)
    loss.backward()
    assert loss.item() > 0.0
    assert pred.grad is not None
    # Gradient descent subtracts the gradient, so a negative gradient increases
    # the underestimated positive displacement toward the positive target.
    assert float(pred.grad.mean().item()) < 0.0


def test_local_only_augmentation_has_no_global_rigid_component():
    hr_msi = torch.rand(3, 4, 16, 16)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(123)
    warped, rotation, target_local, supervised, category, stats = augment_local_repair_batch(
        hr_msi,
        translation_max_px=4.0,
        rotation_max_deg=2.0,
        local_max_displacement_px=0.5,
        control_grid_size=5,
        registered_probability=0.0,
        global_local_fraction=0.0,
        generator=generator,
    )
    assert warped.shape == hr_msi.shape
    assert torch.all(category == LOCAL_ONLY)
    assert torch.all(supervised)
    assert torch.allclose(rotation, torch.zeros_like(rotation))
    assert float(torch.linalg.vector_norm(target_local, dim=1).mean().item()) > 0.0
    assert stats["applied_shift_px"] == 0.0
    assert stats["applied_abs_rotation_deg"] == 0.0


def test_global_local_samples_are_excluded_from_hard_local_field_supervision():
    hr_msi = torch.rand(2, 4, 16, 16)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(321)
    _, _, target_local, supervised, category, stats = augment_local_repair_batch(
        hr_msi,
        translation_max_px=4.0,
        rotation_max_deg=2.0,
        local_max_displacement_px=0.5,
        control_grid_size=5,
        registered_probability=0.0,
        global_local_fraction=1.0,
        generator=generator,
    )
    assert torch.all(category == GLOBAL_LOCAL)
    assert not bool(supervised.any())
    assert float(torch.linalg.vector_norm(target_local, dim=1).mean().item()) > 0.0
    assert stats["global_local_fraction"] == 1.0
