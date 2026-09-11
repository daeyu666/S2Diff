import torch
import torch.nn as nn

from train_v4_local_reconstruction import (
    clear_training_alignment_cache,
    reconstruct_local_only_identity,
)


class DummyProcess:
    total_steps = 2

    def terminal_state(self, lr_hsi, *, target_size):
        return lr_hsi.clone()

    def reverse_update(self, x_t, x0_hat, t):
        return x0_hat


class DummyIdentityLocalModel(nn.Module):
    requires_msi = True
    requires_global_msi_preparation = True

    def __init__(self):
        super().__init__()
        self.prepare_global_calls = 0
        self._training_local_cache = {1: torch.ones(1)}
        self._training_sequence_cache = {1: [torch.ones(1)]}
        self._inference_local_offset = None
        self.last_global_shift_px = None
        self.last_global_rotation_deg = None

    def _reset_inference_local_state(self):
        self._inference_local_offset = None

    def prepare_global_msi(self, *args, **kwargs):
        self.prepare_global_calls += 1
        raise AssertionError("global preparation must be bypassed")

    def forward(self, x_t, hr_msi, t):
        self._inference_local_offset = torch.zeros(
            x_t.shape[0], 2, x_t.shape[-2], x_t.shape[-1],
            device=x_t.device, dtype=x_t.dtype
        )
        return x_t + hr_msi[:, :1]


def test_local_identity_reconstruction_bypasses_global_preparation():
    model = DummyIdentityLocalModel()
    process = DummyProcess()
    lr = torch.zeros(1, 1, 4, 4)
    msi = torch.ones(1, 1, 4, 4)

    out = reconstruct_local_only_identity(
        model,
        process,
        lr,
        target_size=(4, 4),
        warped_msi=msi,
    )

    assert model.prepare_global_calls == 0
    assert model.last_global_shift_px.abs().max().item() == 0.0
    assert model.last_global_rotation_deg.abs().max().item() == 0.0
    assert model._inference_local_offset is not None
    assert torch.isfinite(out).all()


def test_clear_training_alignment_cache():
    model = DummyIdentityLocalModel()
    clear_training_alignment_cache(model)
    assert model._training_local_cache == {}
    assert model._training_sequence_cache == {}
