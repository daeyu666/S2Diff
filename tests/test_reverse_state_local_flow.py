import types

import torch
import torch.nn as nn

from reverse_state_local_flow import rollout_reverse_states_identity


class _DummyProcess:
    total_steps = 12

    def terminal_observation(self, gt):
        return gt * 0.25

    def terminal_state(self, lr, *, target_size):
        assert tuple(lr.shape[-2:]) == tuple(target_size)
        return lr * 4.0

    def reverse_update(self, x_t, x0_hat, t):
        # Make later reverse states observably different from the terminal state.
        return x_t + 0.01 * x0_hat


class _DummyModel(nn.Module):
    requires_msi = False

    def __init__(self):
        super().__init__()
        self.geometry_aligner = types.SimpleNamespace(
            stage_t_by_scale={4: 12, 2: 8, 1: 4}
        )
        self._training_local_cache = {4: torch.ones(1)}
        self._training_sequence_cache = {4: [torch.ones(1)]}
        self._inference_local_offset = torch.ones(1)
        self._inference_local_scale = 4
        self.last_global_shift_px = None
        self.last_global_rotation_deg = None

    def _reset_inference_local_state(self):
        self._inference_local_offset = None
        self._inference_local_scale = None

    def forward(self, x_t, timesteps):
        del timesteps
        return x_t * 0.5


def test_reverse_rollout_captures_terminal_and_later_states():
    model = _DummyModel()
    process = _DummyProcess()
    gt = torch.rand(2, 3, 16, 16)
    msi = torch.rand(2, 4, 16, 16)

    model.train()
    states = rollout_reverse_states_identity(model, process, gt, msi)

    assert set(states) == {4, 2, 1}
    # Terminal closure in this dummy process gives x_12 == gt.
    assert torch.allclose(states[4], gt)
    # Reverse recursion has evolved by t=8 and t=4.
    assert not torch.allclose(states[2], states[4])
    assert not torch.allclose(states[1], states[2])
    # Helper restores caller training mode and clears cached alignment state.
    assert model.training
    assert model._training_local_cache == {}
    assert model._training_sequence_cache == {}
    assert model._inference_local_offset is None
    assert model._inference_local_scale is None
    assert torch.all(model.last_global_shift_px == 0)
    assert torch.all(model.last_global_rotation_deg == 0)
