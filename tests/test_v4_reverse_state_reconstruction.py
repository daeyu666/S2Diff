import torch

import train_v4_reverse_state_reconstruction as mod


class DummyGeometry:
    stage_t_by_scale = {4: 12, 2: 8, 1: 4}


class DummyModel:
    def __init__(self):
        self.geometry_aligner = DummyGeometry()
        self._training_local_cache = {}
        self._training_sequence_cache = {}
        self.last_global_shift_px = None
        self.last_global_rotation_deg = None
        self.reset_called = False

    def _reset_inference_local_state(self):
        self.reset_called = True


class DummySAM:
    def __call__(self, pred, target):
        return (pred - target).abs().mean()


def test_reconstruction_uses_reverse_representative_states_and_reverse_flow_cache(monkeypatch):
    model = DummyModel()
    gt = torch.zeros(2, 3, 8, 8)
    warped = torch.zeros(2, 4, 8, 8)
    reverse_states = {
        4: torch.full_like(gt, 4.0),
        2: torch.full_like(gt, 2.0),
        1: torch.full_like(gt, 1.0),
    }
    finals = {
        4: torch.full((2, 2, 8, 8), 0.4),
        2: torch.full((2, 2, 8, 8), 0.2),
        1: torch.full((2, 2, 8, 8), 0.1),
    }
    sequences = {4: [finals[4]], 2: [finals[2]], 1: [finals[1]]}

    calls = []

    def fake_model_predict(model_arg, x_t, timesteps, hr_msi=None):
        calls.append((x_t.detach().clone(), timesteps.detach().clone(), hr_msi))
        assert model_arg._training_local_cache is finals
        return x_t * 0.0

    monkeypatch.setattr(mod, "model_predict", fake_model_predict)

    l1, sam, predictions = mod.reconstruction_loss_on_reverse_states(
        model,
        reverse_states,
        warped,
        gt,
        finals,
        sequences,
        DummySAM(),
    )

    assert len(calls) == 3
    expected = [(4, 12), (2, 8), (1, 4)]
    for (scale, t_stage), (x_seen, t_seen, msi_seen) in zip(expected, calls):
        assert torch.equal(x_seen, reverse_states[scale])
        assert torch.all(t_seen == t_stage)
        assert msi_seen is warped
        assert scale in predictions

    assert model._training_local_cache is finals
    assert model._training_sequence_cache is sequences
    assert model.reset_called
    assert torch.all(model.last_global_shift_px == 0)
    assert torch.all(model.last_global_rotation_deg == 0)
    assert float(l1.item()) == 0.0
    assert float(sam.item()) == 0.0
