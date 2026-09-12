"""Checkpoint composition utilities for final Innovation-2 joint evaluation.

Late local-only stages deliberately froze / bypassed the global rigid branch.
Their checkpoints are therefore the correct source for the reconstruction
backbone and recurrent local aligner, but not necessarily for the validated
global rigid parameters.  Final Global+Local evaluation composes two sources:

- local checkpoint: full model state (backbone + recurrent local branch);
- global checkpoint: only ``geometry_aligner.global_aligner.*`` is overlaid.

No architecture changes or learned gates are introduced here.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from utils import load_checkpoint


GLOBAL_PREFIX = "geometry_aligner.global_aligner."


def _read_model_state(path: str, map_location: str = "cpu") -> Dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        state = torch.load(path, map_location=map_location)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint {path!r} does not contain a model state_dict")
    return state


def overlay_global_rigid_branch(
    model: torch.nn.Module,
    global_checkpoint: str,
    *,
    map_location: str = "cpu",
) -> int:
    """Overwrite only the validated global rigid branch from another V4 checkpoint."""
    source = _read_model_state(global_checkpoint, map_location=map_location)
    current = model.state_dict()

    copied = 0
    missing = []
    mismatched = []
    for key, tensor in source.items():
        if not key.startswith(GLOBAL_PREFIX):
            continue
        if key not in current:
            missing.append(key)
            continue
        if tuple(current[key].shape) != tuple(tensor.shape):
            mismatched.append((key, tuple(tensor.shape), tuple(current[key].shape)))
            continue
        current[key] = tensor.to(device=current[key].device, dtype=current[key].dtype)
        copied += 1

    if copied == 0:
        raise RuntimeError(
            f"No {GLOBAL_PREFIX} parameters were copied from {global_checkpoint!r}"
        )
    if missing or mismatched:
        details = []
        if missing:
            details.append(f"missing={missing[:5]}")
        if mismatched:
            details.append(f"mismatched={mismatched[:5]}")
        raise RuntimeError("Global checkpoint is incompatible: " + "; ".join(details))

    model.load_state_dict(current, strict=True)
    return copied


def load_joint_v4_checkpoints(
    model: torch.nn.Module,
    local_checkpoint: str,
    global_checkpoint: str,
    *,
    map_location: str = "cpu",
) -> Tuple[int, float, int]:
    """Load local/full state first, then selectively restore validated global rigid state."""
    local_epoch, local_metric = load_checkpoint(
        model,
        local_checkpoint,
        optimizer=None,
        strict=True,
        map_location=map_location,
        load_optimizer=False,
    )
    copied = overlay_global_rigid_branch(
        model,
        global_checkpoint,
        map_location=map_location,
    )
    return int(local_epoch), float(local_metric), int(copied)


__all__ = [
    "GLOBAL_PREFIX",
    "load_joint_v4_checkpoints",
    "overlay_global_rigid_branch",
]
