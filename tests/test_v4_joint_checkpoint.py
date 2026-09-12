import os
import tempfile

import torch
import torch.nn as nn

from v4_joint_checkpoint import overlay_global_rigid_branch


class _Geometry(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_aligner = nn.Linear(3, 2)
        self.local_aligner = nn.Linear(3, 2)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(3, 3)
        self.geometry_aligner = _Geometry()


def test_overlay_changes_only_global_branch():
    target = _Model()
    source = _Model()
    with torch.no_grad():
        for p in target.parameters():
            p.fill_(1.0)
        for p in source.parameters():
            p.fill_(7.0)

    before_backbone = target.backbone.weight.detach().clone()
    before_local = target.geometry_aligner.local_aligner.weight.detach().clone()

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "global.pth")
        torch.save({"model": source.state_dict()}, path)
        copied = overlay_global_rigid_branch(target, path)

    assert copied > 0
    assert torch.allclose(target.backbone.weight, before_backbone)
    assert torch.allclose(target.geometry_aligner.local_aligner.weight, before_local)
    assert torch.allclose(
        target.geometry_aligner.global_aligner.weight,
        source.geometry_aligner.global_aligner.weight,
    )
