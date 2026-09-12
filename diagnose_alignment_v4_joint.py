"""Final composed Global+Local diagnosis for Innovation-2 V4.

Unlike diagnose_alignment_v4_recurrent_flow.py, this launcher never assumes one
checkpoint is authoritative for both geometry branches.  It loads the late
reverse-state local checkpoint as the full recurrent model, then selectively
overlays only ``geometry_aligner.global_aligner.*`` from the validated global
rigid checkpoint before running the existing registered/global-only/local-only/
global+local diagnostic protocol.
"""

from __future__ import annotations

import argparse
import sys

import diagnose_alignment_v4_local as base_diagnostic
from train_v4_recurrent_flow import build_recurrent_model
from v4_joint_checkpoint import load_joint_v4_checkpoints


def _build_recurrent_for_joint(cfg, info, device, process=None):
    if process is None:
        raise ValueError("joint recurrent diagnostic requires progressive process")
    cfg.recurrent_hidden_channels = 64
    cfg.recurrent_correlation_channels = 32
    cfg.recurrent_iterations_scale1 = 2
    cfg.recurrent_iterations_scale2 = 2
    cfg.recurrent_iterations_scale4 = 3
    cfg.recurrent_max_update_scale1 = 0.5
    cfg.recurrent_max_update_scale2 = 1.0
    cfg.recurrent_max_update_scale4 = 2.0
    return build_recurrent_model(cfg, info, device, process)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--joint_local_checkpoint",
        type=str,
        default="./checkpoints/innovation1/PaviaU_v4_reverse_state_reconstruction_l1_oa0p2.pth",
    )
    parser.add_argument("--joint_global_checkpoint", type=str, required=True)
    joint, remaining = parser.parse_known_args()

    original_argv = list(sys.argv)
    try:
        sys.argv = [original_argv[0]] + remaining
        cfg, diagnostic = base_diagnostic.parse_diagnostic_args()
    finally:
        sys.argv = original_argv

    # base diagnostic insists on --resume because historically it used one file.
    # The composed loader below ignores that distinction and treats this as the
    # local/full source before selectively restoring the global branch.
    cfg.resume = joint.joint_local_checkpoint
    base_diagnostic._build_model = _build_recurrent_for_joint

    def _load_composed(
        model,
        path,
        optimizer=None,
        strict=True,
        map_location="cpu",
        load_optimizer=False,
    ):
        del path, optimizer, strict, load_optimizer
        epoch, metric, copied = load_joint_v4_checkpoints(
            model,
            joint.joint_local_checkpoint,
            joint.joint_global_checkpoint,
            map_location=str(map_location),
        )
        print(
            "Composed final V4: "
            f"local/full={joint.joint_local_checkpoint}; "
            f"global={joint.joint_global_checkpoint}; "
            f"global_tensors_copied={copied}"
        )
        return epoch, metric

    base_diagnostic.load_checkpoint = _load_composed
    base_diagnostic.run_local_diagnosis(cfg, diagnostic)


if __name__ == "__main__":
    main()
