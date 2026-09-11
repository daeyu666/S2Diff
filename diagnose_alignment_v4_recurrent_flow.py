"""Run the existing local non-rigid diagnostic with the recurrent-flow V4 model.

This wrapper intentionally reuses diagnose_alignment_v4_local.py so registered,
global-only, local-only and global+local protocols stay identical to the older
soft-expectation/confidence/subpixel ablations.

The launcher uses the default recurrent architecture from train_v4_recurrent_flow.py:
    hidden=64, corr=32, iterations 4/2/1=3/2/2,
    max update 4/2/1=2.0/1.0/0.5 px.
"""

from __future__ import annotations

import diagnose_alignment_v4_local as base_diagnostic
from train_v4_recurrent_flow import build_recurrent_model


def _build_recurrent_for_diagnostic(cfg, info, device, process=None):
    if process is None:
        raise ValueError("recurrent-flow diagnostic requires progressive process")
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
    cfg, diagnostic = base_diagnostic.parse_diagnostic_args()
    base_diagnostic._build_model = _build_recurrent_for_diagnostic
    base_diagnostic.run_local_diagnosis(cfg, diagnostic)


if __name__ == "__main__":
    main()
