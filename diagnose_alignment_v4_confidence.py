"""Run the standard V4 local non-rigid diagnosis with confidence-gated local alignment.

The metric protocol is intentionally identical to diagnose_alignment_v4_local.py
so registered/global-only/local-only/global+local results remain directly
comparable. This wrapper only upgrades the built V4 local aligner before loading
the confidence checkpoint and uses a separate default CSV name.
"""

from __future__ import annotations

import os

import diagnose_alignment_v4_local as base
from main import _build_model as build_plain_v4
from models.predictor_v4_confidence import enable_confidence_gated_local_aligner


def _build_confidence_model(cfg, info, device, process=None):
    model = build_plain_v4(cfg, info, device, process=process)
    enable_confidence_gated_local_aligner(model)
    return model


def main():
    # run_local_diagnosis resolves _build_model from its module globals.
    base._build_model = _build_confidence_model
    cfg, diagnostic = base.parse_diagnostic_args()
    if not diagnostic.misalignment_output:
        diagnostic.misalignment_output = os.path.join(
            cfg.output_root,
            "metrics",
            f"{cfg.dataset}_v4_confidence_local_nonrigid.csv",
        )
    base.run_local_diagnosis(cfg, diagnostic)


if __name__ == "__main__":
    main()
