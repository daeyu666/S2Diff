"""Run the standard V4 local diagnosis with cost-volume sub-pixel refinement."""

from __future__ import annotations

import argparse
import os
import sys

import diagnose_alignment_v4_local as base
from main import _build_model as build_plain_v4
from models.predictor_v4_subpixel import enable_cost_volume_subpixel_refiner


_SUBPIXEL_HIDDEN = 32
_SUBPIXEL_MAX_PX = 0.5


def _build_subpixel_model(cfg, info, device, process=None):
    model = build_plain_v4(cfg, info, device, process=process)
    enable_cost_volume_subpixel_refiner(
        model,
        subpixel_hidden_channels=_SUBPIXEL_HIDDEN,
        subpixel_max_px=_SUBPIXEL_MAX_PX,
    )
    return model


def main():
    global _SUBPIXEL_HIDDEN, _SUBPIXEL_MAX_PX
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--subpixel_hidden_channels", type=int, default=32)
    parser.add_argument("--subpixel_max_px", type=float, default=0.5)
    custom, remaining = parser.parse_known_args()
    if custom.subpixel_hidden_channels < 4:
        raise ValueError("--subpixel_hidden_channels must be >= 4")
    if custom.subpixel_max_px <= 0.0:
        raise ValueError("--subpixel_max_px must be > 0")
    _SUBPIXEL_HIDDEN = int(custom.subpixel_hidden_channels)
    _SUBPIXEL_MAX_PX = float(custom.subpixel_max_px)

    # Remove wrapper-only arguments before the standard diagnostic parser runs.
    sys.argv = [sys.argv[0], *remaining]
    base._build_model = _build_subpixel_model
    cfg, diagnostic = base.parse_diagnostic_args()
    if not diagnostic.misalignment_output:
        diagnostic.misalignment_output = os.path.join(
            cfg.output_root,
            "metrics",
            f"{cfg.dataset}_v4_subpixel_local_nonrigid.csv",
        )
    base.run_local_diagnosis(cfg, diagnostic)


if __name__ == "__main__":
    main()
