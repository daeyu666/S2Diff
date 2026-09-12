"""Compare oracle-state and real reverse-state local-flow accuracy.

The same local-only synthetic MSI warp and inverse sampling target are used for
both branches.  The only difference is the HSI-side matching state:

oracle : x_t = D~_t(X_GT)
reverse: x_t is cached from the actual T->0 reverse recursion.

This isolates train/inference state-domain mismatch without changing the local
flow architecture.
"""

from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np
import torch

from config import parse_args
from data_loader import build_loaders
from degradations.inverse_flow import forward_to_inverse_sampling_field
from degradations.misalignment import make_misaligned_msi
from innovation1 import build_progressive_process
from reverse_state_local_flow import (
    SCALES,
    oracle_states,
    rollout_reverse_states_identity,
    run_local_flow_from_states,
    state_gap_metrics,
)
from train_v4_local_reconstruction_inverse import INVERSE_FIXED_POINT_ITERATIONS
from train_v4_multiscale_bootstrap import field_metrics
from train_v4_recurrent_flow import build_recurrent_model
from utils import get_device, load_checkpoint, set_seed


def parse_diagnostic_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--reverse_state_checkpoint", type=str, required=True)
    parser.add_argument(
        "--reverse_state_eval_severities",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0],
    )
    parser.add_argument("--reverse_state_control_grid", type=int, default=5)
    parser.add_argument("--reverse_state_eval_trials", type=int, default=5)
    parser.add_argument("--reverse_state_eval_seed_offset", type=int, default=15431)
    parser.add_argument("--recurrent_hidden_channels", type=int, default=64)
    parser.add_argument("--recurrent_correlation_channels", type=int, default=32)
    parser.add_argument("--recurrent_iterations_scale4", type=int, default=3)
    parser.add_argument("--recurrent_iterations_scale2", type=int, default=2)
    parser.add_argument("--recurrent_iterations_scale1", type=int, default=2)
    parser.add_argument("--recurrent_max_update_scale4", type=float, default=2.0)
    parser.add_argument("--recurrent_max_update_scale2", type=float, default=1.0)
    parser.add_argument("--recurrent_max_update_scale1", type=float, default=0.5)
    diagnostic, remaining = parser.parse_known_args()
    cfg = parse_args(remaining)

    if cfg.stage != "test":
        raise ValueError("oracle/reverse diagnosis is test-only; use --stage test")
    if str(cfg.predictor_version).lower() != "v4":
        raise ValueError("oracle/reverse diagnosis requires predictor_version=v4")
    if diagnostic.reverse_state_control_grid < 2:
        raise ValueError("reverse_state_control_grid must be >= 2")
    if diagnostic.reverse_state_eval_trials < 1:
        raise ValueError("reverse_state_eval_trials must be >= 1")
    if any(v <= 0.0 for v in diagnostic.reverse_state_eval_severities):
        raise ValueError("all reverse-state severities must be > 0")

    for name in (
        "recurrent_hidden_channels",
        "recurrent_correlation_channels",
        "recurrent_iterations_scale4",
        "recurrent_iterations_scale2",
        "recurrent_iterations_scale1",
        "recurrent_max_update_scale4",
        "recurrent_max_update_scale2",
        "recurrent_max_update_scale1",
    ):
        setattr(cfg, name, getattr(diagnostic, name))
    return cfg, diagnostic


@torch.no_grad()
def evaluate(cfg, diagnostic) -> Dict[float, Dict[str, float]]:
    set_seed(cfg.seed)
    _, test_loader, info = build_loaders(cfg)
    device = get_device(cfg.device)
    process = build_progressive_process(cfg)
    model = build_recurrent_model(cfg, info, device, process)
    epoch, stored = load_checkpoint(
        model,
        diagnostic.reverse_state_checkpoint,
        optimizer=None,
        strict=True,
        map_location=str(device),
        load_optimizer=False,
    )
    model.eval()
    print(
        f"Loaded {diagnostic.reverse_state_checkpoint}: epoch={epoch}, "
        f"stored_metric={stored:.6f}"
    )

    results: Dict[float, Dict[str, float]] = {}
    base_seed = int(cfg.seed) + int(diagnostic.reverse_state_eval_seed_offset)

    for severity in diagnostic.reverse_state_eval_severities:
        accum: Dict[str, List[float]] = {
            "oracle_epe": [], "reverse_epe": [],
            "oracle_pred": [], "reverse_pred": [], "true": [],
            "gap4_mae": [], "gap2_mae": [], "gap1_mae": [],
            "oracle4_epe": [], "oracle2_epe": [], "oracle1_epe": [],
            "reverse4_epe": [], "reverse2_epe": [], "reverse1_epe": [],
        }

        for trial in range(int(diagnostic.reverse_state_eval_trials)):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(base_seed + trial * 100003)

            for batch in test_loader:
                gt = batch["gt"].to(device, non_blocking=True)
                hr_msi = batch["hr_msi"].to(device, non_blocking=True)
                warped, _, params = make_misaligned_msi(
                    hr_msi,
                    translation_max_px=0.0,
                    rotation_max_deg=0.0,
                    local_max_displacement_px=float(severity),
                    control_grid_size=int(diagnostic.reverse_state_control_grid),
                    generator=generator,
                )
                inverse_target = forward_to_inverse_sampling_field(
                    params.local_displacement_px,
                    iterations=INVERSE_FIXED_POINT_ITERATIONS,
                    padding_mode="border",
                )

                oracle = oracle_states(model, process, gt)
                reverse = rollout_reverse_states_identity(model, process, gt, warped)
                gaps = state_gap_metrics(oracle, reverse)

                oracle_final, _ = run_local_flow_from_states(model, oracle, warped)
                reverse_final, _ = run_local_flow_from_states(model, reverse, warped)

                oracle_metrics = {}
                reverse_metrics = {}
                for scale in SCALES:
                    target = (
                        inverse_target
                        if scale == 1
                        else process.state_at(
                            inverse_target,
                            int(model.geometry_aligner.stage_t_by_scale[scale]),
                        )
                    )
                    oracle_metrics[scale] = field_metrics(oracle_final[scale], target)
                    reverse_metrics[scale] = field_metrics(reverse_final[scale], target)

                oe, true_mag, op, _ = oracle_metrics[1]
                re, _, rp, _ = reverse_metrics[1]
                accum["oracle_epe"].append(oe)
                accum["reverse_epe"].append(re)
                accum["oracle_pred"].append(op)
                accum["reverse_pred"].append(rp)
                accum["true"].append(true_mag)
                for scale in SCALES:
                    accum[f"gap{scale}_mae"].append(gaps[scale]["mae"])
                    accum[f"oracle{scale}_epe"].append(oracle_metrics[scale][0])
                    accum[f"reverse{scale}_epe"].append(reverse_metrics[scale][0])

        summary = {key: float(np.mean(values)) for key, values in accum.items()}
        results[float(severity)] = summary
        print(
            f"local<={severity:g}: "
            f"oracle_EPE={summary['oracle_epe']:.4f} "
            f"reverse_EPE={summary['reverse_epe']:.4f} "
            f"pred oracle/reverse/true={summary['oracle_pred']:.3f}/"
            f"{summary['reverse_pred']:.3f}/{summary['true']:.3f}"
        )
        print(
            "  stage EPE oracle->reverse: "
            f"s4 {summary['oracle4_epe']:.4f}->{summary['reverse4_epe']:.4f}, "
            f"s2 {summary['oracle2_epe']:.4f}->{summary['reverse2_epe']:.4f}, "
            f"s1 {summary['oracle1_epe']:.4f}->{summary['reverse1_epe']:.4f}"
        )
        print(
            "  HSI-state MAE oracle-vs-reverse: "
            f"s4={summary['gap4_mae']:.6f}, "
            f"s2={summary['gap2_mae']:.6f}, "
            f"s1={summary['gap1_mae']:.6f}"
        )

    return results


if __name__ == "__main__":
    cfg, diagnostic = parse_diagnostic_args()
    evaluate(cfg, diagnostic)
