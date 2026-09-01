"""Diagnose Innovation 1 predictor capacity versus reverse-trajectory drift.

This script does not change the training protocol. It loads an existing
Innovation 1 checkpoint and compares three views of the same predictor:

1) oracle-state prediction: F_theta(D~_t(X), t) for every t=1..T;
2) terminal one-shot: F_theta(U_T(D_T(X)), T);
3) full reverse: states generated recursively by the fixed physics-consistent
   reverse update.

It also measures how far each recursive reverse state has drifted from the
corresponding oracle forward state D~_t(X), and reports raw value-range
statistics without clamping the model output.
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch

from config import parse_args, print_config
from data_loader import build_loaders
from innovation1 import build_progressive_process
from metrics import MetricAverager, calc_metrics
from models import CleanHSIPredictor
from utils import count_parameters, ensure_dir, get_device, load_checkpoint, set_seed


def _default_checkpoint(cfg) -> str:
    root = os.path.join(cfg.checkpoint_root, "innovation1")
    if cfg.save_name:
        filename = cfg.save_name
        if not filename.endswith(".pth"):
            filename += ".pth"
    else:
        filename = f"{cfg.dataset}_innovation1_{cfg.degradation_mode}.pth"
    return os.path.join(root, filename)


def _build_model(cfg, info, device):
    model = CleanHSIPredictor(
        n_bands=int(info["n_bands"]),
        total_steps=int(cfg.diffusion_steps),
        base_channels=int(cfg.predictor_base_channels),
        time_dim=int(cfg.predictor_time_dim),
        dropout=float(cfg.predictor_dropout),
        residual_prediction=True,
    ).to(device)
    print(f"Predictor trainable params: {count_parameters(model):.3f} M")
    return model


def _range_stats(x: torch.Tensor) -> Dict[str, float]:
    x = x.detach().float()
    total = max(int(x.numel()), 1)
    below = int((x < 0.0).sum().item())
    above = int((x > 1.0).sum().item())
    return {
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "below0_pct": 100.0 * below / total,
        "above1_pct": 100.0 * above / total,
    }


def _mean_dict(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def _fmt(value: float) -> str:
    if value is None:
        return ""
    return f"{float(value):.6f}"


def run_diagnostic(cfg, test_loader, info, device):
    process = build_progressive_process(cfg)
    model = _build_model(cfg, info, device)
    checkpoint = cfg.resume or _default_checkpoint(cfg)

    loaded_epoch, loaded_best = load_checkpoint(
        model,
        checkpoint,
        optimizer=None,
        map_location=str(device),
        load_optimizer=False,
    )
    model.eval()

    print(
        f"Loaded checkpoint: {checkpoint}\n"
        f"  epoch={loaded_epoch}, stored_best_PSNR={loaded_best:.6f}\n"
        f"  degradation={process.operator.mode}, T={process.total_steps}, "
        f"lift={process.default_lift_mode}"
    )

    T = process.total_steps
    oracle_meters = {t: MetricAverager() for t in range(1, T + 1)}
    reverse_pred_meters = {t: MetricAverager() for t in range(1, T + 1)}
    reverse_state_meters = {t: MetricAverager() for t in range(0, T + 1)}
    one_shot_meter = MetricAverager()
    full_reverse_meter = MetricAverager()
    init_meter = MetricAverager()

    drift_rows = defaultdict(list)
    range_rows = defaultdict(list)

    with torch.no_grad():
        for batch in test_loader:
            gt = batch["gt"].to(device, non_blocking=True)
            target_size = tuple(gt.shape[-2:])

            # ---------------------------------------------------------------
            # A. Oracle-state predictor: network always sees exact D~_t(X).
            # ---------------------------------------------------------------
            oracle_states = {}
            for t in range(1, T + 1):
                oracle_state = process.state_at(gt, t)
                oracle_states[t] = oracle_state
                timestep = torch.full(
                    (gt.shape[0],), t, dtype=torch.long, device=device
                )
                oracle_pred = model(oracle_state, timestep)
                oracle_meters[t].update(
                    calc_metrics(oracle_pred, gt, cfg.scale_ratio)
                )

            # ---------------------------------------------------------------
            # B. Terminal one-shot from the actual terminal observation.
            # ---------------------------------------------------------------
            terminal_lr = process.terminal_observation(gt)
            x_t = process.terminal_state(terminal_lr, target_size=target_size)
            init_meter.update(calc_metrics(x_t, gt, cfg.scale_ratio))

            timestep_T = torch.full(
                (gt.shape[0],), T, dtype=torch.long, device=device
            )
            one_shot = model(x_t, timestep_T)
            one_shot_meter.update(calc_metrics(one_shot, gt, cfg.scale_ratio))

            # ---------------------------------------------------------------
            # C. Full reverse. At every t compare the recursively generated
            # state with the exact oracle state D~_t(X).
            # ---------------------------------------------------------------
            reverse_state_meters[T].update(
                calc_metrics(x_t, gt, cfg.scale_ratio)
            )
            oracle_T = oracle_states[T]
            drift_rows[T].append(
                float(torch.mean(torch.abs(x_t - oracle_T)).item())
            )
            range_rows[T].append(_range_stats(x_t))

            for t in range(T, 0, -1):
                timestep = torch.full(
                    (gt.shape[0],), t, dtype=torch.long, device=device
                )
                pred_x0 = model(x_t, timestep)
                reverse_pred_meters[t].update(
                    calc_metrics(pred_x0, gt, cfg.scale_ratio)
                )

                x_prev = process.reverse_update(x_t, pred_x0, t)
                prev_t = t - 1
                reverse_state_meters[prev_t].update(
                    calc_metrics(x_prev, gt, cfg.scale_ratio)
                )
                range_rows[prev_t].append(_range_stats(x_prev))

                if prev_t > 0:
                    oracle_prev = oracle_states[prev_t]
                else:
                    oracle_prev = gt
                drift_rows[prev_t].append(
                    float(torch.mean(torch.abs(x_prev - oracle_prev)).item())
                )
                x_t = x_prev

            full_reverse_meter.update(calc_metrics(x_t, gt, cfg.scale_ratio))

    init_metrics = init_meter.average()
    one_shot_metrics = one_shot_meter.average()
    full_metrics = full_reverse_meter.average()
    oracle_metrics = {t: oracle_meters[t].average() for t in range(1, T + 1)}
    reverse_pred_metrics = {
        t: reverse_pred_meters[t].average() for t in range(1, T + 1)
    }
    reverse_state_metrics = {
        t: reverse_state_meters[t].average() for t in range(0, T + 1)
    }

    # -----------------------------------------------------------------------
    # Console report.
    # -----------------------------------------------------------------------
    print("\n=== Global comparison ===")
    print(
        f"INIT state       : PSNR={init_metrics['PSNR']:.4f} "
        f"SAM={init_metrics['SAM']:.4f}"
    )
    print(
        f"Terminal one-shot: PSNR={one_shot_metrics['PSNR']:.4f} "
        f"SAM={one_shot_metrics['SAM']:.4f}"
    )
    print(
        f"Full reverse     : PSNR={full_metrics['PSNR']:.4f} "
        f"SAM={full_metrics['SAM']:.4f}"
    )
    print(
        f"One-shot - full PSNR gap = "
        f"{one_shot_metrics['PSNR'] - full_metrics['PSNR']:.4f} dB"
    )

    print("\n=== Per-t predictor diagnostic ===")
    print(
        " t scale | oracle_X0_PSNR oracle_X0_SAM | "
        "reverse_X0_PSNR reverse_X0_SAM | state_drift_L1"
    )
    for t in range(1, T + 1):
        scale = process.state(t).scale
        drift = float(np.mean(drift_rows[t]))
        print(
            f"{t:2d} {scale:5d} | "
            f"{oracle_metrics[t]['PSNR']:14.4f} {oracle_metrics[t]['SAM']:13.4f} | "
            f"{reverse_pred_metrics[t]['PSNR']:15.4f} "
            f"{reverse_pred_metrics[t]['SAM']:14.4f} | "
            f"{drift:.6e}"
        )

    print("\n=== Reverse-state trajectory ===")
    print(
        " t scale | state_PSNR state_SAM | min max below0% above1% | drift_L1"
    )
    for t in range(T, -1, -1):
        scale = process.state(t).scale if t > 0 else 1
        r = _mean_dict(range_rows[t])
        drift = float(np.mean(drift_rows[t]))
        m = reverse_state_metrics[t]
        print(
            f"{t:2d} {scale:5d} | {m['PSNR']:10.4f} {m['SAM']:9.4f} | "
            f"{r['min']: .5f} {r['max']: .5f} "
            f"{r['below0_pct']:7.3f} {r['above1_pct']:7.3f} | "
            f"{drift:.6e}"
        )

    # -----------------------------------------------------------------------
    # Save machine-readable CSVs for later comparison across checkpoints.
    # -----------------------------------------------------------------------
    output_dir = os.path.join(cfg.output_root, "metrics")
    ensure_dir(output_dir)
    stem = f"{cfg.dataset}_innovation1_{cfg.degradation_mode}_diagnostic"
    predictor_csv = os.path.join(output_dir, stem + "_predictor.csv")
    trajectory_csv = os.path.join(output_dir, stem + "_trajectory.csv")

    with open(predictor_csv, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "t", "scale",
            "oracle_psnr", "oracle_sam",
            "reverse_pred_psnr", "reverse_pred_sam",
            "state_drift_l1",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in range(1, T + 1):
            writer.writerow({
                "t": t,
                "scale": process.state(t).scale,
                "oracle_psnr": _fmt(oracle_metrics[t]["PSNR"]),
                "oracle_sam": _fmt(oracle_metrics[t]["SAM"]),
                "reverse_pred_psnr": _fmt(reverse_pred_metrics[t]["PSNR"]),
                "reverse_pred_sam": _fmt(reverse_pred_metrics[t]["SAM"]),
                "state_drift_l1": _fmt(float(np.mean(drift_rows[t]))),
            })

    with open(trajectory_csv, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "t", "scale", "state_psnr", "state_sam",
            "min", "max", "below0_pct", "above1_pct", "state_drift_l1",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in range(T, -1, -1):
            r = _mean_dict(range_rows[t])
            m = reverse_state_metrics[t]
            writer.writerow({
                "t": t,
                "scale": process.state(t).scale if t > 0 else 1,
                "state_psnr": _fmt(m["PSNR"]),
                "state_sam": _fmt(m["SAM"]),
                "min": _fmt(r["min"]),
                "max": _fmt(r["max"]),
                "below0_pct": _fmt(r["below0_pct"]),
                "above1_pct": _fmt(r["above1_pct"]),
                "state_drift_l1": _fmt(float(np.mean(drift_rows[t]))),
            })

    print(f"\nSaved predictor diagnostic: {predictor_csv}")
    print(f"Saved trajectory diagnostic: {trajectory_csv}")

    print("\n=== Interpretation hints ===")
    gap = one_shot_metrics["PSNR"] - full_metrics["PSNR"]
    terminal_oracle = oracle_metrics[T]["PSNR"]
    if gap >= 1.0:
        print(
            "Large one-shot/full-reverse gap detected: reverse-state distribution "
            "drift is a strong candidate bottleneck."
        )
    else:
        print(
            "One-shot/full-reverse gap is small: reverse recursion is probably "
            "not the dominant bottleneck."
        )
    if terminal_oracle < 32.0:
        print(
            "Terminal oracle predictor itself is still modest (<32 dB): predictor "
            "capacity / spectral-spatial modeling is also a likely bottleneck."
        )
    else:
        print(
            "Terminal oracle predictor is comparatively strong: prioritize the "
            "gap between oracle-state and recursive-state prediction."
        )


def main():
    # Reuse the exact training configuration parser. No new training settings
    # are introduced by this diagnostic script.
    cfg = parse_args()
    print_config(cfg)
    set_seed(cfg.seed)

    _, test_loader, info = build_loaders(cfg)
    device = get_device(cfg.device)
    print(f"Device: {device}")
    run_diagnostic(cfg, test_loader, info, device)


if __name__ == "__main__":
    main()
