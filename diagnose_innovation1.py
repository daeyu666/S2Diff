"""Diagnose predictor capacity and reverse-state drift for V1/V2/V3 ablations."""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch

from config import parse_args, print_config
from data_loader import build_loaders
from innovation1 import build_progressive_process, model_predict
from metrics import MetricAverager, calc_metrics
from models import (
    CleanHSIPredictor,
    MSIAblationGuidedPredictor,
    SpectralSpatialCleanHSIPredictor,
)
from utils import count_parameters, ensure_dir, get_device, load_checkpoint, set_seed


def _predictor_tag(cfg):
    version = str(getattr(cfg, "predictor_version", "v1")).lower()
    base_channels = int(cfg.predictor_base_channels)
    if version == "v1":
        return "" if base_channels == 64 else f"_bc{base_channels}"
    if version == "v2":
        return "_v2" if base_channels == 64 else f"_v2_bc{base_channels}"
    if version == "v3":
        ablation = str(getattr(cfg, "msi_ablation", "full")).lower()
        mode_tag = "" if ablation == "full" else f"_{ablation}"
        width_tag = "" if base_channels == 64 else f"_bc{base_channels}"
        return f"_v3{mode_tag}{width_tag}"
    raise ValueError(f"Unsupported predictor_version: {version}")


def _default_checkpoint(cfg) -> str:
    root = os.path.join(cfg.checkpoint_root, "innovation1")
    if cfg.save_name:
        filename = cfg.save_name
        if not filename.endswith(".pth"):
            filename += ".pth"
    else:
        filename = (
            f"{cfg.dataset}_innovation1_{cfg.degradation_mode}"
            f"{_predictor_tag(cfg)}.pth"
        )
    return os.path.join(root, filename)


def _build_model(cfg, info, device):
    common = dict(
        n_bands=int(info["n_bands"]),
        total_steps=int(cfg.diffusion_steps),
        base_channels=int(cfg.predictor_base_channels),
        time_dim=int(cfg.predictor_time_dim),
        dropout=float(cfg.predictor_dropout),
        residual_prediction=True,
    )
    version = str(getattr(cfg, "predictor_version", "v1")).lower()
    if version == "v1":
        model = CleanHSIPredictor(**common)
    elif version == "v2":
        model = SpectralSpatialCleanHSIPredictor(
            **common,
            spectral_hidden=int(cfg.spectral_stem_hidden),
        )
    elif version == "v3":
        model = MSIAblationGuidedPredictor(
            **common,
            n_msi_bands=int(info["n_select_bands"]),
            spectral_hidden=int(cfg.spectral_stem_hidden),
            msi_highpass_kernel=int(cfg.msi_highpass_kernel),
            msi_highpass_sigma=float(cfg.msi_highpass_sigma),
            msi_ablation=str(getattr(cfg, "msi_ablation", "full")),
        )
    else:
        raise ValueError(f"Unsupported predictor_version: {version}")
    model = model.to(device)
    ablation = str(getattr(cfg, "msi_ablation", "full")) if version == "v3" else "n/a"
    print(
        f"Predictor version={version}, msi_ablation={ablation}, "
        f"params={count_parameters(model):.3f} M, "
        f"requires_msi={bool(getattr(model, 'requires_msi', False))}"
    )
    return model


def _range_stats(x: torch.Tensor) -> Dict[str, float]:
    x = x.detach().float()
    total = max(int(x.numel()), 1)
    return {
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "below0_pct": 100.0 * int((x < 0.0).sum().item()) / total,
        "above1_pct": 100.0 * int((x > 1.0).sum().item()) / total,
    }


def _mean_dict(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0].keys()
    }


def _fmt(value: float) -> str:
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
        f"  predictor={cfg.predictor_version}, "
        f"msi_ablation={getattr(cfg, 'msi_ablation', 'full')}, "
        f"base_channels={cfg.predictor_base_channels}\n"
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
            hr_msi = None
            if bool(getattr(model, "requires_msi", False)):
                hr_msi = batch["hr_msi"].to(device, non_blocking=True)
            target_size = tuple(gt.shape[-2:])

            oracle_states = {}
            for t in range(1, T + 1):
                oracle_state = process.state_at(gt, t)
                oracle_states[t] = oracle_state
                timestep = torch.full(
                    (gt.shape[0],), t, dtype=torch.long, device=device
                )
                oracle_pred = model_predict(
                    model, oracle_state, timestep, hr_msi=hr_msi
                )
                oracle_meters[t].update(
                    calc_metrics(oracle_pred, gt, cfg.scale_ratio)
                )

            terminal_lr = process.terminal_observation(gt)
            x_t = process.terminal_state(terminal_lr, target_size=target_size)
            init_meter.update(calc_metrics(x_t, gt, cfg.scale_ratio))

            timestep_T = torch.full(
                (gt.shape[0],), T, dtype=torch.long, device=device
            )
            one_shot = model_predict(model, x_t, timestep_T, hr_msi=hr_msi)
            one_shot_meter.update(calc_metrics(one_shot, gt, cfg.scale_ratio))

            reverse_state_meters[T].update(calc_metrics(x_t, gt, cfg.scale_ratio))
            drift_rows[T].append(
                float(torch.mean(torch.abs(x_t - oracle_states[T])).item())
            )
            range_rows[T].append(_range_stats(x_t))

            for t in range(T, 0, -1):
                timestep = torch.full(
                    (gt.shape[0],), t, dtype=torch.long, device=device
                )
                pred_x0 = model_predict(model, x_t, timestep, hr_msi=hr_msi)
                reverse_pred_meters[t].update(
                    calc_metrics(pred_x0, gt, cfg.scale_ratio)
                )

                x_prev = process.reverse_update(x_t, pred_x0, t)
                prev_t = t - 1
                reverse_state_meters[prev_t].update(
                    calc_metrics(x_prev, gt, cfg.scale_ratio)
                )
                range_rows[prev_t].append(_range_stats(x_prev))
                oracle_prev = oracle_states[prev_t] if prev_t > 0 else gt
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

    print("\n=== Global comparison ===")
    print(f"INIT state        : PSNR={init_metrics['PSNR']:.4f} SAM={init_metrics['SAM']:.4f}")
    print(f"Terminal one-shot : PSNR={one_shot_metrics['PSNR']:.4f} SAM={one_shot_metrics['SAM']:.4f}")
    print(f"Full reverse      : PSNR={full_metrics['PSNR']:.4f} SAM={full_metrics['SAM']:.4f}")
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
        print(
            f"{t:2d} {scale:5d} | "
            f"{oracle_metrics[t]['PSNR']:14.4f} {oracle_metrics[t]['SAM']:13.4f} | "
            f"{reverse_pred_metrics[t]['PSNR']:15.4f} "
            f"{reverse_pred_metrics[t]['SAM']:14.4f} | "
            f"{float(np.mean(drift_rows[t])):.6e}"
        )

    print("\n=== Reverse-state trajectory ===")
    print(" t scale | state_PSNR state_SAM | min max below0% above1% | drift_L1")
    for t in range(T, -1, -1):
        scale = process.state(t).scale if t > 0 else 1
        r = _mean_dict(range_rows[t])
        m = reverse_state_metrics[t]
        print(
            f"{t:2d} {scale:5d} | {m['PSNR']:10.4f} {m['SAM']:9.4f} | "
            f"{r['min']: .5f} {r['max']: .5f} "
            f"{r['below0_pct']:7.3f} {r['above1_pct']:7.3f} | "
            f"{float(np.mean(drift_rows[t])):.6e}"
        )

    output_dir = os.path.join(cfg.output_root, "metrics")
    ensure_dir(output_dir)
    stem = (
        f"{cfg.dataset}_innovation1_{cfg.degradation_mode}"
        f"{_predictor_tag(cfg)}_diagnostic"
    )
    predictor_csv = os.path.join(output_dir, stem + "_predictor.csv")
    trajectory_csv = os.path.join(output_dir, stem + "_trajectory.csv")

    with open(predictor_csv, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "t", "scale", "oracle_psnr", "oracle_sam",
            "reverse_pred_psnr", "reverse_pred_sam", "state_drift_l1",
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


def main():
    cfg = parse_args()
    print_config(cfg)
    set_seed(cfg.seed)
    _, test_loader, info = build_loaders(cfg)
    device = get_device(cfg.device)
    print(f"Device: {device}")
    run_diagnostic(cfg, test_loader, info, device)


if __name__ == "__main__":
    main()
