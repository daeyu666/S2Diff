"""Translation diagnosis for learned Innovation-2 V4 alignment.

The external misalignment protocol is identical to the Raw-Direct sensitivity
experiment. Only HR-MSI is warped. V4 performs one learned global rigid
correction, physical-domain matching, and sparse local residual updates only at
the scale-4/2/1 boundaries before Raw-Direct fusion.

Valid-overlap PSNR/SAM remain the primary metrics. This script expects a trained
V4 checkpoint; a V3 Raw-Direct checkpoint is only a warm-start source for V4
training and is not a complete V4 evaluation checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict

import numpy as np
import torch

from config import parse_args
from data_loader import build_loaders
from degradations.misalignment import make_misaligned_msi
from diagnose_misalignment_translation import calc_masked_psnr_sam, fixed_roi
from innovation1 import build_progressive_process, reconstruct_from_terminal_lr
from main import _build_model
from metrics import calc_metrics
from utils import get_device, load_checkpoint, set_seed


DEFAULT_SHIFTS = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0]


def parse_diagnostic_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--misalignment_shifts", type=float, nargs="+", default=DEFAULT_SHIFTS)
    parser.add_argument("--misalignment_trials", type=int, default=5)
    parser.add_argument("--misalignment_fixed_margin", type=int, default=8)
    parser.add_argument("--misalignment_seed", type=int, default=None)
    parser.add_argument("--misalignment_valid_threshold", type=float, default=0.999)
    parser.add_argument("--misalignment_output", type=str, default="")
    diagnostic, remaining = parser.parse_known_args()
    cfg = parse_args(remaining)

    if cfg.stage != "test":
        raise ValueError("Use --stage test")
    if str(cfg.predictor_version).lower() != "v4":
        raise ValueError("This script expects --predictor_version v4")
    if str(cfg.msi_ablation).lower() != "raw_direct":
        raise ValueError("V4 uses --msi_ablation raw_direct")
    if diagnostic.misalignment_trials < 1:
        raise ValueError("misalignment_trials must be >=1")
    if diagnostic.misalignment_fixed_margin < 0:
        raise ValueError("misalignment_fixed_margin must be >=0")
    if not 0.0 < diagnostic.misalignment_valid_threshold <= 1.0:
        raise ValueError("misalignment_valid_threshold must lie in (0,1]")
    if any(float(v) < 0.0 for v in diagnostic.misalignment_shifts):
        raise ValueError("misalignment_shifts must be >=0")
    return cfg, diagnostic


def _write_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def run(cfg, diagnostic):
    set_seed(cfg.seed)
    _, test_loader, info = build_loaders(cfg)
    device = get_device(cfg.device)
    process = build_progressive_process(cfg)
    model = _build_model(cfg, info, device, process=process)

    if not cfg.resume:
        raise ValueError("Pass the trained V4 checkpoint explicitly with --resume")
    loaded_epoch, loaded_best = load_checkpoint(
        model,
        cfg.resume,
        optimizer=None,
        map_location=str(device),
        load_optimizer=False,
    )
    print(
        f"Loaded trained V4 checkpoint {cfg.resume}: epoch={loaded_epoch}, "
        f"stored_best_PSNR={loaded_best:.6f}"
    )

    batches = list(test_loader)
    if not batches:
        raise ValueError("Test loader is empty")

    base_seed = cfg.seed if diagnostic.misalignment_seed is None else int(diagnostic.misalignment_seed)
    details = []

    for max_shift in [float(v) for v in diagnostic.misalignment_shifts]:
        trials = 1 if max_shift == 0.0 else int(diagnostic.misalignment_trials)
        for trial in range(trials):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(base_seed + trial * 100003)

            for sample_index, batch in enumerate(batches):
                gt = batch["gt"].to(device, non_blocking=True)
                hr_msi = batch["hr_msi"].to(device, non_blocking=True)
                shifted_msi, valid_soft, params = make_misaligned_msi(
                    hr_msi,
                    translation_max_px=max_shift,
                    rotation_max_deg=0.0,
                    local_max_displacement_px=0.0,
                    generator=generator,
                )

                terminal_lr = process.terminal_observation(gt)
                pred = reconstruct_from_terminal_lr(
                    model,
                    process,
                    terminal_lr,
                    target_size=tuple(gt.shape[-2:]),
                    hr_msi=shifted_msi,
                )

                full = calc_metrics(pred, gt, cfg.scale_ratio)
                valid_psnr, valid_sam, valid_fraction = calc_masked_psnr_sam(
                    pred,
                    gt,
                    valid_soft,
                    threshold=diagnostic.misalignment_valid_threshold,
                )
                fixed = calc_metrics(
                    fixed_roi(pred, diagnostic.misalignment_fixed_margin),
                    fixed_roi(gt, diagnostic.misalignment_fixed_margin),
                    cfg.scale_ratio,
                )

                dx = float(params.dx_px[0].item())
                dy = float(params.dy_px[0].item())
                global_shift = model.last_global_shift_px
                if global_shift is None:
                    gx = gy = float("nan")
                else:
                    gx = float(global_shift[0, 0].item())
                    gy = float(global_shift[0, 1].item())
                global_rotation = model.last_global_rotation_deg
                grot = (
                    float("nan")
                    if global_rotation is None
                    else float(global_rotation[0].item())
                )

                local_mean = float("nan")
                local_scale = -1
                if model.last_alignment is not None:
                    local_mag = torch.linalg.vector_norm(
                        model.last_alignment.local_offset_px.float(), dim=1
                    )
                    local_mean = float(local_mag.mean().item())
                    local_scale = int(model.last_alignment.local_scale)

                details.append(
                    {
                        "max_shift_px": max_shift,
                        "trial": trial,
                        "sample": sample_index,
                        "dx_px": dx,
                        "dy_px": dy,
                        "actual_shift_px": math.hypot(dx, dy),
                        "global_correction_dx_px": gx,
                        "global_correction_dy_px": gy,
                        "global_correction_rotation_deg": grot,
                        "final_local_scale": local_scale,
                        "final_local_offset_mean_px": local_mean,
                        "valid_fraction": valid_fraction,
                        "PSNR_full": full["PSNR"],
                        "SAM_full": full["SAM"],
                        "PSNR_valid": valid_psnr,
                        "SAM_valid": valid_sam,
                        "PSNR_fixed": fixed["PSNR"],
                        "SAM_fixed": fixed["SAM"],
                    }
                )

    groups = defaultdict(list)
    for row in details:
        groups[float(row["max_shift_px"])].append(row)

    summary = []
    for d in [float(v) for v in diagnostic.misalignment_shifts]:
        rows = groups[d]
        summary.append(
            {
                "max_shift_px": d,
                "n_runs": len(rows),
                "mean_actual_shift_px": float(np.mean([r["actual_shift_px"] for r in rows])),
                "PSNR_valid": float(np.mean([r["PSNR_valid"] for r in rows])),
                "SAM_valid": float(np.mean([r["SAM_valid"] for r in rows])),
                "PSNR_full": float(np.mean([r["PSNR_full"] for r in rows])),
                "SAM_full": float(np.mean([r["SAM_full"] for r in rows])),
                "mean_global_correction_px": float(
                    np.mean([
                        math.hypot(r["global_correction_dx_px"], r["global_correction_dy_px"])
                        for r in rows
                    ])
                ),
                "mean_abs_global_rotation_deg": float(
                    np.mean([abs(r["global_correction_rotation_deg"]) for r in rows])
                ),
                "mean_final_local_offset_px": float(
                    np.mean([r["final_local_offset_mean_px"] for r in rows])
                ),
            }
        )

    print("\nLearned V4 translation alignment summary (valid-overlap primary):")
    for row in summary:
        print(
            f"d={row['max_shift_px']:>4.1f}px "
            f"actual={row['mean_actual_shift_px']:.3f}px "
            f"PSNR_valid={row['PSNR_valid']:.4f} "
            f"SAM_valid={row['SAM_valid']:.4f} "
            f"global_corr={row['mean_global_correction_px']:.3f}px "
            f"global_rot={row['mean_abs_global_rotation_deg']:.3f}deg "
            f"local_final={row['mean_final_local_offset_px']:.3f}px"
        )

    if diagnostic.misalignment_output:
        summary_path = diagnostic.misalignment_output
    else:
        summary_path = os.path.join(
            cfg.output_root,
            "metrics",
            f"{cfg.dataset}_v4_learned_alignment_translation.csv",
        )
    stem, ext = os.path.splitext(summary_path)
    detail_path = f"{stem}_details{ext or '.csv'}"
    _write_csv(summary_path, summary)
    _write_csv(detail_path, details)
    print(f"Summary CSV: {summary_path}")
    print(f"Details CSV: {detail_path}")


if __name__ == "__main__":
    cfg, diagnostic = parse_diagnostic_args()
    run(cfg, diagnostic)
