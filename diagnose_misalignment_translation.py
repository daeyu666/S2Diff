"""Translation-only HSI-MSI misalignment sensitivity diagnosis.

Only HR-MSI is translated. GT-HSI, LR-HSI, the physical degradation trajectory
and reverse recursion remain fixed.

IMPORTANT: severity d is the Euclidean 2-D translation-radius upper bound:
    r ~ U(0, d), theta ~ U(0, 2*pi)
    dx = r*cos(theta), dy = r*sin(theta)
    sqrt(dx^2 + dy^2) <= d

The shared ``degradations.misalignment.make_misaligned_msi`` sampler is used so
training and evaluation cannot silently diverge. Reusing the same trial seed
for every d preserves the same normalized radius/direction and yields a paired
sensitivity curve.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

from config import parse_args
from data_loader import build_loaders
from degradations.misalignment import make_misaligned_msi
from innovation1 import build_progressive_process, reconstruct_from_terminal_lr
from main import _build_model
from metrics import calc_metrics
from utils import get_device, load_checkpoint, set_seed


DEFAULT_SHIFTS = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0]


def parse_diagnostic_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--misalignment_shifts",
        type=float,
        nargs="+",
        default=DEFAULT_SHIFTS,
        help=(
            "Translation severity d in pixels. d bounds total Euclidean shift: "
            "r~U(0,d), theta~U(0,2pi), |shift|<=d."
        ),
    )
    parser.add_argument(
        "--misalignment_trials",
        type=int,
        default=5,
        help="Random paired translation trials for each non-zero d.",
    )
    parser.add_argument(
        "--misalignment_fixed_margin",
        type=int,
        default=8,
        help="Pixels cropped from every border for the common fixed ROI.",
    )
    parser.add_argument(
        "--misalignment_seed",
        type=int,
        default=None,
        help="Seed for paired radial translation. Defaults to normal --seed.",
    )
    parser.add_argument(
        "--misalignment_valid_threshold",
        type=float,
        default=0.999,
        help="Warped all-one mask threshold defining fully valid MSI support.",
    )
    parser.add_argument(
        "--misalignment_output",
        type=str,
        default="",
        help="Optional summary CSV path; *_details.csv is written beside it.",
    )
    diagnostic, remaining = parser.parse_known_args()
    cfg = parse_args(remaining)

    if cfg.stage != "test":
        raise ValueError("Translation diagnosis is test-only; use --stage test")
    if str(cfg.predictor_version).lower() != "v3":
        raise ValueError("Translation diagnosis expects --predictor_version v3")
    if str(cfg.msi_ablation).lower() != "raw_direct":
        raise ValueError("Translation diagnosis currently expects --msi_ablation raw_direct")
    if diagnostic.misalignment_trials < 1:
        raise ValueError("--misalignment_trials must be >= 1")
    if diagnostic.misalignment_fixed_margin < 0:
        raise ValueError("--misalignment_fixed_margin must be >= 0")
    if not 0.0 < diagnostic.misalignment_valid_threshold <= 1.0:
        raise ValueError("--misalignment_valid_threshold must lie in (0,1]")

    shifts = [float(v) for v in diagnostic.misalignment_shifts]
    if not shifts or any(v < 0.0 for v in shifts):
        raise ValueError("Translation severities must be a non-empty list of values >= 0")
    diagnostic.misalignment_shifts = shifts
    return cfg, diagnostic


def calc_masked_psnr_sam(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    threshold: float,
    eps: float = 1e-8,
) -> Tuple[float, float, float]:
    if pred.shape[0] != 1 or target.shape[0] != 1 or valid_mask.shape[0] != 1:
        raise ValueError("Masked metric helper expects batch size 1")

    mask = valid_mask[0, 0] >= float(threshold)
    valid_count = int(mask.sum().item())
    total_count = int(mask.numel())
    if valid_count < 1:
        raise ValueError("No valid pixels remain after MSI translation")

    pred_clamped = torch.clamp(pred.detach().float(), 0.0, 1.0)[0, :, mask]
    target_clamped = torch.clamp(target.detach().float(), 0.0, 1.0)[0, :, mask]
    mse = torch.mean((pred_clamped - target_clamped) ** 2).item()
    psnr = 100.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / max(mse, 1e-12))

    pred_spec = pred.detach().float()[0, :, mask]
    target_spec = target.detach().float()[0, :, mask]
    dot = torch.sum(pred_spec * target_spec, dim=0)
    pred_norm = torch.sqrt(torch.sum(pred_spec * pred_spec, dim=0) + eps)
    target_norm = torch.sqrt(torch.sum(target_spec * target_spec, dim=0) + eps)
    cos = torch.clamp(dot / (pred_norm * target_norm + eps), -1.0 + eps, 1.0 - eps)
    sam = torch.mean(torch.acos(cos) * 180.0 / math.pi).item()
    return float(psnr), float(sam), float(valid_count / total_count)


def fixed_roi(x: torch.Tensor, margin: int) -> torch.Tensor:
    if margin == 0:
        return x
    h, w = x.shape[-2:]
    if 2 * margin >= h or 2 * margin >= w:
        raise ValueError(f"Fixed ROI margin={margin} is too large for {(h, w)}")
    return x[..., margin : h - margin, margin : w - margin]


def _metric_prefix(metrics: Dict[str, float], prefix: str) -> Dict[str, float]:
    return {f"{key}_{prefix}": float(value) for key, value in metrics.items()}


def _mean_rows(rows: Iterable[Dict[str, object]], keys: Iterable[str]) -> Dict[str, float]:
    rows = list(rows)
    return {key: float(np.mean([float(row[key]) for row in rows])) for key in keys}


def _write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _format_row(row: Dict[str, object]) -> str:
    return (
        f"d={float(row['max_shift_px']):>4.1f}px "
        f"actual|shift|={float(row['mean_shift_magnitude_px']):.3f}px "
        f"PSNR_valid={float(row['PSNR_valid']):.4f} "
        f"SAM_valid={float(row['SAM_valid']):.4f} "
        f"PSNR_fixed={float(row['PSNR_fixed']):.4f} "
        f"SAM_fixed={float(row['SAM_fixed']):.4f} "
        f"dPSNR_fixed={float(row['delta_PSNR_fixed']):+.4f}"
    )


@torch.no_grad()
def run_translation_diagnosis(cfg, diagnostic):
    set_seed(cfg.seed)
    _, test_loader, info = build_loaders(cfg)
    device = get_device(cfg.device)
    process = build_progressive_process(cfg)
    model = _build_model(cfg, info, device)

    checkpoint = cfg.resume
    if not checkpoint:
        raise ValueError("Pass the checkpoint explicitly with --resume")
    loaded_epoch, loaded_best = load_checkpoint(
        model,
        checkpoint,
        optimizer=None,
        map_location=str(device),
        load_optimizer=False,
    )
    print(
        f"Loaded checkpoint {checkpoint}: epoch={loaded_epoch}, "
        f"stored_best_PSNR={loaded_best:.6f}"
    )
    print(
        "Radial translation diagnosis: only HR-MSI is warped; "
        "GT/LR-HSI/degradation/reverse stay fixed."
    )

    shift_seed = cfg.seed if diagnostic.misalignment_seed is None else int(diagnostic.misalignment_seed)
    cached_batches = list(test_loader)
    if not cached_batches:
        raise ValueError("Test loader is empty")

    detail_rows: List[Dict[str, object]] = []
    for max_shift in diagnostic.misalignment_shifts:
        trials = 1 if abs(max_shift) < 1e-12 else diagnostic.misalignment_trials
        for trial in range(trials):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(shift_seed + trial * 100003)
            sample_index = 0

            for batch in cached_batches:
                gt = batch["gt"].to(device, non_blocking=True)
                hr_msi = batch["hr_msi"].to(device, non_blocking=True)
                shifted_msi, valid_soft, params = make_misaligned_msi(
                    hr_msi,
                    translation_max_px=float(max_shift),
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

                batch_size = int(gt.shape[0])
                for i in range(batch_size):
                    pred_i = pred[i : i + 1]
                    gt_i = gt[i : i + 1]
                    valid_i = valid_soft[i : i + 1]

                    full_metrics = calc_metrics(pred_i, gt_i, cfg.scale_ratio)
                    valid_psnr, valid_sam, valid_fraction = calc_masked_psnr_sam(
                        pred_i,
                        gt_i,
                        valid_i,
                        threshold=diagnostic.misalignment_valid_threshold,
                    )
                    fixed_metrics = calc_metrics(
                        fixed_roi(pred_i, diagnostic.misalignment_fixed_margin),
                        fixed_roi(gt_i, diagnostic.misalignment_fixed_margin),
                        cfg.scale_ratio,
                    )

                    dx_i = float(params.dx_px[i].item())
                    dy_i = float(params.dy_px[i].item())
                    detail_rows.append(
                        {
                            "max_shift_px": float(max_shift),
                            "trial": int(trial),
                            "sample": int(sample_index + i),
                            "dx_px": dx_i,
                            "dy_px": dy_i,
                            "shift_magnitude_px": float(math.hypot(dx_i, dy_i)),
                            "valid_fraction": valid_fraction,
                            **_metric_prefix(full_metrics, "full"),
                            "PSNR_valid": valid_psnr,
                            "SAM_valid": valid_sam,
                            **_metric_prefix(fixed_metrics, "fixed"),
                        }
                    )
                sample_index += batch_size

    groups: Dict[float, List[Dict[str, object]]] = defaultdict(list)
    for row in detail_rows:
        groups[float(row["max_shift_px"])].append(row)

    metric_keys = [
        "valid_fraction",
        "PSNR_full", "SAM_full", "RMSE_full", "ERGAS_full", "SSIM_full", "CC_full",
        "PSNR_valid", "SAM_valid",
        "PSNR_fixed", "SAM_fixed", "RMSE_fixed", "ERGAS_fixed", "SSIM_fixed", "CC_fixed",
    ]

    summary_rows: List[Dict[str, object]] = []
    for max_shift in diagnostic.misalignment_shifts:
        group = groups[float(max_shift)]
        mean_metrics = _mean_rows(group, metric_keys)
        summary_rows.append(
            {
                "max_shift_px": float(max_shift),
                "n_runs": len(group),
                "mean_abs_dx_px": float(np.mean([abs(float(r["dx_px"])) for r in group])),
                "mean_abs_dy_px": float(np.mean([abs(float(r["dy_px"])) for r in group])),
                "mean_shift_magnitude_px": float(
                    np.mean([float(r["shift_magnitude_px"]) for r in group])
                ),
                **mean_metrics,
            }
        )

    baseline = min(summary_rows, key=lambda row: abs(float(row["max_shift_px"])))
    base_psnr = float(baseline["PSNR_fixed"])
    base_sam = float(baseline["SAM_fixed"])
    for row in summary_rows:
        row["delta_PSNR_fixed"] = float(row["PSNR_fixed"]) - base_psnr
        row["delta_SAM_fixed"] = float(row["SAM_fixed"]) - base_sam

    return summary_rows, detail_rows


def main():
    cfg, diagnostic = parse_diagnostic_args()
    summary_rows, detail_rows = run_translation_diagnosis(cfg, diagnostic)

    if diagnostic.misalignment_output:
        summary_path = diagnostic.misalignment_output
    else:
        summary_path = os.path.join(
            cfg.output_root,
            "metrics",
            f"{cfg.dataset}_translation_radial_sensitivity.csv",
        )
    stem, ext = os.path.splitext(summary_path)
    detail_path = f"{stem}_details{ext or '.csv'}"

    _write_csv(summary_path, summary_rows)
    _write_csv(detail_path, detail_rows)

    print("\nRadial translation sensitivity summary")
    for row in summary_rows:
        print(_format_row(row))
    print(f"Summary CSV: {summary_path}")
    print(f"Details CSV: {detail_path}")


if __name__ == "__main__":
    main()
