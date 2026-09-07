"""Translation-only HSI-MSI misalignment sensitivity diagnosis.

This script intentionally leaves the trained predictor, LR-HSI observation,
physical progressive degradation and reverse recursion unchanged. It perturbs
only the HR-MSI condition at test time and evaluates how a registered
Raw-Direct checkpoint degrades under sub-pixel / pixel translation.

For each maximum shift d, offsets follow the current experiment protocol:
    dx, dy ~ U(-d, d)
Multiple paired trials are supported because the current PaviaU test loader
contains a single center test patch. The same normalized random directions are
reused across d values, so the sensitivity curve is paired across strengths.

Reported metrics:
- full: original full-frame HSI reconstruction metrics;
- valid: PSNR/SAM only on pixels whose warped MSI has fully valid support;
- fixed: metrics on one identical central ROI for every shift strength.
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
import torch.nn.functional as F

from config import parse_args
from data_loader import build_loaders
from innovation1 import build_progressive_process, reconstruct_from_terminal_lr
from main import _build_model
from metrics import calc_metrics
from utils import get_device, load_checkpoint, set_seed


DEFAULT_SHIFTS = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0]


def parse_diagnostic_args():
    """Parse diagnostic-only flags first, then reuse the normal S2Diff config."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--misalignment_shifts",
        type=float,
        nargs="+",
        default=DEFAULT_SHIFTS,
        help="Maximum absolute translation d in pixels; dx,dy are sampled from U(-d,d).",
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
        help="Seed for translation offsets. Defaults to the normal --seed.",
    )
    parser.add_argument(
        "--misalignment_valid_threshold",
        type=float,
        default=0.999,
        help="Warped all-one mask threshold used to define fully valid MSI support.",
    )
    parser.add_argument(
        "--misalignment_output",
        type=str,
        default="",
        help="Optional summary CSV path. A *_details.csv file is written beside it.",
    )
    diagnostic, remaining = parser.parse_known_args()
    cfg = parse_args(remaining)

    if cfg.stage != "test":
        raise ValueError("Translation diagnosis is test-only; use --stage test")
    if str(cfg.predictor_version).lower() != "v3":
        raise ValueError("Translation diagnosis currently expects --predictor_version v3")
    if str(cfg.msi_ablation).lower() != "raw_direct":
        raise ValueError(
            "This first misalignment diagnosis is defined on the frozen Raw-Direct baseline; "
            "use --msi_ablation raw_direct"
        )
    if diagnostic.misalignment_trials < 1:
        raise ValueError("--misalignment_trials must be >= 1")
    if diagnostic.misalignment_fixed_margin < 0:
        raise ValueError("--misalignment_fixed_margin must be >= 0")
    if not 0.0 < diagnostic.misalignment_valid_threshold <= 1.0:
        raise ValueError("--misalignment_valid_threshold must lie in (0,1]")

    shifts = [float(v) for v in diagnostic.misalignment_shifts]
    if not shifts:
        raise ValueError("At least one --misalignment_shifts value is required")
    if any(v < 0.0 for v in shifts):
        raise ValueError("Translation magnitudes must be >= 0")
    diagnostic.misalignment_shifts = shifts
    return cfg, diagnostic


def translate_msi(
    hr_msi: torch.Tensor,
    dx: torch.Tensor,
    dy: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Translate MSI by (dx,dy) pixels and return the warped validity mask.

    Positive dx moves image content to the right; positive dy moves content
    downward. Bilinear sampling enables sub-pixel shifts. Zero padding is used
    only outside the source field of view, and those locations are excluded by
    the validity mask in valid-overlap evaluation.
    """
    if hr_msi.ndim != 4:
        raise ValueError(f"hr_msi must be BxCxHxW, got {tuple(hr_msi.shape)}")
    batch, _, height, width = hr_msi.shape
    if dx.ndim != 1 or dy.ndim != 1 or dx.shape[0] != batch or dy.shape[0] != batch:
        raise ValueError("dx and dy must both have shape [B]")

    theta = torch.zeros(batch, 2, 3, dtype=hr_msi.dtype, device=hr_msi.device)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    # affine_grid maps output coordinates to source coordinates. Subtract the
    # desired image displacement so positive dx/dy move content right/down.
    theta[:, 0, 2] = -2.0 * dx.to(hr_msi.dtype) / float(width)
    theta[:, 1, 2] = -2.0 * dy.to(hr_msi.dtype) / float(height)

    grid = F.affine_grid(theta, size=hr_msi.shape, align_corners=False)
    shifted = F.grid_sample(
        hr_msi,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )

    ones = torch.ones(
        batch, 1, height, width, dtype=hr_msi.dtype, device=hr_msi.device
    )
    valid_soft = F.grid_sample(
        ones,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return shifted, valid_soft


def calc_masked_psnr_sam(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    threshold: float,
    eps: float = 1e-8,
) -> Tuple[float, float, float]:
    """Return PSNR, SAM(deg), and valid fraction for one sample."""
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
    cos = dot / (pred_norm * target_norm + eps)
    cos = torch.clamp(cos, -1.0 + eps, 1.0 - eps)
    sam = torch.mean(torch.acos(cos) * 180.0 / math.pi).item()

    return float(psnr), float(sam), float(valid_count / total_count)


def fixed_roi(x: torch.Tensor, margin: int) -> torch.Tensor:
    if margin == 0:
        return x
    height, width = x.shape[-2:]
    if 2 * margin >= height or 2 * margin >= width:
        raise ValueError(
            f"Fixed ROI margin={margin} is too large for spatial size {(height, width)}"
        )
    return x[..., margin : height - margin, margin : width - margin]


def _metric_prefix(metrics: Dict[str, float], prefix: str) -> Dict[str, float]:
    return {f"{key}_{prefix}": float(value) for key, value in metrics.items()}


def _mean_rows(rows: Iterable[Dict[str, float]], keys: Iterable[str]) -> Dict[str, float]:
    rows = list(rows)
    out = {}
    for key in keys:
        values = [float(row[key]) for row in rows]
        out[key] = float(np.mean(values))
    return out


def _write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _format_row(row: Dict[str, float]) -> str:
    return (
        f"d={row['max_shift_px']:>4.1f}px "
        f"actual|shift|={row['mean_shift_magnitude_px']:.3f}px "
        f"PSNR_full={row['PSNR_full']:.4f} SAM_full={row['SAM_full']:.4f} "
        f"PSNR_valid={row['PSNR_valid']:.4f} SAM_valid={row['SAM_valid']:.4f} "
        f"PSNR_fixed={row['PSNR_fixed']:.4f} SAM_fixed={row['SAM_fixed']:.4f} "
        f"dPSNR_fixed={row['delta_PSNR_fixed']:+.4f} "
        f"dSAM_fixed={row['delta_SAM_fixed']:+.4f}"
    )


@torch.no_grad()
def run_translation_diagnosis(cfg, diagnostic) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    set_seed(cfg.seed)
    _, test_loader, info = build_loaders(cfg)
    device = get_device(cfg.device)
    process = build_progressive_process(cfg)
    model = _build_model(cfg, info, device)

    checkpoint = cfg.resume
    if not checkpoint:
        raise ValueError(
            "Please pass the trained Raw-Direct checkpoint explicitly with --resume"
        )
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
        "Translation diagnosis keeps GT/LR-HSI/degradation/reverse fixed and perturbs "
        "only HR-MSI at inference."
    )

    shift_seed = cfg.seed if diagnostic.misalignment_seed is None else int(diagnostic.misalignment_seed)
    detail_rows: List[Dict[str, object]] = []

    # Preserve the exact same test samples across every strength/trial.
    cached_batches = list(test_loader)
    if not cached_batches:
        raise ValueError("Test loader is empty")
    print(
        f"Test samples={sum(int(batch['gt'].shape[0]) for batch in cached_batches)}, "
        f"trials/nonzero-d={diagnostic.misalignment_trials}, "
        f"fixed_margin={diagnostic.misalignment_fixed_margin}px, seed={shift_seed}"
    )

    for max_shift in diagnostic.misalignment_shifts:
        trials = 1 if abs(max_shift) < 1e-12 else diagnostic.misalignment_trials

        for trial in range(trials):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(shift_seed + trial * 100003)
            sample_index = 0

            for batch in cached_batches:
                gt = batch["gt"].to(device, non_blocking=True)
                hr_msi = batch["hr_msi"].to(device, non_blocking=True)
                batch_size = int(gt.shape[0])

                # Draw normalized offsets first. Reusing the same trial seed for
                # every d gives paired directions; d only scales the amplitude.
                unit = torch.rand(batch_size, 2, generator=generator) * 2.0 - 1.0
                dx = (unit[:, 0] * float(max_shift)).to(device=device, dtype=hr_msi.dtype)
                dy = (unit[:, 1] * float(max_shift)).to(device=device, dtype=hr_msi.dtype)

                shifted_msi, valid_soft = translate_msi(hr_msi, dx, dy)
                terminal_lr = process.terminal_observation(gt)
                pred = reconstruct_from_terminal_lr(
                    model,
                    process,
                    terminal_lr,
                    target_size=tuple(gt.shape[-2:]),
                    hr_msi=shifted_msi,
                )

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
                    pred_fixed = fixed_roi(pred_i, diagnostic.misalignment_fixed_margin)
                    gt_fixed = fixed_roi(gt_i, diagnostic.misalignment_fixed_margin)
                    fixed_metrics = calc_metrics(pred_fixed, gt_fixed, cfg.scale_ratio)

                    dx_i = float(dx[i].item())
                    dy_i = float(dy[i].item())
                    row: Dict[str, object] = {
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
                    detail_rows.append(row)

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

    baseline_candidates = [r for r in summary_rows if abs(float(r["max_shift_px"])) < 1e-12]
    if not baseline_candidates:
        raise ValueError(
            "Include 0 in --misalignment_shifts so delta metrics have a registered baseline"
        )
    baseline = baseline_candidates[0]
    for row in summary_rows:
        row["delta_PSNR_full"] = float(row["PSNR_full"] - baseline["PSNR_full"])
        row["delta_SAM_full"] = float(row["SAM_full"] - baseline["SAM_full"])
        row["delta_PSNR_valid"] = float(row["PSNR_valid"] - baseline["PSNR_valid"])
        row["delta_SAM_valid"] = float(row["SAM_valid"] - baseline["SAM_valid"])
        row["delta_PSNR_fixed"] = float(row["PSNR_fixed"] - baseline["PSNR_fixed"])
        row["delta_SAM_fixed"] = float(row["SAM_fixed"] - baseline["SAM_fixed"])

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
            f"{cfg.dataset}_misalignment_translation_raw_direct.csv",
        )
    stem, ext = os.path.splitext(summary_path)
    detail_path = f"{stem}_details{ext or '.csv'}"

    _write_csv(summary_path, summary_rows)
    _write_csv(detail_path, detail_rows)

    print("\nTranslation sensitivity summary")
    print("-" * 132)
    for row in summary_rows:
        print(_format_row(row))
    print("-" * 132)
    print(f"Summary CSV: {summary_path}")
    print(f"Details CSV: {detail_path}")


if __name__ == "__main__":
    main()
