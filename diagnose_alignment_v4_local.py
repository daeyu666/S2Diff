"""Local non-rigid diagnosis for learned Innovation-2 V4 alignment.

This diagnostic isolates whether the sparse 4->2->1 progressive local aligner
actually handles smooth spatially varying HSI-MSI misregistration.

Only HR-MSI is perturbed. GT-HSI, LR-HSI and the Innovation-1 physical
trajectory remain unchanged. Four scenarios are evaluated:

1. registered: no geometric perturbation;
2. global_only: fixed global translation+rotation, no local warp;
3. local_only: smooth local non-rigid warp, no global rigid perturbation;
4. global_local: the same global rigid perturbation plus the same local control
   pattern as local_only for the corresponding trial/severity.

By default, global_only/global_local use the current trained range d<=4 px and
|rotation|<=2 deg. Local severity sweeps max displacement 0.5/1/2 px using a
5x5 control grid, bicubic interpolation and the shared bounded smooth local
warp from degradations.misalignment.

Valid-overlap PSNR/SAM are primary. The script also logs the true local field,
V4-predicted global correction and the final scale-1 local offset field so we
can determine whether local non-rigid errors are being corrected by the local
branch rather than leaking into the global branch.
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
from diagnose_misalignment_translation import calc_masked_psnr_sam, fixed_roi
from innovation1 import build_progressive_process, reconstruct_from_terminal_lr
from main import _build_model
from metrics import calc_metrics
from utils import get_device, load_checkpoint, set_seed


VALID_SCENARIOS = ("registered", "global_only", "local_only", "global_local")
DEFAULT_LOCAL_MAXIMA = [0.5, 1.0, 2.0]


def parse_diagnostic_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--local_scenarios",
        nargs="+",
        default=list(VALID_SCENARIOS),
        choices=VALID_SCENARIOS,
    )
    parser.add_argument(
        "--local_max_displacements",
        type=float,
        nargs="+",
        default=DEFAULT_LOCAL_MAXIMA,
        help="Maximum smooth local displacement values in HR pixels.",
    )
    parser.add_argument(
        "--diagnostic_global_translation_max_px",
        type=float,
        default=4.0,
        help="Global translation radius used by global_only/global_local.",
    )
    parser.add_argument(
        "--diagnostic_global_rotation_max_deg",
        type=float,
        default=2.0,
        help="Global rotation bound used by global_only/global_local.",
    )
    parser.add_argument("--local_control_grid", type=int, default=5)
    parser.add_argument("--misalignment_trials", type=int, default=5)
    parser.add_argument("--misalignment_seed", type=int, default=None)
    parser.add_argument("--misalignment_fixed_margin", type=int, default=8)
    parser.add_argument("--misalignment_valid_threshold", type=float, default=0.999)
    parser.add_argument("--misalignment_output", type=str, default="")
    diagnostic, remaining = parser.parse_known_args()
    cfg = parse_args(remaining)

    if cfg.stage != "test":
        raise ValueError("V4 local diagnosis is test-only; use --stage test")
    if str(cfg.predictor_version).lower() != "v4":
        raise ValueError("V4 local diagnosis expects --predictor_version v4")
    if str(cfg.msi_ablation).lower() != "raw_direct":
        raise ValueError("V4 uses --msi_ablation raw_direct")
    if diagnostic.misalignment_trials < 1:
        raise ValueError("--misalignment_trials must be >= 1")
    if diagnostic.local_control_grid < 2:
        raise ValueError("--local_control_grid must be >= 2")
    if diagnostic.misalignment_fixed_margin < 0:
        raise ValueError("--misalignment_fixed_margin must be >= 0")
    if not 0.0 < diagnostic.misalignment_valid_threshold <= 1.0:
        raise ValueError("--misalignment_valid_threshold must lie in (0,1]")
    if diagnostic.diagnostic_global_translation_max_px < 0.0:
        raise ValueError("diagnostic global translation max must be >= 0")
    if diagnostic.diagnostic_global_rotation_max_deg < 0.0:
        raise ValueError("diagnostic global rotation max must be >= 0")
    if any(float(v) <= 0.0 for v in diagnostic.local_max_displacements):
        raise ValueError("all --local_max_displacements must be > 0")

    diagnostic.local_max_displacements = [
        float(v) for v in diagnostic.local_max_displacements
    ]
    return cfg, diagnostic


def resolve_geometry(
    scenario: str,
    local_max_px: float,
    global_translation_max_px: float,
    global_rotation_max_deg: float,
) -> Tuple[float, float, float]:
    """Return translation max, rotation max and local max for one scenario."""
    if scenario == "registered":
        return 0.0, 0.0, 0.0
    if scenario == "global_only":
        return global_translation_max_px, global_rotation_max_deg, 0.0
    if scenario == "local_only":
        return 0.0, 0.0, local_max_px
    if scenario == "global_local":
        return global_translation_max_px, global_rotation_max_deg, local_max_px
    raise ValueError(f"unsupported local diagnostic scenario: {scenario}")


def _write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows: Iterable[Dict[str, object]], key: str) -> float:
    rows = list(rows)
    return float(np.mean([float(row[key]) for row in rows]))


def _vector_field_stats(field: torch.Tensor) -> Tuple[float, float, float]:
    """Mean magnitude, max magnitude and RMS magnitude of Bx2xHxW field."""
    magnitude = torch.linalg.vector_norm(field.detach().float(), dim=1)
    mean = float(magnitude.mean().item())
    max_value = float(magnitude.amax().item())
    rms = float(torch.sqrt(torch.mean(magnitude.square())).item())
    return mean, max_value, rms


def _scenario_specs(diagnostic) -> List[Tuple[str, float]]:
    specs: List[Tuple[str, float]] = []
    if "registered" in diagnostic.local_scenarios:
        specs.append(("registered", 0.0))
    if "global_only" in diagnostic.local_scenarios:
        specs.append(("global_only", 0.0))
    for local_max in diagnostic.local_max_displacements:
        if "local_only" in diagnostic.local_scenarios:
            specs.append(("local_only", float(local_max)))
        if "global_local" in diagnostic.local_scenarios:
            specs.append(("global_local", float(local_max)))
    return specs


@torch.no_grad()
def run_local_diagnosis(cfg, diagnostic):
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
    print(
        "V4 local diagnosis: GT/LR-HSI stay fixed; only HR-MSI is warped. "
        f"global range=({diagnostic.diagnostic_global_translation_max_px:g}px, "
        f"{diagnostic.diagnostic_global_rotation_max_deg:g}deg), "
        f"local grid={diagnostic.local_control_grid}x{diagnostic.local_control_grid}."
    )

    batches = list(test_loader)
    if not batches:
        raise ValueError("Test loader is empty")

    specs = _scenario_specs(diagnostic)
    base_seed = cfg.seed if diagnostic.misalignment_seed is None else int(
        diagnostic.misalignment_seed
    )
    detail_rows: List[Dict[str, object]] = []

    for scenario, local_max in specs:
        translation_max, rotation_max, resolved_local_max = resolve_geometry(
            scenario,
            local_max,
            float(diagnostic.diagnostic_global_translation_max_px),
            float(diagnostic.diagnostic_global_rotation_max_deg),
        )
        registered = scenario == "registered"
        trials = 1 if registered else int(diagnostic.misalignment_trials)

        print(
            f"\n[{scenario}] global_d<={translation_max:g}px, "
            f"global_r<={rotation_max:g}deg, local<={resolved_local_max:g}px, "
            f"trials={trials}"
        )

        for trial in range(trials):
            # The same trial seed is reused for local_only/global_local and for
            # every local severity. make_misaligned_msi always consumes the same
            # normalized global random variables before sampling the local field,
            # so the smooth local control pattern is paired across comparisons.
            generator = torch.Generator(device="cpu")
            generator.manual_seed(base_seed + trial * 100003)

            sample_index = 0
            for batch in batches:
                gt = batch["gt"].to(device, non_blocking=True)
                hr_msi = batch["hr_msi"].to(device, non_blocking=True)
                warped_msi, valid_soft, params = make_misaligned_msi(
                    hr_msi,
                    translation_max_px=translation_max,
                    rotation_max_deg=rotation_max,
                    local_max_displacement_px=resolved_local_max,
                    control_grid_size=int(diagnostic.local_control_grid),
                    generator=generator,
                )

                terminal_lr = process.terminal_observation(gt)
                pred = reconstruct_from_terminal_lr(
                    model,
                    process,
                    terminal_lr,
                    target_size=tuple(gt.shape[-2:]),
                    hr_msi=warped_msi,
                )

                batch_size = int(gt.shape[0])
                for i in range(batch_size):
                    pred_i = pred[i : i + 1]
                    gt_i = gt[i : i + 1]
                    valid_i = valid_soft[i : i + 1]

                    full = calc_metrics(pred_i, gt_i, cfg.scale_ratio)
                    valid_psnr, valid_sam, valid_fraction = calc_masked_psnr_sam(
                        pred_i,
                        gt_i,
                        valid_i,
                        threshold=float(diagnostic.misalignment_valid_threshold),
                    )
                    fixed = calc_metrics(
                        fixed_roi(pred_i, int(diagnostic.misalignment_fixed_margin)),
                        fixed_roi(gt_i, int(diagnostic.misalignment_fixed_margin)),
                        cfg.scale_ratio,
                    )

                    dx = float(params.dx_px[i].item())
                    dy = float(params.dy_px[i].item())
                    true_rot = float(params.rotation_deg[i].item())
                    true_local_mean, true_local_max, true_local_rms = _vector_field_stats(
                        params.local_displacement_px[i : i + 1]
                    )

                    global_shift = model.last_global_shift_px
                    if global_shift is None:
                        gx = gy = float("nan")
                    else:
                        gx = float(global_shift[i, 0].item())
                        gy = float(global_shift[i, 1].item())

                    global_rotation = model.last_global_rotation_deg
                    pred_rot = (
                        float("nan")
                        if global_rotation is None
                        else float(global_rotation[i].item())
                    )

                    pred_local_scale = -1
                    pred_local_mean = float("nan")
                    pred_local_max = float("nan")
                    pred_local_rms = float("nan")
                    if model.last_alignment is not None:
                        pred_local_scale = int(model.last_alignment.local_scale)
                        (
                            pred_local_mean,
                            pred_local_max,
                            pred_local_rms,
                        ) = _vector_field_stats(
                            model.last_alignment.local_offset_px[i : i + 1]
                        )

                    detail_rows.append(
                        {
                            "scenario": scenario,
                            "local_max_displacement_px": resolved_local_max,
                            "trial": trial,
                            "sample": sample_index + i,
                            "global_translation_max_px": translation_max,
                            "global_rotation_max_deg": rotation_max,
                            "true_dx_px": dx,
                            "true_dy_px": dy,
                            "true_translation_magnitude_px": math.hypot(dx, dy),
                            "true_rotation_deg": true_rot,
                            "abs_true_rotation_deg": abs(true_rot),
                            "true_local_mean_px": true_local_mean,
                            "true_local_rms_px": true_local_rms,
                            "true_local_max_actual_px": true_local_max,
                            "global_correction_dx_px": gx,
                            "global_correction_dy_px": gy,
                            "global_correction_magnitude_px": math.hypot(gx, gy),
                            "global_correction_rotation_deg": pred_rot,
                            "global_rotation_inverse_error_deg": abs(true_rot + pred_rot),
                            "final_local_scale": pred_local_scale,
                            "pred_local_mean_px": pred_local_mean,
                            "pred_local_rms_px": pred_local_rms,
                            "pred_local_max_px": pred_local_max,
                            "valid_fraction": valid_fraction,
                            "PSNR_valid": valid_psnr,
                            "SAM_valid": valid_sam,
                            "PSNR_full": float(full["PSNR"]),
                            "SAM_full": float(full["SAM"]),
                            "PSNR_fixed": float(fixed["PSNR"]),
                            "SAM_fixed": float(fixed["SAM"]),
                        }
                    )
                sample_index += batch_size

    grouped = defaultdict(list)
    for row in detail_rows:
        key = (str(row["scenario"]), float(row["local_max_displacement_px"]))
        grouped[key].append(row)

    registered_rows = grouped.get(("registered", 0.0), [])
    ref_psnr = _mean(registered_rows, "PSNR_valid") if registered_rows else float("nan")
    ref_sam = _mean(registered_rows, "SAM_valid") if registered_rows else float("nan")

    global_rows = grouped.get(("global_only", 0.0), [])
    global_psnr = _mean(global_rows, "PSNR_valid") if global_rows else float("nan")
    global_sam = _mean(global_rows, "SAM_valid") if global_rows else float("nan")

    summary_rows: List[Dict[str, object]] = []
    for scenario, local_max in specs:
        group = grouped[(scenario, float(local_max))]
        psnr_valid = _mean(group, "PSNR_valid")
        sam_valid = _mean(group, "SAM_valid")

        if scenario == "global_local" and math.isfinite(global_psnr):
            extra_local_drop_psnr = psnr_valid - global_psnr
            extra_local_delta_sam = sam_valid - global_sam
        elif scenario == "local_only" and math.isfinite(ref_psnr):
            extra_local_drop_psnr = psnr_valid - ref_psnr
            extra_local_delta_sam = sam_valid - ref_sam
        else:
            extra_local_drop_psnr = float("nan")
            extra_local_delta_sam = float("nan")

        summary_rows.append(
            {
                "scenario": scenario,
                "local_max_displacement_px": float(local_max),
                "n_runs": len(group),
                "global_translation_max_px": float(group[0]["global_translation_max_px"]),
                "global_rotation_max_deg": float(group[0]["global_rotation_max_deg"]),
                "mean_true_translation_px": _mean(group, "true_translation_magnitude_px"),
                "mean_abs_true_rotation_deg": _mean(group, "abs_true_rotation_deg"),
                "mean_true_local_px": _mean(group, "true_local_mean_px"),
                "mean_true_local_rms_px": _mean(group, "true_local_rms_px"),
                "mean_true_local_max_actual_px": _mean(group, "true_local_max_actual_px"),
                "mean_global_correction_px": _mean(group, "global_correction_magnitude_px"),
                "mean_abs_global_correction_rotation_deg": float(
                    np.mean([abs(float(r["global_correction_rotation_deg"])) for r in group])
                ),
                "mean_global_rotation_inverse_error_deg": _mean(
                    group, "global_rotation_inverse_error_deg"
                ),
                "mean_pred_local_px": _mean(group, "pred_local_mean_px"),
                "mean_pred_local_rms_px": _mean(group, "pred_local_rms_px"),
                "mean_pred_local_max_px": _mean(group, "pred_local_max_px"),
                "valid_fraction": _mean(group, "valid_fraction"),
                "PSNR_valid": psnr_valid,
                "SAM_valid": sam_valid,
                "delta_PSNR_vs_registered": (
                    psnr_valid - ref_psnr if math.isfinite(ref_psnr) else float("nan")
                ),
                "delta_SAM_vs_registered": (
                    sam_valid - ref_sam if math.isfinite(ref_sam) else float("nan")
                ),
                "extra_local_delta_PSNR": extra_local_drop_psnr,
                "extra_local_delta_SAM": extra_local_delta_sam,
                "PSNR_full": _mean(group, "PSNR_full"),
                "SAM_full": _mean(group, "SAM_full"),
                "PSNR_fixed": _mean(group, "PSNR_fixed"),
                "SAM_fixed": _mean(group, "SAM_fixed"),
            }
        )

    if diagnostic.misalignment_output:
        summary_path = diagnostic.misalignment_output
    else:
        summary_path = os.path.join(
            cfg.output_root,
            "metrics",
            f"{cfg.dataset}_v4_local_nonrigid.csv",
        )
    stem, ext = os.path.splitext(summary_path)
    detail_path = f"{stem}_details{ext or '.csv'}"
    _write_csv(summary_path, summary_rows)
    _write_csv(detail_path, detail_rows)

    print("\n=== V4 local non-rigid diagnosis (valid-overlap primary) ===")
    for row in summary_rows:
        local_text = f"local<={row['local_max_displacement_px']:.2f}px"
        print(
            f"{row['scenario']:>12s} {local_text:>14s} | "
            f"true_local={row['mean_true_local_px']:.3f}px "
            f"PSNR={row['PSNR_valid']:.4f} "
            f"SAM={row['SAM_valid']:.4f} "
            f"extra_local_dPSNR={row['extra_local_delta_PSNR']:+.4f} "
            f"pred_local={row['mean_pred_local_px']:.3f}px "
            f"global_corr={row['mean_global_correction_px']:.3f}px "
            f"global_rot_err={row['mean_global_rotation_inverse_error_deg']:.3f}deg"
        )

    print(f"Summary CSV: {summary_path}")
    print(f"Details CSV: {detail_path}")
    return summary_rows, detail_rows


def main():
    cfg, diagnostic = parse_diagnostic_args()
    run_local_diagnosis(cfg, diagnostic)


if __name__ == "__main__":
    main()
