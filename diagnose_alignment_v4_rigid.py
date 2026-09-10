"""Joint translation + rotation diagnosis for learned Innovation-2 V4 alignment.

Only HR-MSI is perturbed. GT-HSI, LR-HSI, the Innovation-1 physical trajectory
and reverse recursion remain unchanged. Each diagnostic level is a paired
(translation_max_px, rotation_max_deg) severity. The same trial seed is reused
across levels so the normalized translation radius/direction and rotation sign
are paired across the severity curve.

Default paired levels:
    (0 px, 0 deg)
    (1 px, 0.5 deg)
    (2 px, 1 deg)
    (4 px, 2 deg)
    (6 px, 3 deg)

Valid-overlap PSNR/SAM are the primary reconstruction metrics. The script also
records the true synthetic rigid perturbation and the V4-predicted global rigid
correction so the translation and rotation branches can be diagnosed directly.
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


DEFAULT_TRANSLATION_MAX = [0.0, 1.0, 2.0, 4.0, 6.0]
DEFAULT_ROTATION_MAX = [0.0, 0.5, 1.0, 2.0, 3.0]


def parse_diagnostic_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--rigid_translation_max_px",
        type=float,
        nargs="+",
        default=DEFAULT_TRANSLATION_MAX,
        help=(
            "Paired maximum Euclidean translation radii in pixels. Must have "
            "the same length as --rigid_rotation_max_deg."
        ),
    )
    parser.add_argument(
        "--rigid_rotation_max_deg",
        type=float,
        nargs="+",
        default=DEFAULT_ROTATION_MAX,
        help=(
            "Paired maximum absolute rotation angles in degrees. Must have the "
            "same length as --rigid_translation_max_px."
        ),
    )
    parser.add_argument("--misalignment_trials", type=int, default=5)
    parser.add_argument("--misalignment_fixed_margin", type=int, default=8)
    parser.add_argument("--misalignment_seed", type=int, default=None)
    parser.add_argument("--misalignment_valid_threshold", type=float, default=0.999)
    parser.add_argument("--misalignment_output", type=str, default="")
    diagnostic, remaining = parser.parse_known_args()
    cfg = parse_args(remaining)

    if cfg.stage != "test":
        raise ValueError("Joint rigid diagnosis is test-only; use --stage test")
    if str(cfg.predictor_version).lower() != "v4":
        raise ValueError("Joint rigid diagnosis expects --predictor_version v4")
    if str(cfg.msi_ablation).lower() != "raw_direct":
        raise ValueError("V4 uses --msi_ablation raw_direct")
    if diagnostic.misalignment_trials < 1:
        raise ValueError("--misalignment_trials must be >= 1")
    if diagnostic.misalignment_fixed_margin < 0:
        raise ValueError("--misalignment_fixed_margin must be >= 0")
    if not 0.0 < diagnostic.misalignment_valid_threshold <= 1.0:
        raise ValueError("--misalignment_valid_threshold must lie in (0,1]")

    translations = [float(v) for v in diagnostic.rigid_translation_max_px]
    rotations = [float(v) for v in diagnostic.rigid_rotation_max_deg]
    if len(translations) != len(rotations):
        raise ValueError(
            "--rigid_translation_max_px and --rigid_rotation_max_deg must have "
            "the same number of paired levels"
        )
    if not translations:
        raise ValueError("At least one rigid severity pair is required")
    if any(v < 0.0 for v in translations):
        raise ValueError("Translation maxima must be >= 0")
    if any(v < 0.0 for v in rotations):
        raise ValueError("Rotation maxima must be >= 0")

    diagnostic.rigid_pairs = list(zip(translations, rotations))
    return cfg, diagnostic


def _write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows: Iterable[Dict[str, object]], key: str) -> float:
    rows = list(rows)
    return float(np.mean([float(row[key]) for row in rows]))


def _local_stats(local_field: torch.Tensor) -> Tuple[float, float]:
    magnitude = torch.linalg.vector_norm(local_field.detach().float(), dim=1)
    return float(magnitude.mean().item()), float(magnitude.amax().item())


@torch.no_grad()
def run_joint_rigid_diagnosis(cfg, diagnostic):
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
        "Joint rigid diagnosis: only HR-MSI is warped; "
        "GT/LR-HSI/Innovation-1 trajectory remain fixed."
    )

    batches = list(test_loader)
    if not batches:
        raise ValueError("Test loader is empty")

    base_seed = cfg.seed if diagnostic.misalignment_seed is None else int(diagnostic.misalignment_seed)
    detail_rows: List[Dict[str, object]] = []

    for level_index, (translation_max, rotation_max) in enumerate(diagnostic.rigid_pairs):
        registered = abs(translation_max) < 1e-12 and abs(rotation_max) < 1e-12
        trials = 1 if registered else int(diagnostic.misalignment_trials)
        print(
            f"\n[level {level_index}] translation_radius<={translation_max:g}px, "
            f"|rotation|<={rotation_max:g}deg, trials={trials}"
        )

        for trial in range(trials):
            # Reuse the same trial seed for every severity level. The shared
            # misalignment sampler consumes the same normalized radius,
            # direction and rotation random variables, then scales them by the
            # current maxima. This gives a paired severity curve.
            generator = torch.Generator(device="cpu")
            generator.manual_seed(base_seed + trial * 100003)

            for sample_index, batch in enumerate(batches):
                gt = batch["gt"].to(device, non_blocking=True)
                hr_msi = batch["hr_msi"].to(device, non_blocking=True)

                warped_msi, valid_soft, params = make_misaligned_msi(
                    hr_msi,
                    translation_max_px=float(translation_max),
                    rotation_max_deg=float(rotation_max),
                    local_max_displacement_px=0.0,
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
                true_rotation = float(params.rotation_deg[0].item())
                actual_shift = math.hypot(dx, dy)

                global_shift = model.last_global_shift_px
                if global_shift is None:
                    gx = gy = float("nan")
                else:
                    gx = float(global_shift[0, 0].item())
                    gy = float(global_shift[0, 1].item())

                global_rotation = model.last_global_rotation_deg
                predicted_rotation = (
                    float("nan")
                    if global_rotation is None
                    else float(global_rotation[0].item())
                )

                local_scale = -1
                local_mean = float("nan")
                local_max = float("nan")
                if model.last_alignment is not None:
                    local_scale = int(model.last_alignment.local_scale)
                    local_mean, local_max = _local_stats(
                        model.last_alignment.local_offset_px
                    )

                # For rotation around the image center, the inverse rigid
                # correction should have approximately the opposite signed
                # angle. This diagnostic is exact for the angle component even
                # though translation and rotation compose jointly.
                rotation_inverse_error = abs(true_rotation + predicted_rotation)

                detail_rows.append(
                    {
                        "level_index": level_index,
                        "translation_max_px": float(translation_max),
                        "rotation_max_deg": float(rotation_max),
                        "trial": trial,
                        "sample": sample_index,
                        "true_dx_px": dx,
                        "true_dy_px": dy,
                        "actual_shift_px": actual_shift,
                        "true_rotation_deg": true_rotation,
                        "abs_true_rotation_deg": abs(true_rotation),
                        "global_correction_dx_px": gx,
                        "global_correction_dy_px": gy,
                        "global_correction_magnitude_px": math.hypot(gx, gy),
                        "global_correction_rotation_deg": predicted_rotation,
                        "abs_global_correction_rotation_deg": abs(predicted_rotation),
                        "rotation_inverse_error_deg": rotation_inverse_error,
                        "final_local_scale": local_scale,
                        "final_local_offset_mean_px": local_mean,
                        "final_local_offset_max_px": local_max,
                        "valid_fraction": valid_fraction,
                        "PSNR_valid": valid_psnr,
                        "SAM_valid": valid_sam,
                        "PSNR_full": full["PSNR"],
                        "SAM_full": full["SAM"],
                        "PSNR_fixed": fixed["PSNR"],
                        "SAM_fixed": fixed["SAM"],
                    }
                )

    grouped = defaultdict(list)
    for row in detail_rows:
        grouped[int(row["level_index"])].append(row)

    summary_rows: List[Dict[str, object]] = []
    for level_index, (translation_max, rotation_max) in enumerate(diagnostic.rigid_pairs):
        rows = grouped[level_index]
        summary_rows.append(
            {
                "level_index": level_index,
                "translation_max_px": float(translation_max),
                "rotation_max_deg": float(rotation_max),
                "n_runs": len(rows),
                "mean_actual_shift_px": _mean(rows, "actual_shift_px"),
                "mean_abs_actual_rotation_deg": _mean(rows, "abs_true_rotation_deg"),
                "PSNR_valid": _mean(rows, "PSNR_valid"),
                "SAM_valid": _mean(rows, "SAM_valid"),
                "valid_fraction": _mean(rows, "valid_fraction"),
                "PSNR_full": _mean(rows, "PSNR_full"),
                "SAM_full": _mean(rows, "SAM_full"),
                "mean_global_correction_px": _mean(rows, "global_correction_magnitude_px"),
                "mean_abs_global_rotation_deg": _mean(rows, "abs_global_correction_rotation_deg"),
                "mean_rotation_inverse_error_deg": _mean(rows, "rotation_inverse_error_deg"),
                "mean_final_local_offset_px": _mean(rows, "final_local_offset_mean_px"),
                "max_final_local_offset_px": float(
                    np.max([float(row["final_local_offset_max_px"]) for row in rows])
                ),
            }
        )

    print("\nLearned V4 joint rigid alignment summary (valid-overlap primary):")
    for row in summary_rows:
        print(
            f"d<={row['translation_max_px']:>4.1f}px "
            f"r<={row['rotation_max_deg']:>3.1f}deg "
            f"actual_shift={row['mean_actual_shift_px']:.3f}px "
            f"actual|rot|={row['mean_abs_actual_rotation_deg']:.3f}deg "
            f"PSNR_valid={row['PSNR_valid']:.4f} "
            f"SAM_valid={row['SAM_valid']:.4f} "
            f"global_corr={row['mean_global_correction_px']:.3f}px "
            f"global|rot|={row['mean_abs_global_rotation_deg']:.3f}deg "
            f"rot_inv_err={row['mean_rotation_inverse_error_deg']:.3f}deg "
            f"local_final={row['mean_final_local_offset_px']:.3f}px"
        )

    if diagnostic.misalignment_output:
        summary_path = diagnostic.misalignment_output
    else:
        summary_path = os.path.join(
            cfg.output_root,
            "metrics",
            f"{cfg.dataset}_v4_learned_alignment_rigid.csv",
        )
    stem, ext = os.path.splitext(summary_path)
    if not ext:
        ext = ".csv"
        summary_path = stem + ext
    detail_path = f"{stem}_details{ext}"

    _write_csv(summary_path, summary_rows)
    _write_csv(detail_path, detail_rows)
    print(f"Summary CSV: {summary_path}")
    print(f"Details CSV: {detail_path}")


if __name__ == "__main__":
    cfg, diagnostic = parse_diagnostic_args()
    run_joint_rigid_diagnosis(cfg, diagnostic)
