"""Complete non-registered HSI-MSI sensitivity diagnosis.

The trained model is never changed. Only the HR-MSI observation is perturbed at
inference. This script uses ``degradations.misalignment`` and evaluates the
same frozen Raw-Direct checkpoint under registered, translation, rotation,
global rigid, local smooth non-rigid, and global+local deformation.

Valid-overlap PSNR/SAM are the primary metrics. Full-frame metrics are retained
as secondary end-to-end references.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch

from config import parse_args
from data_loader import build_loaders
from degradations.misalignment import make_misaligned_msi
from diagnose_misalignment_translation import calc_masked_psnr_sam
from innovation1 import build_progressive_process, reconstruct_from_terminal_lr
from main import _build_model
from metrics import calc_metrics
from utils import get_device, load_checkpoint, set_seed


LEVEL_PRESETS = {
    "registered": (0.0, 0.0, 0.0),
    "mild": (1.0, 0.5, 0.5),
    "medium": (2.0, 1.0, 1.0),
    "strong": (4.0, 2.0, 2.0),
    "very_strong": (6.0, 3.0, 3.0),
}

VALID_SCENARIOS = (
    "registered",
    "translation",
    "rotation",
    "global",
    "local",
    "global_local",
)


def parse_diagnostic_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--misalignment_scenarios",
        nargs="+",
        default=["registered", "translation", "rotation", "global", "local", "global_local"],
        choices=VALID_SCENARIOS,
    )
    parser.add_argument(
        "--misalignment_levels",
        nargs="+",
        default=["mild", "medium", "strong", "very_strong"],
        choices=["mild", "medium", "strong", "very_strong"],
    )
    parser.add_argument("--misalignment_trials", type=int, default=5)
    parser.add_argument("--misalignment_seed", type=int, default=None)
    parser.add_argument("--local_control_grid", type=int, default=5)
    parser.add_argument("--misalignment_valid_threshold", type=float, default=0.999)
    parser.add_argument("--misalignment_output", type=str, default="")
    diagnostic, remaining = parser.parse_known_args()
    cfg = parse_args(remaining)

    if cfg.stage != "test":
        raise ValueError("Complete misalignment diagnosis is test-only; use --stage test")
    if str(cfg.predictor_version).lower() != "v3":
        raise ValueError("Misalignment diagnosis expects --predictor_version v3")
    if str(cfg.msi_ablation).lower() != "raw_direct":
        raise ValueError("Use the frozen Raw-Direct baseline: --msi_ablation raw_direct")
    if diagnostic.misalignment_trials < 1:
        raise ValueError("--misalignment_trials must be >= 1")
    if diagnostic.local_control_grid < 2:
        raise ValueError("--local_control_grid must be >= 2")
    if not 0.0 < diagnostic.misalignment_valid_threshold <= 1.0:
        raise ValueError("--misalignment_valid_threshold must lie in (0,1]")
    return cfg, diagnostic


def resolve_severity(scenario: str, level: str) -> Tuple[float, float, float]:
    """Return translation max px, rotation max deg, local max displacement px."""
    if scenario == "registered":
        return LEVEL_PRESETS["registered"]
    translation, rotation, local = LEVEL_PRESETS[level]
    if scenario == "translation":
        return translation, 0.0, 0.0
    if scenario == "rotation":
        return 0.0, rotation, 0.0
    if scenario == "global":
        return translation, rotation, 0.0
    if scenario == "local":
        return 0.0, 0.0, local
    if scenario == "global_local":
        return translation, rotation, local
    raise ValueError(f"Unsupported scenario: {scenario}")


def _write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows: List[Dict[str, object]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _local_stats(local_field: torch.Tensor) -> Tuple[float, float]:
    magnitude = torch.linalg.vector_norm(local_field.detach().float(), dim=1)
    return float(magnitude.mean().item()), float(magnitude.amax().item())


@torch.no_grad()
def run_complete_diagnosis(cfg, diagnostic):
    set_seed(cfg.seed)
    _, test_loader, info = build_loaders(cfg)
    device = get_device(cfg.device)
    process = build_progressive_process(cfg)
    model = _build_model(cfg, info, device)

    if not cfg.resume:
        raise ValueError("Pass the trained Raw-Direct checkpoint explicitly with --resume")
    loaded_epoch, loaded_best = load_checkpoint(
        model,
        cfg.resume,
        optimizer=None,
        map_location=str(device),
        load_optimizer=False,
    )
    print(
        f"Loaded checkpoint {cfg.resume}: epoch={loaded_epoch}, "
        f"stored_best_PSNR={loaded_best:.6f}"
    )

    cached_batches = list(test_loader)
    if not cached_batches:
        raise ValueError("Test loader is empty")

    base_seed = cfg.seed if diagnostic.misalignment_seed is None else int(diagnostic.misalignment_seed)
    detail_rows: List[Dict[str, object]] = []

    run_specs: List[Tuple[str, str]] = []
    if "registered" in diagnostic.misalignment_scenarios:
        run_specs.append(("registered", "registered"))
    for scenario in diagnostic.misalignment_scenarios:
        if scenario == "registered":
            continue
        for level in diagnostic.misalignment_levels:
            run_specs.append((scenario, level))

    for scenario, level in run_specs:
        translation_max, rotation_max, local_max = resolve_severity(scenario, level)
        trials = 1 if scenario == "registered" else int(diagnostic.misalignment_trials)

        print(
            f"\n[{scenario}/{level}] translation<=±{translation_max:g}px, "
            f"rotation<=±{rotation_max:g}deg, local<={local_max:g}px"
        )

        for trial in range(trials):
            # Same trial seed is reused across severity levels. The degradation
            # sampler always consumes the same global RNG variables, so global
            # directions and local control patterns are paired across levels.
            generator = torch.Generator(device="cpu")
            generator.manual_seed(base_seed + trial * 100003)
            sample_index = 0

            for batch in cached_batches:
                gt = batch["gt"].to(device, non_blocking=True)
                hr_msi = batch["hr_msi"].to(device, non_blocking=True)

                warped_msi, valid_soft, params = make_misaligned_msi(
                    hr_msi,
                    translation_max_px=translation_max,
                    rotation_max_deg=rotation_max,
                    local_max_displacement_px=local_max,
                    control_grid_size=diagnostic.local_control_grid,
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
                        threshold=diagnostic.misalignment_valid_threshold,
                    )
                    local_mean, local_max_actual = _local_stats(
                        params.local_displacement_px[i : i + 1]
                    )
                    dx = float(params.dx_px[i].item())
                    dy = float(params.dy_px[i].item())
                    angle = float(params.rotation_deg[i].item())

                    detail_rows.append(
                        {
                            "scenario": scenario,
                            "level": level,
                            "trial": trial,
                            "sample": sample_index + i,
                            "translation_max_px": translation_max,
                            "rotation_max_deg": rotation_max,
                            "local_max_displacement_px": local_max,
                            "dx_px": dx,
                            "dy_px": dy,
                            "translation_magnitude_px": math.hypot(dx, dy),
                            "rotation_deg": angle,
                            "local_mean_displacement_px": local_mean,
                            "local_max_actual_px": local_max_actual,
                            "valid_fraction": valid_fraction,
                            "PSNR_valid": valid_psnr,
                            "SAM_valid": valid_sam,
                            "PSNR_full": float(full["PSNR"]),
                            "SAM_full": float(full["SAM"]),
                            "RMSE_full": float(full["RMSE"]),
                            "ERGAS_full": float(full["ERGAS"]),
                            "SSIM_full": float(full["SSIM"]),
                            "CC_full": float(full["CC"]),
                        }
                    )
                sample_index += batch_size

    grouped = defaultdict(list)
    for row in detail_rows:
        grouped[(row["scenario"], row["level"])].append(row)

    registered_key = ("registered", "registered")
    if registered_key in grouped:
        ref_psnr = _mean(grouped[registered_key], "PSNR_valid")
        ref_sam = _mean(grouped[registered_key], "SAM_valid")
    else:
        ref_psnr = float("nan")
        ref_sam = float("nan")

    summary_rows: List[Dict[str, object]] = []
    for scenario, level in run_specs:
        group = grouped[(scenario, level)]
        psnr_valid = _mean(group, "PSNR_valid")
        sam_valid = _mean(group, "SAM_valid")
        row = {
            "scenario": scenario,
            "level": level,
            "n_runs": len(group),
            "translation_max_px": float(group[0]["translation_max_px"]),
            "rotation_max_deg": float(group[0]["rotation_max_deg"]),
            "local_max_displacement_px": float(group[0]["local_max_displacement_px"]),
            "mean_translation_magnitude_px": _mean(group, "translation_magnitude_px"),
            "mean_abs_rotation_deg": float(np.mean([abs(float(r["rotation_deg"])) for r in group])),
            "mean_local_displacement_px": _mean(group, "local_mean_displacement_px"),
            "mean_local_max_actual_px": _mean(group, "local_max_actual_px"),
            "valid_fraction": _mean(group, "valid_fraction"),
            "PSNR_valid": psnr_valid,
            "SAM_valid": sam_valid,
            "delta_PSNR_valid": psnr_valid - ref_psnr if math.isfinite(ref_psnr) else float("nan"),
            "delta_SAM_valid": sam_valid - ref_sam if math.isfinite(ref_sam) else float("nan"),
            "PSNR_full": _mean(group, "PSNR_full"),
            "SAM_full": _mean(group, "SAM_full"),
            "SSIM_full": _mean(group, "SSIM_full"),
            "CC_full": _mean(group, "CC_full"),
        }
        summary_rows.append(row)

    if diagnostic.misalignment_output:
        summary_path = diagnostic.misalignment_output
    else:
        summary_path = os.path.join(
            cfg.output_root,
            "metrics",
            f"{cfg.dataset}_misalignment_full_raw_direct.csv",
        )
    stem, ext = os.path.splitext(summary_path)
    detail_path = f"{stem}_details{ext or '.csv'}"
    _write_csv(summary_path, summary_rows)
    _write_csv(detail_path, detail_rows)

    print("\n=== Misalignment valid-overlap summary ===")
    for row in summary_rows:
        print(
            f"{row['scenario']:>12s} {row['level']:>11s} | "
            f"valid={row['valid_fraction']:.4f} "
            f"PSNR_valid={row['PSNR_valid']:.4f} "
            f"SAM_valid={row['SAM_valid']:.4f} "
            f"dPSNR={row['delta_PSNR_valid']:+.4f} "
            f"dSAM={row['delta_SAM_valid']:+.4f}"
        )
    print(f"Summary CSV: {summary_path}")
    print(f"Details CSV: {detail_path}")
    return summary_rows, detail_rows


def main():
    cfg, diagnostic = parse_diagnostic_args()
    run_complete_diagnosis(cfg, diagnostic)


if __name__ == "__main__":
    main()
