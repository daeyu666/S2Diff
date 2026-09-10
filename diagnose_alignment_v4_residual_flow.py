"""Diagnose where V4 local displacement amplitude is being lost.

The geometric perturbation protocol matches diagnose_alignment_v4_local.py,
but this script additionally records the local-search flow at scales 4 -> 2 -> 1:

    raw expected residual
    confidence gate
    gated residual
    accumulated dense offset

If raw and gated residuals are close while the accumulated offset remains much
smaller than the known synthetic local field, the candidate soft-expectation is
the bottleneck. If gated residual is much smaller than raw residual, confidence
suppression is still the bottleneck.
"""

from __future__ import annotations

import csv
import math
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch

import diagnose_alignment_v4_local as base
from data_loader import build_loaders
from degradations.misalignment import make_misaligned_msi
from innovation1 import build_progressive_process, reconstruct_from_terminal_lr
from main import _build_model
from models.predictor_v4_residual_telemetry import enable_residual_telemetry_aligner
from utils import get_device, load_checkpoint, set_seed


SCALES = (4, 2, 1)


def _field_mean(field: torch.Tensor, sample: int) -> float:
    value = field[sample : sample + 1].detach().float()
    return float(torch.linalg.vector_norm(value, dim=1).mean().item())


def _map_mean(value: torch.Tensor, sample: int) -> float:
    return float(value[sample : sample + 1].detach().float().mean().item())


def _write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        raise ValueError("cannot write empty residual-flow CSV")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows: List[Dict[str, object]], key: str) -> float:
    values = [float(row[key]) for row in rows if math.isfinite(float(row[key]))]
    return float(np.mean(values)) if values else float("nan")


@torch.no_grad()
def run(cfg, diagnostic):
    set_seed(cfg.seed)
    _, test_loader, info = build_loaders(cfg)
    device = get_device(cfg.device)
    process = build_progressive_process(cfg)
    model = _build_model(cfg, info, device, process=process)
    enable_residual_telemetry_aligner(model)

    if not cfg.resume:
        raise ValueError("Pass the confidence/gate-presence checkpoint with --resume")
    loaded_epoch, loaded_best = load_checkpoint(
        model,
        cfg.resume,
        optimizer=None,
        strict=True,
        map_location=str(device),
        load_optimizer=False,
    )
    print(
        f"Loaded V4 confidence checkpoint {cfg.resume}: epoch={loaded_epoch}, "
        f"stored_best_PSNR={loaded_best:.6f}"
    )

    batches = list(test_loader)
    if not batches:
        raise ValueError("Test loader is empty")

    specs = base._scenario_specs(diagnostic)
    base_seed = cfg.seed if diagnostic.misalignment_seed is None else int(
        diagnostic.misalignment_seed
    )
    detail_rows: List[Dict[str, object]] = []

    for scenario, local_max in specs:
        translation_max, rotation_max, resolved_local_max = base.resolve_geometry(
            scenario,
            local_max,
            float(diagnostic.diagnostic_global_translation_max_px),
            float(diagnostic.diagnostic_global_rotation_max_deg),
        )
        trials = 1 if scenario == "registered" else int(diagnostic.misalignment_trials)
        print(
            f"\n[{scenario}] global_d<={translation_max:g}px, "
            f"global_r<={rotation_max:g}deg, local<={resolved_local_max:g}px"
        )

        for trial in range(trials):
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

                local = model.geometry_aligner.local_aligner
                local.clear_residual_telemetry()
                terminal_lr = process.terminal_observation(gt)
                pred = reconstruct_from_terminal_lr(
                    model,
                    process,
                    terminal_lr,
                    target_size=tuple(gt.shape[-2:]),
                    hr_msi=warped_msi,
                )

                for i in range(int(gt.shape[0])):
                    psnr, sam, valid_fraction = base.calc_masked_psnr_sam(
                        pred[i : i + 1],
                        gt[i : i + 1],
                        valid_soft[i : i + 1],
                        threshold=float(diagnostic.misalignment_valid_threshold),
                    )
                    true_local = _field_mean(params.local_displacement_px, i)
                    row: Dict[str, object] = {
                        "scenario": scenario,
                        "local_max_displacement_px": float(resolved_local_max),
                        "trial": trial,
                        "sample": sample_index + i,
                        "true_local_mean_px": true_local,
                        "PSNR_valid": float(psnr),
                        "SAM_valid": float(sam),
                        "valid_fraction": float(valid_fraction),
                    }

                    for scale in SCALES:
                        raw = local.last_raw_residual_by_scale.get(scale)
                        gated = local.last_gated_residual_by_scale.get(scale)
                        accumulated = local.last_accumulated_dense_by_scale.get(scale)
                        gate = local.last_confidence_by_scale.get(scale)
                        margin = local.last_margin_by_scale.get(scale)

                        raw_mean = _field_mean(raw, i) if raw is not None else float("nan")
                        gated_mean = (
                            _field_mean(gated, i) if gated is not None else float("nan")
                        )
                        accumulated_mean = (
                            _field_mean(accumulated, i)
                            if accumulated is not None
                            else float("nan")
                        )
                        gate_mean = _map_mean(gate, i) if gate is not None else float("nan")
                        margin_mean = (
                            _map_mean(margin, i) if margin is not None else float("nan")
                        )
                        retention = (
                            gated_mean / raw_mean
                            if math.isfinite(raw_mean)
                            and math.isfinite(gated_mean)
                            and raw_mean > 1e-9
                            else float("nan")
                        )
                        row[f"raw_residual_s{scale}_px"] = raw_mean
                        row[f"gate_s{scale}"] = gate_mean
                        row[f"margin_s{scale}"] = margin_mean
                        row[f"gated_residual_s{scale}_px"] = gated_mean
                        row[f"gate_retention_s{scale}"] = retention
                        row[f"accumulated_offset_s{scale}_px"] = accumulated_mean

                    final_pred = float(row["accumulated_offset_s1_px"])
                    row["final_pred_local_px"] = final_pred
                    row["final_pred_to_true_ratio"] = (
                        final_pred / true_local if true_local > 1e-9 else float("nan")
                    )
                    detail_rows.append(row)

                sample_index += int(gt.shape[0])

    grouped = defaultdict(list)
    for row in detail_rows:
        grouped[(str(row["scenario"]), float(row["local_max_displacement_px"]))].append(row)

    summary_rows: List[Dict[str, object]] = []
    for scenario, local_max in specs:
        rows = grouped[(scenario, float(local_max))]
        summary: Dict[str, object] = {
            "scenario": scenario,
            "local_max_displacement_px": float(local_max),
            "mean_true_local_px": _mean(rows, "true_local_mean_px"),
            "mean_final_pred_local_px": _mean(rows, "final_pred_local_px"),
            "mean_final_pred_to_true_ratio": _mean(rows, "final_pred_to_true_ratio"),
            "PSNR_valid": _mean(rows, "PSNR_valid"),
            "SAM_valid": _mean(rows, "SAM_valid"),
        }
        for scale in SCALES:
            for key in (
                f"raw_residual_s{scale}_px",
                f"gate_s{scale}",
                f"margin_s{scale}",
                f"gated_residual_s{scale}_px",
                f"gate_retention_s{scale}",
                f"accumulated_offset_s{scale}_px",
            ):
                summary[f"mean_{key}"] = _mean(rows, key)
        summary_rows.append(summary)

    if diagnostic.misalignment_output:
        stem, _ = os.path.splitext(diagnostic.misalignment_output)
        summary_path = stem + "_residual_flow.csv"
    else:
        summary_path = os.path.join(
            cfg.output_root,
            "metrics",
            f"{cfg.dataset}_v4_local_residual_flow.csv",
        )
    stem, ext = os.path.splitext(summary_path)
    detail_path = f"{stem}_details{ext or '.csv'}"
    _write_csv(summary_path, summary_rows)
    _write_csv(detail_path, detail_rows)

    print("\n=== V4 local residual-flow diagnosis ===")
    for row in summary_rows:
        print(
            f"{row['scenario']:>12s} local<={row['local_max_displacement_px']:.2f}px | "
            f"true={row['mean_true_local_px']:.3f}px "
            f"final={row['mean_final_pred_local_px']:.3f}px "
            f"PSNR={row['PSNR_valid']:.3f} SAM={row['SAM_valid']:.3f}"
        )
        for scale in SCALES:
            print(
                f"  s{scale}: raw={row[f'mean_raw_residual_s{scale}_px']:.4f}px "
                f"gate={row[f'mean_gate_s{scale}']:.3f} "
                f"gated={row[f'mean_gated_residual_s{scale}_px']:.4f}px "
                f"retain={row[f'mean_gate_retention_s{scale}']:.3f} "
                f"accum={row[f'mean_accumulated_offset_s{scale}_px']:.4f}px"
            )

    print(f"Summary CSV: {summary_path}")
    print(f"Details CSV: {detail_path}")
    print(
        "Interpretation: raw≈gated with small final offset => candidate soft-expectation "
        "is amplitude-limited; gated<<raw => confidence suppression remains dominant."
    )
    return summary_rows, detail_rows


def main():
    cfg, diagnostic = base.parse_diagnostic_args()
    run(cfg, diagnostic)


if __name__ == "__main__":
    main()
