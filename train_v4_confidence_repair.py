"""Confidence-gate repair stage for Innovation-2 V4 local alignment.

Start from the pre-zero-constraint local-repair checkpoint (the model that still
responds to true non-rigid motion), insert the confidence-gated 4->2->1 local
aligner, and train the gate jointly with the existing local descriptor.

Default mixture:
    20% registered
    20% global_only
    30% local_only
    30% global_local

Loss weights follow the diagnosed compromise:
    lambda_local = 0.10
    lambda_zero  = 0.05
    lambda_rot   = 0.10

The smaller zero loss is intentionally auxiliary. Whether a local residual is
suppressed is primarily learned by the confidence gate from center-vs-move
matching evidence, rather than by globally shrinking every displacement field.
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Dict

import torch

from config import parse_args
from data_loader import build_loaders
from innovation1 import build_progressive_process, evaluate
from main import _build_model, _compact_float_tag, _format_metrics
from models.predictor_v4_confidence import enable_confidence_gated_local_aligner
from train_v4_local_zero_repair import (
    evaluate_nonregistered_scenarios,
    train_zero_repair_epoch,
)
from utils import (
    CSVLogger,
    ensure_dir,
    get_device,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)


def parse_confidence_repair_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--confidence_repair_init_checkpoint",
        type=str,
        required=True,
        help="Pre-zero-constraint V4 local-repair checkpoint used as warm start.",
    )
    parser.add_argument("--train_msi_local_max_px", type=float, default=0.5)
    parser.add_argument("--confidence_repair_control_grid", type=int, default=5)
    parser.add_argument("--lambda_local", type=float, default=0.1)
    parser.add_argument("--lambda_zero", type=float, default=0.05)
    parser.add_argument("--lambda_rot", type=float, default=0.1)
    parser.add_argument("--local_loss_norm_px", type=float, default=0.0)
    parser.add_argument("--rotation_loss_norm_deg", type=float, default=0.0)
    parser.add_argument("--confidence_gate_gain_init", type=float, default=4.0)
    parser.add_argument("--confidence_gate_bias_init", type=float, default=1.5)
    parser.add_argument("--confidence_registered_probability", type=float, default=0.2)
    parser.add_argument("--confidence_global_only_probability", type=float, default=0.2)
    parser.add_argument("--confidence_local_only_probability", type=float, default=0.3)
    parser.add_argument("--confidence_global_local_probability", type=float, default=0.3)
    parser.add_argument("--confidence_global_downsample", type=int, default=2)
    parser.add_argument("--confidence_registered_tolerance_db", type=float, default=0.5)
    parser.add_argument("--confidence_eval_seed_offset", type=int, default=15431)
    parser.add_argument("--confidence_valid_threshold", type=float, default=0.999)
    parser.add_argument("--confidence_repair_save_name", type=str, default="")
    repair, remaining = parser.parse_known_args()
    cfg = parse_args(remaining)

    if cfg.stage != "train":
        raise ValueError("Confidence repair is train-only; use --stage train")
    if str(cfg.predictor_version).lower() != "v4":
        raise ValueError("Confidence repair requires --predictor_version v4")
    if str(cfg.msi_ablation).lower() != "raw_direct":
        raise ValueError("Confidence repair requires --msi_ablation raw_direct")
    if repair.train_msi_local_max_px <= 0.0:
        raise ValueError("--train_msi_local_max_px must be > 0")
    if repair.confidence_repair_control_grid < 2:
        raise ValueError("--confidence_repair_control_grid must be >= 2")
    if min(repair.lambda_local, repair.lambda_zero, repair.lambda_rot) < 0.0:
        raise ValueError("lambda_local/lambda_zero/lambda_rot must be >= 0")
    if repair.confidence_gate_gain_init <= 0.0:
        raise ValueError("--confidence_gate_gain_init must be > 0")
    if repair.confidence_global_downsample < 1:
        raise ValueError("--confidence_global_downsample must be >= 1")
    if repair.confidence_registered_tolerance_db < 0.0:
        raise ValueError("--confidence_registered_tolerance_db must be >= 0")
    if not 0.0 < repair.confidence_valid_threshold <= 1.0:
        raise ValueError("--confidence_valid_threshold must lie in (0,1]")

    probs = [
        repair.confidence_registered_probability,
        repair.confidence_global_only_probability,
        repair.confidence_local_only_probability,
        repair.confidence_global_local_probability,
    ]
    if any(float(p) < 0.0 for p in probs):
        raise ValueError("confidence category probabilities must be nonnegative")
    if abs(sum(float(p) for p in probs) - 1.0) > 1e-6:
        raise ValueError("confidence category probabilities must sum to 1")

    if repair.local_loss_norm_px <= 0.0:
        repair.local_loss_norm_px = float(repair.train_msi_local_max_px)
    if repair.rotation_loss_norm_deg <= 0.0:
        repair.rotation_loss_norm_deg = float(cfg.train_msi_rotation_max_deg)
    if repair.lambda_rot > 0.0 and repair.rotation_loss_norm_deg <= 0.0:
        raise ValueError("Positive lambda_rot requires positive rotation normalization")

    cfg.alignment_global_feature_downsample = int(repair.confidence_global_downsample)
    return cfg, repair


def _paths(cfg, repair):
    root = os.path.join(cfg.checkpoint_root, "innovation1")
    ensure_dir(root)
    if repair.confidence_repair_save_name:
        stem = repair.confidence_repair_save_name
    else:
        stem = (
            f"{cfg.dataset}_v4_confidence_repair"
            f"_d{_compact_float_tag(cfg.train_msi_translation_max_px)}"
            f"_r{_compact_float_tag(cfg.train_msi_rotation_max_deg)}"
            f"_l{_compact_float_tag(repair.train_msi_local_max_px)}"
            f"_loc{_compact_float_tag(repair.lambda_local)}"
            f"_zero{_compact_float_tag(repair.lambda_zero)}"
            f"_rot{_compact_float_tag(repair.lambda_rot)}"
            f"_g{_compact_float_tag(repair.confidence_gate_gain_init)}"
            f"_b{_compact_float_tag(repair.confidence_gate_bias_init)}"
            f"_gds{int(cfg.alignment_global_feature_downsample)}"
        )
    if stem.endswith(".pth"):
        stem = stem[:-4]
    return (
        os.path.join(root, stem + ".pth"),
        os.path.join(root, stem + "_best_registered.pth"),
        os.path.join(root, stem + "_last.pth"),
        os.path.join(cfg.log_root, stem + ".csv"),
    )


def _gate_snapshot(model) -> Dict[int, float]:
    local = model.geometry_aligner.local_aligner
    if not hasattr(local, "confidence_snapshot"):
        return {}
    return local.confidence_snapshot()


def run(cfg, repair):
    set_seed(cfg.seed)
    train_loader, test_loader, info = build_loaders(cfg)
    device = get_device(cfg.device)
    process = build_progressive_process(cfg)
    model = _build_model(cfg, info, device, process=process)
    enable_confidence_gated_local_aligner(
        model,
        confidence_gain_init=repair.confidence_gate_gain_init,
        confidence_bias_init=repair.confidence_gate_bias_init,
    )

    # The source checkpoint predates the gate. strict=False should report only
    # the new confidence gain/bias keys as missing; all old V4/local weights load.
    source_epoch, source_best = load_checkpoint(
        model,
        repair.confidence_repair_init_checkpoint,
        optimizer=None,
        strict=False,
        map_location=str(device),
        load_optimizer=False,
    )
    print(
        "Confidence-repair warm start: "
        f"{repair.confidence_repair_init_checkpoint} "
        f"(epoch={source_epoch}, stored_registered_PSNR={source_best:.6f})"
    )
    print(
        "Confidence gate: margin=best_move-center, "
        f"gain_init={repair.confidence_gate_gain_init:g}, "
        f"bias_init={repair.confidence_gate_bias_init:g}; "
        f"lambda_local={repair.lambda_local:g}, "
        f"lambda_zero={repair.lambda_zero:g}, lambda_rot={repair.lambda_rot:g}."
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        int(cfg.seed) + int(getattr(cfg, "train_misalignment_seed_offset", 7919))
    )

    best_path, best_registered_path, last_path, log_path = _paths(cfg, repair)
    logger = CSVLogger(
        log_path,
        fieldnames=[
            "epoch", "loss", "l1", "sam_loss", "rot_loss", "local_loss",
            "zero_loss", "local_epe_px", "true_local_px", "pred_local_px",
            "registered_pred_local_px", "global_only_pred_local_px",
            "gate_scale4", "gate_scale2", "gate_scale1", "registered_PSNR",
            "registered_SAM", "global_only_PSNR", "global_only_SAM",
            "local_only_PSNR", "local_only_SAM", "global_local_PSNR",
            "global_local_SAM", "robust_PSNR", "selection_score",
            "best_selection_score", "best_registered_PSNR",
        ],
    )

    eval_seed = int(cfg.seed) + int(repair.confidence_eval_seed_offset)
    best_selection_score = float("-inf")
    best_registered_psnr = float("-inf")

    for epoch in range(1, cfg.epochs + 1):
        stats = train_zero_repair_epoch(
            model,
            train_loader,
            optimizer,
            process,
            device,
            lambda_l1=cfg.lambda_l1,
            lambda_sam=cfg.lambda_sam,
            lambda_deg=cfg.lambda_deg,
            lambda_rot=repair.lambda_rot,
            lambda_local=repair.lambda_local,
            lambda_zero=repair.lambda_zero,
            rotation_loss_norm_deg=repair.rotation_loss_norm_deg,
            local_loss_norm_px=repair.local_loss_norm_px,
            boundary_probability=cfg.boundary_probability,
            boundary_radius=cfg.boundary_radius,
            grad_clip=cfg.grad_clip,
            translation_max_px=cfg.train_msi_translation_max_px,
            rotation_max_deg=cfg.train_msi_rotation_max_deg,
            local_max_displacement_px=repair.train_msi_local_max_px,
            local_control_grid=repair.confidence_repair_control_grid,
            registered_probability=repair.confidence_registered_probability,
            global_only_probability=repair.confidence_global_only_probability,
            local_only_probability=repair.confidence_local_only_probability,
            global_local_probability=repair.confidence_global_local_probability,
            generator=generator,
        )
        gate = _gate_snapshot(model)
        print(
            f"Epoch {epoch:04d}/{cfg.epochs:04d} "
            f"loss={stats.loss:.6f} local={stats.local:.6f} zero={stats.zero:.6f} "
            f"local_epe={stats.local_epe_px:.4f}px "
            f"true_local={stats.target_local_mean_px:.4f}px "
            f"pred_local={stats.predicted_local_mean_px:.4f}px "
            f"reg_pred={stats.registered_pred_local_px:.4f}px "
            f"global_only_pred={stats.global_only_pred_local_px:.4f}px "
            f"gate4/2/1={gate.get(4, float('nan')):.3f}/"
            f"{gate.get(2, float('nan')):.3f}/{gate.get(1, float('nan')):.3f}"
        )

        registered_metrics: Dict[str, float] = {}
        nonreg_metrics: Dict[str, float] = {}
        robust_psnr = float("nan")
        selection_score = float("nan")

        if epoch % cfg.eval_interval == 0 or epoch == cfg.epochs:
            registered_metrics = evaluate(
                model, test_loader, process, device, scale_ratio=cfg.scale_ratio
            )
            nonreg_metrics = evaluate_nonregistered_scenarios(
                model,
                test_loader,
                process,
                device,
                scale_ratio=cfg.scale_ratio,
                translation_max_px=cfg.train_msi_translation_max_px,
                rotation_max_deg=cfg.train_msi_rotation_max_deg,
                local_max_displacement_px=repair.train_msi_local_max_px,
                control_grid_size=repair.confidence_repair_control_grid,
                valid_threshold=repair.confidence_valid_threshold,
                seed=eval_seed,
            )
            registered_psnr = float(registered_metrics["PSNR"])
            robust_psnr = (
                float(nonreg_metrics["global_only_PSNR"])
                + float(nonreg_metrics["local_only_PSNR"])
                + float(nonreg_metrics["global_local_PSNR"])
            ) / 3.0
            floor_psnr = float(source_best) - float(
                repair.confidence_registered_tolerance_db
            )
            registered_gap = max(0.0, floor_psnr - registered_psnr)
            selection_score = robust_psnr - 5.0 * registered_gap

            print(f"  registered: {_format_metrics(registered_metrics)}")
            print(
                "  confidence eval: "
                f"global-only={nonreg_metrics['global_only_PSNR']:.4f}/"
                f"{nonreg_metrics['global_only_SAM']:.4f}, "
                f"local-only={nonreg_metrics['local_only_PSNR']:.4f}/"
                f"{nonreg_metrics['local_only_SAM']:.4f}, "
                f"global+local={nonreg_metrics['global_local_PSNR']:.4f}/"
                f"{nonreg_metrics['global_local_SAM']:.4f}, "
                f"robust={robust_psnr:.4f}, selection={selection_score:.4f}"
            )

            if registered_psnr > best_registered_psnr:
                best_registered_psnr = registered_psnr
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    best_registered_psnr,
                    best_registered_path,
                    extra={"config": vars(cfg), "repair": vars(repair)},
                )

            if selection_score > best_selection_score:
                best_selection_score = selection_score
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    registered_psnr,
                    best_path,
                    extra={
                        "config": vars(cfg),
                        "repair": vars(repair),
                        "registered_metrics": registered_metrics,
                        "nonregistered_metrics": nonreg_metrics,
                        "robust_psnr": robust_psnr,
                        "selection_score": selection_score,
                        "source_checkpoint": repair.confidence_repair_init_checkpoint,
                    },
                )
                print(f"  saved best confidence checkpoint -> {best_path}")

        logger.write(
            {
                "epoch": epoch,
                "loss": stats.loss,
                "l1": stats.l1,
                "sam_loss": stats.sam,
                "rot_loss": stats.rot,
                "local_loss": stats.local,
                "zero_loss": stats.zero,
                "local_epe_px": stats.local_epe_px,
                "true_local_px": stats.target_local_mean_px,
                "pred_local_px": stats.predicted_local_mean_px,
                "registered_pred_local_px": stats.registered_pred_local_px,
                "global_only_pred_local_px": stats.global_only_pred_local_px,
                "gate_scale4": gate.get(4, ""),
                "gate_scale2": gate.get(2, ""),
                "gate_scale1": gate.get(1, ""),
                "registered_PSNR": registered_metrics.get("PSNR", ""),
                "registered_SAM": registered_metrics.get("SAM", ""),
                "global_only_PSNR": nonreg_metrics.get("global_only_PSNR", ""),
                "global_only_SAM": nonreg_metrics.get("global_only_SAM", ""),
                "local_only_PSNR": nonreg_metrics.get("local_only_PSNR", ""),
                "local_only_SAM": nonreg_metrics.get("local_only_SAM", ""),
                "global_local_PSNR": nonreg_metrics.get("global_local_PSNR", ""),
                "global_local_SAM": nonreg_metrics.get("global_local_SAM", ""),
                "robust_PSNR": robust_psnr if math.isfinite(robust_psnr) else "",
                "selection_score": selection_score if math.isfinite(selection_score) else "",
                "best_selection_score": best_selection_score,
                "best_registered_PSNR": best_registered_psnr,
            }
        )

        if epoch % cfg.save_interval == 0 or epoch == cfg.epochs:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best_registered_psnr,
                last_path,
                extra={"config": vars(cfg), "repair": vars(repair)},
            )

    print("Confidence repair complete.")
    print(f"Best confidence checkpoint: {best_path}")
    print(f"Best registered checkpoint: {best_registered_path}")
    print(f"Last checkpoint: {last_path}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    cfg, repair = parse_confidence_repair_args()
    run(cfg, repair)
