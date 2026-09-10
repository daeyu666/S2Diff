"""Head-only sub-pixel repair stage for Innovation-2 V4 alignment.

This stage starts from the trained confidence/gate-presence checkpoint and adds
only cost-volume-conditioned continuous residual heads.  By default all legacy
parameters are frozen and only ``subpixel_heads`` are optimized, making the
ablation attribution clean and protecting registered/global-rigid behavior.

Default local range remains 0.5 px.  Local-only samples supervise the final
scale-1 dense field; registered/global-only samples provide a weak zero target;
global+local samples are trained through reconstruction only.
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
from models.predictor_v4_subpixel import enable_cost_volume_subpixel_refiner
from train_v4_gate_presence_repair import train_gate_presence_epoch
from train_v4_local_zero_repair import evaluate_nonregistered_scenarios
from utils import (
    CSVLogger,
    ensure_dir,
    get_device,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)


def parse_subpixel_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--subpixel_init_checkpoint", type=str, required=True)
    parser.add_argument("--train_msi_local_max_px", type=float, default=0.5)
    parser.add_argument("--subpixel_control_grid", type=int, default=5)
    parser.add_argument("--subpixel_hidden_channels", type=int, default=32)
    parser.add_argument("--subpixel_max_px", type=float, default=0.5)
    parser.add_argument("--lambda_local", type=float, default=0.2)
    parser.add_argument("--lambda_zero", type=float, default=0.02)
    parser.add_argument("--lambda_gate", type=float, default=0.0)
    parser.add_argument("--lambda_rot", type=float, default=0.0)
    parser.add_argument("--local_loss_norm_px", type=float, default=0.0)
    parser.add_argument("--rotation_loss_norm_deg", type=float, default=0.0)
    parser.add_argument("--confidence_gate_gain_init", type=float, default=4.0)
    parser.add_argument("--confidence_gate_bias_init", type=float, default=1.5)
    parser.add_argument("--subpixel_registered_probability", type=float, default=0.2)
    parser.add_argument("--subpixel_global_only_probability", type=float, default=0.2)
    parser.add_argument("--subpixel_local_only_probability", type=float, default=0.3)
    parser.add_argument("--subpixel_global_local_probability", type=float, default=0.3)
    parser.add_argument("--subpixel_global_downsample", type=int, default=2)
    parser.add_argument("--subpixel_registered_tolerance_db", type=float, default=0.35)
    parser.add_argument("--subpixel_eval_seed_offset", type=int, default=15431)
    parser.add_argument("--subpixel_valid_threshold", type=float, default=0.999)
    parser.add_argument("--subpixel_save_name", type=str, default="")
    repair, remaining = parser.parse_known_args()
    cfg = parse_args(remaining)

    if cfg.stage != "train":
        raise ValueError("Subpixel repair is train-only; use --stage train")
    if str(cfg.predictor_version).lower() != "v4":
        raise ValueError("Subpixel repair requires --predictor_version v4")
    if str(cfg.msi_ablation).lower() != "raw_direct":
        raise ValueError("Subpixel repair requires --msi_ablation raw_direct")
    if repair.train_msi_local_max_px <= 0.0:
        raise ValueError("--train_msi_local_max_px must be > 0")
    if repair.subpixel_control_grid < 2:
        raise ValueError("--subpixel_control_grid must be >= 2")
    if repair.subpixel_hidden_channels < 4:
        raise ValueError("--subpixel_hidden_channels must be >= 4")
    if repair.subpixel_max_px <= 0.0:
        raise ValueError("--subpixel_max_px must be > 0")
    if min(repair.lambda_local, repair.lambda_zero, repair.lambda_gate, repair.lambda_rot) < 0.0:
        raise ValueError("auxiliary loss weights must be >= 0")
    if repair.subpixel_global_downsample < 1:
        raise ValueError("--subpixel_global_downsample must be >= 1")
    if repair.subpixel_registered_tolerance_db < 0.0:
        raise ValueError("--subpixel_registered_tolerance_db must be >= 0")
    if not 0.0 < repair.subpixel_valid_threshold <= 1.0:
        raise ValueError("--subpixel_valid_threshold must lie in (0,1]")

    probs = [
        repair.subpixel_registered_probability,
        repair.subpixel_global_only_probability,
        repair.subpixel_local_only_probability,
        repair.subpixel_global_local_probability,
    ]
    if any(float(p) < 0.0 for p in probs) or abs(sum(float(p) for p in probs) - 1.0) > 1e-6:
        raise ValueError("subpixel category probabilities must be nonnegative and sum to 1")

    if repair.local_loss_norm_px <= 0.0:
        repair.local_loss_norm_px = float(repair.train_msi_local_max_px)
    if repair.rotation_loss_norm_deg <= 0.0:
        repair.rotation_loss_norm_deg = max(float(cfg.train_msi_rotation_max_deg), 1e-6)

    cfg.alignment_global_feature_downsample = int(repair.subpixel_global_downsample)
    return cfg, repair


def _paths(cfg, repair):
    root = os.path.join(cfg.checkpoint_root, "innovation1")
    ensure_dir(root)
    if repair.subpixel_save_name:
        stem = repair.subpixel_save_name
    else:
        stem = (
            f"{cfg.dataset}_v4_subpixel"
            f"_d{_compact_float_tag(cfg.train_msi_translation_max_px)}"
            f"_r{_compact_float_tag(cfg.train_msi_rotation_max_deg)}"
            f"_l{_compact_float_tag(repair.train_msi_local_max_px)}"
            f"_loc{_compact_float_tag(repair.lambda_local)}"
            f"_zero{_compact_float_tag(repair.lambda_zero)}"
            f"_sp{_compact_float_tag(repair.subpixel_max_px)}"
            f"_h{int(repair.subpixel_hidden_channels)}"
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


def _freeze_legacy_parameters(model: torch.nn.Module) -> int:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    local = model.geometry_aligner.local_aligner
    if not hasattr(local, "subpixel_heads"):
        raise RuntimeError("Subpixel aligner was not installed")
    for parameter in local.subpixel_heads.parameters():
        parameter.requires_grad_(True)
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if count <= 0:
        raise RuntimeError("No trainable subpixel parameters")
    return int(count)


def _subpixel_snapshot(model) -> Dict[int, float]:
    local = model.geometry_aligner.local_aligner
    if not hasattr(local, "subpixel_snapshot"):
        return {}
    return local.subpixel_snapshot()


def run(cfg, repair):
    set_seed(cfg.seed)
    train_loader, test_loader, info = build_loaders(cfg)
    device = get_device(cfg.device)
    process = build_progressive_process(cfg)
    model = _build_model(cfg, info, device, process=process)
    enable_cost_volume_subpixel_refiner(
        model,
        confidence_gain_init=repair.confidence_gate_gain_init,
        confidence_bias_init=repair.confidence_gate_bias_init,
        subpixel_hidden_channels=repair.subpixel_hidden_channels,
        subpixel_max_px=repair.subpixel_max_px,
    )

    # Source contains all legacy V4/confidence parameters; only subpixel head
    # parameters are new and remain at their exact-zero-output initialization.
    source_epoch, source_best = load_checkpoint(
        model,
        repair.subpixel_init_checkpoint,
        optimizer=None,
        strict=False,
        map_location=str(device),
        load_optimizer=False,
    )
    trainable_count = _freeze_legacy_parameters(model)
    print(
        "Subpixel warm start: "
        f"{repair.subpixel_init_checkpoint} "
        f"(epoch={source_epoch}, stored_registered_PSNR={source_best:.6f})"
    )
    print(
        "Training scope=head-only; "
        f"trainable_parameters={trainable_count}, "
        f"subpixel_max=±{repair.subpixel_max_px:g}px, "
        f"lambda_local={repair.lambda_local:g}, lambda_zero={repair.lambda_zero:g}."
    )

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        int(cfg.seed) + int(getattr(cfg, "train_misalignment_seed_offset", 7919))
    )

    best_path, best_registered_path, last_path, log_path = _paths(cfg, repair)
    logger = CSVLogger(
        log_path,
        fieldnames=[
            "epoch", "loss", "l1", "sam_loss", "local_loss", "zero_loss",
            "local_epe_px", "true_local_px", "pred_local_px",
            "registered_pred_local_px", "global_only_pred_local_px",
            "subpixel_scale4", "subpixel_scale2", "subpixel_scale1",
            "registered_PSNR", "registered_SAM", "global_only_PSNR",
            "global_only_SAM", "local_only_PSNR", "local_only_SAM",
            "global_local_PSNR", "global_local_SAM", "robust_PSNR",
            "selection_score", "best_selection_score", "best_registered_PSNR",
        ],
    )

    eval_seed = int(cfg.seed) + int(repair.subpixel_eval_seed_offset)
    best_selection_score = float("-inf")
    best_registered_psnr = float("-inf")

    for epoch in range(1, cfg.epochs + 1):
        stats = train_gate_presence_epoch(
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
            lambda_gate=repair.lambda_gate,
            rotation_loss_norm_deg=repair.rotation_loss_norm_deg,
            local_loss_norm_px=repair.local_loss_norm_px,
            boundary_probability=cfg.boundary_probability,
            boundary_radius=cfg.boundary_radius,
            grad_clip=cfg.grad_clip,
            translation_max_px=cfg.train_msi_translation_max_px,
            rotation_max_deg=cfg.train_msi_rotation_max_deg,
            local_max_displacement_px=repair.train_msi_local_max_px,
            local_control_grid=repair.subpixel_control_grid,
            registered_probability=repair.subpixel_registered_probability,
            global_only_probability=repair.subpixel_global_only_probability,
            local_only_probability=repair.subpixel_local_only_probability,
            global_local_probability=repair.subpixel_global_local_probability,
            generator=generator,
        )
        sp = _subpixel_snapshot(model)
        print(
            f"Epoch {epoch:04d}/{cfg.epochs:04d} "
            f"loss={stats.loss:.6f} local={stats.local:.6f} zero={stats.zero:.6f} "
            f"local_epe={stats.local_epe_px:.4f}px "
            f"true_local={stats.target_local_mean_px:.4f}px "
            f"pred_local={stats.predicted_local_mean_px:.4f}px "
            f"reg_pred={stats.registered_pred_local_px:.4f}px "
            f"global_only_pred={stats.global_only_pred_local_px:.4f}px "
            f"subpix4/2/1={sp.get(4, float('nan')):.3f}/"
            f"{sp.get(2, float('nan')):.3f}/{sp.get(1, float('nan')):.3f}"
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
                control_grid_size=repair.subpixel_control_grid,
                valid_threshold=repair.subpixel_valid_threshold,
                seed=eval_seed,
            )
            registered_psnr = float(registered_metrics["PSNR"])
            robust_psnr = (
                float(nonreg_metrics["global_only_PSNR"])
                + float(nonreg_metrics["local_only_PSNR"])
                + float(nonreg_metrics["global_local_PSNR"])
            ) / 3.0
            floor_psnr = float(source_best) - float(repair.subpixel_registered_tolerance_db)
            registered_gap = max(0.0, floor_psnr - registered_psnr)
            selection_score = robust_psnr - 5.0 * registered_gap

            print(f"  registered: {_format_metrics(registered_metrics)}")
            print(
                "  subpixel eval: "
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
                        "source_checkpoint": repair.subpixel_init_checkpoint,
                    },
                )
                print(f"  saved best subpixel checkpoint -> {best_path}")

        logger.write(
            {
                "epoch": epoch,
                "loss": stats.loss,
                "l1": stats.l1,
                "sam_loss": stats.sam,
                "local_loss": stats.local,
                "zero_loss": stats.zero,
                "local_epe_px": stats.local_epe_px,
                "true_local_px": stats.target_local_mean_px,
                "pred_local_px": stats.predicted_local_mean_px,
                "registered_pred_local_px": stats.registered_pred_local_px,
                "global_only_pred_local_px": stats.global_only_pred_local_px,
                "subpixel_scale4": sp.get(4, ""),
                "subpixel_scale2": sp.get(2, ""),
                "subpixel_scale1": sp.get(1, ""),
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
                extra={
                    "config": vars(cfg),
                    "repair": vars(repair),
                    "source_checkpoint": repair.subpixel_init_checkpoint,
                },
            )

    print("Subpixel repair complete.")
    print(f"Best subpixel checkpoint: {best_path}")
    print(f"Best registered checkpoint: {best_registered_path}")
    print(f"Last checkpoint: {last_path}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    cfg, repair = parse_subpixel_args()
    run(cfg, repair)
