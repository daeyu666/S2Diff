"""Gate-presence supervision stage for Innovation-2 V4 confidence alignment.

This stage continues from a trained confidence-gated V4 checkpoint.  It keeps
all geometry and fusion structure unchanged and adds one auxiliary objective:

    registered/global_only -> gate target 0
    local_only/global_local -> gate target 1

The target is applied to the differentiable gate logits at physical scales
4, 2 and 1 using BCEWithLogitsLoss.  This teaches the gate to separate local
motion presence from absence instead of relying on a strong zero-field loss.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from config import parse_args
from data_loader import build_loaders
from innovation1 import (
    batch_state_at,
    build_progressive_process,
    degradation_consistency_loss,
    evaluate,
    model_predict,
)
from losses import SAMLoss
from main import _build_model, _compact_float_tag, _format_metrics
from models.predictor_v4_gate_presence import (
    enable_presence_supervised_confidence_aligner,
)
from train_v4_local_repair import (
    normalized_local_field_loss,
    prepare_v4_alignment_with_local,
)
from train_v4_local_zero_repair import (
    GLOBAL_LOCAL,
    GLOBAL_ONLY,
    LOCAL_ONLY,
    REGISTERED,
    _field_mean_on_category,
    _local_only_stats,
    augment_zero_repair_batch,
    evaluate_nonregistered_scenarios,
)
from train_v4_rotation_repair import normalized_inverse_rotation_loss
from utils import (
    AverageMeter,
    CSVLogger,
    ensure_dir,
    get_device,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)


@dataclass
class GatePresenceStats:
    loss: float
    l1: float
    sam: float
    deg: float
    rot: float
    local: float
    zero: float
    gate: float
    local_epe_px: float
    target_local_mean_px: float
    predicted_local_mean_px: float
    registered_pred_local_px: float
    global_only_pred_local_px: float
    gate_off_mean: float
    gate_on_mean: float
    gate_scale4: float
    gate_scale2: float
    gate_scale1: float


def gate_presence_loss(
    gate_logits_by_scale: Dict[int, torch.Tensor],
    category: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """BCE presence supervision averaged equally over available physical scales."""
    if not gate_logits_by_scale:
        raise RuntimeError("No differentiable confidence-gate logits were produced")
    if category.ndim != 1:
        raise ValueError("category must have shape [B]")

    presence = ((category == LOCAL_ONLY) | (category == GLOBAL_LOCAL)).float()
    losses = []
    on_values = []
    off_values = []
    scale_means: Dict[int, float] = {}

    for scale in sorted(gate_logits_by_scale, reverse=True):
        logits = gate_logits_by_scale[scale]
        if logits.ndim != 3 or logits.shape[0] != category.shape[0]:
            raise ValueError(
                f"gate logits at scale {scale} must have shape BxHc xWc; "
                f"got {tuple(logits.shape)}"
            )
        target = presence[:, None, None].expand_as(logits)
        losses.append(F.binary_cross_entropy_with_logits(logits, target))

        with torch.no_grad():
            gate = torch.sigmoid(logits.detach())
            scale_means[int(scale)] = float(gate.mean().item())
            on_mask = presence.bool()
            off_mask = ~on_mask
            if bool(on_mask.any()):
                on_values.append(float(gate[on_mask].mean().item()))
            if bool(off_mask.any()):
                off_values.append(float(gate[off_mask].mean().item()))

    loss = torch.stack(losses).mean()
    diagnostics = {
        "gate_on_mean": float(np.mean(on_values)) if on_values else 0.0,
        "gate_off_mean": float(np.mean(off_values)) if off_values else 0.0,
        "gate_scale4": scale_means.get(4, float("nan")),
        "gate_scale2": scale_means.get(2, float("nan")),
        "gate_scale1": scale_means.get(1, float("nan")),
    }
    return loss, diagnostics


def train_gate_presence_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    process,
    device: torch.device,
    *,
    lambda_l1: float,
    lambda_sam: float,
    lambda_deg: float,
    lambda_rot: float,
    lambda_local: float,
    lambda_zero: float,
    lambda_gate: float,
    rotation_loss_norm_deg: float,
    local_loss_norm_px: float,
    boundary_probability: float,
    boundary_radius: int,
    grad_clip: float,
    translation_max_px: float,
    rotation_max_deg: float,
    local_max_displacement_px: float,
    local_control_grid: int,
    registered_probability: float,
    global_only_probability: float,
    local_only_probability: float,
    global_local_probability: float,
    generator: Optional[torch.Generator],
) -> GatePresenceStats:
    model.train()
    sam_loss_fn = SAMLoss()
    meter_names = (
        "loss", "l1", "sam", "deg", "rot", "local", "zero", "gate",
        "local_epe", "true_local", "pred_local", "reg_pred", "go_pred",
        "gate_off", "gate_on", "gate4", "gate2", "gate1",
    )
    meters = {name: AverageMeter() for name in meter_names}

    for batch in loader:
        gt = batch["gt"].to(device, non_blocking=True)
        hr_msi = batch["hr_msi"].to(device, non_blocking=True)
        batch_size = int(gt.shape[0])

        (
            warped_msi,
            applied_rotation_deg,
            target_local_field,
            local_supervision_mask,
            zero_supervision_mask,
            category,
            _,
        ) = augment_zero_repair_batch(
            hr_msi,
            translation_max_px=translation_max_px,
            rotation_max_deg=rotation_max_deg,
            local_max_displacement_px=local_max_displacement_px,
            control_grid_size=local_control_grid,
            registered_probability=registered_probability,
            global_only_probability=global_only_probability,
            local_only_probability=local_only_probability,
            global_local_probability=global_local_probability,
            generator=generator,
        )

        local_aligner = model.geometry_aligner.local_aligner
        if hasattr(local_aligner, "clear_current_gate_logits"):
            local_aligner.clear_current_gate_logits()

        globally_aligned_msi, predicted_rotation_deg, predicted_local_field = (
            prepare_v4_alignment_with_local(model, gt, warped_msi)
        )

        gate_loss, gate_diag = gate_presence_loss(
            local_aligner.current_gate_logits_by_scale,
            category,
        )

        timesteps = process.sample_timesteps(
            batch_size,
            boundary_probability=boundary_probability,
            boundary_radius=boundary_radius,
            device=device,
        )
        with torch.no_grad():
            x_t = batch_state_at(process, gt, timesteps)

        pred_x0 = model_predict(
            model,
            x_t,
            timesteps,
            hr_msi=globally_aligned_msi,
        )

        l1 = F.l1_loss(pred_x0, gt)
        sam = sam_loss_fn(pred_x0, gt)
        if lambda_deg > 0.0:
            deg = degradation_consistency_loss(process, pred_x0, gt, timesteps)
        else:
            deg = pred_x0.new_zeros(())

        if lambda_rot > 0.0:
            rot = normalized_inverse_rotation_loss(
                predicted_rotation_deg,
                applied_rotation_deg,
                normalization_deg=rotation_loss_norm_deg,
            )
        else:
            rot = pred_x0.new_zeros(())

        if lambda_local > 0.0:
            local = normalized_local_field_loss(
                predicted_local_field,
                target_local_field,
                local_supervision_mask,
                normalization_px=local_loss_norm_px,
            )
        else:
            local = pred_x0.new_zeros(())

        if lambda_zero > 0.0:
            zero = normalized_local_field_loss(
                predicted_local_field,
                torch.zeros_like(predicted_local_field),
                zero_supervision_mask,
                normalization_px=local_loss_norm_px,
            )
        else:
            zero = pred_x0.new_zeros(())

        loss = (
            lambda_l1 * l1
            + lambda_sam * sam
            + lambda_deg * deg
            + lambda_rot * rot
            + lambda_local * local
            + lambda_zero * zero
            + lambda_gate * gate_loss
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite gate-presence repair loss")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        max_norm = (
            float(grad_clip)
            if grad_clip is not None and grad_clip > 0.0
            else float("inf")
        )
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=max_norm,
            error_if_nonfinite=True,
        )
        optimizer.step()

        with torch.no_grad():
            local_epe, target_local_mean, pred_local_mean = _local_only_stats(
                predicted_local_field,
                target_local_field,
                category,
            )
            reg_pred = _field_mean_on_category(
                predicted_local_field, category, REGISTERED
            )
            go_pred = _field_mean_on_category(
                predicted_local_field, category, GLOBAL_ONLY
            )

        values = {
            "loss": float(loss.item()),
            "l1": float(l1.item()),
            "sam": float(sam.item()),
            "deg": float(deg.item()),
            "rot": float(rot.item()),
            "local": float(local.item()),
            "zero": float(zero.item()),
            "gate": float(gate_loss.item()),
            "local_epe": local_epe,
            "true_local": target_local_mean,
            "pred_local": pred_local_mean,
            "reg_pred": reg_pred,
            "go_pred": go_pred,
            "gate_off": gate_diag["gate_off_mean"],
            "gate_on": gate_diag["gate_on_mean"],
            "gate4": gate_diag["gate_scale4"],
            "gate2": gate_diag["gate_scale2"],
            "gate1": gate_diag["gate_scale1"],
        }
        for name, value in values.items():
            meters[name].update(value, batch_size)

    return GatePresenceStats(
        loss=meters["loss"].avg,
        l1=meters["l1"].avg,
        sam=meters["sam"].avg,
        deg=meters["deg"].avg,
        rot=meters["rot"].avg,
        local=meters["local"].avg,
        zero=meters["zero"].avg,
        gate=meters["gate"].avg,
        local_epe_px=meters["local_epe"].avg,
        target_local_mean_px=meters["true_local"].avg,
        predicted_local_mean_px=meters["pred_local"].avg,
        registered_pred_local_px=meters["reg_pred"].avg,
        global_only_pred_local_px=meters["go_pred"].avg,
        gate_off_mean=meters["gate_off"].avg,
        gate_on_mean=meters["gate_on"].avg,
        gate_scale4=meters["gate4"].avg,
        gate_scale2=meters["gate2"].avg,
        gate_scale1=meters["gate1"].avg,
    )


def parse_args_presence():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gate_presence_init_checkpoint", type=str, required=True)
    parser.add_argument("--train_msi_local_max_px", type=float, default=0.5)
    parser.add_argument("--gate_presence_control_grid", type=int, default=5)
    parser.add_argument("--lambda_local", type=float, default=0.1)
    parser.add_argument("--lambda_zero", type=float, default=0.02)
    parser.add_argument("--lambda_gate", type=float, default=0.05)
    parser.add_argument("--lambda_rot", type=float, default=0.1)
    parser.add_argument("--local_loss_norm_px", type=float, default=0.0)
    parser.add_argument("--rotation_loss_norm_deg", type=float, default=0.0)
    parser.add_argument("--confidence_gate_gain_init", type=float, default=4.0)
    parser.add_argument("--confidence_gate_bias_init", type=float, default=1.5)
    parser.add_argument("--gate_presence_registered_probability", type=float, default=0.2)
    parser.add_argument("--gate_presence_global_only_probability", type=float, default=0.2)
    parser.add_argument("--gate_presence_local_only_probability", type=float, default=0.3)
    parser.add_argument("--gate_presence_global_local_probability", type=float, default=0.3)
    parser.add_argument("--gate_presence_global_downsample", type=int, default=2)
    parser.add_argument("--gate_presence_registered_tolerance_db", type=float, default=0.5)
    parser.add_argument("--gate_presence_eval_seed_offset", type=int, default=15431)
    parser.add_argument("--gate_presence_valid_threshold", type=float, default=0.999)
    parser.add_argument("--gate_presence_save_name", type=str, default="")
    repair, remaining = parser.parse_known_args()
    cfg = parse_args(remaining)

    if cfg.stage != "train":
        raise ValueError("Gate presence repair is train-only; use --stage train")
    if str(cfg.predictor_version).lower() != "v4":
        raise ValueError("Gate presence repair requires --predictor_version v4")
    if str(cfg.msi_ablation).lower() != "raw_direct":
        raise ValueError("Gate presence repair requires --msi_ablation raw_direct")
    if repair.train_msi_local_max_px <= 0.0:
        raise ValueError("--train_msi_local_max_px must be > 0")
    if repair.gate_presence_control_grid < 2:
        raise ValueError("--gate_presence_control_grid must be >= 2")
    if min(
        repair.lambda_local,
        repair.lambda_zero,
        repair.lambda_gate,
        repair.lambda_rot,
    ) < 0.0:
        raise ValueError("all auxiliary loss weights must be >= 0")

    probs = [
        repair.gate_presence_registered_probability,
        repair.gate_presence_global_only_probability,
        repair.gate_presence_local_only_probability,
        repair.gate_presence_global_local_probability,
    ]
    if any(float(p) < 0.0 for p in probs) or abs(sum(float(p) for p in probs) - 1.0) > 1e-6:
        raise ValueError("gate-presence category probabilities must be nonnegative and sum to 1")

    if repair.local_loss_norm_px <= 0.0:
        repair.local_loss_norm_px = float(repair.train_msi_local_max_px)
    if repair.rotation_loss_norm_deg <= 0.0:
        repair.rotation_loss_norm_deg = float(cfg.train_msi_rotation_max_deg)
    if repair.lambda_rot > 0.0 and repair.rotation_loss_norm_deg <= 0.0:
        raise ValueError("Positive lambda_rot requires positive rotation normalization")
    if repair.gate_presence_global_downsample < 1:
        raise ValueError("--gate_presence_global_downsample must be >= 1")

    cfg.alignment_global_feature_downsample = int(repair.gate_presence_global_downsample)
    return cfg, repair


def _paths(cfg, repair):
    root = os.path.join(cfg.checkpoint_root, "innovation1")
    ensure_dir(root)
    if repair.gate_presence_save_name:
        stem = repair.gate_presence_save_name
    else:
        stem = (
            f"{cfg.dataset}_v4_gate_presence"
            f"_d{_compact_float_tag(cfg.train_msi_translation_max_px)}"
            f"_r{_compact_float_tag(cfg.train_msi_rotation_max_deg)}"
            f"_l{_compact_float_tag(repair.train_msi_local_max_px)}"
            f"_loc{_compact_float_tag(repair.lambda_local)}"
            f"_zero{_compact_float_tag(repair.lambda_zero)}"
            f"_gate{_compact_float_tag(repair.lambda_gate)}"
            f"_rot{_compact_float_tag(repair.lambda_rot)}"
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


def run(cfg, repair):
    set_seed(cfg.seed)
    train_loader, test_loader, info = build_loaders(cfg)
    device = get_device(cfg.device)
    process = build_progressive_process(cfg)
    model = _build_model(cfg, info, device, process=process)
    enable_presence_supervised_confidence_aligner(
        model,
        confidence_gain_init=repair.confidence_gate_gain_init,
        confidence_bias_init=repair.confidence_gate_bias_init,
    )

    # This stage is intended to continue from a trained confidence checkpoint;
    # parameter names are identical, so strict loading protects the experiment.
    source_epoch, source_best = load_checkpoint(
        model,
        repair.gate_presence_init_checkpoint,
        optimizer=None,
        strict=True,
        map_location=str(device),
        load_optimizer=False,
    )
    print(
        "Gate-presence warm start: "
        f"{repair.gate_presence_init_checkpoint} "
        f"(epoch={source_epoch}, stored_registered_PSNR={source_best:.6f})"
    )
    print(
        "Loss weights: "
        f"local={repair.lambda_local:g}, zero={repair.lambda_zero:g}, "
        f"gate={repair.lambda_gate:g}, rot={repair.lambda_rot:g}."
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
            "zero_loss", "gate_loss", "local_epe_px", "true_local_px",
            "pred_local_px", "registered_pred_local_px",
            "global_only_pred_local_px", "gate_off_mean", "gate_on_mean",
            "gate_scale4", "gate_scale2", "gate_scale1", "registered_PSNR",
            "registered_SAM", "global_only_PSNR", "global_only_SAM",
            "local_only_PSNR", "local_only_SAM", "global_local_PSNR",
            "global_local_SAM", "robust_PSNR", "selection_score",
            "best_selection_score", "best_registered_PSNR",
        ],
    )

    eval_seed = int(cfg.seed) + int(repair.gate_presence_eval_seed_offset)
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
            local_control_grid=repair.gate_presence_control_grid,
            registered_probability=repair.gate_presence_registered_probability,
            global_only_probability=repair.gate_presence_global_only_probability,
            local_only_probability=repair.gate_presence_local_only_probability,
            global_local_probability=repair.gate_presence_global_local_probability,
            generator=generator,
        )

        print(
            f"Epoch {epoch:04d}/{cfg.epochs:04d} "
            f"loss={stats.loss:.6f} local={stats.local:.6f} "
            f"zero={stats.zero:.6f} gate={stats.gate:.6f} "
            f"true/pred={stats.target_local_mean_px:.4f}/"
            f"{stats.predicted_local_mean_px:.4f}px "
            f"reg/go={stats.registered_pred_local_px:.4f}/"
            f"{stats.global_only_pred_local_px:.4f}px "
            f"gate_off/on={stats.gate_off_mean:.3f}/{stats.gate_on_mean:.3f} "
            f"gate4/2/1={stats.gate_scale4:.3f}/"
            f"{stats.gate_scale2:.3f}/{stats.gate_scale1:.3f}"
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
                control_grid_size=repair.gate_presence_control_grid,
                valid_threshold=repair.gate_presence_valid_threshold,
                seed=eval_seed,
            )
            registered_psnr = float(registered_metrics["PSNR"])
            robust_psnr = (
                float(nonreg_metrics["global_only_PSNR"])
                + float(nonreg_metrics["local_only_PSNR"])
                + float(nonreg_metrics["global_local_PSNR"])
            ) / 3.0
            floor_psnr = float(source_best) - float(
                repair.gate_presence_registered_tolerance_db
            )
            registered_gap = max(0.0, floor_psnr - registered_psnr)
            selection_score = robust_psnr - 5.0 * registered_gap

            print(f"  registered: {_format_metrics(registered_metrics)}")
            print(
                "  presence eval: "
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
                        "source_checkpoint": repair.gate_presence_init_checkpoint,
                    },
                )
                print(f"  saved best gate-presence checkpoint -> {best_path}")

        logger.write(
            {
                "epoch": epoch,
                "loss": stats.loss,
                "l1": stats.l1,
                "sam_loss": stats.sam,
                "rot_loss": stats.rot,
                "local_loss": stats.local,
                "zero_loss": stats.zero,
                "gate_loss": stats.gate,
                "local_epe_px": stats.local_epe_px,
                "true_local_px": stats.target_local_mean_px,
                "pred_local_px": stats.predicted_local_mean_px,
                "registered_pred_local_px": stats.registered_pred_local_px,
                "global_only_pred_local_px": stats.global_only_pred_local_px,
                "gate_off_mean": stats.gate_off_mean,
                "gate_on_mean": stats.gate_on_mean,
                "gate_scale4": stats.gate_scale4,
                "gate_scale2": stats.gate_scale2,
                "gate_scale1": stats.gate_scale1,
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

    print("Gate-presence repair complete.")
    print(f"Best checkpoint: {best_path}")
    print(f"Best registered checkpoint: {best_registered_path}")
    print(f"Last checkpoint: {last_path}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    cfg, repair = parse_args_presence()
    run(cfg, repair)
