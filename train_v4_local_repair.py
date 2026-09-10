"""Local non-rigid repair stage for Innovation-2 V4 alignment.

This stage starts from the trained global/rotation V4 checkpoint and teaches the
existing 4->2->1 sparse progressive local aligner to recover smooth non-rigid
misregistration without changing the architecture.

Training deliberately mixes three sample types:

    registered   : no global warp, no local warp
    local_only   : local smooth warp only
    global_local : global rigid warp + local smooth warp

The synthetic local displacement field is an exact supervision target only for
``local_only`` (and zero-valued ``registered``) samples. For ``global_local``
samples, a preceding rigid transform changes the local field's coordinate frame,
so the original field is *not* used as a hard local target. Those samples still
train combined robustness through reconstruction loss and global rotation
supervision. This avoids injecting a geometrically inconsistent target.

Loss:

    L = lambda_l1 * L1
      + lambda_sam * L_SAM
      + lambda_deg * L_deg
      + lambda_rot * L_rot
      + lambda_local * L_local

where L_local is a normalized Smooth-L1 field loss on the final scale-1 dense
local offset for exactly-supervisable samples.
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
from degradations import MisalignmentParameters, apply_misalignment, sample_misalignment_parameters
from diagnose_misalignment_translation import calc_masked_psnr_sam
from innovation1 import (
    batch_state_at,
    build_progressive_process,
    degradation_consistency_loss,
    evaluate,
    model_predict,
    reconstruct_from_terminal_lr,
)
from losses import SAMLoss
from main import _build_model, _compact_float_tag, _format_metrics
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


REGISTERED = 0
LOCAL_ONLY = 1
GLOBAL_LOCAL = 2


@dataclass
class LocalRepairStats:
    loss: float
    l1: float
    sam: float
    deg: float
    rot: float
    local: float
    rot_inverse_mae_deg: float
    local_epe_px: float
    target_local_mean_px: float
    predicted_local_mean_px: float
    registered_fraction: float
    local_only_fraction: float
    global_local_fraction: float
    applied_shift_px: float
    applied_abs_rotation_deg: float


def normalized_local_field_loss(
    predicted_offset_px: torch.Tensor,
    target_offset_px: torch.Tensor,
    supervised_mask: torch.Tensor,
    normalization_px: float,
) -> torch.Tensor:
    """Normalized Smooth-L1 loss for exactly-supervisable dense local fields."""
    if predicted_offset_px.shape != target_offset_px.shape:
        raise ValueError("predicted and target local fields must share shape")
    if predicted_offset_px.ndim != 4 or predicted_offset_px.shape[1] != 2:
        raise ValueError("local fields must have shape Bx2xHxW")
    if supervised_mask.ndim != 1 or supervised_mask.shape[0] != predicted_offset_px.shape[0]:
        raise ValueError("supervised_mask must have shape [B]")
    norm = float(normalization_px)
    if norm <= 0.0:
        raise ValueError("normalization_px must be > 0")
    mask = supervised_mask.to(device=predicted_offset_px.device, dtype=torch.bool)
    if not bool(mask.any()):
        return predicted_offset_px.sum() * 0.0
    return F.smooth_l1_loss(
        predicted_offset_px[mask] / norm,
        target_offset_px[mask] / norm,
    )


def category_masks_from_unit(
    unit: torch.Tensor,
    *,
    registered_probability: float,
    global_local_fraction: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map one U(0,1) draw per sample to registered/local-only/global+local."""
    if unit.ndim != 1:
        raise ValueError("unit must have shape [B]")
    p_reg = float(registered_probability)
    p_gl = float(global_local_fraction)
    if not 0.0 <= p_reg < 1.0:
        raise ValueError("registered_probability must lie in [0,1)")
    if not 0.0 <= p_gl <= 1.0:
        raise ValueError("global_local_fraction must lie in [0,1]")

    registered = unit < p_reg
    warped_unit = (unit - p_reg) / max(1.0 - p_reg, 1e-12)
    global_local = (~registered) & (warped_unit < p_gl)
    local_only = (~registered) & (~global_local)

    category = torch.full_like(unit, LOCAL_ONLY, dtype=torch.long)
    category[registered] = REGISTERED
    category[global_local] = GLOBAL_LOCAL
    return category, registered, local_only, global_local


def augment_local_repair_batch(
    hr_msi: torch.Tensor,
    *,
    translation_max_px: float,
    rotation_max_deg: float,
    local_max_displacement_px: float,
    control_grid_size: int,
    registered_probability: float,
    global_local_fraction: float,
    generator: Optional[torch.Generator],
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Dict[str, float],
]:
    """Build mixed registered/local-only/global+local training samples.

    Returns warped MSI, signed applied rotation, local target field, local-loss
    supervision mask, category ids, and scalar augmentation diagnostics.
    """
    if hr_msi.ndim != 4:
        raise ValueError("hr_msi must be BxCxHxW")
    b, _, h, w = hr_msi.shape
    if local_max_displacement_px <= 0.0:
        raise ValueError("local_max_displacement_px must be > 0 for local repair")

    sampled = sample_misalignment_parameters(
        b,
        h,
        w,
        translation_max_px=float(translation_max_px),
        rotation_max_deg=float(rotation_max_deg),
        local_max_displacement_px=float(local_max_displacement_px),
        control_grid_size=int(control_grid_size),
        generator=generator,
        device=hr_msi.device,
        dtype=hr_msi.dtype,
    )

    unit = torch.rand(
        b,
        generator=generator,
        device="cpu",
        dtype=torch.float32,
    )
    category_cpu, registered_cpu, local_only_cpu, global_local_cpu = category_masks_from_unit(
        unit,
        registered_probability=registered_probability,
        global_local_fraction=global_local_fraction,
    )
    category = category_cpu.to(device=hr_msi.device)
    registered = registered_cpu.to(device=hr_msi.device)
    local_only = local_only_cpu.to(device=hr_msi.device)
    global_local = global_local_cpu.to(device=hr_msi.device)

    global_mask = global_local.to(dtype=hr_msi.dtype)
    local_mask = (local_only | global_local).to(dtype=hr_msi.dtype)

    dx = sampled.dx_px * global_mask
    dy = sampled.dy_px * global_mask
    rotation = sampled.rotation_deg * global_mask
    local = sampled.local_displacement_px * local_mask[:, None, None, None]

    effective = MisalignmentParameters(
        dx_px=dx,
        dy_px=dy,
        rotation_deg=rotation,
        local_displacement_px=local,
    )
    warped, _ = apply_misalignment(hr_msi, effective)

    # The original local field is an exact target in local-only coordinates and
    # zero is exact for registered samples. Global+local is excluded because the
    # preceding rigid transform changes the residual field coordinate frame.
    local_supervision_mask = registered | local_only
    local_target = local

    with torch.no_grad():
        shift_mag = torch.sqrt(dx.square() + dy.square())
        local_mag = torch.linalg.vector_norm(local.float(), dim=1)
        stats = {
            "registered_fraction": float(registered.float().mean().item()),
            "local_only_fraction": float(local_only.float().mean().item()),
            "global_local_fraction": float(global_local.float().mean().item()),
            "applied_shift_px": float(shift_mag.mean().item()),
            "applied_abs_rotation_deg": float(rotation.abs().mean().item()),
            "applied_local_mean_px": float(local_mag.mean().item()),
        }

    return (
        warped,
        rotation,
        local_target,
        local_supervision_mask,
        category,
        stats,
    )


def prepare_v4_alignment_with_local(
    model: torch.nn.Module,
    gt_hsi: torch.Tensor,
    hr_msi: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare differentiable V4 pyramid and expose theta_corr + final local field."""
    if not bool(getattr(model, "supports_training_alignment_pyramid", False)):
        raise ValueError("Local-repair launcher requires predictor_version=v4")

    (
        global_msi,
        shift,
        predicted_rotation,
        local_cache,
        _,
    ) = model.geometry_aligner.prepare_training_pyramid(gt_hsi, hr_msi)
    if 1 not in local_cache:
        raise RuntimeError("V4 local pyramid does not contain physical scale 1")

    model.last_global_shift_px = shift.detach()
    model.last_global_rotation_deg = predicted_rotation.detach()
    model._training_local_cache = local_cache
    return global_msi, predicted_rotation, local_cache[1]


def _local_only_field_stats(
    predicted: torch.Tensor,
    target: torch.Tensor,
    category: torch.Tensor,
) -> Tuple[float, float, float]:
    """Return EPE, target magnitude mean and predicted magnitude mean on local-only."""
    mask = category == LOCAL_ONLY
    if not bool(mask.any()):
        return 0.0, 0.0, 0.0
    pred = predicted.detach()[mask].float()
    tgt = target.detach()[mask].float()
    epe = torch.linalg.vector_norm(pred - tgt, dim=1).mean()
    tgt_mag = torch.linalg.vector_norm(tgt, dim=1).mean()
    pred_mag = torch.linalg.vector_norm(pred, dim=1).mean()
    return float(epe.item()), float(tgt_mag.item()), float(pred_mag.item())


def train_local_repair_epoch(
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
    global_local_fraction: float,
    generator: Optional[torch.Generator],
) -> LocalRepairStats:
    model.train()
    sam_loss_fn = SAMLoss()

    meters = {
        name: AverageMeter()
        for name in (
            "loss", "l1", "sam", "deg", "rot", "local", "rot_mae",
            "local_epe", "target_local", "pred_local", "registered",
            "local_only", "global_local", "shift", "rotation"
        )
    }

    for batch in loader:
        gt = batch["gt"].to(device, non_blocking=True)
        hr_msi = batch["hr_msi"].to(device, non_blocking=True)
        batch_size = int(gt.shape[0])

        (
            warped_msi,
            applied_rotation_deg,
            target_local_field,
            local_supervision_mask,
            category,
            aug_stats,
        ) = augment_local_repair_batch(
            hr_msi,
            translation_max_px=translation_max_px,
            rotation_max_deg=rotation_max_deg,
            local_max_displacement_px=local_max_displacement_px,
            control_grid_size=local_control_grid,
            registered_probability=registered_probability,
            global_local_fraction=global_local_fraction,
            generator=generator,
        )

        globally_aligned_msi, predicted_rotation_deg, predicted_local_field = (
            prepare_v4_alignment_with_local(model, gt, warped_msi)
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

        loss = (
            lambda_l1 * l1
            + lambda_sam * sam
            + lambda_deg * deg
            + lambda_rot * rot
            + lambda_local * local
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite local-repair loss; "
                f"timesteps={timesteps.detach().cpu().tolist()}"
            )

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
            rot_mae = (predicted_rotation_deg + applied_rotation_deg).abs().mean()
            local_epe, target_local_mean, predicted_local_mean = _local_only_field_stats(
                predicted_local_field,
                target_local_field,
                category,
            )

        meters["loss"].update(loss.item(), batch_size)
        meters["l1"].update(l1.item(), batch_size)
        meters["sam"].update(sam.item(), batch_size)
        meters["deg"].update(deg.item(), batch_size)
        meters["rot"].update(rot.item(), batch_size)
        meters["local"].update(local.item(), batch_size)
        meters["rot_mae"].update(rot_mae.item(), batch_size)
        meters["local_epe"].update(local_epe, batch_size)
        meters["target_local"].update(target_local_mean, batch_size)
        meters["pred_local"].update(predicted_local_mean, batch_size)
        meters["registered"].update(aug_stats["registered_fraction"], batch_size)
        meters["local_only"].update(aug_stats["local_only_fraction"], batch_size)
        meters["global_local"].update(aug_stats["global_local_fraction"], batch_size)
        meters["shift"].update(aug_stats["applied_shift_px"], batch_size)
        meters["rotation"].update(aug_stats["applied_abs_rotation_deg"], batch_size)

    return LocalRepairStats(
        loss=meters["loss"].avg,
        l1=meters["l1"].avg,
        sam=meters["sam"].avg,
        deg=meters["deg"].avg,
        rot=meters["rot"].avg,
        local=meters["local"].avg,
        rot_inverse_mae_deg=meters["rot_mae"].avg,
        local_epe_px=meters["local_epe"].avg,
        target_local_mean_px=meters["target_local"].avg,
        predicted_local_mean_px=meters["pred_local"].avg,
        registered_fraction=meters["registered"].avg,
        local_only_fraction=meters["local_only"].avg,
        global_local_fraction=meters["global_local"].avg,
        applied_shift_px=meters["shift"].avg,
        applied_abs_rotation_deg=meters["rotation"].avg,
    )


@torch.no_grad()
def evaluate_local_repair_scenarios(
    model: torch.nn.Module,
    loader,
    process,
    device: torch.device,
    *,
    scale_ratio: int,
    translation_max_px: float,
    rotation_max_deg: float,
    local_max_displacement_px: float,
    control_grid_size: int,
    valid_threshold: float,
    seed: int,
) -> Dict[str, float]:
    """Fixed one-trial local-only and global+local valid-overlap evaluation."""
    model.eval()
    results: Dict[str, float] = {}

    for scenario in ("local_only", "global_local"):
        psnr_values = []
        sam_values = []
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))

        for batch in loader:
            gt = batch["gt"].to(device, non_blocking=True)
            hr_msi = batch["hr_msi"].to(device, non_blocking=True)
            tmax = 0.0 if scenario == "local_only" else float(translation_max_px)
            rmax = 0.0 if scenario == "local_only" else float(rotation_max_deg)

            params = sample_misalignment_parameters(
                int(hr_msi.shape[0]),
                int(hr_msi.shape[-2]),
                int(hr_msi.shape[-1]),
                translation_max_px=tmax,
                rotation_max_deg=rmax,
                local_max_displacement_px=float(local_max_displacement_px),
                control_grid_size=int(control_grid_size),
                generator=generator,
                device=hr_msi.device,
                dtype=hr_msi.dtype,
            )
            warped, valid = apply_misalignment(hr_msi, params)
            terminal_lr = process.terminal_observation(gt)
            pred = reconstruct_from_terminal_lr(
                model,
                process,
                terminal_lr,
                target_size=tuple(gt.shape[-2:]),
                hr_msi=warped,
            )
            psnr, sam, _ = calc_masked_psnr_sam(
                pred,
                gt,
                valid,
                threshold=float(valid_threshold),
            )
            psnr_values.append(float(psnr))
            sam_values.append(float(sam))

        results[f"{scenario}_PSNR"] = float(np.mean(psnr_values))
        results[f"{scenario}_SAM"] = float(np.mean(sam_values))

    return results


def parse_local_repair_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--local_repair_init_checkpoint",
        type=str,
        required=True,
        help="Trained V4 global/rotation checkpoint used as a weights-only warm start.",
    )
    parser.add_argument("--train_msi_local_max_px", type=float, default=0.5)
    parser.add_argument("--local_repair_control_grid", type=int, default=5)
    parser.add_argument("--lambda_local", type=float, default=0.1)
    parser.add_argument("--lambda_rot", type=float, default=0.1)
    parser.add_argument(
        "--local_loss_norm_px",
        type=float,
        default=0.0,
        help="Normalization inside L_local; 0 uses --train_msi_local_max_px.",
    )
    parser.add_argument(
        "--rotation_loss_norm_deg",
        type=float,
        default=0.0,
        help="Normalization inside L_rot; 0 uses --train_msi_rotation_max_deg.",
    )
    parser.add_argument(
        "--local_repair_registered_probability",
        type=float,
        default=-1.0,
        help=(
            "Registered-sample probability. Negative means "
            "1 - --train_msi_translation_probability."
        ),
    )
    parser.add_argument(
        "--local_repair_global_local_fraction",
        type=float,
        default=0.5,
        help="Among non-registered samples, fraction using global+local; the rest are local-only.",
    )
    parser.add_argument("--local_repair_global_downsample", type=int, default=2)
    parser.add_argument("--local_repair_registered_tolerance_db", type=float, default=0.75)
    parser.add_argument("--local_repair_eval_seed_offset", type=int, default=15431)
    parser.add_argument("--local_repair_valid_threshold", type=float, default=0.999)
    parser.add_argument("--local_repair_save_name", type=str, default="")
    repair, remaining = parser.parse_known_args()
    cfg = parse_args(remaining)

    if cfg.stage != "train":
        raise ValueError("Local repair is train-only; use --stage train")
    if str(cfg.predictor_version).lower() != "v4":
        raise ValueError("Local repair requires --predictor_version v4")
    if str(cfg.msi_ablation).lower() != "raw_direct":
        raise ValueError("V4 local repair requires --msi_ablation raw_direct")
    if repair.train_msi_local_max_px <= 0.0:
        raise ValueError("--train_msi_local_max_px must be > 0")
    if repair.local_repair_control_grid < 2:
        raise ValueError("--local_repair_control_grid must be >= 2")
    if repair.lambda_local < 0.0 or repair.lambda_rot < 0.0:
        raise ValueError("lambda_local/lambda_rot must be >= 0")
    if repair.local_repair_global_downsample < 1:
        raise ValueError("--local_repair_global_downsample must be >= 1")
    if not 0.0 <= repair.local_repair_global_local_fraction <= 1.0:
        raise ValueError("--local_repair_global_local_fraction must lie in [0,1]")
    if not 0.0 < repair.local_repair_valid_threshold <= 1.0:
        raise ValueError("--local_repair_valid_threshold must lie in (0,1]")
    if repair.local_repair_registered_tolerance_db < 0.0:
        raise ValueError("--local_repair_registered_tolerance_db must be >= 0")

    p_reg = float(repair.local_repair_registered_probability)
    if p_reg < 0.0:
        p_reg = 1.0 - float(cfg.train_msi_translation_probability)
    if not 0.0 <= p_reg < 1.0:
        raise ValueError("Resolved registered probability must lie in [0,1)")
    repair.local_repair_registered_probability = p_reg

    local_norm = float(repair.local_loss_norm_px)
    if local_norm <= 0.0:
        local_norm = float(repair.train_msi_local_max_px)
    if local_norm <= 0.0:
        raise ValueError("Local loss normalization must be > 0")
    repair.local_loss_norm_px = local_norm

    rot_norm = float(repair.rotation_loss_norm_deg)
    if rot_norm <= 0.0:
        rot_norm = float(cfg.train_msi_rotation_max_deg)
    if repair.lambda_rot > 0.0 and rot_norm <= 0.0:
        raise ValueError("Positive lambda_rot requires positive rotation normalization")
    repair.rotation_loss_norm_deg = max(rot_norm, 1e-6)

    cfg.alignment_global_feature_downsample = int(repair.local_repair_global_downsample)
    return cfg, repair


def _repair_paths(cfg, repair):
    root = os.path.join(cfg.checkpoint_root, "innovation1")
    ensure_dir(root)
    if repair.local_repair_save_name:
        stem = repair.local_repair_save_name
    else:
        stem = (
            f"{cfg.dataset}_v4_local_repair"
            f"_d{_compact_float_tag(cfg.train_msi_translation_max_px)}"
            f"_r{_compact_float_tag(cfg.train_msi_rotation_max_deg)}"
            f"_l{_compact_float_tag(repair.train_msi_local_max_px)}"
            f"_locsup{_compact_float_tag(repair.lambda_local)}"
            f"_rotsup{_compact_float_tag(repair.lambda_rot)}"
            f"_mix{_compact_float_tag(repair.local_repair_global_local_fraction)}"
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

    source_epoch, source_best = load_checkpoint(
        model,
        repair.local_repair_init_checkpoint,
        optimizer=None,
        strict=True,
        map_location=str(device),
        load_optimizer=False,
    )
    print(
        "Local-repair warm start: "
        f"{repair.local_repair_init_checkpoint} "
        f"(epoch={source_epoch}, best_registered_PSNR={source_best:.6f})"
    )
    print(
        "Local-repair settings: "
        f"global d<={cfg.train_msi_translation_max_px:g}px, "
        f"r<=±{cfg.train_msi_rotation_max_deg:g}deg, "
        f"local<={repair.train_msi_local_max_px:g}px, "
        f"grid={repair.local_repair_control_grid}x{repair.local_repair_control_grid}, "
        f"p_registered={repair.local_repair_registered_probability:.3f}, "
        f"nonreg_global_local_fraction={repair.local_repair_global_local_fraction:.3f}, "
        f"lambda_local={repair.lambda_local:g}, lambda_rot={repair.lambda_rot:g}, "
        f"global_downsample={cfg.alignment_global_feature_downsample}"
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        int(cfg.seed) + int(getattr(cfg, "train_misalignment_seed_offset", 7919))
    )

    best_path, best_registered_path, last_path, log_path = _repair_paths(cfg, repair)
    logger = CSVLogger(
        log_path,
        fieldnames=[
            "epoch", "loss", "l1", "sam_loss", "deg_loss", "rot_loss",
            "local_loss", "rot_inverse_mae_deg", "local_epe_px",
            "target_local_mean_px", "predicted_local_mean_px",
            "registered_fraction", "local_only_fraction", "global_local_fraction",
            "applied_shift_px", "applied_abs_rotation_deg", "registered_PSNR",
            "registered_SAM", "local_only_PSNR", "local_only_SAM",
            "global_local_PSNR", "global_local_SAM", "robust_PSNR",
            "selection_score", "best_selection_score", "best_registered_PSNR",
        ],
    )

    eval_seed = int(cfg.seed) + int(repair.local_repair_eval_seed_offset)
    best_selection_score = float("-inf")
    best_registered_psnr = float("-inf")

    for epoch in range(1, cfg.epochs + 1):
        stats = train_local_repair_epoch(
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
            rotation_loss_norm_deg=repair.rotation_loss_norm_deg,
            local_loss_norm_px=repair.local_loss_norm_px,
            boundary_probability=cfg.boundary_probability,
            boundary_radius=cfg.boundary_radius,
            grad_clip=cfg.grad_clip,
            translation_max_px=cfg.train_msi_translation_max_px,
            rotation_max_deg=cfg.train_msi_rotation_max_deg,
            local_max_displacement_px=repair.train_msi_local_max_px,
            local_control_grid=repair.local_repair_control_grid,
            registered_probability=repair.local_repair_registered_probability,
            global_local_fraction=repair.local_repair_global_local_fraction,
            generator=generator,
        )

        print(
            f"Epoch {epoch:04d}/{cfg.epochs:04d} "
            f"loss={stats.loss:.6f} l1={stats.l1:.6f} sam={stats.sam:.6f} "
            f"rot={stats.rot:.6f} local={stats.local:.6f} "
            f"local_epe={stats.local_epe_px:.4f}px "
            f"true_local={stats.target_local_mean_px:.4f}px "
            f"pred_local={stats.predicted_local_mean_px:.4f}px "
            f"rot_inv_mae={stats.rot_inverse_mae_deg:.4f}deg"
        )

        registered_metrics: Dict[str, float] = {}
        repair_metrics: Dict[str, float] = {}
        robust_psnr = float("nan")
        selection_score = float("nan")

        if epoch % cfg.eval_interval == 0 or epoch == cfg.epochs:
            registered_metrics = evaluate(
                model,
                test_loader,
                process,
                device,
                scale_ratio=cfg.scale_ratio,
            )
            repair_metrics = evaluate_local_repair_scenarios(
                model,
                test_loader,
                process,
                device,
                scale_ratio=cfg.scale_ratio,
                translation_max_px=cfg.train_msi_translation_max_px,
                rotation_max_deg=cfg.train_msi_rotation_max_deg,
                local_max_displacement_px=repair.train_msi_local_max_px,
                control_grid_size=repair.local_repair_control_grid,
                valid_threshold=repair.local_repair_valid_threshold,
                seed=eval_seed,
            )
            registered_psnr = float(registered_metrics["PSNR"])
            robust_psnr = 0.5 * (
                float(repair_metrics["local_only_PSNR"])
                + float(repair_metrics["global_local_PSNR"])
            )
            floor_psnr = float(source_best) - float(
                repair.local_repair_registered_tolerance_db
            )
            registered_gap = max(0.0, floor_psnr - registered_psnr)
            selection_score = robust_psnr - 5.0 * registered_gap

            print(f"  registered: {_format_metrics(registered_metrics)}")
            print(
                "  local repair eval: "
                f"local-only PSNR={repair_metrics['local_only_PSNR']:.4f} "
                f"SAM={repair_metrics['local_only_SAM']:.4f}; "
                f"global+local PSNR={repair_metrics['global_local_PSNR']:.4f} "
                f"SAM={repair_metrics['global_local_SAM']:.4f}; "
                f"robust_PSNR={robust_psnr:.4f} selection={selection_score:.4f}"
            )

            if registered_psnr > best_registered_psnr:
                best_registered_psnr = registered_psnr
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    best_registered_psnr,
                    best_registered_path,
                    extra={
                        "config": vars(cfg),
                        "repair": vars(repair),
                        "registered_metrics": registered_metrics,
                        "repair_metrics": repair_metrics,
                        "source_checkpoint": repair.local_repair_init_checkpoint,
                    },
                )
                print(f"  saved best registered -> {best_registered_path}")

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
                        "repair_metrics": repair_metrics,
                        "robust_psnr": robust_psnr,
                        "selection_score": selection_score,
                        "source_checkpoint": repair.local_repair_init_checkpoint,
                    },
                )
                print(f"  saved best local-robust checkpoint -> {best_path}")

        logger.write(
            {
                "epoch": epoch,
                "loss": stats.loss,
                "l1": stats.l1,
                "sam_loss": stats.sam,
                "deg_loss": stats.deg,
                "rot_loss": stats.rot,
                "local_loss": stats.local,
                "rot_inverse_mae_deg": stats.rot_inverse_mae_deg,
                "local_epe_px": stats.local_epe_px,
                "target_local_mean_px": stats.target_local_mean_px,
                "predicted_local_mean_px": stats.predicted_local_mean_px,
                "registered_fraction": stats.registered_fraction,
                "local_only_fraction": stats.local_only_fraction,
                "global_local_fraction": stats.global_local_fraction,
                "applied_shift_px": stats.applied_shift_px,
                "applied_abs_rotation_deg": stats.applied_abs_rotation_deg,
                "registered_PSNR": registered_metrics.get("PSNR", ""),
                "registered_SAM": registered_metrics.get("SAM", ""),
                "local_only_PSNR": repair_metrics.get("local_only_PSNR", ""),
                "local_only_SAM": repair_metrics.get("local_only_SAM", ""),
                "global_local_PSNR": repair_metrics.get("global_local_PSNR", ""),
                "global_local_SAM": repair_metrics.get("global_local_SAM", ""),
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
                    "registered_metrics": registered_metrics,
                    "repair_metrics": repair_metrics,
                    "source_checkpoint": repair.local_repair_init_checkpoint,
                },
            )

    print("Local repair complete.")
    print(f"Best local-robust checkpoint: {best_path}")
    print(f"Best registered checkpoint: {best_registered_path}")
    print(f"Last checkpoint: {last_path}")
    print(f"Log: {log_path}")
    print(
        "Next: run diagnose_alignment_v4_local.py on the best local-robust "
        "checkpoint before increasing local curriculum severity."
    )


if __name__ == "__main__":
    cfg, repair = parse_local_repair_args()
    run(cfg, repair)
