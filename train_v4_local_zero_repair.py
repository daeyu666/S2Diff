"""Zero-response repair stage for Innovation-2 V4 local alignment.

This stage continues from a trained V4 local-repair checkpoint and addresses the
observed false local motion on samples that contain no non-rigid deformation.
The existing architecture is unchanged.

Training mixture (defaults):
    20% registered    : no global warp, no local warp
    20% global_only   : global rigid warp only
    30% local_only    : smooth local warp only
    30% global_local  : global rigid + smooth local warp

Supervision is deliberately separated:
    L_local : non-zero dense-field supervision on local_only only
    L_zero  : zero-field supervision on registered + global_only only

The synthetic local field is not used as a hard target for global_local because
its coordinate frame is changed by the preceding rigid transform. Those samples
still train the joint system through reconstruction and rotation supervision.

Loss:
    L = lambda_l1 * L1
      + lambda_sam * L_SAM
      + lambda_deg * L_deg
      + lambda_rot * L_rot
      + lambda_local * L_local
      + lambda_zero * L_zero
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
from train_v4_local_repair import (
    normalized_local_field_loss,
    prepare_v4_alignment_with_local,
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


REGISTERED = 0
GLOBAL_ONLY = 1
LOCAL_ONLY = 2
GLOBAL_LOCAL = 3


@dataclass
class ZeroRepairStats:
    loss: float
    l1: float
    sam: float
    deg: float
    rot: float
    local: float
    zero: float
    rot_inverse_mae_deg: float
    local_epe_px: float
    target_local_mean_px: float
    predicted_local_mean_px: float
    registered_pred_local_px: float
    global_only_pred_local_px: float
    registered_fraction: float
    global_only_fraction: float
    local_only_fraction: float
    global_local_fraction: float


def category_masks_from_unit(
    unit: torch.Tensor,
    *,
    registered_probability: float,
    global_only_probability: float,
    local_only_probability: float,
    global_local_probability: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split one U(0,1) draw into the four mutually-exclusive training modes."""
    if unit.ndim != 1:
        raise ValueError("unit must have shape [B]")
    probs = [
        float(registered_probability),
        float(global_only_probability),
        float(local_only_probability),
        float(global_local_probability),
    ]
    if any(p < 0.0 for p in probs):
        raise ValueError("all category probabilities must be >= 0")
    if abs(sum(probs) - 1.0) > 1e-6:
        raise ValueError(f"category probabilities must sum to 1, got {sum(probs):.8f}")

    p0 = probs[0]
    p1 = p0 + probs[1]
    p2 = p1 + probs[2]

    registered = unit < p0
    global_only = (unit >= p0) & (unit < p1)
    local_only = (unit >= p1) & (unit < p2)
    global_local = unit >= p2

    category = torch.full_like(unit, GLOBAL_LOCAL, dtype=torch.long)
    category[registered] = REGISTERED
    category[global_only] = GLOBAL_ONLY
    category[local_only] = LOCAL_ONLY
    return category, registered, global_only, local_only, global_local


def augment_zero_repair_batch(
    hr_msi: torch.Tensor,
    *,
    translation_max_px: float,
    rotation_max_deg: float,
    local_max_displacement_px: float,
    control_grid_size: int,
    registered_probability: float,
    global_only_probability: float,
    local_only_probability: float,
    global_local_probability: float,
    generator: Optional[torch.Generator],
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Dict[str, float],
]:
    """Create registered/global-only/local-only/global+local mixed samples."""
    if hr_msi.ndim != 4:
        raise ValueError("hr_msi must be BxCxHxW")
    if local_max_displacement_px <= 0.0:
        raise ValueError("local_max_displacement_px must be > 0")

    b, _, h, w = hr_msi.shape
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
    category_cpu, reg_cpu, go_cpu, lo_cpu, gl_cpu = category_masks_from_unit(
        unit,
        registered_probability=registered_probability,
        global_only_probability=global_only_probability,
        local_only_probability=local_only_probability,
        global_local_probability=global_local_probability,
    )

    category = category_cpu.to(device=hr_msi.device)
    registered = reg_cpu.to(device=hr_msi.device)
    global_only = go_cpu.to(device=hr_msi.device)
    local_only = lo_cpu.to(device=hr_msi.device)
    global_local = gl_cpu.to(device=hr_msi.device)

    global_mask = (global_only | global_local).to(dtype=hr_msi.dtype)
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

    # Non-zero field target is geometrically exact only for local-only samples.
    local_target = sampled.local_displacement_px
    local_supervision_mask = local_only

    # Registered and global-only samples have exactly zero non-rigid target.
    zero_supervision_mask = registered | global_only

    with torch.no_grad():
        stats = {
            "registered_fraction": float(registered.float().mean().item()),
            "global_only_fraction": float(global_only.float().mean().item()),
            "local_only_fraction": float(local_only.float().mean().item()),
            "global_local_fraction": float(global_local.float().mean().item()),
        }

    return (
        warped,
        rotation,
        local_target,
        local_supervision_mask,
        zero_supervision_mask,
        category,
        stats,
    )


def _field_mean_on_category(
    field: torch.Tensor,
    category: torch.Tensor,
    category_id: int,
) -> float:
    mask = category == int(category_id)
    if not bool(mask.any()):
        return 0.0
    magnitude = torch.linalg.vector_norm(field.detach()[mask].float(), dim=1)
    return float(magnitude.mean().item())


def _local_only_stats(
    predicted: torch.Tensor,
    target: torch.Tensor,
    category: torch.Tensor,
) -> Tuple[float, float, float]:
    mask = category == LOCAL_ONLY
    if not bool(mask.any()):
        return 0.0, 0.0, 0.0
    pred = predicted.detach()[mask].float()
    tgt = target.detach()[mask].float()
    epe = torch.linalg.vector_norm(pred - tgt, dim=1).mean()
    tgt_mag = torch.linalg.vector_norm(tgt, dim=1).mean()
    pred_mag = torch.linalg.vector_norm(pred, dim=1).mean()
    return float(epe.item()), float(tgt_mag.item()), float(pred_mag.item())


def train_zero_repair_epoch(
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
) -> ZeroRepairStats:
    model.train()
    sam_loss_fn = SAMLoss()

    names = (
        "loss", "l1", "sam", "deg", "rot", "local", "zero", "rot_mae",
        "local_epe", "target_local", "pred_local", "reg_pred", "go_pred",
        "registered", "global_only", "local_only", "global_local",
    )
    meters = {name: AverageMeter() for name in names}

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
            aug_stats,
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
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite local-zero-repair loss; "
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
            "loss": loss.item(),
            "l1": l1.item(),
            "sam": sam.item(),
            "deg": deg.item(),
            "rot": rot.item(),
            "local": local.item(),
            "zero": zero.item(),
            "rot_mae": rot_mae.item(),
            "local_epe": local_epe,
            "target_local": target_local_mean,
            "pred_local": pred_local_mean,
            "reg_pred": reg_pred,
            "go_pred": go_pred,
            "registered": aug_stats["registered_fraction"],
            "global_only": aug_stats["global_only_fraction"],
            "local_only": aug_stats["local_only_fraction"],
            "global_local": aug_stats["global_local_fraction"],
        }
        for name, value in values.items():
            meters[name].update(value, batch_size)

    return ZeroRepairStats(
        loss=meters["loss"].avg,
        l1=meters["l1"].avg,
        sam=meters["sam"].avg,
        deg=meters["deg"].avg,
        rot=meters["rot"].avg,
        local=meters["local"].avg,
        zero=meters["zero"].avg,
        rot_inverse_mae_deg=meters["rot_mae"].avg,
        local_epe_px=meters["local_epe"].avg,
        target_local_mean_px=meters["target_local"].avg,
        predicted_local_mean_px=meters["pred_local"].avg,
        registered_pred_local_px=meters["reg_pred"].avg,
        global_only_pred_local_px=meters["go_pred"].avg,
        registered_fraction=meters["registered"].avg,
        global_only_fraction=meters["global_only"].avg,
        local_only_fraction=meters["local_only"].avg,
        global_local_fraction=meters["global_local"].avg,
    )


@torch.no_grad()
def evaluate_nonregistered_scenarios(
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
    """Paired one-trial global-only/local-only/global+local evaluation."""
    model.eval()
    results: Dict[str, float] = {}

    specs = {
        "global_only": (translation_max_px, rotation_max_deg, 0.0),
        "local_only": (0.0, 0.0, local_max_displacement_px),
        "global_local": (translation_max_px, rotation_max_deg, local_max_displacement_px),
    }

    for scenario, (tmax, rmax, lmax) in specs.items():
        psnr_values = []
        sam_values = []
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))

        for batch in loader:
            gt = batch["gt"].to(device, non_blocking=True)
            hr_msi = batch["hr_msi"].to(device, non_blocking=True)
            params = sample_misalignment_parameters(
                int(hr_msi.shape[0]),
                int(hr_msi.shape[-2]),
                int(hr_msi.shape[-1]),
                translation_max_px=float(tmax),
                rotation_max_deg=float(rmax),
                local_max_displacement_px=float(lmax),
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


def parse_zero_repair_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--zero_repair_init_checkpoint",
        type=str,
        required=True,
        help="Existing V4 local-repair checkpoint used as a weights-only warm start.",
    )
    parser.add_argument("--train_msi_local_max_px", type=float, default=0.5)
    parser.add_argument("--zero_repair_control_grid", type=int, default=5)
    parser.add_argument("--lambda_local", type=float, default=0.1)
    parser.add_argument("--lambda_zero", type=float, default=0.2)
    parser.add_argument("--lambda_rot", type=float, default=0.1)
    parser.add_argument("--local_loss_norm_px", type=float, default=0.0)
    parser.add_argument("--rotation_loss_norm_deg", type=float, default=0.0)
    parser.add_argument("--zero_repair_registered_probability", type=float, default=0.2)
    parser.add_argument("--zero_repair_global_only_probability", type=float, default=0.2)
    parser.add_argument("--zero_repair_local_only_probability", type=float, default=0.3)
    parser.add_argument("--zero_repair_global_local_probability", type=float, default=0.3)
    parser.add_argument("--zero_repair_global_downsample", type=int, default=2)
    parser.add_argument("--zero_repair_registered_tolerance_db", type=float, default=0.5)
    parser.add_argument("--zero_repair_eval_seed_offset", type=int, default=15431)
    parser.add_argument("--zero_repair_valid_threshold", type=float, default=0.999)
    parser.add_argument("--zero_repair_save_name", type=str, default="")
    repair, remaining = parser.parse_known_args()
    cfg = parse_args(remaining)

    if cfg.stage != "train":
        raise ValueError("Local zero repair is train-only; use --stage train")
    if str(cfg.predictor_version).lower() != "v4":
        raise ValueError("Local zero repair requires --predictor_version v4")
    if str(cfg.msi_ablation).lower() != "raw_direct":
        raise ValueError("V4 local zero repair requires --msi_ablation raw_direct")
    if repair.train_msi_local_max_px <= 0.0:
        raise ValueError("--train_msi_local_max_px must be > 0")
    if repair.zero_repair_control_grid < 2:
        raise ValueError("--zero_repair_control_grid must be >= 2")
    if min(repair.lambda_local, repair.lambda_zero, repair.lambda_rot) < 0.0:
        raise ValueError("lambda_local/lambda_zero/lambda_rot must be >= 0")
    if repair.zero_repair_global_downsample < 1:
        raise ValueError("--zero_repair_global_downsample must be >= 1")
    if repair.zero_repair_registered_tolerance_db < 0.0:
        raise ValueError("--zero_repair_registered_tolerance_db must be >= 0")
    if not 0.0 < repair.zero_repair_valid_threshold <= 1.0:
        raise ValueError("--zero_repair_valid_threshold must lie in (0,1]")

    probs = [
        repair.zero_repair_registered_probability,
        repair.zero_repair_global_only_probability,
        repair.zero_repair_local_only_probability,
        repair.zero_repair_global_local_probability,
    ]
    if any(float(p) < 0.0 for p in probs) or abs(sum(float(p) for p in probs) - 1.0) > 1e-6:
        raise ValueError("zero-repair category probabilities must be nonnegative and sum to 1")

    local_norm = float(repair.local_loss_norm_px)
    if local_norm <= 0.0:
        local_norm = float(repair.train_msi_local_max_px)
    repair.local_loss_norm_px = local_norm

    rot_norm = float(repair.rotation_loss_norm_deg)
    if rot_norm <= 0.0:
        rot_norm = float(cfg.train_msi_rotation_max_deg)
    if repair.lambda_rot > 0.0 and rot_norm <= 0.0:
        raise ValueError("Positive lambda_rot requires positive rotation normalization")
    repair.rotation_loss_norm_deg = max(rot_norm, 1e-6)

    cfg.alignment_global_feature_downsample = int(repair.zero_repair_global_downsample)
    return cfg, repair


def _paths(cfg, repair):
    root = os.path.join(cfg.checkpoint_root, "innovation1")
    ensure_dir(root)
    if repair.zero_repair_save_name:
        stem = repair.zero_repair_save_name
    else:
        stem = (
            f"{cfg.dataset}_v4_local_zero_repair"
            f"_d{_compact_float_tag(cfg.train_msi_translation_max_px)}"
            f"_r{_compact_float_tag(cfg.train_msi_rotation_max_deg)}"
            f"_l{_compact_float_tag(repair.train_msi_local_max_px)}"
            f"_loc{_compact_float_tag(repair.lambda_local)}"
            f"_zero{_compact_float_tag(repair.lambda_zero)}"
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

    source_epoch, source_best = load_checkpoint(
        model,
        repair.zero_repair_init_checkpoint,
        optimizer=None,
        strict=True,
        map_location=str(device),
        load_optimizer=False,
    )
    print(
        "Local zero-repair warm start: "
        f"{repair.zero_repair_init_checkpoint} "
        f"(epoch={source_epoch}, stored_registered_PSNR={source_best:.6f})"
    )
    print(
        "Mixture: registered/global-only/local-only/global+local="
        f"{repair.zero_repair_registered_probability:.2f}/"
        f"{repair.zero_repair_global_only_probability:.2f}/"
        f"{repair.zero_repair_local_only_probability:.2f}/"
        f"{repair.zero_repair_global_local_probability:.2f}; "
        f"lambda_local={repair.lambda_local:g}, lambda_zero={repair.lambda_zero:g}, "
        f"lambda_rot={repair.lambda_rot:g}"
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
            "epoch", "loss", "l1", "sam_loss", "deg_loss", "rot_loss",
            "local_loss", "zero_loss", "rot_inverse_mae_deg", "local_epe_px",
            "target_local_mean_px", "predicted_local_mean_px",
            "registered_pred_local_px", "global_only_pred_local_px",
            "registered_fraction", "global_only_fraction", "local_only_fraction",
            "global_local_fraction", "registered_PSNR", "registered_SAM",
            "global_only_PSNR", "global_only_SAM", "local_only_PSNR",
            "local_only_SAM", "global_local_PSNR", "global_local_SAM",
            "robust_PSNR", "selection_score", "best_selection_score",
            "best_registered_PSNR",
        ],
    )

    eval_seed = int(cfg.seed) + int(repair.zero_repair_eval_seed_offset)
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
            local_control_grid=repair.zero_repair_control_grid,
            registered_probability=repair.zero_repair_registered_probability,
            global_only_probability=repair.zero_repair_global_only_probability,
            local_only_probability=repair.zero_repair_local_only_probability,
            global_local_probability=repair.zero_repair_global_local_probability,
            generator=generator,
        )

        print(
            f"Epoch {epoch:04d}/{cfg.epochs:04d} "
            f"loss={stats.loss:.6f} l1={stats.l1:.6f} sam={stats.sam:.6f} "
            f"rot={stats.rot:.6f} local={stats.local:.6f} zero={stats.zero:.6f} "
            f"local_epe={stats.local_epe_px:.4f}px "
            f"true_local={stats.target_local_mean_px:.4f}px "
            f"pred_local={stats.predicted_local_mean_px:.4f}px "
            f"reg_pred={stats.registered_pred_local_px:.4f}px "
            f"global_only_pred={stats.global_only_pred_local_px:.4f}px"
        )

        registered_metrics: Dict[str, float] = {}
        scenario_metrics: Dict[str, float] = {}
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
            scenario_metrics = evaluate_nonregistered_scenarios(
                model,
                test_loader,
                process,
                device,
                scale_ratio=cfg.scale_ratio,
                translation_max_px=cfg.train_msi_translation_max_px,
                rotation_max_deg=cfg.train_msi_rotation_max_deg,
                local_max_displacement_px=repair.train_msi_local_max_px,
                control_grid_size=repair.zero_repair_control_grid,
                valid_threshold=repair.zero_repair_valid_threshold,
                seed=eval_seed,
            )

            registered_psnr = float(registered_metrics["PSNR"])
            robust_psnr = float(np.mean([
                scenario_metrics["global_only_PSNR"],
                scenario_metrics["local_only_PSNR"],
                scenario_metrics["global_local_PSNR"],
            ]))
            floor_psnr = float(source_best) - float(
                repair.zero_repair_registered_tolerance_db
            )
            registered_gap = max(0.0, floor_psnr - registered_psnr)
            selection_score = robust_psnr - 5.0 * registered_gap

            print(f"  registered: {_format_metrics(registered_metrics)}")
            print(
                "  zero/local eval: "
                f"global-only={scenario_metrics['global_only_PSNR']:.4f}/"
                f"{scenario_metrics['global_only_SAM']:.4f}; "
                f"local-only={scenario_metrics['local_only_PSNR']:.4f}/"
                f"{scenario_metrics['local_only_SAM']:.4f}; "
                f"global+local={scenario_metrics['global_local_PSNR']:.4f}/"
                f"{scenario_metrics['global_local_SAM']:.4f}; "
                f"robust={robust_psnr:.4f} selection={selection_score:.4f}"
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
                        "scenario_metrics": scenario_metrics,
                        "source_checkpoint": repair.zero_repair_init_checkpoint,
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
                        "scenario_metrics": scenario_metrics,
                        "robust_psnr": robust_psnr,
                        "selection_score": selection_score,
                        "source_checkpoint": repair.zero_repair_init_checkpoint,
                    },
                )
                print(f"  saved best zero/local-robust -> {best_path}")

        logger.write(
            {
                "epoch": epoch,
                "loss": stats.loss,
                "l1": stats.l1,
                "sam_loss": stats.sam,
                "deg_loss": stats.deg,
                "rot_loss": stats.rot,
                "local_loss": stats.local,
                "zero_loss": stats.zero,
                "rot_inverse_mae_deg": stats.rot_inverse_mae_deg,
                "local_epe_px": stats.local_epe_px,
                "target_local_mean_px": stats.target_local_mean_px,
                "predicted_local_mean_px": stats.predicted_local_mean_px,
                "registered_pred_local_px": stats.registered_pred_local_px,
                "global_only_pred_local_px": stats.global_only_pred_local_px,
                "registered_fraction": stats.registered_fraction,
                "global_only_fraction": stats.global_only_fraction,
                "local_only_fraction": stats.local_only_fraction,
                "global_local_fraction": stats.global_local_fraction,
                "registered_PSNR": registered_metrics.get("PSNR", ""),
                "registered_SAM": registered_metrics.get("SAM", ""),
                "global_only_PSNR": scenario_metrics.get("global_only_PSNR", ""),
                "global_only_SAM": scenario_metrics.get("global_only_SAM", ""),
                "local_only_PSNR": scenario_metrics.get("local_only_PSNR", ""),
                "local_only_SAM": scenario_metrics.get("local_only_SAM", ""),
                "global_local_PSNR": scenario_metrics.get("global_local_PSNR", ""),
                "global_local_SAM": scenario_metrics.get("global_local_SAM", ""),
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
                    "scenario_metrics": scenario_metrics,
                    "source_checkpoint": repair.zero_repair_init_checkpoint,
                },
            )

    print("Local zero-response repair complete.")
    print(f"Best zero/local-robust checkpoint: {best_path}")
    print(f"Best registered checkpoint: {best_registered_path}")
    print(f"Last checkpoint: {last_path}")
    print(f"Log: {log_path}")
    print(
        "Next: run diagnose_alignment_v4_local.py on the best zero/local-robust "
        "checkpoint before increasing local curriculum to 1 px."
    )


if __name__ == "__main__":
    cfg, repair = parse_zero_repair_args()
    run(cfg, repair)
