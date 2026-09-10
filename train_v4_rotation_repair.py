"""Rotation-repair training stage for learnable Innovation-2 V4 alignment.

This launcher keeps the current V4 architecture and the Innovation-1 physical
trajectory unchanged, but targets the diagnosed global-rotation under-correction:

1. warm-start all V4 weights from an existing trained V4 checkpoint;
2. use a finer global matching descriptor resolution (default downsample=2);
3. supervise the predicted global correction angle with the known synthetic
   augmentation angle:
       L_rot = SmoothL1(theta_corr / r_norm, -theta_aug / r_norm)
4. retain registered samples through the existing global-warp probability.

The stage saves its own checkpoints/logs and does not overwrite the source V4
checkpoint. Use diagnose_alignment_v4_rigid.py after training.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from config import parse_args
from data_loader import build_loaders
from degradations import make_misaligned_msi
from innovation1 import (
    batch_state_at,
    build_progressive_process,
    degradation_consistency_loss,
    evaluate,
    model_predict,
)
from losses import SAMLoss
from main import _build_model, _compact_float_tag, _format_metrics
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
class RotationRepairStats:
    loss: float
    l1: float
    sam: float
    deg: float
    rot: float
    rot_inverse_mae_deg: float
    predicted_abs_rotation_deg: float
    applied_shift_px: float
    applied_abs_rotation_deg: float


def normalized_inverse_rotation_loss(
    predicted_correction_deg: torch.Tensor,
    applied_rotation_deg: torch.Tensor,
    normalization_deg: float,
) -> torch.Tensor:
    """Smooth-L1 supervision for the inverse rotation, normalized by severity."""
    norm = float(normalization_deg)
    if norm <= 0.0:
        raise ValueError("normalization_deg must be > 0")
    if predicted_correction_deg.shape != applied_rotation_deg.shape:
        raise ValueError(
            "predicted_correction_deg and applied_rotation_deg must share shape"
        )
    return F.smooth_l1_loss(
        predicted_correction_deg / norm,
        -applied_rotation_deg / norm,
    )


def augment_global_with_targets(
    hr_msi: torch.Tensor,
    *,
    max_shift_px: float,
    max_rotation_deg: float,
    probability: float,
    generator: Optional[torch.Generator],
) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    """Warp only HR-MSI and return the signed applied rotation per sample."""
    max_shift = float(max_shift_px)
    max_rotation = float(max_rotation_deg)
    prob = float(probability)
    if max_shift < 0.0 or max_rotation < 0.0:
        raise ValueError("augmentation maxima must be >= 0")
    if not 0.0 <= prob <= 1.0:
        raise ValueError("probability must lie in [0,1]")

    batch_size = hr_msi.shape[0]
    zero_rotation = torch.zeros(
        batch_size, device=hr_msi.device, dtype=hr_msi.dtype
    )
    if (max_shift == 0.0 and max_rotation == 0.0) or prob == 0.0:
        return hr_msi, zero_rotation, 0.0, 0.0

    warped, _, params = make_misaligned_msi(
        hr_msi,
        translation_max_px=max_shift,
        rotation_max_deg=max_rotation,
        local_max_displacement_px=0.0,
        generator=generator,
    )
    shift_mag = torch.sqrt(params.dx_px.square() + params.dy_px.square())
    applied_rotation = params.rotation_deg.to(
        device=hr_msi.device, dtype=hr_msi.dtype
    )

    if prob < 1.0:
        use_aug_cpu = (
            torch.rand(
                batch_size,
                generator=generator,
                device="cpu",
                dtype=torch.float32,
            )
            < prob
        )
        use_aug = use_aug_cpu.to(device=hr_msi.device)
        warped = torch.where(
            use_aug[:, None, None, None],
            warped,
            hr_msi,
        )
        factor = use_aug.to(dtype=hr_msi.dtype)
        shift_mag = shift_mag.to(device=hr_msi.device, dtype=hr_msi.dtype) * factor
        applied_rotation = applied_rotation * factor
    else:
        shift_mag = shift_mag.to(device=hr_msi.device, dtype=hr_msi.dtype)

    return (
        warped,
        applied_rotation,
        float(shift_mag.mean().item()),
        float(applied_rotation.abs().mean().item()),
    )


def prepare_v4_alignment_with_rotation(
    model: torch.nn.Module,
    gt_hsi: torch.Tensor,
    hr_msi: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare one differentiable V4 alignment pyramid and expose theta_corr."""
    if not bool(getattr(model, "supports_training_alignment_pyramid", False)):
        raise ValueError("Rotation-repair launcher requires predictor_version=v4")

    (
        global_msi,
        shift,
        predicted_rotation,
        local_cache,
        _,
    ) = model.geometry_aligner.prepare_training_pyramid(gt_hsi, hr_msi)

    # Keep diagnostics detached, while predicted_rotation itself remains in the
    # graph and is used by L_rot below.
    model.last_global_shift_px = shift.detach()
    model.last_global_rotation_deg = predicted_rotation.detach()
    model._training_local_cache = local_cache
    return global_msi, predicted_rotation


def train_rotation_repair_epoch(
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
    rotation_loss_norm_deg: float,
    boundary_probability: float,
    boundary_radius: int,
    grad_clip: float,
    msi_translation_max_px: float,
    msi_rotation_max_deg: float,
    msi_warp_probability: float,
    generator: Optional[torch.Generator],
) -> RotationRepairStats:
    model.train()
    sam_loss_fn = SAMLoss()

    loss_meter = AverageMeter()
    l1_meter = AverageMeter()
    sam_meter = AverageMeter()
    deg_meter = AverageMeter()
    rot_meter = AverageMeter()
    rot_inv_mae_meter = AverageMeter()
    pred_abs_rot_meter = AverageMeter()
    shift_meter = AverageMeter()
    applied_rot_meter = AverageMeter()

    for batch in loader:
        gt = batch["gt"].to(device, non_blocking=True)
        hr_msi = batch["hr_msi"].to(device, non_blocking=True)
        batch_size = gt.shape[0]

        (
            warped_msi,
            applied_rotation_deg,
            mean_shift,
            mean_abs_rotation,
        ) = augment_global_with_targets(
            hr_msi,
            max_shift_px=msi_translation_max_px,
            max_rotation_deg=msi_rotation_max_deg,
            probability=msi_warp_probability,
            generator=generator,
        )

        globally_aligned_msi, predicted_rotation_deg = (
            prepare_v4_alignment_with_rotation(model, gt, warped_msi)
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

        loss = (
            lambda_l1 * l1
            + lambda_sam * sam
            + lambda_deg * deg
            + lambda_rot * rot
        )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite rotation-repair loss; "
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
            inverse_error = (
                predicted_rotation_deg + applied_rotation_deg
            ).abs().mean()
            predicted_abs = predicted_rotation_deg.abs().mean()

        loss_meter.update(loss.item(), batch_size)
        l1_meter.update(l1.item(), batch_size)
        sam_meter.update(sam.item(), batch_size)
        deg_meter.update(deg.item(), batch_size)
        rot_meter.update(rot.item(), batch_size)
        rot_inv_mae_meter.update(inverse_error.item(), batch_size)
        pred_abs_rot_meter.update(predicted_abs.item(), batch_size)
        shift_meter.update(mean_shift, batch_size)
        applied_rot_meter.update(mean_abs_rotation, batch_size)

    return RotationRepairStats(
        loss=loss_meter.avg,
        l1=l1_meter.avg,
        sam=sam_meter.avg,
        deg=deg_meter.avg,
        rot=rot_meter.avg,
        rot_inverse_mae_deg=rot_inv_mae_meter.avg,
        predicted_abs_rotation_deg=pred_abs_rot_meter.avg,
        applied_shift_px=shift_meter.avg,
        applied_abs_rotation_deg=applied_rot_meter.avg,
    )


def parse_rotation_repair_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--rotation_repair_init_checkpoint",
        type=str,
        required=True,
        help="Existing trained V4 checkpoint used as weights-only warm start.",
    )
    parser.add_argument("--lambda_rot", type=float, default=0.1)
    parser.add_argument(
        "--rotation_loss_norm_deg",
        type=float,
        default=0.0,
        help=(
            "Normalization used inside L_rot. 0 means use "
            "--train_msi_rotation_max_deg."
        ),
    )
    parser.add_argument(
        "--rotation_repair_global_downsample",
        type=int,
        default=2,
        help="Global descriptor downsample for the repair stage.",
    )
    parser.add_argument(
        "--rotation_repair_save_name",
        type=str,
        default="",
        help="Optional checkpoint stem for this repair stage.",
    )
    repair, remaining = parser.parse_known_args()
    cfg = parse_args(remaining)

    if cfg.stage != "train":
        raise ValueError("Rotation repair is train-only; use --stage train")
    if str(cfg.predictor_version).lower() != "v4":
        raise ValueError("Rotation repair requires --predictor_version v4")
    if str(cfg.msi_ablation).lower() != "raw_direct":
        raise ValueError("V4 rotation repair requires --msi_ablation raw_direct")
    if repair.lambda_rot < 0.0:
        raise ValueError("--lambda_rot must be >= 0")
    if repair.rotation_repair_global_downsample < 1:
        raise ValueError("--rotation_repair_global_downsample must be >= 1")
    if cfg.train_msi_rotation_max_deg <= 0.0 and repair.lambda_rot > 0.0:
        raise ValueError(
            "Positive --lambda_rot requires --train_msi_rotation_max_deg > 0"
        )

    norm_deg = float(repair.rotation_loss_norm_deg)
    if norm_deg <= 0.0:
        norm_deg = float(cfg.train_msi_rotation_max_deg)
    if norm_deg <= 0.0:
        raise ValueError("Rotation loss normalization must be > 0")
    repair.rotation_loss_norm_deg = norm_deg

    # The only architecture-runtime change in this repair stage.
    cfg.alignment_global_feature_downsample = int(
        repair.rotation_repair_global_downsample
    )
    return cfg, repair


def _repair_paths(cfg, repair):
    root = os.path.join(cfg.checkpoint_root, "innovation1")
    ensure_dir(root)
    if repair.rotation_repair_save_name:
        stem = repair.rotation_repair_save_name
    else:
        stem = (
            f"{cfg.dataset}_v4_rotation_repair"
            f"_d{_compact_float_tag(cfg.train_msi_translation_max_px)}"
            f"_r{_compact_float_tag(cfg.train_msi_rotation_max_deg)}"
            f"_rotsup{_compact_float_tag(repair.lambda_rot)}"
            f"_gds{int(cfg.alignment_global_feature_downsample)}"
        )
        probability = float(cfg.train_msi_translation_probability)
        if abs(probability - 1.0) > 1e-12:
            stem += f"_p{_compact_float_tag(probability)}"
    if stem.endswith(".pth"):
        stem = stem[:-4]
    return (
        os.path.join(root, stem + ".pth"),
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
        repair.rotation_repair_init_checkpoint,
        optimizer=None,
        strict=True,
        map_location=str(device),
        load_optimizer=False,
    )
    print(
        "Rotation-repair warm start: "
        f"{repair.rotation_repair_init_checkpoint} "
        f"(epoch={source_epoch}, best_registered_PSNR={source_best:.6f})"
    )
    print(
        "Repair settings: "
        f"global_feature_downsample={cfg.alignment_global_feature_downsample}, "
        f"lambda_rot={repair.lambda_rot:g}, "
        f"rotation_norm={repair.rotation_loss_norm_deg:g}deg"
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

    best_path, last_path, log_path = _repair_paths(cfg, repair)
    logger = CSVLogger(
        log_path,
        fieldnames=[
            "epoch",
            "loss",
            "l1",
            "sam_loss",
            "deg_loss",
            "rot_loss",
            "rot_inverse_mae_deg",
            "predicted_abs_rotation_deg",
            "applied_shift_px",
            "applied_abs_rotation_deg",
            "PSNR",
            "SAM",
            "RMSE",
            "ERGAS",
            "SSIM",
            "CC",
            "best_PSNR",
        ],
    )

    best_psnr = float("-inf")
    for epoch in range(1, cfg.epochs + 1):
        stats = train_rotation_repair_epoch(
            model,
            train_loader,
            optimizer,
            process,
            device,
            lambda_l1=cfg.lambda_l1,
            lambda_sam=cfg.lambda_sam,
            lambda_deg=cfg.lambda_deg,
            lambda_rot=repair.lambda_rot,
            rotation_loss_norm_deg=repair.rotation_loss_norm_deg,
            boundary_probability=cfg.boundary_probability,
            boundary_radius=cfg.boundary_radius,
            grad_clip=cfg.grad_clip,
            msi_translation_max_px=cfg.train_msi_translation_max_px,
            msi_rotation_max_deg=cfg.train_msi_rotation_max_deg,
            msi_warp_probability=cfg.train_msi_translation_probability,
            generator=generator,
        )

        print(
            f"Epoch {epoch:04d}/{cfg.epochs:04d} "
            f"loss={stats.loss:.6f} l1={stats.l1:.6f} "
            f"sam={stats.sam:.6f} rot={stats.rot:.6f} "
            f"rot_inv_mae={stats.rot_inverse_mae_deg:.4f}deg "
            f"pred|rot|={stats.predicted_abs_rotation_deg:.4f}deg "
            f"aug_shift={stats.applied_shift_px:.4f}px "
            f"aug|rot|={stats.applied_abs_rotation_deg:.4f}deg"
        )

        metrics = {}
        if epoch % cfg.eval_interval == 0 or epoch == cfg.epochs:
            metrics = evaluate(
                model,
                test_loader,
                process,
                device,
                scale_ratio=cfg.scale_ratio,
            )
            print(f"  registered eval: {_format_metrics(metrics)}")
            psnr = float(metrics["PSNR"])
            if psnr > best_psnr:
                best_psnr = psnr
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    best_psnr,
                    best_path,
                    extra={
                        "config": vars(cfg),
                        "repair": vars(repair),
                        "metrics": metrics,
                        "source_checkpoint": repair.rotation_repair_init_checkpoint,
                    },
                )
                print(f"  saved repair best -> {best_path}")

        logger.write(
            {
                "epoch": epoch,
                "loss": stats.loss,
                "l1": stats.l1,
                "sam_loss": stats.sam,
                "deg_loss": stats.deg,
                "rot_loss": stats.rot,
                "rot_inverse_mae_deg": stats.rot_inverse_mae_deg,
                "predicted_abs_rotation_deg": stats.predicted_abs_rotation_deg,
                "applied_shift_px": stats.applied_shift_px,
                "applied_abs_rotation_deg": stats.applied_abs_rotation_deg,
                "PSNR": metrics.get("PSNR", ""),
                "SAM": metrics.get("SAM", ""),
                "RMSE": metrics.get("RMSE", ""),
                "ERGAS": metrics.get("ERGAS", ""),
                "SSIM": metrics.get("SSIM", ""),
                "CC": metrics.get("CC", ""),
                "best_PSNR": best_psnr,
            }
        )

        if epoch % cfg.save_interval == 0 or epoch == cfg.epochs:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best_psnr,
                last_path,
                extra={
                    "config": vars(cfg),
                    "repair": vars(repair),
                    "metrics": metrics,
                    "source_checkpoint": repair.rotation_repair_init_checkpoint,
                },
            )

    print("Rotation repair complete.")
    print(f"Best registered PSNR in repair stage: {best_psnr:.6f}")
    print(f"Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")
    print(f"Log: {log_path}")
    print(
        "Next: run diagnose_alignment_v4_rigid.py on the repair best checkpoint "
        "before increasing curriculum severity."
    )


if __name__ == "__main__":
    cfg, repair = parse_rotation_repair_args()
    run(cfg, repair)
