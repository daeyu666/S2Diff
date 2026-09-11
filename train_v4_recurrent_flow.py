"""Train the physical-domain recurrent residual-flow local alignment branch.

This is the replacement experiment for the failed local soft-expectation /
confidence-gate / sub-pixel-head line.  The already validated V4 Raw-Direct
backbone and global rigid correction are warm-started from an existing V4
checkpoint.  By default only the new recurrent local branch is trainable.

Synthetic training mixture:
    registered   : global=0, local=0       -> flow target 0
    global_only  : global!=0, local=0      -> flow target 0
    local_only   : global=0, local!=0      -> exact local-flow target
    global_local : global!=0, local!=0     -> reconstruction only for local flow

The same sequence-flow loss supervises zero and non-zero exact targets.  There
is no separate hard zero-response loss, no confidence gate, and no sub-pixel
MLP.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from config import parse_args
from data_loader import build_loaders
from degradations import (
    MisalignmentParameters,
    apply_misalignment,
    sample_misalignment_parameters,
)
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
from main import _compact_float_tag, _format_metrics
from models.predictor_v4_recurrent_flow import StateMatchedRecurrentFlowPredictor
from train_v4_rotation_repair import normalized_inverse_rotation_loss
from utils import (
    AverageMeter,
    CSVLogger,
    count_parameters,
    ensure_dir,
    get_device,
    save_checkpoint,
    set_seed,
)

REGISTERED = 0
GLOBAL_ONLY = 1
LOCAL_ONLY = 2
GLOBAL_LOCAL = 3


@dataclass
class RecurrentFlowStats:
    loss: float
    l1: float
    sam: float
    deg: float
    rot: float
    flow: float
    smooth: float
    local_epe_px: float
    target_local_mean_px: float
    predicted_local_mean_px: float
    zero_predicted_mean_px: float
    applied_shift_px: float
    applied_abs_rotation_deg: float
    applied_local_mean_px: float


def category_masks(
    unit: torch.Tensor,
    *,
    registered_probability: float,
    global_only_probability: float,
    local_only_probability: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split one U(0,1) draw into four mutually exclusive categories."""
    probs = [
        float(registered_probability),
        float(global_only_probability),
        float(local_only_probability),
    ]
    if any(p < 0.0 for p in probs) or sum(probs) > 1.0 + 1e-8:
        raise ValueError("category probabilities must be non-negative and sum <= 1")
    p0, p1, p2 = probs
    registered = unit < p0
    global_only = (unit >= p0) & (unit < p0 + p1)
    local_only = (unit >= p0 + p1) & (unit < p0 + p1 + p2)
    global_local = ~(registered | global_only | local_only)
    category = torch.full_like(unit, GLOBAL_LOCAL, dtype=torch.long)
    category[registered] = REGISTERED
    category[global_only] = GLOBAL_ONLY
    category[local_only] = LOCAL_ONLY
    return category, registered, global_only, local_only, global_local


def augment_recurrent_flow_batch(
    hr_msi: torch.Tensor,
    *,
    translation_max_px: float,
    rotation_max_deg: float,
    local_max_displacement_px: float,
    control_grid_size: int,
    registered_probability: float,
    global_only_probability: float,
    local_only_probability: float,
    generator: Optional[torch.Generator],
):
    """Generate four-way misalignment mixture and exact flow targets where valid."""
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
    unit = torch.rand(b, generator=generator, device="cpu")
    cat_cpu, reg_cpu, go_cpu, lo_cpu, gl_cpu = category_masks(
        unit,
        registered_probability=registered_probability,
        global_only_probability=global_only_probability,
        local_only_probability=local_only_probability,
    )
    category = cat_cpu.to(hr_msi.device)
    registered = reg_cpu.to(hr_msi.device)
    global_only = go_cpu.to(hr_msi.device)
    local_only = lo_cpu.to(hr_msi.device)
    global_local = gl_cpu.to(hr_msi.device)

    global_mask = (global_only | global_local).to(hr_msi.dtype)
    local_mask = (local_only | global_local).to(hr_msi.dtype)

    dx = sampled.dx_px * global_mask
    dy = sampled.dy_px * global_mask
    rotation = sampled.rotation_deg * global_mask
    local = sampled.local_displacement_px * local_mask[:, None, None, None]
    params = MisalignmentParameters(dx, dy, rotation, local)
    warped, valid = apply_misalignment(hr_msi, params)

    # One unified flow target: exact zeros for registered/global-only and the
    # exact synthetic field for local-only.  Global+local is excluded because
    # the preceding rigid warp changes the local field coordinate frame.
    flow_supervised = registered | global_only | local_only
    flow_target = local

    with torch.no_grad():
        shift = torch.sqrt(dx.square() + dy.square())
        local_mag = torch.linalg.vector_norm(local.float(), dim=1)
        stats = {
            "shift": float(shift.mean().item()),
            "rotation": float(rotation.abs().mean().item()),
            "local": float(local_mag.mean().item()),
            "registered_fraction": float(registered.float().mean().item()),
            "global_only_fraction": float(global_only.float().mean().item()),
            "local_only_fraction": float(local_only.float().mean().item()),
            "global_local_fraction": float(global_local.float().mean().item()),
        }
    return (
        warped,
        valid,
        rotation,
        flow_target,
        flow_supervised,
        category,
        stats,
    )


def sequence_flow_loss(
    sequence_by_scale: Dict[int, List[torch.Tensor]],
    target_flow: torch.Tensor,
    supervised_mask: torch.Tensor,
    *,
    normalization_px: float,
    gamma: float = 0.8,
) -> torch.Tensor:
    """RAFT-style discounted loss over all 4->2->1 recurrent predictions."""
    norm = float(normalization_px)
    if norm <= 0.0:
        raise ValueError("normalization_px must be > 0")
    if not 0.0 < float(gamma) <= 1.0:
        raise ValueError("gamma must lie in (0,1]")
    mask = supervised_mask.to(device=target_flow.device, dtype=torch.bool)
    predictions: List[torch.Tensor] = []
    for scale in sorted(sequence_by_scale.keys(), reverse=True):
        predictions.extend(sequence_by_scale[scale])
    if not predictions:
        raise ValueError("empty recurrent flow sequence")
    if not bool(mask.any()):
        return predictions[-1].sum() * 0.0

    total = predictions[-1].new_zeros(())
    weight_sum = 0.0
    n = len(predictions)
    for index, pred in enumerate(predictions):
        weight = float(gamma) ** float(n - index - 1)
        total = total + weight * F.smooth_l1_loss(
            pred[mask] / norm,
            target_flow[mask] / norm,
        )
        weight_sum += weight
    return total / max(weight_sum, 1e-12)


def smooth_flow_loss(flow: torch.Tensor, normalization_px: float) -> torch.Tensor:
    """Weak total-variation regularization; does not directly shrink flow amplitude."""
    norm = max(float(normalization_px), 1e-6)
    dx = (flow[:, :, :, 1:] - flow[:, :, :, :-1]).abs().mean()
    dy = (flow[:, :, 1:, :] - flow[:, :, :-1, :]).abs().mean()
    return (dx + dy) / norm


def build_recurrent_model(cfg, info, device, process):
    if info.get("srf_weights") is None:
        raise ValueError("Recurrent V4 requires SRF weights; use --msi_mode srf")
    model = StateMatchedRecurrentFlowPredictor(
        n_bands=int(info["n_bands"]),
        n_msi_bands=int(info["n_select_bands"]),
        total_steps=int(cfg.diffusion_steps),
        base_channels=int(cfg.predictor_base_channels),
        time_dim=int(cfg.predictor_time_dim),
        dropout=float(cfg.predictor_dropout),
        residual_prediction=True,
        spectral_hidden=int(getattr(cfg, "spectral_stem_hidden", 8)),
        msi_highpass_kernel=int(getattr(cfg, "msi_highpass_kernel", 5)),
        msi_highpass_sigma=float(getattr(cfg, "msi_highpass_sigma", 1.0)),
        progressive_process=process,
        srf_weights=torch.as_tensor(info["srf_weights"], dtype=torch.float32),
        alignment_descriptor_channels=int(cfg.alignment_descriptor_channels),
        alignment_global_search_radius=int(cfg.alignment_global_search_radius),
        alignment_global_rotation_max_deg=float(cfg.alignment_global_rotation_max_deg),
        alignment_global_rotation_step_deg=float(cfg.alignment_global_rotation_step_deg),
        alignment_global_feature_downsample=int(cfg.alignment_global_feature_downsample),
        alignment_global_candidate_chunk=int(cfg.alignment_global_candidate_chunk),
        alignment_control_stride=int(cfg.alignment_control_stride),
        alignment_local_radius_scale1=int(cfg.alignment_local_radius_scale1),
        alignment_local_radius_scale2=int(cfg.alignment_local_radius_scale2),
        alignment_local_radius_scale4=int(cfg.alignment_local_radius_scale4),
        recurrent_hidden_channels=int(cfg.recurrent_hidden_channels),
        recurrent_correlation_channels=int(cfg.recurrent_correlation_channels),
        recurrent_iterations_scale1=int(cfg.recurrent_iterations_scale1),
        recurrent_iterations_scale2=int(cfg.recurrent_iterations_scale2),
        recurrent_iterations_scale4=int(cfg.recurrent_iterations_scale4),
        recurrent_max_update_scale1=float(cfg.recurrent_max_update_scale1),
        recurrent_max_update_scale2=float(cfg.recurrent_max_update_scale2),
        recurrent_max_update_scale4=float(cfg.recurrent_max_update_scale4),
    ).to(device)
    return model


def warm_start_compatible(model: torch.nn.Module, path: str, device: torch.device):
    """Load every shape-compatible V4 weight and leave recurrent heads fresh."""
    try:
        state = torch.load(path, map_location=str(device), weights_only=False)
    except TypeError:
        state = torch.load(path, map_location=str(device))
    source = state.get("model", state)
    target = model.state_dict()
    compatible = {
        key: value
        for key, value in source.items()
        if key in target and tuple(value.shape) == tuple(target[key].shape)
    }
    skipped = [key for key in source if key not in compatible]
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    print(
        f"Warm start compatible tensors={len(compatible)}, "
        f"source skipped={len(skipped)}, new/missing={len(missing)}, "
        f"unexpected={len(unexpected)}"
    )
    return int(state.get("epoch", 0)), float(state.get("best_metric", 0.0))


def set_train_scope(model: torch.nn.Module, scope: str) -> None:
    scope = str(scope).lower()
    if scope not in {"local", "all"}:
        raise ValueError("train scope must be local or all")
    if scope == "all":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        return
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.geometry_aligner.local_aligner.parameters():
        parameter.requires_grad_(True)


def prepare_recurrent_alignment(model, gt, warped_msi):
    (
        global_msi,
        shift,
        rotation,
        local_cache,
        _,
        sequence_cache,
    ) = model.geometry_aligner.prepare_training_pyramid(gt, warped_msi)
    model.last_global_shift_px = shift.detach()
    model.last_global_rotation_deg = rotation.detach()
    model._training_local_cache = local_cache
    model._training_sequence_cache = sequence_cache
    return global_msi, rotation, local_cache[1], sequence_cache


def _field_stats(pred, target, category):
    with torch.no_grad():
        local_mask = category == LOCAL_ONLY
        zero_mask = (category == REGISTERED) | (category == GLOBAL_ONLY)
        if bool(local_mask.any()):
            p = pred[local_mask].float()
            t = target[local_mask].float()
            epe = torch.linalg.vector_norm(p - t, dim=1).mean().item()
            pred_mag = torch.linalg.vector_norm(p, dim=1).mean().item()
            tgt_mag = torch.linalg.vector_norm(t, dim=1).mean().item()
        else:
            epe = pred_mag = tgt_mag = 0.0
        if bool(zero_mask.any()):
            zero_mag = torch.linalg.vector_norm(
                pred[zero_mask].float(), dim=1
            ).mean().item()
        else:
            zero_mag = 0.0
    return float(epe), float(tgt_mag), float(pred_mag), float(zero_mag)


def train_epoch(
    model,
    loader,
    optimizer,
    process,
    device,
    cfg,
    repair,
    generator,
):
    model.train()
    sam_fn = SAMLoss()
    meters = {name: AverageMeter() for name in (
        "loss", "l1", "sam", "deg", "rot", "flow", "smooth",
        "epe", "target", "pred", "zero", "shift", "rotation", "local"
    )}

    for batch in loader:
        gt = batch["gt"].to(device, non_blocking=True)
        hr_msi = batch["hr_msi"].to(device, non_blocking=True)
        b = int(gt.shape[0])
        (
            warped,
            _,
            applied_rotation,
            target_flow,
            supervised_flow,
            category,
            aug,
        ) = augment_recurrent_flow_batch(
            hr_msi,
            translation_max_px=cfg.train_msi_translation_max_px,
            rotation_max_deg=cfg.train_msi_rotation_max_deg,
            local_max_displacement_px=repair.train_msi_local_max_px,
            control_grid_size=repair.recurrent_synthetic_control_grid,
            registered_probability=repair.recurrent_registered_probability,
            global_only_probability=repair.recurrent_global_only_probability,
            local_only_probability=repair.recurrent_local_only_probability,
            generator=generator,
        )

        global_msi, predicted_rotation, final_flow, flow_sequence = (
            prepare_recurrent_alignment(model, gt, warped)
        )
        timesteps = process.sample_timesteps(
            b,
            boundary_probability=cfg.boundary_probability,
            boundary_radius=cfg.boundary_radius,
            device=device,
        )
        with torch.no_grad():
            x_t = batch_state_at(process, gt, timesteps)
        pred_x0 = model_predict(model, x_t, timesteps, hr_msi=global_msi)

        l1 = F.l1_loss(pred_x0, gt)
        sam = sam_fn(pred_x0, gt)
        deg = (
            degradation_consistency_loss(process, pred_x0, gt, timesteps)
            if cfg.lambda_deg > 0.0
            else pred_x0.new_zeros(())
        )
        rot = (
            normalized_inverse_rotation_loss(
                predicted_rotation,
                applied_rotation,
                normalization_deg=max(cfg.train_msi_rotation_max_deg, 1e-6),
            )
            if repair.lambda_rot > 0.0
            else pred_x0.new_zeros(())
        )
        flow = sequence_flow_loss(
            flow_sequence,
            target_flow,
            supervised_flow,
            normalization_px=repair.flow_loss_norm_px,
            gamma=repair.flow_sequence_gamma,
        )
        smooth = smooth_flow_loss(final_flow, repair.flow_loss_norm_px)
        loss = (
            cfg.lambda_l1 * l1
            + cfg.lambda_sam * sam
            + cfg.lambda_deg * deg
            + repair.lambda_rot * rot
            + repair.lambda_flow * flow
            + repair.lambda_flow_smooth * smooth
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite recurrent-flow training loss")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=float(cfg.grad_clip) if cfg.grad_clip > 0 else float("inf"),
            error_if_nonfinite=True,
        )
        optimizer.step()

        epe, target_mag, pred_mag, zero_mag = _field_stats(
            final_flow, target_flow, category
        )
        values = {
            "loss": loss.item(), "l1": l1.item(), "sam": sam.item(),
            "deg": deg.item(), "rot": rot.item(), "flow": flow.item(),
            "smooth": smooth.item(), "epe": epe, "target": target_mag,
            "pred": pred_mag, "zero": zero_mag, "shift": aug["shift"],
            "rotation": aug["rotation"], "local": aug["local"],
        }
        for name, value in values.items():
            meters[name].update(value, b)

    return RecurrentFlowStats(
        loss=meters["loss"].avg,
        l1=meters["l1"].avg,
        sam=meters["sam"].avg,
        deg=meters["deg"].avg,
        rot=meters["rot"].avg,
        flow=meters["flow"].avg,
        smooth=meters["smooth"].avg,
        local_epe_px=meters["epe"].avg,
        target_local_mean_px=meters["target"].avg,
        predicted_local_mean_px=meters["pred"].avg,
        zero_predicted_mean_px=meters["zero"].avg,
        applied_shift_px=meters["shift"].avg,
        applied_abs_rotation_deg=meters["rotation"].avg,
        applied_local_mean_px=meters["local"].avg,
    )


@torch.no_grad()
def evaluate_scenarios(
    model,
    loader,
    process,
    device,
    *,
    scale_ratio: int,
    translation_max_px: float,
    rotation_max_deg: float,
    local_max_displacement_px: float,
    control_grid_size: int,
    valid_threshold: float,
    seed: int,
):
    model.eval()
    results = {}
    for scenario in ("global_only", "local_only", "global_local"):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        psnrs, sams, predicted_local = [], [], []
        true_local = []
        for batch in loader:
            gt = batch["gt"].to(device, non_blocking=True)
            msi = batch["hr_msi"].to(device, non_blocking=True)
            use_global = scenario in {"global_only", "global_local"}
            use_local = scenario in {"local_only", "global_local"}
            params = sample_misalignment_parameters(
                int(msi.shape[0]), int(msi.shape[-2]), int(msi.shape[-1]),
                translation_max_px=float(translation_max_px) if use_global else 0.0,
                rotation_max_deg=float(rotation_max_deg) if use_global else 0.0,
                local_max_displacement_px=float(local_max_displacement_px) if use_local else 0.0,
                control_grid_size=int(control_grid_size),
                generator=generator,
                device=msi.device,
                dtype=msi.dtype,
            )
            warped, valid = apply_misalignment(msi, params)
            terminal_lr = process.terminal_observation(gt)
            pred = reconstruct_from_terminal_lr(
                model,
                process,
                terminal_lr,
                target_size=tuple(gt.shape[-2:]),
                hr_msi=warped,
            )
            psnr, sam, _ = calc_masked_psnr_sam(
                pred, gt, valid, threshold=float(valid_threshold)
            )
            psnrs.append(float(psnr)); sams.append(float(sam))
            flow = model._inference_local_offset
            predicted_local.append(
                float(torch.linalg.vector_norm(flow.float(), dim=1).mean().item())
            )
            true_local.append(
                float(torch.linalg.vector_norm(
                    params.local_displacement_px.float(), dim=1
                ).mean().item())
            )
        results[f"{scenario}_PSNR"] = float(np.mean(psnrs))
        results[f"{scenario}_SAM"] = float(np.mean(sams))
        results[f"{scenario}_pred_local"] = float(np.mean(predicted_local))
        results[f"{scenario}_true_local"] = float(np.mean(true_local))
    return results


def parse_repair_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--recurrent_init_checkpoint", type=str, required=True)
    parser.add_argument("--train_msi_local_max_px", type=float, default=1.0)
    parser.add_argument("--recurrent_synthetic_control_grid", type=int, default=5)
    parser.add_argument("--recurrent_registered_probability", type=float, default=0.20)
    parser.add_argument("--recurrent_global_only_probability", type=float, default=0.20)
    parser.add_argument("--recurrent_local_only_probability", type=float, default=0.30)
    parser.add_argument("--lambda_flow", type=float, default=1.0)
    parser.add_argument("--lambda_flow_smooth", type=float, default=0.01)
    parser.add_argument("--lambda_rot", type=float, default=0.0)
    parser.add_argument("--flow_sequence_gamma", type=float, default=0.8)
    parser.add_argument("--flow_loss_norm_px", type=float, default=0.0)
    parser.add_argument("--recurrent_hidden_channels", type=int, default=64)
    parser.add_argument("--recurrent_correlation_channels", type=int, default=32)
    parser.add_argument("--recurrent_iterations_scale1", type=int, default=2)
    parser.add_argument("--recurrent_iterations_scale2", type=int, default=2)
    parser.add_argument("--recurrent_iterations_scale4", type=int, default=3)
    parser.add_argument("--recurrent_max_update_scale1", type=float, default=0.5)
    parser.add_argument("--recurrent_max_update_scale2", type=float, default=1.0)
    parser.add_argument("--recurrent_max_update_scale4", type=float, default=2.0)
    parser.add_argument("--recurrent_train_scope", choices=["local", "all"], default="local")
    parser.add_argument("--recurrent_valid_threshold", type=float, default=0.999)
    parser.add_argument("--recurrent_registered_tolerance_db", type=float, default=0.75)
    parser.add_argument("--recurrent_eval_seed_offset", type=int, default=20831)
    parser.add_argument("--recurrent_save_name", type=str, default="")
    repair, remaining = parser.parse_known_args()
    cfg = parse_args(remaining)

    if cfg.stage != "train" or str(cfg.predictor_version).lower() != "v4":
        raise ValueError("Use --stage train --predictor_version v4")
    if repair.train_msi_local_max_px <= 0.0:
        raise ValueError("train_msi_local_max_px must be > 0")
    if repair.flow_loss_norm_px <= 0.0:
        repair.flow_loss_norm_px = float(repair.train_msi_local_max_px)
    if repair.lambda_flow <= 0.0 or repair.lambda_flow_smooth < 0.0:
        raise ValueError("invalid flow loss weights")
    if not 0.0 < repair.flow_sequence_gamma <= 1.0:
        raise ValueError("flow_sequence_gamma must lie in (0,1]")
    probability_sum = (
        repair.recurrent_registered_probability
        + repair.recurrent_global_only_probability
        + repair.recurrent_local_only_probability
    )
    if probability_sum > 1.0 + 1e-8 or min(
        repair.recurrent_registered_probability,
        repair.recurrent_global_only_probability,
        repair.recurrent_local_only_probability,
    ) < 0.0:
        raise ValueError("invalid recurrent category probabilities")

    for name in (
        "recurrent_hidden_channels", "recurrent_correlation_channels",
        "recurrent_iterations_scale1", "recurrent_iterations_scale2",
        "recurrent_iterations_scale4", "recurrent_max_update_scale1",
        "recurrent_max_update_scale2", "recurrent_max_update_scale4",
    ):
        setattr(cfg, name, getattr(repair, name))
    return cfg, repair


def paths(cfg, repair):
    root = os.path.join(cfg.checkpoint_root, "innovation1")
    ensure_dir(root)
    if repair.recurrent_save_name:
        stem = repair.recurrent_save_name.removesuffix(".pth")
    else:
        stem = (
            f"{cfg.dataset}_v4_recurrent_flow"
            f"_d{_compact_float_tag(cfg.train_msi_translation_max_px)}"
            f"_r{_compact_float_tag(cfg.train_msi_rotation_max_deg)}"
            f"_l{_compact_float_tag(repair.train_msi_local_max_px)}"
            f"_it{repair.recurrent_iterations_scale4}-{repair.recurrent_iterations_scale2}-{repair.recurrent_iterations_scale1}"
        )
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
    model = build_recurrent_model(cfg, info, device, process)
    source_epoch, source_best = warm_start_compatible(
        model, repair.recurrent_init_checkpoint, device
    )
    set_train_scope(model, repair.recurrent_train_scope)
    print(
        f"Recurrent-flow warm start epoch={source_epoch}, source_best={source_best:.6f}; "
        f"scope={repair.recurrent_train_scope}; trainable={count_parameters(model):.3f}M"
    )
    print(
        "Local update: full correlation evidence -> ConvGRU residual flow; "
        f"iterations 4/2/1={repair.recurrent_iterations_scale4}/"
        f"{repair.recurrent_iterations_scale2}/{repair.recurrent_iterations_scale1}."
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(cfg.seed) + int(cfg.train_misalignment_seed_offset))
    best_path, last_path, log_path = paths(cfg, repair)
    logger = CSVLogger(log_path, fieldnames=[
        "epoch", "loss", "l1", "sam", "flow", "smooth", "local_epe",
        "target_local", "pred_local", "zero_pred_local", "registered_PSNR",
        "local_only_PSNR", "local_only_SAM", "local_only_pred",
        "global_only_PSNR", "global_only_pred", "global_local_PSNR",
        "global_local_SAM", "global_local_pred", "robust_score"
    ])

    best_score = float("-inf")
    eval_seed = int(cfg.seed) + int(repair.recurrent_eval_seed_offset)
    for epoch in range(1, cfg.epochs + 1):
        stats = train_epoch(
            model, train_loader, optimizer, process, device, cfg, repair, generator
        )
        print(
            f"Epoch {epoch:04d}/{cfg.epochs:04d} loss={stats.loss:.6f} "
            f"flow={stats.flow:.6f} EPE={stats.local_epe_px:.4f}px "
            f"true_local={stats.target_local_mean_px:.4f}px "
            f"pred_local={stats.predicted_local_mean_px:.4f}px "
            f"zero_pred={stats.zero_predicted_mean_px:.4f}px"
        )

        registered = {}
        scenarios = {}
        robust_score = float("nan")
        if epoch % cfg.eval_interval == 0 or epoch == cfg.epochs:
            registered = evaluate(
                model, test_loader, process, device, scale_ratio=cfg.scale_ratio
            )
            scenarios = evaluate_scenarios(
                model, test_loader, process, device,
                scale_ratio=cfg.scale_ratio,
                translation_max_px=cfg.train_msi_translation_max_px,
                rotation_max_deg=cfg.train_msi_rotation_max_deg,
                local_max_displacement_px=repair.train_msi_local_max_px,
                control_grid_size=repair.recurrent_synthetic_control_grid,
                valid_threshold=repair.recurrent_valid_threshold,
                seed=eval_seed,
            )
            robust_score = 0.5 * (
                scenarios["local_only_PSNR"] + scenarios["global_local_PSNR"]
            )
            print(f"  registered: {_format_metrics(registered)}")
            print(
                "  local-only: "
                f"PSNR={scenarios['local_only_PSNR']:.4f} "
                f"SAM={scenarios['local_only_SAM']:.4f} "
                f"true={scenarios['local_only_true_local']:.4f}px "
                f"pred={scenarios['local_only_pred_local']:.4f}px"
            )
            print(
                "  global+local: "
                f"PSNR={scenarios['global_local_PSNR']:.4f} "
                f"SAM={scenarios['global_local_SAM']:.4f} "
                f"pred={scenarios['global_local_pred_local']:.4f}px"
            )

            registered_ok = float(registered["PSNR"]) >= (
                float(source_best) - float(repair.recurrent_registered_tolerance_db)
            )
            if registered_ok and robust_score > best_score:
                best_score = robust_score
                save_checkpoint(
                    model, optimizer, epoch, best_score, best_path,
                    extra={
                        "config": vars(cfg), "repair": vars(repair),
                        "registered": registered, "scenarios": scenarios,
                        "source_checkpoint": repair.recurrent_init_checkpoint,
                    },
                )
                print(f"  saved recurrent-flow best -> {best_path}")

        logger.write({
            "epoch": epoch, "loss": stats.loss, "l1": stats.l1,
            "sam": stats.sam, "flow": stats.flow, "smooth": stats.smooth,
            "local_epe": stats.local_epe_px, "target_local": stats.target_local_mean_px,
            "pred_local": stats.predicted_local_mean_px,
            "zero_pred_local": stats.zero_predicted_mean_px,
            "registered_PSNR": registered.get("PSNR", ""),
            "local_only_PSNR": scenarios.get("local_only_PSNR", ""),
            "local_only_SAM": scenarios.get("local_only_SAM", ""),
            "local_only_pred": scenarios.get("local_only_pred_local", ""),
            "global_only_PSNR": scenarios.get("global_only_PSNR", ""),
            "global_only_pred": scenarios.get("global_only_pred_local", ""),
            "global_local_PSNR": scenarios.get("global_local_PSNR", ""),
            "global_local_SAM": scenarios.get("global_local_SAM", ""),
            "global_local_pred": scenarios.get("global_local_pred_local", ""),
            "robust_score": robust_score,
        })
        if epoch % cfg.save_interval == 0 or epoch == cfg.epochs:
            save_checkpoint(
                model, optimizer, epoch, best_score, last_path,
                extra={"config": vars(cfg), "repair": vars(repair)},
            )

    print(f"Best recurrent-flow checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    cfg, repair = parse_repair_args()
    run(cfg, repair)
