"""Registered-input self-motion audit for the final V4 Global+Local model.

This diagnostic answers one question before any combined-misalignment result is
trusted: when HR-MSI is already registered, which alignment component damages
it?  The same composed model is evaluated in four inference modes:

A. OFF/OFF      : bypass global and local alignment;
B. Global/OFF   : apply the learned global rigid correction only;
C. OFF/Local    : force global identity and run recurrent 4->2->1 local flow;
D. Global/Local : run the complete V4 alignment pipeline.

The model itself is composed from two validated checkpoints: a late local-only
checkpoint supplies the reconstruction backbone + recurrent local branch, while
only ``geometry_aligner.global_aligner.*`` is restored from the validated global
rigid checkpoint.
"""

from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np
import torch

from config import parse_args
from data_loader import build_loaders
from innovation1 import build_progressive_process, reconstruct_from_terminal_lr
from metrics import calc_metrics
from models.predictor_v3_ablation import MSIAblationGuidedPredictor
from train_v4_local_reconstruction import reconstruct_local_only_identity
from train_v4_recurrent_flow import build_recurrent_model
from utils import get_device, set_seed
from v4_joint_checkpoint import load_joint_v4_checkpoints


@torch.no_grad()
def reconstruct_no_alignment(model, process, lr_hsi, *, target_size, hr_msi):
    """Raw-Direct reverse recursion with both geometry branches bypassed."""
    model.eval()
    if hasattr(model, "_reset_inference_local_state"):
        model._reset_inference_local_state()
    x_t = process.terminal_state(lr_hsi, target_size=target_size)
    for t in range(int(process.total_steps), 0, -1):
        timestep = torch.full(
            (x_t.shape[0],), int(t), dtype=torch.long, device=x_t.device
        )
        pred_x0 = MSIAblationGuidedPredictor.forward(
            model, x_t, hr_msi, timestep
        )
        x_t = process.reverse_update(x_t, pred_x0, int(t))
    return x_t


@torch.no_grad()
def reconstruct_global_only(model, process, lr_hsi, *, target_size, hr_msi):
    """Apply global rigid correction once, then bypass recurrent local flow."""
    model.eval()
    if hasattr(model, "_reset_inference_local_state"):
        model._reset_inference_local_state()
    x_t = process.terminal_state(lr_hsi, target_size=target_size)
    aligned, shift, rotation = model.geometry_aligner.global_coarse_correction(
        x_t, hr_msi, int(process.total_steps)
    )
    model.last_global_shift_px = shift.detach()
    model.last_global_rotation_deg = rotation.detach()
    for t in range(int(process.total_steps), 0, -1):
        timestep = torch.full(
            (x_t.shape[0],), int(t), dtype=torch.long, device=x_t.device
        )
        pred_x0 = MSIAblationGuidedPredictor.forward(
            model, x_t, aligned, timestep
        )
        x_t = process.reverse_update(x_t, pred_x0, int(t))
    return x_t


def _mean_local_magnitude(model) -> float:
    flow = getattr(model, "_inference_local_offset", None)
    if flow is None:
        return 0.0
    return float(torch.linalg.vector_norm(flow.detach().float(), dim=1).mean().item())


def _mean_global_shift(model) -> float:
    shift = getattr(model, "last_global_shift_px", None)
    if shift is None:
        return 0.0
    return float(torch.linalg.vector_norm(shift.detach().float(), dim=1).mean().item())


def _mean_global_rotation(model) -> float:
    rotation = getattr(model, "last_global_rotation_deg", None)
    if rotation is None:
        return 0.0
    return float(rotation.detach().float().abs().mean().item())


def parse_joint_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--joint_local_checkpoint",
        type=str,
        default="./checkpoints/innovation1/PaviaU_v4_reverse_state_reconstruction_l1_oa0p2.pth",
    )
    parser.add_argument("--joint_global_checkpoint", type=str, required=True)
    joint, remaining = parser.parse_known_args()
    cfg = parse_args(remaining)
    if cfg.stage != "test":
        raise ValueError("registered component audit is test-only; use --stage test")
    if str(cfg.predictor_version).lower() != "v4":
        raise ValueError("registered component audit requires predictor_version=v4")
    return cfg, joint


@torch.no_grad()
def main():
    cfg, joint = parse_joint_args()
    set_seed(cfg.seed)
    _, test_loader, info = build_loaders(cfg)
    device = get_device(cfg.device)
    process = build_progressive_process(cfg)
    model = build_recurrent_model(cfg, info, device, process)
    epoch, metric, copied = load_joint_v4_checkpoints(
        model,
        joint.joint_local_checkpoint,
        joint.joint_global_checkpoint,
        map_location=str(device),
    )
    print(
        "Composed V4 checkpoints:\n"
        f"  local/full = {joint.joint_local_checkpoint}\n"
        f"  global     = {joint.joint_global_checkpoint}\n"
        f"  local_epoch={epoch}, local_metric={metric:.6f}, global_tensors_copied={copied}"
    )

    accum: Dict[str, Dict[str, List[float]]] = {
        name: {"PSNR": [], "SAM": [], "gshift": [], "grot": [], "local": []}
        for name in ("off_off", "global_only", "local_only", "global_local")
    }

    for batch in test_loader:
        gt = batch["gt"].to(device, non_blocking=True)
        hr_msi = batch["hr_msi"].to(device, non_blocking=True)
        terminal_lr = process.terminal_observation(gt)
        target_size = tuple(gt.shape[-2:])

        pred = reconstruct_no_alignment(
            model, process, terminal_lr, target_size=target_size, hr_msi=hr_msi
        )
        met = calc_metrics(pred, gt, cfg.scale_ratio)
        accum["off_off"]["PSNR"].append(float(met["PSNR"]))
        accum["off_off"]["SAM"].append(float(met["SAM"]))
        accum["off_off"]["gshift"].append(0.0)
        accum["off_off"]["grot"].append(0.0)
        accum["off_off"]["local"].append(0.0)

        pred = reconstruct_global_only(
            model, process, terminal_lr, target_size=target_size, hr_msi=hr_msi
        )
        met = calc_metrics(pred, gt, cfg.scale_ratio)
        accum["global_only"]["PSNR"].append(float(met["PSNR"]))
        accum["global_only"]["SAM"].append(float(met["SAM"]))
        accum["global_only"]["gshift"].append(_mean_global_shift(model))
        accum["global_only"]["grot"].append(_mean_global_rotation(model))
        accum["global_only"]["local"].append(0.0)

        pred = reconstruct_local_only_identity(
            model,
            process,
            terminal_lr,
            target_size=target_size,
            warped_msi=hr_msi,
        )
        met = calc_metrics(pred, gt, cfg.scale_ratio)
        accum["local_only"]["PSNR"].append(float(met["PSNR"]))
        accum["local_only"]["SAM"].append(float(met["SAM"]))
        accum["local_only"]["gshift"].append(0.0)
        accum["local_only"]["grot"].append(0.0)
        accum["local_only"]["local"].append(_mean_local_magnitude(model))

        pred = reconstruct_from_terminal_lr(
            model,
            process,
            terminal_lr,
            target_size=target_size,
            hr_msi=hr_msi,
        )
        met = calc_metrics(pred, gt, cfg.scale_ratio)
        accum["global_local"]["PSNR"].append(float(met["PSNR"]))
        accum["global_local"]["SAM"].append(float(met["SAM"]))
        accum["global_local"]["gshift"].append(_mean_global_shift(model))
        accum["global_local"]["grot"].append(_mean_global_rotation(model))
        accum["global_local"]["local"].append(_mean_local_magnitude(model))

    print("\nRegistered-input component audit")
    for name in ("off_off", "global_only", "local_only", "global_local"):
        row = accum[name]
        print(
            f"{name:12s} "
            f"PSNR={np.mean(row['PSNR']):.4f} "
            f"SAM={np.mean(row['SAM']):.4f} "
            f"pred_global_shift={np.mean(row['gshift']):.4f}px "
            f"pred_global_rot={np.mean(row['grot']):.4f}deg "
            f"pred_local={np.mean(row['local']):.4f}px"
        )


if __name__ == "__main__":
    main()
