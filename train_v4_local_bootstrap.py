"""Scale-1 local-flow bootstrap for Innovation-2 recurrent alignment.

Purpose
-------
This is a diagnostic pretraining stage, not the final 4->2->1 model.

It removes the three confounders found in the first recurrent-flow experiment:
1. local-only samples only: no registered/global-only/global+local mixture;
2. global rigid correction is forced to identity (completely bypassed);
3. only physical scale=1 is trained, using pure sequence flow supervision.

The residual output head is reinitialized with a tiny N(0, 1e-3) weight instead
of exact zeros so flow gradients reach the recurrent/correlation stack from the
first optimization step.

If this stage cannot recover the known synthetic local field, the recurrent
local estimator itself is not viable and should not be pushed into the complete
reconstruction pipeline.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from config import parse_args
from data_loader import build_loaders
from degradations.misalignment import make_misaligned_msi
from innovation1 import build_progressive_process
from main import _compact_float_tag
from train_v4_recurrent_flow import (
    build_recurrent_model,
    sequence_flow_loss,
    warm_start_compatible,
)
from utils import (
    AverageMeter,
    CSVLogger,
    count_parameters,
    ensure_dir,
    get_device,
    save_checkpoint,
    set_seed,
)


@dataclass
class BootstrapStats:
    loss: float
    epe_px: float
    true_mean_px: float
    pred_mean_px: float
    pred_to_true_ratio: float
    first_pred_mean_px: float
    last_pred_mean_px: float


def tiny_initialize_delta_head(model: torch.nn.Module, std: float = 1e-3) -> None:
    """Near-zero but non-degenerate output initialization."""
    std = float(std)
    if std <= 0.0:
        raise ValueError("bootstrap delta-head std must be > 0")
    head = model.geometry_aligner.local_aligner.delta_head[-1]
    if not isinstance(head, nn.Conv2d):
        raise TypeError("expected final recurrent delta head to be Conv2d")
    nn.init.normal_(head.weight, mean=0.0, std=std)
    nn.init.zeros_(head.bias)


def freeze_to_local_branch(model: torch.nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.geometry_aligner.local_aligner.parameters():
        parameter.requires_grad_(True)


def scale1_state_and_matches(model, process, gt, warped_msi):
    """Build matched physical-domain inputs at the representative scale-1 state.

    Global alignment is deliberately bypassed.  Passing warped_msi directly as
    the MSI input is equivalent to forcing the learned global rigid transform to
    identity in this bootstrap experiment.
    """
    scale1_t = int(model.geometry_aligner.stage_t_by_scale[1])
    x_scale1 = process.state_at(gt, scale1_t)
    timesteps = torch.full(
        (gt.shape[0],),
        scale1_t,
        dtype=torch.long,
        device=gt.device,
    )
    z_h, z_m = model.geometry_aligner.matched_states(
        x_scale1,
        warped_msi,
        timesteps,
    )
    return z_h, z_m, scale1_t


def predict_scale1_sequence(model, process, gt, warped_msi) -> List[torch.Tensor]:
    z_h, z_m, _ = scale1_state_and_matches(model, process, gt, warped_msi)
    _, _, sequence = model.geometry_aligner.local_aligner(
        z_h,
        z_m,
        previous_dense_offset=None,
        scale=1,
    )
    return sequence


def field_stats(pred: torch.Tensor, target: torch.Tensor):
    pred = pred.float()
    target = target.float()
    epe = torch.linalg.vector_norm(pred - target, dim=1).mean()
    pred_mag = torch.linalg.vector_norm(pred, dim=1).mean()
    true_mag = torch.linalg.vector_norm(target, dim=1).mean()
    ratio = pred_mag / true_mag.clamp_min(1e-8)
    return (
        float(epe.item()),
        float(true_mag.item()),
        float(pred_mag.item()),
        float(ratio.item()),
    )


def train_epoch(
    model,
    loader,
    optimizer,
    process,
    device,
    *,
    local_max_px: float,
    control_grid: int,
    gamma: float,
    generator: Optional[torch.Generator],
):
    model.train()
    meters = {
        key: AverageMeter()
        for key in ("loss", "epe", "true", "pred", "ratio", "first", "last")
    }

    for batch in loader:
        gt = batch["gt"].to(device, non_blocking=True)
        hr_msi = batch["hr_msi"].to(device, non_blocking=True)
        batch_size = int(gt.shape[0])

        warped, _, params = make_misaligned_msi(
            hr_msi,
            translation_max_px=0.0,
            rotation_max_deg=0.0,
            local_max_displacement_px=float(local_max_px),
            control_grid_size=int(control_grid),
            generator=generator,
        )
        target = params.local_displacement_px

        sequence = predict_scale1_sequence(model, process, gt, warped)
        supervised = torch.ones(batch_size, dtype=torch.bool, device=device)
        loss = sequence_flow_loss(
            {1: sequence},
            target,
            supervised,
            normalization_px=float(local_max_px),
            gamma=float(gamma),
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite local bootstrap loss")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=1.0,
            error_if_nonfinite=True,
        )
        optimizer.step()

        epe, true_mag, pred_mag, ratio = field_stats(sequence[-1], target)
        first_mag = float(
            torch.linalg.vector_norm(sequence[0].detach().float(), dim=1)
            .mean()
            .item()
        )
        last_mag = float(
            torch.linalg.vector_norm(sequence[-1].detach().float(), dim=1)
            .mean()
            .item()
        )
        values = {
            "loss": float(loss.item()),
            "epe": epe,
            "true": true_mag,
            "pred": pred_mag,
            "ratio": ratio,
            "first": first_mag,
            "last": last_mag,
        }
        for key, value in values.items():
            meters[key].update(value, batch_size)

    return BootstrapStats(
        loss=meters["loss"].avg,
        epe_px=meters["epe"].avg,
        true_mean_px=meters["true"].avg,
        pred_mean_px=meters["pred"].avg,
        pred_to_true_ratio=meters["ratio"].avg,
        first_pred_mean_px=meters["first"].avg,
        last_pred_mean_px=meters["last"].avg,
    )


@torch.no_grad()
def evaluate_flow_severities(
    model,
    loader,
    process,
    device,
    *,
    severities: List[float],
    control_grid: int,
    trials: int,
    seed: int,
) -> Dict[float, Dict[str, float]]:
    model.eval()
    results: Dict[float, Dict[str, float]] = {}

    for local_max in severities:
        epe_values, true_values, pred_values, ratio_values = [], [], [], []
        iter_values: Optional[List[List[float]]] = None

        for trial in range(int(trials)):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(seed) + trial * 100003)

            for batch in loader:
                gt = batch["gt"].to(device, non_blocking=True)
                hr_msi = batch["hr_msi"].to(device, non_blocking=True)
                warped, _, params = make_misaligned_msi(
                    hr_msi,
                    translation_max_px=0.0,
                    rotation_max_deg=0.0,
                    local_max_displacement_px=float(local_max),
                    control_grid_size=int(control_grid),
                    generator=generator,
                )
                sequence = predict_scale1_sequence(model, process, gt, warped)
                epe, true_mag, pred_mag, ratio = field_stats(
                    sequence[-1],
                    params.local_displacement_px,
                )
                epe_values.append(epe)
                true_values.append(true_mag)
                pred_values.append(pred_mag)
                ratio_values.append(ratio)

                if iter_values is None:
                    iter_values = [[] for _ in sequence]
                for index, prediction in enumerate(sequence):
                    iter_values[index].append(
                        float(
                            torch.linalg.vector_norm(
                                prediction.float(), dim=1
                            ).mean().item()
                        )
                    )

        summary = {
            "epe_px": float(np.mean(epe_values)),
            "true_mean_px": float(np.mean(true_values)),
            "pred_mean_px": float(np.mean(pred_values)),
            "pred_to_true_ratio": float(np.mean(ratio_values)),
        }
        if iter_values is not None:
            for index, values in enumerate(iter_values, start=1):
                summary[f"iter{index}_pred_mean_px"] = float(np.mean(values))
        results[float(local_max)] = summary

    return results


def parse_bootstrap_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--bootstrap_init_checkpoint", type=str, required=True)
    parser.add_argument("--bootstrap_local_max_px", type=float, default=1.0)
    parser.add_argument("--bootstrap_control_grid", type=int, default=5)
    parser.add_argument("--bootstrap_iterations", type=int, default=4)
    parser.add_argument("--bootstrap_max_update_px", type=float, default=0.5)
    parser.add_argument("--bootstrap_delta_init_std", type=float, default=1e-3)
    parser.add_argument("--bootstrap_gamma", type=float, default=0.8)
    parser.add_argument(
        "--bootstrap_eval_severities",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0],
    )
    parser.add_argument("--bootstrap_eval_trials", type=int, default=3)
    parser.add_argument("--bootstrap_eval_interval", type=int, default=5)
    parser.add_argument("--bootstrap_save_name", type=str, default="")
    bootstrap, remaining = parser.parse_known_args()
    cfg = parse_args(remaining)

    if cfg.stage != "train":
        raise ValueError("bootstrap is train-only; use --stage train")
    if str(cfg.predictor_version).lower() != "v4":
        raise ValueError("bootstrap requires --predictor_version v4")
    if bootstrap.bootstrap_local_max_px <= 0.0:
        raise ValueError("--bootstrap_local_max_px must be > 0")
    if bootstrap.bootstrap_control_grid < 2:
        raise ValueError("--bootstrap_control_grid must be >= 2")
    if bootstrap.bootstrap_iterations < 1:
        raise ValueError("--bootstrap_iterations must be >= 1")
    if bootstrap.bootstrap_max_update_px <= 0.0:
        raise ValueError("--bootstrap_max_update_px must be > 0")
    if bootstrap.bootstrap_delta_init_std <= 0.0:
        raise ValueError("--bootstrap_delta_init_std must be > 0")
    if not 0.0 < bootstrap.bootstrap_gamma <= 1.0:
        raise ValueError("--bootstrap_gamma must lie in (0,1]")
    if bootstrap.bootstrap_eval_trials < 1:
        raise ValueError("--bootstrap_eval_trials must be >= 1")
    if bootstrap.bootstrap_eval_interval < 1:
        raise ValueError("--bootstrap_eval_interval must be >= 1")
    if any(v <= 0.0 for v in bootstrap.bootstrap_eval_severities):
        raise ValueError("bootstrap eval severities must all be > 0")

    cfg.recurrent_hidden_channels = 64
    cfg.recurrent_correlation_channels = 32
    cfg.recurrent_iterations_scale1 = int(bootstrap.bootstrap_iterations)
    cfg.recurrent_iterations_scale2 = 2
    cfg.recurrent_iterations_scale4 = 3
    cfg.recurrent_max_update_scale1 = float(bootstrap.bootstrap_max_update_px)
    cfg.recurrent_max_update_scale2 = 1.0
    cfg.recurrent_max_update_scale4 = 2.0
    return cfg, bootstrap


def output_paths(cfg, bootstrap):
    root = os.path.join(cfg.checkpoint_root, "innovation1")
    ensure_dir(root)
    if bootstrap.bootstrap_save_name:
        stem = bootstrap.bootstrap_save_name.removesuffix(".pth")
    else:
        stem = (
            f"{cfg.dataset}_v4_local_bootstrap"
            f"_l{_compact_float_tag(bootstrap.bootstrap_local_max_px)}"
            f"_it{bootstrap.bootstrap_iterations}"
            f"_upd{_compact_float_tag(bootstrap.bootstrap_max_update_px)}"
        )
    return (
        os.path.join(root, stem + ".pth"),
        os.path.join(root, stem + "_last.pth"),
        os.path.join(cfg.log_root, stem + ".csv"),
    )


def run(cfg, bootstrap):
    set_seed(cfg.seed)
    train_loader, test_loader, info = build_loaders(cfg)
    device = get_device(cfg.device)
    process = build_progressive_process(cfg)
    model = build_recurrent_model(cfg, info, device, process)

    source_epoch, source_best = warm_start_compatible(
        model,
        bootstrap.bootstrap_init_checkpoint,
        device,
    )
    tiny_initialize_delta_head(
        model,
        std=float(bootstrap.bootstrap_delta_init_std),
    )
    freeze_to_local_branch(model)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(
        "Scale-1 local bootstrap warm start: "
        f"{bootstrap.bootstrap_init_checkpoint} "
        f"(source_epoch={source_epoch}, source_best={source_best:.6f})"
    )
    print(
        "Bootstrap protocol: local-only=100%, global=identity, scale=1 only, "
        "loss=sequence flow only."
    )
    print(
        f"scale1_t={model.geometry_aligner.stage_t_by_scale[1]}, "
        f"iterations={bootstrap.bootstrap_iterations}, "
        f"max_update={bootstrap.bootstrap_max_update_px:g}px, "
        f"delta_init_std={bootstrap.bootstrap_delta_init_std:g}, "
        f"trainable={trainable:.3f}M / total={count_parameters(model):.3f}M"
    )

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        int(cfg.seed) + int(getattr(cfg, "train_misalignment_seed_offset", 7919))
    )

    best_path, last_path, log_path = output_paths(cfg, bootstrap)
    eval_keys = []
    for severity in bootstrap.bootstrap_eval_severities:
        tag = _compact_float_tag(severity)
        eval_keys.extend(
            [
                f"eval_l{tag}_epe",
                f"eval_l{tag}_true",
                f"eval_l{tag}_pred",
                f"eval_l{tag}_ratio",
            ]
        )
    logger = CSVLogger(
        log_path,
        fieldnames=[
            "epoch",
            "loss",
            "train_epe_px",
            "train_true_mean_px",
            "train_pred_mean_px",
            "train_ratio",
            "train_first_pred_px",
            "train_last_pred_px",
            *eval_keys,
            "best_eval_epe_px",
        ],
    )

    best_epe = float("inf")
    eval_seed = int(cfg.seed) + 15431

    for epoch in range(1, int(cfg.epochs) + 1):
        stats = train_epoch(
            model,
            train_loader,
            optimizer,
            process,
            device,
            local_max_px=float(bootstrap.bootstrap_local_max_px),
            control_grid=int(bootstrap.bootstrap_control_grid),
            gamma=float(bootstrap.bootstrap_gamma),
            generator=generator,
        )
        print(
            f"Epoch {epoch:04d}/{cfg.epochs:04d} "
            f"flow={stats.loss:.6f} epe={stats.epe_px:.4f}px "
            f"true={stats.true_mean_px:.4f}px "
            f"pred={stats.pred_mean_px:.4f}px "
            f"ratio={stats.pred_to_true_ratio:.3f} "
            f"iter1={stats.first_pred_mean_px:.4f}px "
            f"iterN={stats.last_pred_mean_px:.4f}px"
        )

        eval_results = {}
        should_eval = (
            epoch % int(bootstrap.bootstrap_eval_interval) == 0
            or epoch == int(cfg.epochs)
        )
        if should_eval:
            eval_results = evaluate_flow_severities(
                model,
                test_loader,
                process,
                device,
                severities=[float(v) for v in bootstrap.bootstrap_eval_severities],
                control_grid=int(bootstrap.bootstrap_control_grid),
                trials=int(bootstrap.bootstrap_eval_trials),
                seed=eval_seed,
            )
            print("  scale-1 local-only flow diagnosis:")
            for severity, result in eval_results.items():
                iter_text = " ".join(
                    f"{key}={value:.4f}"
                    for key, value in result.items()
                    if key.startswith("iter")
                )
                print(
                    f"    local<={severity:g}px "
                    f"true={result['true_mean_px']:.4f} "
                    f"pred={result['pred_mean_px']:.4f} "
                    f"ratio={result['pred_to_true_ratio']:.3f} "
                    f"EPE={result['epe_px']:.4f} {iter_text}"
                )

            train_severity = min(
                eval_results,
                key=lambda value: abs(
                    float(value) - float(bootstrap.bootstrap_local_max_px)
                ),
            )
            current_epe = float(eval_results[train_severity]["epe_px"])
            if current_epe < best_epe:
                best_epe = current_epe
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    -best_epe,
                    best_path,
                    extra={
                        "bootstrap": vars(bootstrap),
                        "source_checkpoint": bootstrap.bootstrap_init_checkpoint,
                        "scale1_t": int(model.geometry_aligner.stage_t_by_scale[1]),
                        "best_eval_epe_px": best_epe,
                        "eval_results": eval_results,
                    },
                )
                print(f"  saved bootstrap best -> {best_path}")

        row = {
            "epoch": epoch,
            "loss": stats.loss,
            "train_epe_px": stats.epe_px,
            "train_true_mean_px": stats.true_mean_px,
            "train_pred_mean_px": stats.pred_mean_px,
            "train_ratio": stats.pred_to_true_ratio,
            "train_first_pred_px": stats.first_pred_mean_px,
            "train_last_pred_px": stats.last_pred_mean_px,
            "best_eval_epe_px": best_epe if np.isfinite(best_epe) else "",
        }
        for severity, result in eval_results.items():
            tag = _compact_float_tag(severity)
            row[f"eval_l{tag}_epe"] = result["epe_px"]
            row[f"eval_l{tag}_true"] = result["true_mean_px"]
            row[f"eval_l{tag}_pred"] = result["pred_mean_px"]
            row[f"eval_l{tag}_ratio"] = result["pred_to_true_ratio"]
        logger.write(row)

        if epoch % int(cfg.save_interval) == 0 or epoch == int(cfg.epochs):
            save_checkpoint(
                model,
                optimizer,
                epoch,
                -best_epe if np.isfinite(best_epe) else 0.0,
                last_path,
                extra={
                    "bootstrap": vars(bootstrap),
                    "source_checkpoint": bootstrap.bootstrap_init_checkpoint,
                    "scale1_t": int(model.geometry_aligner.stage_t_by_scale[1]),
                    "best_eval_epe_px": best_epe,
                },
            )

    print("Scale-1 local bootstrap complete.")
    print(f"Best eval EPE: {best_epe:.6f}px")
    print(f"Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    cfg, bootstrap = parse_bootstrap_args()
    run(cfg, bootstrap)
