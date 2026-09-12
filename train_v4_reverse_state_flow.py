"""Reverse-state-aware flow-only fine-tuning for Innovation-2.

This stage addresses the train/inference mismatch left after the oracle-state
4->2->1 bootstrap.  Local-only synthetic MSI warps and geometrically correct
inverse sampling targets are retained, but the HSI side is no longer built from
D~_t(X_GT).  Instead, x_12/x_8/x_4 are cached from the model's real deterministic
reverse recursion under global-identity inference.

The reverse rollout is no-grad/eval and therefore does not update Innovation-1
or the reconstruction backbone.  Cached states are detached and only the local
recurrent branch is optimized with the same physical multiscale inverse-flow
loss.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from config import parse_args
from data_loader import build_loaders
from degradations.inverse_flow import forward_to_inverse_sampling_field
from degradations.misalignment import make_misaligned_msi
from innovation1 import build_progressive_process
from main import _compact_float_tag
from reverse_state_local_flow import (
    SCALES,
    oracle_states,
    rollout_reverse_states_identity,
    run_local_flow_from_states,
)
from train_v4_local_bootstrap import freeze_to_local_branch
from train_v4_local_reconstruction_inverse import INVERSE_FIXED_POINT_ITERATIONS
from train_v4_multiscale_bootstrap import (
    field_metrics,
    physical_flow_targets,
    stage_sequence_loss,
)
from train_v4_recurrent_flow import build_recurrent_model
from utils import (
    AverageMeter,
    CSVLogger,
    count_parameters,
    ensure_dir,
    get_device,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)


@dataclass
class ReverseStateStats:
    loss: float
    final_epe_px: float
    final_true_mean_px: float
    final_pred_mean_px: float
    final_ratio: float
    scale4_epe_px: float
    scale2_epe_px: float
    scale1_epe_px: float


def inverse_targets(process, model, forward_field: torch.Tensor):
    with torch.no_grad():
        inverse_full = forward_to_inverse_sampling_field(
            forward_field,
            iterations=INVERSE_FIXED_POINT_ITERATIONS,
            padding_mode="border",
        )
        targets = physical_flow_targets(
            process,
            model.geometry_aligner.stage_t_by_scale,
            inverse_full,
        )
    return inverse_full, targets


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
    stage_weights: Dict[int, float],
    grad_clip: float,
    generator: Optional[torch.Generator],
) -> ReverseStateStats:
    model.train()
    meters = {
        key: AverageMeter()
        for key in ("loss", "epe", "true", "pred", "ratio", "epe4", "epe2", "epe1")
    }

    for batch in loader:
        gt = batch["gt"].to(device, non_blocking=True)
        hr_msi = batch["hr_msi"].to(device, non_blocking=True)
        b = int(gt.shape[0])

        warped, _, params = make_misaligned_msi(
            hr_msi,
            translation_max_px=0.0,
            rotation_max_deg=0.0,
            local_max_displacement_px=float(local_max_px),
            control_grid_size=int(control_grid),
            generator=generator,
        )
        _, targets = inverse_targets(process, model, params.local_displacement_px)

        # Real inference-state distribution; no gradient through the reverse rollout.
        reverse_states = rollout_reverse_states_identity(
            model,
            process,
            gt,
            warped,
        )
        finals, sequences = run_local_flow_from_states(
            model,
            reverse_states,
            warped,
        )
        loss = stage_sequence_loss(
            sequences,
            targets,
            normalization_px=float(local_max_px),
            gamma=float(gamma),
            stage_weights=stage_weights,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite reverse-state flow loss")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=float(grad_clip) if float(grad_clip) > 0.0 else float("inf"),
            error_if_nonfinite=True,
        )
        optimizer.step()

        e1, t1, p1, r1 = field_metrics(finals[1], targets[1])
        e2, _, _, _ = field_metrics(finals[2], targets[2])
        e4, _, _, _ = field_metrics(finals[4], targets[4])
        for key, value in {
            "loss": float(loss.item()),
            "epe": e1,
            "true": t1,
            "pred": p1,
            "ratio": r1,
            "epe4": e4,
            "epe2": e2,
            "epe1": e1,
        }.items():
            meters[key].update(value, b)

    return ReverseStateStats(
        loss=meters["loss"].avg,
        final_epe_px=meters["epe"].avg,
        final_true_mean_px=meters["true"].avg,
        final_pred_mean_px=meters["pred"].avg,
        final_ratio=meters["ratio"].avg,
        scale4_epe_px=meters["epe4"].avg,
        scale2_epe_px=meters["epe2"].avg,
        scale1_epe_px=meters["epe1"].avg,
    )


@torch.no_grad()
def evaluate_state_modes(
    model,
    loader,
    process,
    device,
    *,
    severities: List[float],
    control_grid: int,
    trials: int,
    seed: int,
):
    model.eval()
    results = {}
    for local_max in severities:
        accum = {
            "oracle_epe": [], "reverse_epe": [],
            "oracle_pred": [], "reverse_pred": [], "true": [],
            "oracle4_epe": [], "oracle2_epe": [], "oracle1_epe": [],
            "reverse4_epe": [], "reverse2_epe": [], "reverse1_epe": [],
        }
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
                _, targets = inverse_targets(process, model, params.local_displacement_px)
                oracle = oracle_states(model, process, gt)
                reverse = rollout_reverse_states_identity(model, process, gt, warped)
                oracle_final, _ = run_local_flow_from_states(model, oracle, warped)
                reverse_final, _ = run_local_flow_from_states(model, reverse, warped)

                oracle_metrics = {
                    scale: field_metrics(oracle_final[scale], targets[scale])
                    for scale in SCALES
                }
                reverse_metrics = {
                    scale: field_metrics(reverse_final[scale], targets[scale])
                    for scale in SCALES
                }
                oe, true_mag, op, _ = oracle_metrics[1]
                re, _, rp, _ = reverse_metrics[1]
                accum["oracle_epe"].append(oe)
                accum["reverse_epe"].append(re)
                accum["oracle_pred"].append(op)
                accum["reverse_pred"].append(rp)
                accum["true"].append(true_mag)
                for scale in SCALES:
                    accum[f"oracle{scale}_epe"].append(oracle_metrics[scale][0])
                    accum[f"reverse{scale}_epe"].append(reverse_metrics[scale][0])

        results[float(local_max)] = {
            key: float(np.mean(values)) for key, values in accum.items()
        }
    return results


def parse_reverse_state_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--reverse_state_init_checkpoint", type=str, required=True)
    parser.add_argument("--reverse_state_local_max_px", type=float, default=1.0)
    parser.add_argument("--reverse_state_control_grid", type=int, default=5)
    parser.add_argument("--reverse_state_gamma", type=float, default=0.8)
    parser.add_argument("--reverse_state_weight_scale4", type=float, default=0.5)
    parser.add_argument("--reverse_state_weight_scale2", type=float, default=0.7)
    parser.add_argument("--reverse_state_weight_scale1", type=float, default=1.0)
    parser.add_argument("--reverse_state_eval_severities", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--reverse_state_eval_trials", type=int, default=3)
    parser.add_argument("--reverse_state_eval_interval", type=int, default=5)
    parser.add_argument("--reverse_state_save_name", type=str, default="")
    parser.add_argument("--recurrent_hidden_channels", type=int, default=64)
    parser.add_argument("--recurrent_correlation_channels", type=int, default=32)
    parser.add_argument("--recurrent_iterations_scale4", type=int, default=3)
    parser.add_argument("--recurrent_iterations_scale2", type=int, default=2)
    parser.add_argument("--recurrent_iterations_scale1", type=int, default=2)
    parser.add_argument("--recurrent_max_update_scale4", type=float, default=2.0)
    parser.add_argument("--recurrent_max_update_scale2", type=float, default=1.0)
    parser.add_argument("--recurrent_max_update_scale1", type=float, default=0.5)
    args, remaining = parser.parse_known_args()
    cfg = parse_args(remaining)

    if cfg.stage != "train":
        raise ValueError("reverse-state flow fine-tune is train-only; use --stage train")
    if str(cfg.predictor_version).lower() != "v4":
        raise ValueError("reverse-state flow fine-tune requires predictor_version=v4")
    if args.reverse_state_local_max_px <= 0.0:
        raise ValueError("reverse_state_local_max_px must be > 0")
    if args.reverse_state_control_grid < 2:
        raise ValueError("reverse_state_control_grid must be >= 2")
    if not 0.0 < args.reverse_state_gamma <= 1.0:
        raise ValueError("reverse_state_gamma must lie in (0,1]")
    if min(
        args.reverse_state_weight_scale4,
        args.reverse_state_weight_scale2,
        args.reverse_state_weight_scale1,
    ) <= 0.0:
        raise ValueError("reverse-state stage weights must be > 0")
    if args.reverse_state_eval_trials < 1 or args.reverse_state_eval_interval < 1:
        raise ValueError("reverse-state eval counts must be >= 1")
    if any(v <= 0.0 for v in args.reverse_state_eval_severities):
        raise ValueError("all reverse-state eval severities must be > 0")

    for name in (
        "recurrent_hidden_channels", "recurrent_correlation_channels",
        "recurrent_iterations_scale4", "recurrent_iterations_scale2", "recurrent_iterations_scale1",
        "recurrent_max_update_scale4", "recurrent_max_update_scale2", "recurrent_max_update_scale1",
    ):
        setattr(cfg, name, getattr(args, name))
    return cfg, args


def output_paths(cfg, args):
    root = os.path.join(cfg.checkpoint_root, "innovation1")
    ensure_dir(root)
    if args.reverse_state_save_name:
        stem = args.reverse_state_save_name.removesuffix(".pth")
    else:
        stem = (
            f"{cfg.dataset}_v4_reverse_state_flow"
            f"_l{_compact_float_tag(args.reverse_state_local_max_px)}"
        )
    return (
        os.path.join(root, stem + ".pth"),
        os.path.join(root, stem + "_last.pth"),
        os.path.join(cfg.log_root, stem + ".csv"),
    )


def run(cfg, args):
    set_seed(cfg.seed)
    train_loader, test_loader, info = build_loaders(cfg)
    device = get_device(cfg.device)
    process = build_progressive_process(cfg)
    model = build_recurrent_model(cfg, info, device, process)
    source_epoch, source_metric = load_checkpoint(
        model,
        args.reverse_state_init_checkpoint,
        optimizer=None,
        strict=True,
        map_location=str(device),
        load_optimizer=False,
    )
    freeze_to_local_branch(model)

    stage_weights = {
        4: float(args.reverse_state_weight_scale4),
        2: float(args.reverse_state_weight_scale2),
        1: float(args.reverse_state_weight_scale1),
    }
    print(
        "Reverse-state flow fine-tune warm start: "
        f"{args.reverse_state_init_checkpoint} "
        f"(epoch={source_epoch}, stored_metric={source_metric:.6f})"
    )
    print(
        "Protocol: local-only=100%, global=identity, HSI states come from real reverse rollout, "
        "loss=inverse physical 4->2->1 flow only."
    )
    print(
        f"trainable={sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.3f}M "
        f"/ total={count_parameters(model):.3f}M"
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
    best_path, last_path, log_path = output_paths(cfg, args)
    logger = CSVLogger(
        log_path,
        fieldnames=[
            "epoch", "loss", "train_reverse_epe", "train_true", "train_pred", "train_ratio",
            "train_s4_epe", "train_s2_epe", "train_s1_epe",
            "eval_l05_oracle_epe", "eval_l05_reverse_epe",
            "eval_l1_oracle_epe", "eval_l1_reverse_epe",
            "eval_l2_oracle_epe", "eval_l2_reverse_epe", "best_reverse_epe",
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
            local_max_px=float(args.reverse_state_local_max_px),
            control_grid=int(args.reverse_state_control_grid),
            gamma=float(args.reverse_state_gamma),
            stage_weights=stage_weights,
            grad_clip=float(cfg.grad_clip),
            generator=generator,
        )
        print(
            f"Epoch {epoch:04d}/{cfg.epochs:04d} loss={stats.loss:.6f} "
            f"reverse_EPE={stats.final_epe_px:.4f} "
            f"pred/true={stats.final_pred_mean_px:.3f}/{stats.final_true_mean_px:.3f} "
            f"s4/s2/s1 EPE={stats.scale4_epe_px:.4f}/"
            f"{stats.scale2_epe_px:.4f}/{stats.scale1_epe_px:.4f}"
        )

        evaluation = None
        if epoch % int(args.reverse_state_eval_interval) == 0 or epoch == int(cfg.epochs):
            evaluation = evaluate_state_modes(
                model,
                test_loader,
                process,
                device,
                severities=[float(v) for v in args.reverse_state_eval_severities],
                control_grid=int(args.reverse_state_control_grid),
                trials=int(args.reverse_state_eval_trials),
                seed=eval_seed,
            )
            for severity, result in evaluation.items():
                print(
                    f"  local<={severity:g}: oracle_EPE={result['oracle_epe']:.4f} "
                    f"reverse_EPE={result['reverse_epe']:.4f} "
                    f"reverse pred/true={result['reverse_pred']:.3f}/{result['true']:.3f}"
                )

            ref_key = float(args.reverse_state_local_max_px)
            if ref_key not in evaluation:
                ref_key = min(evaluation, key=lambda x: abs(x - ref_key))
            ref_epe = float(evaluation[ref_key]["reverse_epe"])
            if ref_epe < best_epe:
                best_epe = ref_epe
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    best_epe,
                    best_path,
                    extra={
                        "config": vars(cfg),
                        "reverse_state_flow": vars(args),
                        "evaluation": evaluation,
                        "source_checkpoint": args.reverse_state_init_checkpoint,
                    },
                )
                print(f"  saved best reverse-state flow -> {best_path}")

        row = {
            "epoch": epoch,
            "loss": stats.loss,
            "train_reverse_epe": stats.final_epe_px,
            "train_true": stats.final_true_mean_px,
            "train_pred": stats.final_pred_mean_px,
            "train_ratio": stats.final_ratio,
            "train_s4_epe": stats.scale4_epe_px,
            "train_s2_epe": stats.scale2_epe_px,
            "train_s1_epe": stats.scale1_epe_px,
            "best_reverse_epe": best_epe,
        }
        if evaluation:
            tags = {0.5: "05", 1.0: "1", 2.0: "2"}
            for severity, result in evaluation.items():
                if severity in tags:
                    tag = tags[severity]
                    row[f"eval_l{tag}_oracle_epe"] = result["oracle_epe"]
                    row[f"eval_l{tag}_reverse_epe"] = result["reverse_epe"]
        logger.write(row)

        if epoch % int(cfg.save_interval) == 0 or epoch == int(cfg.epochs):
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best_epe,
                last_path,
                extra={
                    "config": vars(cfg),
                    "reverse_state_flow": vars(args),
                    "evaluation": evaluation,
                    "source_checkpoint": args.reverse_state_init_checkpoint,
                },
            )

    print("Reverse-state flow fine-tune complete.")
    print(f"Best reverse-state EPE: {best_epe:.6f}px")
    print(f"Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    cfg, args = parse_reverse_state_args()
    run(cfg, args)
