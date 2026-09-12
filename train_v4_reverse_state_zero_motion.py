"""Zero-motion preservation repair for the final recurrent Innovation-2 local branch.

This stage starts from the validated reverse-state reconstruction checkpoint and
changes only the training distribution.  The architecture, global-identity
protocol, inverse-flow geometry, physical 4->2->1 recurrent updates, Raw-MSI
reconstruction loss and weak oracle-state anchor are unchanged.

Default training distribution (sampled once per batch):
    30% registered : pristine HR-MSI, inverse-flow target = 0 at every scale
    70% local-only : smooth local warp <= 1 px, inverse-flow target as before

The purpose is to teach the recurrent updater the missing identity behavior:
    already aligned evidence -> residual flow update -> 0
without introducing a gate or any new alignment structure.

A checkpoint is considered valid only when all of the following hold:
    registered PSNR >= 43.8 dB
    local<=1 PSNR  >= 39.3 dB
    local flow magnitude ratio in [0.8, 1.2]
Among valid checkpoints, registered_PSNR + local_PSNR is maximized.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch

import train_v4_reverse_state_reconstruction as base
from degradations.misalignment import make_misaligned_msi
from losses import SAMLoss
from metrics import calc_metrics
from reverse_state_local_flow import oracle_states, rollout_reverse_states_identity, run_local_flow_from_states
from train_v4_local_reconstruction import reconstruct_local_only_identity
from train_v4_local_reconstruction_inverse import evaluate_inverse_local_only_reconstruction
from train_v4_reverse_state_flow import inverse_targets


_REGISTERED_PROBABILITY = 0.30


@dataclass
class ZeroMotionStats:
    loss: float
    l1: float
    sam: float
    reverse_flow: float
    oracle_flow: float
    registered_fraction: float
    registered_pred_flow_px: float
    local_reverse_epe_px: float
    local_true_mean_px: float
    local_pred_mean_px: float
    local_ratio: float


def validate_registered_probability(value: float) -> float:
    value = float(value)
    if not 0.0 < value < 1.0:
        raise ValueError("registered probability must lie strictly in (0,1)")
    return value


def sample_registered_batch(
    probability: float,
    *,
    generator: Optional[torch.Generator],
) -> bool:
    probability = validate_registered_probability(probability)
    draw = torch.rand((), generator=generator, device="cpu", dtype=torch.float32)
    return bool(draw.item() < probability)


def zero_or_local_forward_target(
    hr_msi: torch.Tensor,
    *,
    registered: bool,
    local_max_px: float,
    control_grid: int,
    generator: Optional[torch.Generator],
):
    """Return training MSI and forward local field for one batch mode."""
    if registered:
        b, _, h, w = hr_msi.shape
        forward = torch.zeros(
            b, 2, h, w, device=hr_msi.device, dtype=hr_msi.dtype
        )
        return hr_msi, forward

    warped, _, params = make_misaligned_msi(
        hr_msi,
        translation_max_px=0.0,
        rotation_max_deg=0.0,
        local_max_displacement_px=float(local_max_px),
        control_grid_size=int(control_grid),
        generator=generator,
    )
    return warped, params.local_displacement_px


def zero_motion_train_epoch(
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
    lambda_l1: float,
    lambda_sam: float,
    lambda_flow: float,
    lambda_oracle_anchor: float,
    grad_clip: float,
    generator: Optional[torch.Generator],
) -> ZeroMotionStats:
    model.train()
    sam_fn = SAMLoss()
    meters = {
        key: base.AverageMeter()
        for key in (
            "loss", "l1", "sam", "reverse_flow", "oracle_flow",
            "registered_fraction", "registered_pred",
            "local_epe", "local_true", "local_pred", "local_ratio",
        )
    }

    for batch in loader:
        gt = batch["gt"].to(device, non_blocking=True)
        hr_msi = batch["hr_msi"].to(device, non_blocking=True)
        b = int(gt.shape[0])
        registered = sample_registered_batch(
            _REGISTERED_PROBABILITY,
            generator=generator,
        )
        training_msi, forward_target = zero_or_local_forward_target(
            hr_msi,
            registered=registered,
            local_max_px=float(local_max_px),
            control_grid=int(control_grid),
            generator=generator,
        )
        _, targets = inverse_targets(process, model, forward_target)

        # Main domain: actual reverse states seen by the local branch at inference.
        reverse_states = rollout_reverse_states_identity(
            model, process, gt, training_msi
        )
        reverse_finals, reverse_sequences = run_local_flow_from_states(
            model, reverse_states, training_msi
        )
        reverse_flow_loss = base.stage_sequence_loss(
            reverse_sequences,
            targets,
            normalization_px=float(local_max_px),
            gamma=float(gamma),
            stage_weights=stage_weights,
        )

        # Weak oracle-state anchor uses the same zero/non-zero target definition.
        with torch.no_grad():
            oracle = oracle_states(model, process, gt)
        oracle_finals, oracle_sequences = run_local_flow_from_states(
            model, oracle, training_msi
        )
        oracle_flow_loss = base.stage_sequence_loss(
            oracle_sequences,
            targets,
            normalization_px=float(local_max_px),
            gamma=float(gamma),
            stage_weights=stage_weights,
        )

        l1, sam, _ = base.reconstruction_loss_on_reverse_states(
            model,
            reverse_states,
            training_msi,
            gt,
            reverse_finals,
            reverse_sequences,
            sam_fn,
        )
        loss = (
            float(lambda_l1) * l1
            + float(lambda_sam) * sam
            + float(lambda_flow) * reverse_flow_loss
            + float(lambda_oracle_anchor) * oracle_flow_loss
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite reverse-state zero-motion repair loss")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=float(grad_clip) if float(grad_clip) > 0.0 else float("inf"),
            error_if_nonfinite=True,
        )
        optimizer.step()
        base._clear_training_cache(model)

        meters["loss"].update(float(loss.item()), b)
        meters["l1"].update(float(l1.item()), b)
        meters["sam"].update(float(sam.item()), b)
        meters["reverse_flow"].update(float(reverse_flow_loss.item()), b)
        meters["oracle_flow"].update(float(oracle_flow_loss.item()), b)
        meters["registered_fraction"].update(1.0 if registered else 0.0, b)

        if registered:
            pred_mag = float(
                torch.linalg.vector_norm(
                    reverse_finals[1].detach().float(), dim=1
                ).mean().item()
            )
            meters["registered_pred"].update(pred_mag, b)
        else:
            epe, true_mag, pred_mag, ratio = base.field_metrics(
                reverse_finals[1], targets[1]
            )
            meters["local_epe"].update(epe, b)
            meters["local_true"].update(true_mag, b)
            meters["local_pred"].update(pred_mag, b)
            meters["local_ratio"].update(ratio, b)

    return ZeroMotionStats(
        loss=meters["loss"].avg,
        l1=meters["l1"].avg,
        sam=meters["sam"].avg,
        reverse_flow=meters["reverse_flow"].avg,
        oracle_flow=meters["oracle_flow"].avg,
        registered_fraction=meters["registered_fraction"].avg,
        registered_pred_flow_px=meters["registered_pred"].avg,
        local_reverse_epe_px=meters["local_epe"].avg,
        local_true_mean_px=meters["local_true"].avg,
        local_pred_mean_px=meters["local_pred"].avg,
        local_ratio=meters["local_ratio"].avg,
    )


@torch.no_grad()
def evaluate_registered_zero_motion(model, loader, process, device, *, scale_ratio: int):
    """Evaluate pristine registered MSI with global identity and local branch active."""
    model.eval()
    psnrs = []
    sams = []
    pred_flows = []

    for batch in loader:
        gt = batch["gt"].to(device, non_blocking=True)
        hr_msi = batch["hr_msi"].to(device, non_blocking=True)
        terminal_lr = process.terminal_observation(gt)
        pred = reconstruct_local_only_identity(
            model,
            process,
            terminal_lr,
            target_size=tuple(gt.shape[-2:]),
            warped_msi=hr_msi,
        )
        metrics = calc_metrics(pred, gt, int(scale_ratio))
        psnrs.append(float(metrics["PSNR"]))
        sams.append(float(metrics["SAM"]))
        flow = model._inference_local_offset
        if flow is None:
            raise RuntimeError("registered evaluation did not produce final local flow")
        pred_flows.append(
            float(torch.linalg.vector_norm(flow.float(), dim=1).mean().item())
        )

    return {
        "PSNR": float(np.mean(psnrs)),
        "SAM": float(np.mean(sams)),
        "pred_flow": float(np.mean(pred_flows)),
    }


def parse_zero_motion_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--zero_motion_registered_probability", type=float, default=0.30)
    parser.add_argument("--zero_motion_min_registered_psnr", type=float, default=43.8)
    parser.add_argument("--zero_motion_min_local_psnr", type=float, default=39.3)
    parser.add_argument("--zero_motion_eval_interval", type=int, default=2)
    zero_args, remaining = parser.parse_known_args()

    original_argv = list(sys.argv)
    try:
        sys.argv = [original_argv[0]] + remaining
        cfg, args = base.parse_reverse_reconstruction_args()
    finally:
        sys.argv = original_argv

    zero_args.zero_motion_registered_probability = validate_registered_probability(
        zero_args.zero_motion_registered_probability
    )
    if zero_args.zero_motion_eval_interval < 1:
        raise ValueError("zero_motion_eval_interval must be >= 1")
    return cfg, args, zero_args


def _output_paths(cfg, args):
    root = os.path.join(cfg.checkpoint_root, "innovation1")
    base.ensure_dir(root)
    if args.reverse_recon_save_name:
        stem = args.reverse_recon_save_name.removesuffix(".pth")
    else:
        stem = (
            f"{cfg.dataset}_v4_reverse_state_zero_motion"
            f"_r{int(round(100.0 * _REGISTERED_PROBABILITY))}"
            f"_l{base._compact_float_tag(args.reverse_recon_local_max_px)}"
        )
    return (
        os.path.join(root, stem + ".pth"),
        os.path.join(root, stem + "_last.pth"),
        os.path.join(cfg.log_root, stem + ".csv"),
    )


def run(cfg, args, zero_args):
    global _REGISTERED_PROBABILITY
    _REGISTERED_PROBABILITY = float(zero_args.zero_motion_registered_probability)

    base.set_seed(cfg.seed)
    train_loader, test_loader, info = base.build_loaders(cfg)
    device = base.get_device(cfg.device)
    process = base.build_progressive_process(cfg)
    model = base.build_recurrent_model(cfg, info, device, process)
    source_epoch, source_metric = base.load_checkpoint(
        model,
        args.reverse_recon_init_checkpoint,
        optimizer=None,
        strict=True,
        map_location=str(device),
        load_optimizer=False,
    )
    base.freeze_to_local_branch(model)

    stage_weights = {
        4: float(args.reverse_recon_weight_scale4),
        2: float(args.reverse_recon_weight_scale2),
        1: float(args.reverse_recon_weight_scale1),
    }
    print(
        "Zero-motion repair warm start: "
        f"{args.reverse_recon_init_checkpoint} "
        f"(epoch={source_epoch}, stored_metric={source_metric:.6f})"
    )
    print(
        f"Protocol: registered={100.0*_REGISTERED_PROBABILITY:.0f}% + "
        f"local<={args.reverse_recon_local_max_px:g}px="
        f"{100.0*(1.0-_REGISTERED_PROBABILITY):.0f}%, global=identity."
    )
    print(
        "Architecture unchanged: reverse-state + inverse-flow + physical 4->2->1 + "
        "Raw-MSI reconstruction + weak oracle anchor; only recurrent local branch trains."
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

    best_path, last_path, log_path = _output_paths(cfg, args)
    logger = base.CSVLogger(
        log_path,
        fieldnames=[
            "epoch", "loss", "l1", "sam", "reverse_flow", "oracle_flow",
            "train_registered_fraction", "train_registered_pred_flow",
            "train_local_epe", "train_local_true", "train_local_pred", "train_local_ratio",
            "registered_PSNR", "registered_SAM", "registered_pred_flow",
            "local_PSNR", "local_SAM", "local_EPE", "local_true", "local_pred", "local_ratio",
            "joint_score", "best_joint_score",
        ],
    )

    eval_seed = int(cfg.seed) + 15431
    reference_severity = float(args.reverse_recon_local_max_px)
    best_score = float("-inf")

    # Epoch-0 diagnosis confirms the failure mode before repair begins.
    registered_eval = evaluate_registered_zero_motion(
        model, test_loader, process, device, scale_ratio=int(cfg.scale_ratio)
    )
    local_eval_all = evaluate_inverse_local_only_reconstruction(
        model,
        test_loader,
        process,
        device,
        severities=[reference_severity],
        control_grid=int(args.reverse_recon_control_grid),
        trials=int(args.reverse_recon_eval_trials),
        valid_threshold=float(args.reverse_recon_valid_threshold),
        seed=eval_seed,
    )
    local_eval = local_eval_all[reference_severity]
    print(
        "Epoch 0000 baseline | "
        f"registered={registered_eval['PSNR']:.3f}/{registered_eval['SAM']:.3f} "
        f"flow={registered_eval['pred_flow']:.4f}px | "
        f"local<={reference_severity:g}={local_eval['PSNR']:.3f}/{local_eval['SAM']:.3f} "
        f"EPE={local_eval['EPE']:.4f} ratio={local_eval['ratio']:.3f}"
    )

    for epoch in range(1, int(cfg.epochs) + 1):
        stats = zero_motion_train_epoch(
            model,
            train_loader,
            optimizer,
            process,
            device,
            local_max_px=reference_severity,
            control_grid=int(args.reverse_recon_control_grid),
            gamma=float(args.reverse_recon_gamma),
            stage_weights=stage_weights,
            lambda_l1=float(cfg.lambda_l1),
            lambda_sam=float(cfg.lambda_sam),
            lambda_flow=float(args.lambda_flow),
            lambda_oracle_anchor=float(args.lambda_oracle_anchor),
            grad_clip=float(cfg.grad_clip),
            generator=generator,
        )
        print(
            f"Epoch {epoch:04d}/{cfg.epochs:04d} loss={stats.loss:.6f} "
            f"reg_frac={stats.registered_fraction:.2f} "
            f"reg_pred={stats.registered_pred_flow_px:.4f}px "
            f"local_EPE={stats.local_reverse_epe_px:.4f} "
            f"local_pred/true={stats.local_pred_mean_px:.3f}/{stats.local_true_mean_px:.3f}"
        )

        evaluation_due = (
            epoch % int(zero_args.zero_motion_eval_interval) == 0
            or epoch == int(cfg.epochs)
        )
        row = {
            "epoch": epoch,
            "loss": stats.loss,
            "l1": stats.l1,
            "sam": stats.sam,
            "reverse_flow": stats.reverse_flow,
            "oracle_flow": stats.oracle_flow,
            "train_registered_fraction": stats.registered_fraction,
            "train_registered_pred_flow": stats.registered_pred_flow_px,
            "train_local_epe": stats.local_reverse_epe_px,
            "train_local_true": stats.local_true_mean_px,
            "train_local_pred": stats.local_pred_mean_px,
            "train_local_ratio": stats.local_ratio,
            "best_joint_score": best_score,
        }

        if evaluation_due:
            registered_eval = evaluate_registered_zero_motion(
                model, test_loader, process, device, scale_ratio=int(cfg.scale_ratio)
            )
            local_eval_all = evaluate_inverse_local_only_reconstruction(
                model,
                test_loader,
                process,
                device,
                severities=[reference_severity],
                control_grid=int(args.reverse_recon_control_grid),
                trials=int(args.reverse_recon_eval_trials),
                valid_threshold=float(args.reverse_recon_valid_threshold),
                seed=eval_seed,
            )
            local_eval = local_eval_all[reference_severity]
            score = float(registered_eval["PSNR"]) + float(local_eval["PSNR"])
            flow_ok = (
                float(args.reverse_recon_flow_ratio_min)
                <= float(local_eval["ratio"])
                <= float(args.reverse_recon_flow_ratio_max)
            )
            reg_ok = float(registered_eval["PSNR"]) >= float(
                zero_args.zero_motion_min_registered_psnr
            )
            local_ok = float(local_eval["PSNR"]) >= float(
                zero_args.zero_motion_min_local_psnr
            )
            print(
                f"  registered: PSNR={registered_eval['PSNR']:.3f} "
                f"SAM={registered_eval['SAM']:.3f} "
                f"pred_flow={registered_eval['pred_flow']:.4f}px"
            )
            print(
                f"  local<={reference_severity:g}: PSNR={local_eval['PSNR']:.3f} "
                f"SAM={local_eval['SAM']:.3f} EPE={local_eval['EPE']:.4f} "
                f"flow={local_eval['pred']:.3f}/{local_eval['true']:.3f} "
                f"ratio={local_eval['ratio']:.3f}"
            )
            print(
                f"  guards: registered={reg_ok} local={local_ok} flow={flow_ok} "
                f"joint_score={score:.3f}"
            )

            if reg_ok and local_ok and flow_ok and score > best_score:
                best_score = score
                base.save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    best_score,
                    best_path,
                    extra={
                        "config": vars(cfg),
                        "zero_motion": vars(zero_args),
                        "reverse_state_reconstruction": vars(args),
                        "registered_evaluation": registered_eval,
                        "local_evaluation": local_eval,
                        "source_checkpoint": args.reverse_recon_init_checkpoint,
                    },
                )
                print(f"  saved best zero-motion checkpoint -> {best_path}")

            row.update({
                "registered_PSNR": registered_eval["PSNR"],
                "registered_SAM": registered_eval["SAM"],
                "registered_pred_flow": registered_eval["pred_flow"],
                "local_PSNR": local_eval["PSNR"],
                "local_SAM": local_eval["SAM"],
                "local_EPE": local_eval["EPE"],
                "local_true": local_eval["true"],
                "local_pred": local_eval["pred"],
                "local_ratio": local_eval["ratio"],
                "joint_score": score,
                "best_joint_score": best_score,
            })

        logger.write(row)

        if epoch % int(cfg.save_interval) == 0 or epoch == int(cfg.epochs):
            base.save_checkpoint(
                model,
                optimizer,
                epoch,
                best_score,
                last_path,
                extra={
                    "config": vars(cfg),
                    "zero_motion": vars(zero_args),
                    "reverse_state_reconstruction": vars(args),
                    "source_checkpoint": args.reverse_recon_init_checkpoint,
                },
            )

    print("Zero-motion preservation repair complete.")
    print(f"Best joint score: {best_score:.6f}")
    print(f"Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")
    print(f"Log: {log_path}")


def main():
    cfg, args, zero_args = parse_zero_motion_args()
    if not args.reverse_recon_save_name:
        # Leave output naming to _output_paths, which includes registered ratio.
        args.reverse_recon_save_name = ""
    run(cfg, args, zero_args)


if __name__ == "__main__":
    main()
