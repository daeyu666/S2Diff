"""Paired zero-motion repair for the final recurrent Innovation-2 local branch.

Every optimization step uses the same HSI/MSI batch twice:
1. local branch: synthesize local-only non-rigid warp and retain the complete
   reverse-state inverse-flow + Raw-MSI reconstruction objective;
2. registered branch: keep pristine HR-MSI and supervise the recurrent 4->2->1
   flow to remain zero, with a weak registered reconstruction term.

This avoids the task alternation/forgetting observed with the earlier 30/70
batch-mixture repair.  Architecture, global-identity protocol, recurrent GRU,
correlation evidence, inverse-flow geometry and 4->2->1 schedule are unchanged.

Loss:
    L_local = L1_local + 0.1*SAM_local + lambda_flow*Lflow_local
              + lambda_oracle_anchor*Loracle_local
    L_zero  = lambda_flow*Lflow_zero
              + lambda_oracle_anchor*Loracle_zero
    L_reg   = L1_registered + 0.1*SAM_registered
    L       = L_local + lambda_zero*L_zero
              + lambda_registered_recon*L_reg

Defaults:
    lambda_zero = 1.5
    lambda_registered_recon = 0.25

Checkpoint selection is continuous rather than hard-gated.  The launcher saves
best-joint (registered PSNR + local PSNR), best-registered, best-local and last.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional

import torch

import train_v4_reverse_state_reconstruction as base
from degradations.misalignment import make_misaligned_msi
from losses import SAMLoss
from reverse_state_local_flow import (
    oracle_states,
    rollout_reverse_states_identity,
    run_local_flow_from_states,
)
from train_v4_local_reconstruction_inverse import (
    evaluate_inverse_local_only_reconstruction,
)
from train_v4_reverse_state_flow import inverse_targets
from train_v4_reverse_state_zero_motion import evaluate_registered_zero_motion


@dataclass
class PairedZeroStats:
    loss: float
    local_loss: float
    zero_loss: float
    registered_recon_loss: float
    local_l1: float
    local_sam: float
    local_reverse_flow: float
    local_oracle_flow: float
    registered_l1: float
    registered_sam: float
    registered_reverse_flow: float
    registered_oracle_flow: float
    registered_pred_flow_px: float
    local_epe_px: float
    local_true_mean_px: float
    local_pred_mean_px: float
    local_ratio: float


def validate_paired_weights(lambda_zero: float, lambda_registered_recon: float):
    lambda_zero = float(lambda_zero)
    lambda_registered_recon = float(lambda_registered_recon)
    if lambda_zero <= 0.0:
        raise ValueError("paired lambda_zero must be > 0")
    if lambda_registered_recon < 0.0:
        raise ValueError("paired lambda_registered_recon must be >= 0")
    return lambda_zero, lambda_registered_recon


def compose_paired_loss(
    local_loss: torch.Tensor,
    zero_loss: torch.Tensor,
    registered_recon_loss: torch.Tensor,
    *,
    lambda_zero: float,
    lambda_registered_recon: float,
) -> torch.Tensor:
    lambda_zero, lambda_registered_recon = validate_paired_weights(
        lambda_zero, lambda_registered_recon
    )
    return (
        local_loss
        + lambda_zero * zero_loss
        + lambda_registered_recon * registered_recon_loss
    )


def paired_joint_score(registered_psnr: float, local_psnr: float) -> float:
    return float(registered_psnr) + float(local_psnr)


def _flow_branch(
    model,
    process,
    gt: torch.Tensor,
    msi: torch.Tensor,
    targets,
    oracle,
    *,
    normalization_px: float,
    gamma: float,
    stage_weights: Dict[int, float],
):
    reverse_states = rollout_reverse_states_identity(model, process, gt, msi)
    finals, sequences = run_local_flow_from_states(model, reverse_states, msi)
    reverse_flow = base.stage_sequence_loss(
        sequences,
        targets,
        normalization_px=float(normalization_px),
        gamma=float(gamma),
        stage_weights=stage_weights,
    )
    oracle_finals, oracle_sequences = run_local_flow_from_states(
        model, oracle, msi
    )
    oracle_flow = base.stage_sequence_loss(
        oracle_sequences,
        targets,
        normalization_px=float(normalization_px),
        gamma=float(gamma),
        stage_weights=stage_weights,
    )
    return reverse_states, finals, sequences, reverse_flow, oracle_flow


def paired_train_epoch(
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
    lambda_zero: float,
    lambda_registered_recon: float,
    grad_clip: float,
    generator: Optional[torch.Generator],
) -> PairedZeroStats:
    model.train()
    sam_fn = SAMLoss()
    meters = {
        key: base.AverageMeter()
        for key in (
            "loss", "local_loss", "zero_loss", "reg_recon_loss",
            "local_l1", "local_sam", "local_rev", "local_oracle",
            "reg_l1", "reg_sam", "reg_rev", "reg_oracle", "reg_pred",
            "local_epe", "local_true", "local_pred", "local_ratio",
        )
    }

    for batch in loader:
        gt = batch["gt"].to(device, non_blocking=True)
        hr_msi = batch["hr_msi"].to(device, non_blocking=True)
        b = int(gt.shape[0])

        # One oracle HSI pyramid is shared; matching still uses branch-specific MSI.
        with torch.no_grad():
            oracle = oracle_states(model, process, gt)

        # ----- Local branch: keep the validated non-rigid task on every step. -----
        warped, _, params = make_misaligned_msi(
            hr_msi,
            translation_max_px=0.0,
            rotation_max_deg=0.0,
            local_max_displacement_px=float(local_max_px),
            control_grid_size=int(control_grid),
            generator=generator,
        )
        _, local_targets = inverse_targets(
            process, model, params.local_displacement_px
        )
        (
            local_states,
            local_finals,
            local_sequences,
            local_reverse_flow,
            local_oracle_flow,
        ) = _flow_branch(
            model,
            process,
            gt,
            warped,
            local_targets,
            oracle,
            normalization_px=float(local_max_px),
            gamma=float(gamma),
            stage_weights=stage_weights,
        )
        local_l1, local_sam, _ = base.reconstruction_loss_on_reverse_states(
            model,
            local_states,
            warped,
            gt,
            local_finals,
            local_sequences,
            sam_fn,
        )
        local_loss = (
            float(lambda_l1) * local_l1
            + float(lambda_sam) * local_sam
            + float(lambda_flow) * local_reverse_flow
            + float(lambda_oracle_anchor) * local_oracle_flow
        )
        base._clear_training_cache(model)

        # ----- Registered branch: identity evidence must produce zero flow. -----
        zero_forward = torch.zeros(
            b,
            2,
            hr_msi.shape[-2],
            hr_msi.shape[-1],
            device=hr_msi.device,
            dtype=hr_msi.dtype,
        )
        _, zero_targets = inverse_targets(process, model, zero_forward)
        (
            reg_states,
            reg_finals,
            reg_sequences,
            reg_reverse_flow,
            reg_oracle_flow,
        ) = _flow_branch(
            model,
            process,
            gt,
            hr_msi,
            zero_targets,
            oracle,
            normalization_px=float(local_max_px),
            gamma=float(gamma),
            stage_weights=stage_weights,
        )
        reg_l1, reg_sam, _ = base.reconstruction_loss_on_reverse_states(
            model,
            reg_states,
            hr_msi,
            gt,
            reg_finals,
            reg_sequences,
            sam_fn,
        )
        zero_loss = (
            float(lambda_flow) * reg_reverse_flow
            + float(lambda_oracle_anchor) * reg_oracle_flow
        )
        registered_recon_loss = (
            float(lambda_l1) * reg_l1 + float(lambda_sam) * reg_sam
        )

        loss = compose_paired_loss(
            local_loss,
            zero_loss,
            registered_recon_loss,
            lambda_zero=float(lambda_zero),
            lambda_registered_recon=float(lambda_registered_recon),
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite paired zero-motion repair loss")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=float(grad_clip) if float(grad_clip) > 0.0 else float("inf"),
            error_if_nonfinite=True,
        )
        optimizer.step()
        base._clear_training_cache(model)

        local_epe, local_true, local_pred, local_ratio = base.field_metrics(
            local_finals[1], local_targets[1]
        )
        reg_pred = float(
            torch.linalg.vector_norm(
                reg_finals[1].detach().float(), dim=1
            ).mean().item()
        )
        values = {
            "loss": float(loss.item()),
            "local_loss": float(local_loss.item()),
            "zero_loss": float(zero_loss.item()),
            "reg_recon_loss": float(registered_recon_loss.item()),
            "local_l1": float(local_l1.item()),
            "local_sam": float(local_sam.item()),
            "local_rev": float(local_reverse_flow.item()),
            "local_oracle": float(local_oracle_flow.item()),
            "reg_l1": float(reg_l1.item()),
            "reg_sam": float(reg_sam.item()),
            "reg_rev": float(reg_reverse_flow.item()),
            "reg_oracle": float(reg_oracle_flow.item()),
            "reg_pred": reg_pred,
            "local_epe": local_epe,
            "local_true": local_true,
            "local_pred": local_pred,
            "local_ratio": local_ratio,
        }
        for key, value in values.items():
            meters[key].update(value, b)

    return PairedZeroStats(
        loss=meters["loss"].avg,
        local_loss=meters["local_loss"].avg,
        zero_loss=meters["zero_loss"].avg,
        registered_recon_loss=meters["reg_recon_loss"].avg,
        local_l1=meters["local_l1"].avg,
        local_sam=meters["local_sam"].avg,
        local_reverse_flow=meters["local_rev"].avg,
        local_oracle_flow=meters["local_oracle"].avg,
        registered_l1=meters["reg_l1"].avg,
        registered_sam=meters["reg_sam"].avg,
        registered_reverse_flow=meters["reg_rev"].avg,
        registered_oracle_flow=meters["reg_oracle"].avg,
        registered_pred_flow_px=meters["reg_pred"].avg,
        local_epe_px=meters["local_epe"].avg,
        local_true_mean_px=meters["local_true"].avg,
        local_pred_mean_px=meters["local_pred"].avg,
        local_ratio=meters["local_ratio"].avg,
    )


def parse_paired_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--paired_lambda_zero", type=float, default=1.5)
    parser.add_argument("--paired_lambda_registered_recon", type=float, default=0.25)
    parser.add_argument("--paired_eval_interval", type=int, default=1)
    paired, remaining = parser.parse_known_args()

    original_argv = list(sys.argv)
    try:
        sys.argv = [original_argv[0]] + remaining
        cfg, args = base.parse_reverse_reconstruction_args()
    finally:
        sys.argv = original_argv

    (
        paired.paired_lambda_zero,
        paired.paired_lambda_registered_recon,
    ) = validate_paired_weights(
        paired.paired_lambda_zero,
        paired.paired_lambda_registered_recon,
    )
    if paired.paired_eval_interval < 1:
        raise ValueError("paired_eval_interval must be >= 1")
    return cfg, args, paired


def _output_paths(cfg, args, paired):
    root = os.path.join(cfg.checkpoint_root, "innovation1")
    base.ensure_dir(root)
    if args.reverse_recon_save_name:
        stem = args.reverse_recon_save_name.removesuffix(".pth")
    else:
        stem = (
            f"{cfg.dataset}_v4_reverse_state_zero_motion_paired"
            f"_l{base._compact_float_tag(args.reverse_recon_local_max_px)}"
            f"_z{base._compact_float_tag(paired.paired_lambda_zero)}"
            f"_rr{base._compact_float_tag(paired.paired_lambda_registered_recon)}"
        )
    return {
        "joint": os.path.join(root, stem + ".pth"),
        "registered": os.path.join(root, stem + "_best_registered.pth"),
        "local": os.path.join(root, stem + "_best_local.pth"),
        "last": os.path.join(root, stem + "_last.pth"),
        "log": os.path.join(cfg.log_root, stem + ".csv"),
    }


def run(cfg, args, paired):
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
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        int(cfg.seed) + int(getattr(cfg, "train_misalignment_seed_offset", 7919))
    )

    paths = _output_paths(cfg, args, paired)
    logger = base.CSVLogger(
        paths["log"],
        fieldnames=[
            "epoch", "loss", "local_loss", "zero_loss", "registered_recon_loss",
            "train_registered_pred_flow", "train_local_epe", "train_local_true",
            "train_local_pred", "train_local_ratio",
            "registered_PSNR", "registered_SAM", "registered_pred_flow",
            "local_PSNR", "local_SAM", "local_EPE", "local_true", "local_pred",
            "local_ratio", "joint_score", "best_joint_score",
            "best_registered_PSNR", "best_local_PSNR",
        ],
    )

    print(
        "Paired zero-motion warm start: "
        f"{args.reverse_recon_init_checkpoint} "
        f"(epoch={source_epoch}, stored_metric={source_metric:.6f})"
    )
    print(
        "Protocol: every batch = local<=%.3gpx branch + registered identity branch; "
        "global=identity." % float(args.reverse_recon_local_max_px)
    )
    print(
        f"Weights: lambda_zero={paired.paired_lambda_zero:g}, "
        f"lambda_registered_recon={paired.paired_lambda_registered_recon:g}, "
        f"lr={cfg.lr:g}."
    )

    eval_seed = int(cfg.seed) + 15431
    severity = float(args.reverse_recon_local_max_px)
    best_joint = float("-inf")
    best_registered = float("-inf")
    best_local = float("-inf")

    def evaluate_pair():
        reg = evaluate_registered_zero_motion(
            model, test_loader, process, device, scale_ratio=int(cfg.scale_ratio)
        )
        loc = evaluate_inverse_local_only_reconstruction(
            model,
            test_loader,
            process,
            device,
            severities=[severity],
            control_grid=int(args.reverse_recon_control_grid),
            trials=int(args.reverse_recon_eval_trials),
            valid_threshold=float(args.reverse_recon_valid_threshold),
            seed=eval_seed,
        )[severity]
        return reg, loc

    registered_eval, local_eval = evaluate_pair()
    print(
        "Epoch 0000 baseline | "
        f"registered={registered_eval['PSNR']:.3f}/{registered_eval['SAM']:.3f} "
        f"flow={registered_eval['pred_flow']:.4f}px | "
        f"local<={severity:g}={local_eval['PSNR']:.3f}/{local_eval['SAM']:.3f} "
        f"EPE={local_eval['EPE']:.4f} ratio={local_eval['ratio']:.3f}"
    )

    for epoch in range(1, int(cfg.epochs) + 1):
        stats = paired_train_epoch(
            model,
            train_loader,
            optimizer,
            process,
            device,
            local_max_px=severity,
            control_grid=int(args.reverse_recon_control_grid),
            gamma=float(args.reverse_recon_gamma),
            stage_weights=stage_weights,
            lambda_l1=float(cfg.lambda_l1),
            lambda_sam=float(cfg.lambda_sam),
            lambda_flow=float(args.lambda_flow),
            lambda_oracle_anchor=float(args.lambda_oracle_anchor),
            lambda_zero=float(paired.paired_lambda_zero),
            lambda_registered_recon=float(paired.paired_lambda_registered_recon),
            grad_clip=float(cfg.grad_clip),
            generator=generator,
        )
        print(
            f"Epoch {epoch:04d}/{cfg.epochs:04d} loss={stats.loss:.6f} "
            f"local={stats.local_loss:.6f} zero={stats.zero_loss:.6f} "
            f"reg_recon={stats.registered_recon_loss:.6f} "
            f"reg_pred={stats.registered_pred_flow_px:.4f}px "
            f"local_EPE={stats.local_epe_px:.4f} "
            f"local_pred/true={stats.local_pred_mean_px:.3f}/{stats.local_true_mean_px:.3f}"
        )

        row = {
            "epoch": epoch,
            "loss": stats.loss,
            "local_loss": stats.local_loss,
            "zero_loss": stats.zero_loss,
            "registered_recon_loss": stats.registered_recon_loss,
            "train_registered_pred_flow": stats.registered_pred_flow_px,
            "train_local_epe": stats.local_epe_px,
            "train_local_true": stats.local_true_mean_px,
            "train_local_pred": stats.local_pred_mean_px,
            "train_local_ratio": stats.local_ratio,
            "best_joint_score": best_joint,
            "best_registered_PSNR": best_registered,
            "best_local_PSNR": best_local,
        }

        evaluation_due = (
            epoch % int(paired.paired_eval_interval) == 0
            or epoch == int(cfg.epochs)
        )
        if evaluation_due:
            registered_eval, local_eval = evaluate_pair()
            score = paired_joint_score(
                registered_eval["PSNR"], local_eval["PSNR"]
            )
            print(
                f"  registered: PSNR={registered_eval['PSNR']:.3f} "
                f"SAM={registered_eval['SAM']:.3f} "
                f"pred_flow={registered_eval['pred_flow']:.4f}px"
            )
            print(
                f"  local<={severity:g}: PSNR={local_eval['PSNR']:.3f} "
                f"SAM={local_eval['SAM']:.3f} EPE={local_eval['EPE']:.4f} "
                f"flow={local_eval['pred']:.3f}/{local_eval['true']:.3f} "
                f"ratio={local_eval['ratio']:.3f} | joint={score:.3f}"
            )

            extra = {
                "config": vars(cfg),
                "paired_zero_motion": vars(paired),
                "reverse_state_reconstruction": vars(args),
                "registered_evaluation": registered_eval,
                "local_evaluation": local_eval,
                "source_checkpoint": args.reverse_recon_init_checkpoint,
            }
            if score > best_joint:
                best_joint = score
                base.save_checkpoint(
                    model, optimizer, epoch, best_joint, paths["joint"], extra=extra
                )
                print(f"  saved best joint -> {paths['joint']}")
            if float(registered_eval["PSNR"]) > best_registered:
                best_registered = float(registered_eval["PSNR"])
                base.save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    best_registered,
                    paths["registered"],
                    extra=extra,
                )
            if float(local_eval["PSNR"]) > best_local:
                best_local = float(local_eval["PSNR"])
                base.save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    best_local,
                    paths["local"],
                    extra=extra,
                )

            row.update(
                {
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
                    "best_joint_score": best_joint,
                    "best_registered_PSNR": best_registered,
                    "best_local_PSNR": best_local,
                }
            )

        logger.write(row)
        if epoch % int(cfg.save_interval) == 0 or epoch == int(cfg.epochs):
            base.save_checkpoint(
                model,
                optimizer,
                epoch,
                best_joint,
                paths["last"],
                extra={
                    "config": vars(cfg),
                    "paired_zero_motion": vars(paired),
                    "reverse_state_reconstruction": vars(args),
                    "registered_evaluation": registered_eval,
                    "local_evaluation": local_eval,
                    "source_checkpoint": args.reverse_recon_init_checkpoint,
                },
            )

    print("Paired zero-motion repair complete.")
    print(f"Best joint checkpoint: {paths['joint']} ({best_joint:.6f})")
    print(f"Best registered checkpoint: {paths['registered']} ({best_registered:.6f})")
    print(f"Best local checkpoint: {paths['local']} ({best_local:.6f})")
    print(f"Last checkpoint: {paths['last']}")
    print(f"Log: {paths['log']}")


if __name__ == "__main__":
    cfg, args, paired = parse_paired_args()
    run(cfg, args, paired)
