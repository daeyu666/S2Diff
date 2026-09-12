"""Reverse-state reconstruction fine-tune for Innovation-2 local alignment.

This is the final local-only validation stage after reverse-state flow adaptation.
It keeps the global rigid branch at identity and trains only the recurrent local
branch.

For each batch:
1. synthesize a local-only MSI warp and convert forward displacement d to the
   geometrically correct inverse sampling target u;
2. run the model's real deterministic reverse trajectory under no-grad/eval and
   cache reverse HSI states at the 4/2/1 representative timesteps;
3. estimate differentiable recurrent local flow from those reverse states;
4. supervise the reverse-state flow with physical 4/2/1 inverse-flow targets;
5. cache the reverse-state flows in the predictor and compute Raw-MSI
   reconstruction loss on the same reverse states at t4/t2/t1;
6. add a weak oracle-state flow anchor to preserve the already learned clean
   physical-domain matching ability.

Loss:
    L = lambda_l1 * L1_reverse_reconstruction
      + lambda_sam * SAM_reverse_reconstruction
      + lambda_flow * L_reverse_flow
      + lambda_oracle_anchor * L_oracle_flow

Innovation-1 and the reconstruction backbone remain frozen; gradients reach the
local recurrent branch through both flow supervision and differentiable Raw-MSI
sampling.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from config import parse_args
from data_loader import build_loaders
from degradations.misalignment import make_misaligned_msi
from innovation1 import build_progressive_process, model_predict
from losses import SAMLoss
from main import _compact_float_tag
from reverse_state_local_flow import (
    SCALES,
    oracle_states,
    rollout_reverse_states_identity,
    run_local_flow_from_states,
)
from train_v4_local_bootstrap import freeze_to_local_branch
from train_v4_local_reconstruction_inverse import (
    evaluate_inverse_local_only_reconstruction,
)
from train_v4_multiscale_bootstrap import field_metrics, stage_sequence_loss
from train_v4_recurrent_flow import build_recurrent_model
from train_v4_reverse_state_flow import evaluate_state_modes, inverse_targets
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
class ReverseReconstructionStats:
    loss: float
    l1: float
    sam: float
    reverse_flow: float
    oracle_flow: float
    reverse_epe_px: float
    reverse_true_mean_px: float
    reverse_pred_mean_px: float
    reverse_ratio: float
    oracle_epe_px: float


def _set_reverse_training_cache(model, finals, sequences, reference: torch.Tensor) -> None:
    model._training_local_cache = finals
    if hasattr(model, "_training_sequence_cache"):
        model._training_sequence_cache = sequences
    model.last_global_shift_px = torch.zeros(
        reference.shape[0], 2, device=reference.device, dtype=reference.dtype
    )
    model.last_global_rotation_deg = torch.zeros(
        reference.shape[0], device=reference.device, dtype=reference.dtype
    )
    if hasattr(model, "_reset_inference_local_state"):
        model._reset_inference_local_state()


def _clear_training_cache(model) -> None:
    model._training_local_cache = {}
    if hasattr(model, "_training_sequence_cache"):
        model._training_sequence_cache = {}


def reconstruction_loss_on_reverse_states(
    model,
    reverse_states: Dict[int, torch.Tensor],
    warped_msi: torch.Tensor,
    gt: torch.Tensor,
    reverse_finals: Dict[int, torch.Tensor],
    reverse_sequences,
    sam_fn: SAMLoss,
):
    """Reconstruct X from the same reverse-state domain used by local flow."""
    _set_reverse_training_cache(model, reverse_finals, reverse_sequences, gt)
    l1_terms = []
    sam_terms = []
    predictions = {}
    for scale in SCALES:
        x_state = reverse_states[scale]
        t_stage = int(model.geometry_aligner.stage_t_by_scale[scale])
        timesteps = torch.full(
            (gt.shape[0],), t_stage, dtype=torch.long, device=gt.device
        )
        pred = model_predict(
            model,
            x_state,
            timesteps,
            hr_msi=warped_msi,
        )
        predictions[scale] = pred
        l1_terms.append(F.l1_loss(pred, gt))
        sam_terms.append(sam_fn(pred, gt))
    l1 = torch.stack(l1_terms).mean()
    sam = torch.stack(sam_terms).mean()
    return l1, sam, predictions


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
    lambda_l1: float,
    lambda_sam: float,
    lambda_flow: float,
    lambda_oracle_anchor: float,
    grad_clip: float,
    generator: Optional[torch.Generator],
) -> ReverseReconstructionStats:
    model.train()
    sam_fn = SAMLoss()
    meters = {
        key: AverageMeter()
        for key in (
            "loss", "l1", "sam", "reverse_flow", "oracle_flow",
            "reverse_epe", "reverse_true", "reverse_pred", "reverse_ratio",
            "oracle_epe",
        )
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

        # Main domain: the exact HSI state distribution seen during inference.
        reverse_states = rollout_reverse_states_identity(model, process, gt, warped)
        reverse_finals, reverse_sequences = run_local_flow_from_states(
            model, reverse_states, warped
        )
        reverse_flow_loss = stage_sequence_loss(
            reverse_sequences,
            targets,
            normalization_px=float(local_max_px),
            gamma=float(gamma),
            stage_weights=stage_weights,
        )

        # Weak clean-domain anchor. It stabilizes matching but never drives reconstruction.
        with torch.no_grad():
            oracle = oracle_states(model, process, gt)
        oracle_finals, oracle_sequences = run_local_flow_from_states(model, oracle, warped)
        oracle_flow_loss = stage_sequence_loss(
            oracle_sequences,
            targets,
            normalization_px=float(local_max_px),
            gamma=float(gamma),
            stage_weights=stage_weights,
        )

        l1, sam, _ = reconstruction_loss_on_reverse_states(
            model,
            reverse_states,
            warped,
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
            raise FloatingPointError("non-finite reverse-state reconstruction loss")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=float(grad_clip) if float(grad_clip) > 0.0 else float("inf"),
            error_if_nonfinite=True,
        )
        optimizer.step()
        _clear_training_cache(model)

        re, true_mag, pred_mag, ratio = field_metrics(
            reverse_finals[1], targets[1]
        )
        oe, _, _, _ = field_metrics(oracle_finals[1], targets[1])
        for key, value in {
            "loss": float(loss.item()),
            "l1": float(l1.item()),
            "sam": float(sam.item()),
            "reverse_flow": float(reverse_flow_loss.item()),
            "oracle_flow": float(oracle_flow_loss.item()),
            "reverse_epe": re,
            "reverse_true": true_mag,
            "reverse_pred": pred_mag,
            "reverse_ratio": ratio,
            "oracle_epe": oe,
        }.items():
            meters[key].update(value, b)

    return ReverseReconstructionStats(
        loss=meters["loss"].avg,
        l1=meters["l1"].avg,
        sam=meters["sam"].avg,
        reverse_flow=meters["reverse_flow"].avg,
        oracle_flow=meters["oracle_flow"].avg,
        reverse_epe_px=meters["reverse_epe"].avg,
        reverse_true_mean_px=meters["reverse_true"].avg,
        reverse_pred_mean_px=meters["reverse_pred"].avg,
        reverse_ratio=meters["reverse_ratio"].avg,
        oracle_epe_px=meters["oracle_epe"].avg,
    )


def parse_reverse_reconstruction_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--reverse_recon_init_checkpoint", type=str, required=True)
    parser.add_argument("--reverse_recon_local_max_px", type=float, default=1.0)
    parser.add_argument("--reverse_recon_control_grid", type=int, default=5)
    parser.add_argument("--reverse_recon_gamma", type=float, default=0.8)
    parser.add_argument("--reverse_recon_weight_scale4", type=float, default=0.5)
    parser.add_argument("--reverse_recon_weight_scale2", type=float, default=0.7)
    parser.add_argument("--reverse_recon_weight_scale1", type=float, default=1.0)
    parser.add_argument("--lambda_flow", type=float, default=1.0)
    parser.add_argument("--lambda_oracle_anchor", type=float, default=0.2)
    parser.add_argument(
        "--reverse_recon_eval_severities",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0],
    )
    parser.add_argument("--reverse_recon_eval_trials", type=int, default=3)
    parser.add_argument("--reverse_recon_eval_interval", type=int, default=5)
    parser.add_argument("--reverse_recon_valid_threshold", type=float, default=0.999)
    parser.add_argument("--reverse_recon_flow_ratio_min", type=float, default=0.80)
    parser.add_argument("--reverse_recon_flow_ratio_max", type=float, default=1.20)
    parser.add_argument("--reverse_recon_save_name", type=str, default="")
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
        raise ValueError("reverse-state reconstruction is train-only; use --stage train")
    if str(cfg.predictor_version).lower() != "v4":
        raise ValueError("reverse-state reconstruction requires predictor_version=v4")
    if args.reverse_recon_local_max_px <= 0.0:
        raise ValueError("reverse_recon_local_max_px must be > 0")
    if args.reverse_recon_control_grid < 2:
        raise ValueError("reverse_recon_control_grid must be >= 2")
    if not 0.0 < args.reverse_recon_gamma <= 1.0:
        raise ValueError("reverse_recon_gamma must lie in (0,1]")
    if min(
        args.reverse_recon_weight_scale4,
        args.reverse_recon_weight_scale2,
        args.reverse_recon_weight_scale1,
    ) <= 0.0:
        raise ValueError("reverse reconstruction stage weights must be > 0")
    if args.lambda_flow <= 0.0:
        raise ValueError("lambda_flow must be > 0")
    if args.lambda_oracle_anchor < 0.0:
        raise ValueError("lambda_oracle_anchor must be >= 0")
    if args.reverse_recon_eval_trials < 1 or args.reverse_recon_eval_interval < 1:
        raise ValueError("reverse reconstruction eval counts must be >= 1")
    if not 0.0 < args.reverse_recon_valid_threshold <= 1.0:
        raise ValueError("reverse_recon_valid_threshold must lie in (0,1]")
    if args.reverse_recon_flow_ratio_min <= 0.0:
        raise ValueError("flow ratio min must be > 0")
    if args.reverse_recon_flow_ratio_max < args.reverse_recon_flow_ratio_min:
        raise ValueError("flow ratio max must be >= flow ratio min")
    if any(v <= 0.0 for v in args.reverse_recon_eval_severities):
        raise ValueError("all reverse reconstruction eval severities must be > 0")

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
    if args.reverse_recon_save_name:
        stem = args.reverse_recon_save_name.removesuffix(".pth")
    else:
        stem = (
            f"{cfg.dataset}_v4_reverse_state_reconstruction"
            f"_l{_compact_float_tag(args.reverse_recon_local_max_px)}"
            f"_oa{_compact_float_tag(args.lambda_oracle_anchor)}"
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
        args.reverse_recon_init_checkpoint,
        optimizer=None,
        strict=True,
        map_location=str(device),
        load_optimizer=False,
    )
    freeze_to_local_branch(model)

    stage_weights = {
        4: float(args.reverse_recon_weight_scale4),
        2: float(args.reverse_recon_weight_scale2),
        1: float(args.reverse_recon_weight_scale1),
    }
    print(
        "Reverse-state reconstruction warm start: "
        f"{args.reverse_recon_init_checkpoint} "
        f"(epoch={source_epoch}, stored_metric={source_metric:.6f})"
    )
    print(
        "Protocol: local-only=100%, global=identity, reverse-state 4->2->1 inverse-flow + "
        "Raw-MSI reconstruction + weak oracle-flow anchor."
    )
    print(
        f"lambda_l1={cfg.lambda_l1:g}, lambda_sam={cfg.lambda_sam:g}, "
        f"lambda_flow={args.lambda_flow:g}, oracle_anchor={args.lambda_oracle_anchor:g}, "
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
            "epoch", "loss", "l1", "sam", "reverse_flow", "oracle_flow",
            "train_reverse_epe", "train_oracle_epe", "train_true", "train_pred", "train_ratio",
            "eval_l05_PSNR", "eval_l05_SAM", "eval_l05_EPE",
            "eval_l1_PSNR", "eval_l1_SAM", "eval_l1_EPE",
            "eval_l2_PSNR", "eval_l2_SAM", "eval_l2_EPE",
            "eval_l1_oracle_epe", "eval_l1_reverse_epe", "best_guarded_PSNR",
        ],
    )

    best_psnr = float("-inf")
    eval_seed = int(cfg.seed) + 15431
    for epoch in range(1, int(cfg.epochs) + 1):
        stats = train_epoch(
            model,
            train_loader,
            optimizer,
            process,
            device,
            local_max_px=float(args.reverse_recon_local_max_px),
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
            f"l1={stats.l1:.6f} sam={stats.sam:.6f} "
            f"flow_rev={stats.reverse_flow:.6f} flow_oracle={stats.oracle_flow:.6f} "
            f"EPE rev/oracle={stats.reverse_epe_px:.4f}/{stats.oracle_epe_px:.4f} "
            f"pred/true={stats.reverse_pred_mean_px:.3f}/{stats.reverse_true_mean_px:.3f}"
        )

        evaluation = None
        state_eval = None
        if epoch % int(args.reverse_recon_eval_interval) == 0 or epoch == int(cfg.epochs):
            evaluation = evaluate_inverse_local_only_reconstruction(
                model,
                test_loader,
                process,
                device,
                severities=[float(v) for v in args.reverse_recon_eval_severities],
                control_grid=int(args.reverse_recon_control_grid),
                trials=int(args.reverse_recon_eval_trials),
                valid_threshold=float(args.reverse_recon_valid_threshold),
                seed=eval_seed,
            )
            state_eval = evaluate_state_modes(
                model,
                test_loader,
                process,
                device,
                severities=[float(v) for v in args.reverse_recon_eval_severities],
                control_grid=int(args.reverse_recon_control_grid),
                trials=int(args.reverse_recon_eval_trials),
                seed=eval_seed,
            )
            for severity, result in evaluation.items():
                states = state_eval[severity]
                print(
                    f"  local<={severity:g}: PSNR={result['PSNR']:.3f} SAM={result['SAM']:.3f} "
                    f"reverse_EPE={result['EPE']:.4f} flow={result['pred']:.3f}/{result['true']:.3f} "
                    f"oracle_EPE={states['oracle_epe']:.4f}"
                )

            reference_key = float(args.reverse_recon_local_max_px)
            if reference_key not in evaluation:
                reference_key = min(
                    evaluation,
                    key=lambda x: abs(x - float(args.reverse_recon_local_max_px)),
                )
            reference = evaluation[reference_key]
            flow_ok = (
                float(args.reverse_recon_flow_ratio_min)
                <= float(reference["ratio"])
                <= float(args.reverse_recon_flow_ratio_max)
            )
            if flow_ok and float(reference["PSNR"]) > best_psnr:
                best_psnr = float(reference["PSNR"])
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    best_psnr,
                    best_path,
                    extra={
                        "config": vars(cfg),
                        "reverse_state_reconstruction": vars(args),
                        "evaluation": evaluation,
                        "state_evaluation": state_eval,
                        "source_checkpoint": args.reverse_recon_init_checkpoint,
                    },
                )
                print(f"  saved best guarded reverse reconstruction -> {best_path}")
            elif not flow_ok:
                print(
                    "  best checkpoint guard rejected: "
                    f"flow ratio={reference['ratio']:.3f} outside "
                    f"[{args.reverse_recon_flow_ratio_min:.2f}, {args.reverse_recon_flow_ratio_max:.2f}]"
                )

        row = {
            "epoch": epoch,
            "loss": stats.loss,
            "l1": stats.l1,
            "sam": stats.sam,
            "reverse_flow": stats.reverse_flow,
            "oracle_flow": stats.oracle_flow,
            "train_reverse_epe": stats.reverse_epe_px,
            "train_oracle_epe": stats.oracle_epe_px,
            "train_true": stats.reverse_true_mean_px,
            "train_pred": stats.reverse_pred_mean_px,
            "train_ratio": stats.reverse_ratio,
            "best_guarded_PSNR": best_psnr,
        }
        if evaluation:
            tags = {0.5: "05", 1.0: "1", 2.0: "2"}
            for severity, result in evaluation.items():
                if severity in tags:
                    tag = tags[severity]
                    row[f"eval_l{tag}_PSNR"] = result["PSNR"]
                    row[f"eval_l{tag}_SAM"] = result["SAM"]
                    row[f"eval_l{tag}_EPE"] = result["EPE"]
            if 1.0 in state_eval:
                row["eval_l1_oracle_epe"] = state_eval[1.0]["oracle_epe"]
                row["eval_l1_reverse_epe"] = state_eval[1.0]["reverse_epe"]
        logger.write(row)

        if epoch % int(cfg.save_interval) == 0 or epoch == int(cfg.epochs):
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best_psnr,
                last_path,
                extra={
                    "config": vars(cfg),
                    "reverse_state_reconstruction": vars(args),
                    "evaluation": evaluation,
                    "state_evaluation": state_eval,
                    "source_checkpoint": args.reverse_recon_init_checkpoint,
                },
            )

    print("Reverse-state reconstruction fine-tune complete.")
    print(f"Best guarded local PSNR: {best_psnr:.6f}")
    print(f"Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    cfg, args = parse_reverse_reconstruction_args()
    run(cfg, args)
