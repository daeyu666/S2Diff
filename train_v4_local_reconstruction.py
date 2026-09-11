"""Stage-3A local-only reconstruction fine-tune for recurrent Innovation-2 flow.

This stage starts from a successful physical 4->2->1 multiscale bootstrap.
It keeps the geometry protocol deliberately simple:

- local non-rigid MSI warp only;
- global rigid correction is bypassed (identity);
- recurrent local flow runs progressively at physical scales 4 -> 2 -> 1;
- stage-specific physical flow supervision is retained;
- aligned Raw MSI is fed back into the original Raw-Direct reconstruction
  backbone and reconstruction loss is restored.

Default optimization keeps the already validated reconstruction/global modules
frozen and trains only the recurrent local branch. Reconstruction gradients
still reach local flow through differentiable Raw-MSI sampling.

Loss:
    L = lambda_l1 * L1
      + lambda_sam * L_SAM
      + lambda_flow * L_physical_multiscale_flow

Validation uses the same local-only + global-identity protocol and reports
valid-overlap PSNR/SAM together with final scale-1 flow EPE/magnitude.
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
from diagnose_misalignment_translation import calc_masked_psnr_sam
from innovation1 import (
    batch_state_at,
    build_progressive_process,
    model_predict,
)
from losses import SAMLoss
from main import _compact_float_tag
from train_v4_multiscale_bootstrap import (
    field_metrics,
    physical_flow_targets,
    run_multiscale_sequence,
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
class ReconstructionFineTuneStats:
    loss: float
    l1: float
    sam: float
    flow: float
    final_epe_px: float
    final_true_mean_px: float
    final_pred_mean_px: float
    final_ratio: float
    scale4_pred_mean_px: float
    scale4_target_mean_px: float
    scale2_pred_mean_px: float
    scale2_target_mean_px: float
    scale1_pred_mean_px: float
    scale1_target_mean_px: float


def set_reconstruction_training_scope(model: torch.nn.Module, scope: str) -> None:
    """Select trainable parameters while always freezing learned global geometry."""
    scope = str(scope).lower()
    if scope not in {"local", "local_fusion", "all_except_global"}:
        raise ValueError(
            "reconstruction train scope must be local, local_fusion, or all_except_global"
        )

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    for parameter in model.geometry_aligner.local_aligner.parameters():
        parameter.requires_grad_(True)

    if scope == "local":
        return

    if scope == "local_fusion":
        enabled = 0
        for name, parameter in model.named_parameters():
            if name.startswith("geometry_aligner."):
                continue
            lname = name.lower()
            if "msi" in lname or "fusion" in lname or "guide" in lname:
                parameter.requires_grad_(True)
                enabled += parameter.numel()
        if enabled == 0:
            raise RuntimeError(
                "local_fusion scope found no MSI/fusion parameters; use scope=local"
            )
        return

    for name, parameter in model.named_parameters():
        if name.startswith("geometry_aligner.global_aligner."):
            parameter.requires_grad_(False)
        else:
            parameter.requires_grad_(True)


def prepare_local_only_training_alignment(
    model,
    process,
    gt: torch.Tensor,
    warped_msi: torch.Tensor,
):
    """Run differentiable 4->2->1 local flow with global correction fixed identity."""
    final_by_scale, sequences = run_multiscale_sequence(
        model,
        process,
        gt,
        warped_msi,
    )
    model._training_local_cache = final_by_scale
    model._training_sequence_cache = sequences
    model.last_global_shift_px = torch.zeros(
        gt.shape[0], 2, device=gt.device, dtype=gt.dtype
    )
    model.last_global_rotation_deg = torch.zeros(
        gt.shape[0], device=gt.device, dtype=gt.dtype
    )
    return final_by_scale, sequences


def clear_training_alignment_cache(model) -> None:
    model._training_local_cache = {}
    if hasattr(model, "_training_sequence_cache"):
        model._training_sequence_cache = {}


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
    boundary_probability: float,
    boundary_radius: int,
    grad_clip: float,
    generator: Optional[torch.Generator],
) -> ReconstructionFineTuneStats:
    model.train()
    sam_fn = SAMLoss()
    names = [
        "loss", "l1", "sam", "flow",
        "epe", "true", "pred", "ratio",
        "pred4", "true4", "pred2", "true2", "pred1", "true1",
    ]
    meters = {name: AverageMeter() for name in names}

    for batch in loader:
        gt = batch["gt"].to(device, non_blocking=True)
        hr_msi = batch["hr_msi"].to(device, non_blocking=True)
        b = int(gt.shape[0])

        warped_msi, _, params = make_misaligned_msi(
            hr_msi,
            translation_max_px=0.0,
            rotation_max_deg=0.0,
            local_max_displacement_px=float(local_max_px),
            control_grid_size=int(control_grid),
            generator=generator,
        )
        target_full = params.local_displacement_px
        targets = physical_flow_targets(
            process,
            model.geometry_aligner.stage_t_by_scale,
            target_full,
        )

        finals, sequences = prepare_local_only_training_alignment(
            model,
            process,
            gt,
            warped_msi,
        )

        timesteps = process.sample_timesteps(
            b,
            boundary_probability=float(boundary_probability),
            boundary_radius=int(boundary_radius),
            device=device,
        )
        with torch.no_grad():
            x_t = batch_state_at(process, gt, timesteps)

        pred_x0 = model_predict(
            model,
            x_t,
            timesteps,
            hr_msi=warped_msi,
        )

        l1 = F.l1_loss(pred_x0, gt)
        sam = sam_fn(pred_x0, gt)
        flow = stage_sequence_loss(
            sequences,
            targets,
            normalization_px=float(local_max_px),
            gamma=float(gamma),
            stage_weights=stage_weights,
        )
        loss = (
            float(lambda_l1) * l1
            + float(lambda_sam) * sam
            + float(lambda_flow) * flow
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite local reconstruction fine-tune loss")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=float(grad_clip) if float(grad_clip) > 0.0 else float("inf"),
            error_if_nonfinite=True,
        )
        optimizer.step()

        epe1, true1, pred1, ratio1 = field_metrics(finals[1], targets[1])
        _, true2, pred2, _ = field_metrics(finals[2], targets[2])
        _, true4, pred4, _ = field_metrics(finals[4], targets[4])
        values = {
            "loss": float(loss.item()),
            "l1": float(l1.item()),
            "sam": float(sam.item()),
            "flow": float(flow.item()),
            "epe": epe1,
            "true": true1,
            "pred": pred1,
            "ratio": ratio1,
            "pred4": pred4,
            "true4": true4,
            "pred2": pred2,
            "true2": true2,
            "pred1": pred1,
            "true1": true1,
        }
        for key, value in values.items():
            meters[key].update(value, b)

    clear_training_alignment_cache(model)
    return ReconstructionFineTuneStats(
        loss=meters["loss"].avg,
        l1=meters["l1"].avg,
        sam=meters["sam"].avg,
        flow=meters["flow"].avg,
        final_epe_px=meters["epe"].avg,
        final_true_mean_px=meters["true"].avg,
        final_pred_mean_px=meters["pred"].avg,
        final_ratio=meters["ratio"].avg,
        scale4_pred_mean_px=meters["pred4"].avg,
        scale4_target_mean_px=meters["true4"].avg,
        scale2_pred_mean_px=meters["pred2"].avg,
        scale2_target_mean_px=meters["true2"].avg,
        scale1_pred_mean_px=meters["pred1"].avg,
        scale1_target_mean_px=meters["true1"].avg,
    )


@torch.no_grad()
def reconstruct_local_only_identity(
    model,
    process,
    lr_hsi: torch.Tensor,
    *,
    target_size,
    warped_msi: torch.Tensor,
) -> torch.Tensor:
    """Full T->0 reverse inference with global correction explicitly bypassed."""
    model.eval()
    clear_training_alignment_cache(model)
    if hasattr(model, "_reset_inference_local_state"):
        model._reset_inference_local_state()

    x_t = process.terminal_state(lr_hsi, target_size=target_size)
    model.last_global_shift_px = torch.zeros(
        x_t.shape[0], 2, device=x_t.device, dtype=x_t.dtype
    )
    model.last_global_rotation_deg = torch.zeros(
        x_t.shape[0], device=x_t.device, dtype=x_t.dtype
    )

    for t in range(process.total_steps, 0, -1):
        timestep = torch.full(
            (x_t.shape[0],),
            t,
            dtype=torch.long,
            device=x_t.device,
        )
        pred_x0 = model_predict(
            model,
            x_t,
            timestep,
            hr_msi=warped_msi,
        )
        x_t = process.reverse_update(x_t, pred_x0, t)
    return x_t


@torch.no_grad()
def evaluate_local_only_reconstruction(
    model,
    loader,
    process,
    device,
    *,
    severities: List[float],
    control_grid: int,
    trials: int,
    valid_threshold: float,
    seed: int,
):
    """Local-only valid-overlap PSNR/SAM plus final flow diagnostics."""
    model.eval()
    results = {}

    for local_max in severities:
        psnrs: List[float] = []
        sams: List[float] = []
        epes: List[float] = []
        true_values: List[float] = []
        pred_values: List[float] = []
        ratios: List[float] = []

        for trial in range(int(trials)):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(seed) + trial * 100003)

            for batch in loader:
                gt = batch["gt"].to(device, non_blocking=True)
                hr_msi = batch["hr_msi"].to(device, non_blocking=True)
                warped, valid, params = make_misaligned_msi(
                    hr_msi,
                    translation_max_px=0.0,
                    rotation_max_deg=0.0,
                    local_max_displacement_px=float(local_max),
                    control_grid_size=int(control_grid),
                    generator=generator,
                )

                terminal_lr = process.terminal_observation(gt)
                pred = reconstruct_local_only_identity(
                    model,
                    process,
                    terminal_lr,
                    target_size=tuple(gt.shape[-2:]),
                    warped_msi=warped,
                )
                psnr, sam, _ = calc_masked_psnr_sam(
                    pred,
                    gt,
                    valid,
                    threshold=float(valid_threshold),
                )
                psnrs.append(float(psnr))
                sams.append(float(sam))

                final_flow = model._inference_local_offset
                if final_flow is None:
                    raise RuntimeError("missing final inference local flow")
                epe, true_mag, pred_mag, ratio = field_metrics(
                    final_flow,
                    params.local_displacement_px,
                )
                epes.append(epe)
                true_values.append(true_mag)
                pred_values.append(pred_mag)
                ratios.append(ratio)

        results[float(local_max)] = {
            "PSNR": float(np.mean(psnrs)),
            "SAM": float(np.mean(sams)),
            "EPE": float(np.mean(epes)),
            "true": float(np.mean(true_values)),
            "pred": float(np.mean(pred_values)),
            "ratio": float(np.mean(ratios)),
        }

    return results


def parse_reconstruction_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--reconstruction_init_checkpoint", type=str, required=True)
    parser.add_argument("--reconstruction_local_max_px", type=float, default=1.0)
    parser.add_argument("--reconstruction_control_grid", type=int, default=5)
    parser.add_argument("--reconstruction_gamma", type=float, default=0.8)
    parser.add_argument("--reconstruction_weight_scale4", type=float, default=0.5)
    parser.add_argument("--reconstruction_weight_scale2", type=float, default=0.7)
    parser.add_argument("--reconstruction_weight_scale1", type=float, default=1.0)
    parser.add_argument("--lambda_flow", type=float, default=1.0)
    parser.add_argument(
        "--reconstruction_train_scope",
        choices=["local", "local_fusion", "all_except_global"],
        default="local",
    )
    parser.add_argument("--reconstruction_iterations_scale4", type=int, default=3)
    parser.add_argument("--reconstruction_iterations_scale2", type=int, default=2)
    parser.add_argument("--reconstruction_iterations_scale1", type=int, default=2)
    parser.add_argument("--reconstruction_max_update_scale4", type=float, default=2.0)
    parser.add_argument("--reconstruction_max_update_scale2", type=float, default=1.0)
    parser.add_argument("--reconstruction_max_update_scale1", type=float, default=0.5)
    parser.add_argument(
        "--reconstruction_eval_severities",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0],
    )
    parser.add_argument("--reconstruction_eval_trials", type=int, default=3)
    parser.add_argument("--reconstruction_eval_interval", type=int, default=5)
    parser.add_argument("--reconstruction_valid_threshold", type=float, default=0.999)
    parser.add_argument(
        "--reconstruction_flow_ratio_min",
        type=float,
        default=0.80,
    )
    parser.add_argument(
        "--reconstruction_flow_ratio_max",
        type=float,
        default=1.20,
    )
    parser.add_argument("--reconstruction_save_name", type=str, default="")
    args, remaining = parser.parse_known_args()
    cfg = parse_args(remaining)

    if cfg.stage != "train":
        raise ValueError("local reconstruction fine-tune is train-only; use --stage train")
    if str(cfg.predictor_version).lower() != "v4":
        raise ValueError("local reconstruction fine-tune requires predictor_version=v4")
    if args.reconstruction_local_max_px <= 0.0:
        raise ValueError("reconstruction_local_max_px must be > 0")
    if args.reconstruction_control_grid < 2:
        raise ValueError("reconstruction_control_grid must be >= 2")
    if not 0.0 < args.reconstruction_gamma <= 1.0:
        raise ValueError("reconstruction_gamma must lie in (0,1]")
    if min(
        args.reconstruction_weight_scale4,
        args.reconstruction_weight_scale2,
        args.reconstruction_weight_scale1,
    ) <= 0.0:
        raise ValueError("reconstruction stage weights must be > 0")
    if args.lambda_flow <= 0.0:
        raise ValueError("lambda_flow must be > 0")
    if args.reconstruction_eval_trials < 1 or args.reconstruction_eval_interval < 1:
        raise ValueError("reconstruction eval counts must be >= 1")
    if not 0.0 < args.reconstruction_valid_threshold <= 1.0:
        raise ValueError("reconstruction_valid_threshold must lie in (0,1]")
    if any(v <= 0.0 for v in args.reconstruction_eval_severities):
        raise ValueError("all reconstruction eval severities must be > 0")
    if args.reconstruction_flow_ratio_min <= 0.0:
        raise ValueError("flow ratio minimum must be > 0")
    if args.reconstruction_flow_ratio_max < args.reconstruction_flow_ratio_min:
        raise ValueError("flow ratio max must be >= min")

    cfg.recurrent_hidden_channels = 64
    cfg.recurrent_correlation_channels = 32
    cfg.recurrent_iterations_scale4 = int(args.reconstruction_iterations_scale4)
    cfg.recurrent_iterations_scale2 = int(args.reconstruction_iterations_scale2)
    cfg.recurrent_iterations_scale1 = int(args.reconstruction_iterations_scale1)
    cfg.recurrent_max_update_scale4 = float(args.reconstruction_max_update_scale4)
    cfg.recurrent_max_update_scale2 = float(args.reconstruction_max_update_scale2)
    cfg.recurrent_max_update_scale1 = float(args.reconstruction_max_update_scale1)
    return cfg, args


def output_paths(cfg, args):
    root = os.path.join(cfg.checkpoint_root, "innovation1")
    ensure_dir(root)
    if args.reconstruction_save_name:
        stem = args.reconstruction_save_name.removesuffix(".pth")
    else:
        stem = (
            f"{cfg.dataset}_v4_local_reconstruction"
            f"_l{_compact_float_tag(args.reconstruction_local_max_px)}"
            f"_flow{_compact_float_tag(args.lambda_flow)}"
            f"_{args.reconstruction_train_scope}"
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

    source_epoch, source_best = load_checkpoint(
        model,
        args.reconstruction_init_checkpoint,
        optimizer=None,
        strict=True,
        map_location=str(device),
        load_optimizer=False,
    )
    set_reconstruction_training_scope(model, args.reconstruction_train_scope)

    stage_weights = {
        4: float(args.reconstruction_weight_scale4),
        2: float(args.reconstruction_weight_scale2),
        1: float(args.reconstruction_weight_scale1),
    }
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    print(
        "Local-only reconstruction fine-tune warm start: "
        f"{args.reconstruction_init_checkpoint} "
        f"(epoch={source_epoch}, stored_metric={source_best:.6f})"
    )
    print(
        "Protocol: local-only=100%, global=identity, physical 4->2->1 flow + "
        "Raw-MSI reconstruction."
    )
    print(
        f"train_scope={args.reconstruction_train_scope}, "
        f"lambda_l1={cfg.lambda_l1:g}, lambda_sam={cfg.lambda_sam:g}, "
        f"lambda_flow={args.lambda_flow:g}, "
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

    best_path, last_path, log_path = output_paths(cfg, args)
    logger = CSVLogger(
        log_path,
        fieldnames=[
            "epoch", "loss", "l1", "sam", "flow",
            "train_epe", "train_true", "train_pred", "train_ratio",
            "train_s4_pred", "train_s4_true",
            "train_s2_pred", "train_s2_true",
            "train_s1_pred", "train_s1_true",
            "eval_l05_PSNR", "eval_l05_SAM", "eval_l05_EPE", "eval_l05_pred", "eval_l05_true",
            "eval_l1_PSNR", "eval_l1_SAM", "eval_l1_EPE", "eval_l1_pred", "eval_l1_true",
            "eval_l2_PSNR", "eval_l2_SAM", "eval_l2_EPE", "eval_l2_pred", "eval_l2_true",
            "best_guarded_PSNR",
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
            local_max_px=float(args.reconstruction_local_max_px),
            control_grid=int(args.reconstruction_control_grid),
            gamma=float(args.reconstruction_gamma),
            stage_weights=stage_weights,
            lambda_l1=float(cfg.lambda_l1),
            lambda_sam=float(cfg.lambda_sam),
            lambda_flow=float(args.lambda_flow),
            boundary_probability=float(cfg.boundary_probability),
            boundary_radius=int(cfg.boundary_radius),
            grad_clip=float(cfg.grad_clip),
            generator=generator,
        )

        print(
            f"Epoch {epoch:04d}/{cfg.epochs:04d} "
            f"loss={stats.loss:.6f} l1={stats.l1:.6f} sam={stats.sam:.6f} "
            f"flow={stats.flow:.6f} "
            f"s4={stats.scale4_pred_mean_px:.3f}/{stats.scale4_target_mean_px:.3f} "
            f"s2={stats.scale2_pred_mean_px:.3f}/{stats.scale2_target_mean_px:.3f} "
            f"s1={stats.scale1_pred_mean_px:.3f}/{stats.scale1_target_mean_px:.3f} "
            f"EPE={stats.final_epe_px:.4f}"
        )

        evaluation = None
        if (
            epoch % int(args.reconstruction_eval_interval) == 0
            or epoch == int(cfg.epochs)
        ):
            evaluation = evaluate_local_only_reconstruction(
                model,
                test_loader,
                process,
                device,
                severities=[float(v) for v in args.reconstruction_eval_severities],
                control_grid=int(args.reconstruction_control_grid),
                trials=int(args.reconstruction_eval_trials),
                valid_threshold=float(args.reconstruction_valid_threshold),
                seed=eval_seed,
            )
            for severity, result in evaluation.items():
                print(
                    f"  local<={severity:g}: "
                    f"PSNR={result['PSNR']:.3f} SAM={result['SAM']:.3f} "
                    f"flow={result['pred']:.3f}/{result['true']:.3f} "
                    f"EPE={result['EPE']:.4f}"
                )

            reference_key = float(args.reconstruction_local_max_px)
            if reference_key not in evaluation:
                reference_key = min(
                    evaluation,
                    key=lambda x: abs(x - float(args.reconstruction_local_max_px)),
                )
            reference = evaluation[reference_key]
            flow_ok = (
                float(args.reconstruction_flow_ratio_min)
                <= float(reference["ratio"])
                <= float(args.reconstruction_flow_ratio_max)
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
                        "local_reconstruction": vars(args),
                        "evaluation": evaluation,
                        "source_checkpoint": args.reconstruction_init_checkpoint,
                    },
                )
                print(f"  saved best guarded reconstruction -> {best_path}")
            elif not flow_ok:
                print(
                    "  best checkpoint guard rejected: "
                    f"flow ratio={reference['ratio']:.3f} outside "
                    f"[{args.reconstruction_flow_ratio_min:.2f}, "
                    f"{args.reconstruction_flow_ratio_max:.2f}]"
                )

        row = {
            "epoch": epoch,
            "loss": stats.loss,
            "l1": stats.l1,
            "sam": stats.sam,
            "flow": stats.flow,
            "train_epe": stats.final_epe_px,
            "train_true": stats.final_true_mean_px,
            "train_pred": stats.final_pred_mean_px,
            "train_ratio": stats.final_ratio,
            "train_s4_pred": stats.scale4_pred_mean_px,
            "train_s4_true": stats.scale4_target_mean_px,
            "train_s2_pred": stats.scale2_pred_mean_px,
            "train_s2_true": stats.scale2_target_mean_px,
            "train_s1_pred": stats.scale1_pred_mean_px,
            "train_s1_true": stats.scale1_target_mean_px,
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
                    row[f"eval_l{tag}_pred"] = result["pred"]
                    row[f"eval_l{tag}_true"] = result["true"]
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
                    "local_reconstruction": vars(args),
                    "evaluation": evaluation,
                    "source_checkpoint": args.reconstruction_init_checkpoint,
                },
            )

    print("Local-only reconstruction fine-tune complete.")
    print(f"Best guarded local PSNR: {best_psnr:.6f}")
    print(f"Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    cfg, args = parse_reconstruction_args()
    run(cfg, args)
