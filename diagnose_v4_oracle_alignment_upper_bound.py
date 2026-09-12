"""Oracle upper-bound diagnosis for Innovation-2 local non-rigid alignment.

The experiment isolates the ceiling imposed by synthetic warp + resampling.
For every local-only perturbation it compares four paths under the same frozen
Raw-Direct reconstruction backbone:

1. pristine_registered: original registered HR-MSI, no alignment front end;
2. misaligned_noalign: synthetically warped HR-MSI, no correction;
3. learned_alignment: current recurrent 4->2->1 local alignment;
4. oracle_inverse: warped HR-MSI resampled with the exact inverse sampling field.

Misaligned/learned/oracle metrics use the same validity mask.  Pristine is
reported both on the full image and on that common mask.  The script also
reports MSI-domain residual error after learned/oracle realignment and an
oracle-gap recovery ratio:

    recovery = (PSNR_learned - PSNR_noalign)
               / (PSNR_oracle - PSNR_noalign)

A learned result close to oracle indicates that further flow optimization is
unlikely to improve reconstruction under the current twice-resampled synthetic
misregistration protocol.
"""

from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np
import torch

from config import parse_args
from data_loader import build_loaders
from degradations.inverse_flow import forward_to_inverse_sampling_field
from degradations.misalignment import make_misaligned_msi
from diagnose_misalignment_translation import calc_masked_psnr_sam
from innovation1 import build_progressive_process
from models.predictor_v3_ablation import MSIAblationGuidedPredictor
from models.predictor_v4_alignment import _sample_with_source_offset
from train_v4_local_reconstruction import reconstruct_local_only_identity
from train_v4_local_reconstruction_inverse import INVERSE_FIXED_POINT_ITERATIONS
from train_v4_recurrent_flow import build_recurrent_model
from utils import get_device, load_checkpoint, set_seed


@torch.no_grad()
def reconstruct_raw_direct_bypass_alignment(
    model,
    process,
    lr_hsi: torch.Tensor,
    *,
    target_size,
    raw_msi: torch.Tensor,
) -> torch.Tensor:
    """Run Innovation-1 reverse recursion through the frozen Raw-Direct backbone.

    Calling ``MSIAblationGuidedPredictor.forward`` explicitly bypasses the V4
    geometry wrapper while reusing exactly the same reconstruction weights.
    """
    model.eval()
    x_t = process.terminal_state(lr_hsi, target_size=target_size)
    for t in range(int(process.total_steps), 0, -1):
        timestep = torch.full(
            (x_t.shape[0],), int(t), dtype=torch.long, device=x_t.device
        )
        pred_x0 = MSIAblationGuidedPredictor.forward(
            model,
            x_t,
            raw_msi,
            timestep,
        )
        x_t = process.reverse_update(x_t, pred_x0, int(t))
    return x_t


def _masked_l1(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor, threshold: float) -> float:
    valid = (mask >= float(threshold)).to(dtype=a.dtype)
    if valid.shape[1] == 1 and a.shape[1] != 1:
        valid = valid.expand(-1, a.shape[1], -1, -1)
    denom = valid.sum().clamp_min(1.0)
    return float(((a - b).abs() * valid).sum().item() / denom.item())


def _safe_recovery(noalign: float, learned: float, oracle: float) -> float:
    denom = float(oracle) - float(noalign)
    if abs(denom) < 1e-8:
        return float("nan")
    return (float(learned) - float(noalign)) / denom


@torch.no_grad()
def evaluate_upper_bound(
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
) -> Dict[float, Dict[str, float]]:
    model.eval()
    results: Dict[float, Dict[str, float]] = {}

    for local_max in severities:
        accum = {
            "registered_full_psnr": [], "registered_full_sam": [],
            "registered_common_psnr": [], "registered_common_sam": [],
            "noalign_psnr": [], "noalign_sam": [],
            "learned_psnr": [], "learned_sam": [],
            "oracle_psnr": [], "oracle_sam": [],
            "oracle_msi_l1": [], "learned_msi_l1": [],
        }

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
                inverse = forward_to_inverse_sampling_field(
                    params.local_displacement_px,
                    iterations=INVERSE_FIXED_POINT_ITERATIONS,
                    padding_mode="border",
                )
                oracle_aligned_msi = _sample_with_source_offset(warped, inverse)

                terminal_lr = process.terminal_observation(gt)
                target_size = tuple(gt.shape[-2:])

                registered_pred = reconstruct_raw_direct_bypass_alignment(
                    model, process, terminal_lr,
                    target_size=target_size,
                    raw_msi=hr_msi,
                )
                noalign_pred = reconstruct_raw_direct_bypass_alignment(
                    model, process, terminal_lr,
                    target_size=target_size,
                    raw_msi=warped,
                )
                oracle_pred = reconstruct_raw_direct_bypass_alignment(
                    model, process, terminal_lr,
                    target_size=target_size,
                    raw_msi=oracle_aligned_msi,
                )
                learned_pred = reconstruct_local_only_identity(
                    model, process, terminal_lr,
                    target_size=target_size,
                    warped_msi=warped,
                )

                ones = torch.ones_like(valid)
                reg_full_psnr, reg_full_sam, _ = calc_masked_psnr_sam(
                    registered_pred, gt, ones, threshold=0.5
                )
                reg_common_psnr, reg_common_sam, _ = calc_masked_psnr_sam(
                    registered_pred, gt, valid, threshold=float(valid_threshold)
                )
                noalign_psnr, noalign_sam, _ = calc_masked_psnr_sam(
                    noalign_pred, gt, valid, threshold=float(valid_threshold)
                )
                learned_psnr, learned_sam, _ = calc_masked_psnr_sam(
                    learned_pred, gt, valid, threshold=float(valid_threshold)
                )
                oracle_psnr, oracle_sam, _ = calc_masked_psnr_sam(
                    oracle_pred, gt, valid, threshold=float(valid_threshold)
                )

                values = {
                    "registered_full_psnr": reg_full_psnr,
                    "registered_full_sam": reg_full_sam,
                    "registered_common_psnr": reg_common_psnr,
                    "registered_common_sam": reg_common_sam,
                    "noalign_psnr": noalign_psnr,
                    "noalign_sam": noalign_sam,
                    "learned_psnr": learned_psnr,
                    "learned_sam": learned_sam,
                    "oracle_psnr": oracle_psnr,
                    "oracle_sam": oracle_sam,
                    "oracle_msi_l1": _masked_l1(
                        oracle_aligned_msi, hr_msi, valid, float(valid_threshold)
                    ),
                }

                learned_flow = model._inference_local_offset
                if learned_flow is None:
                    raise RuntimeError("learned alignment did not produce final local flow")
                learned_aligned_msi = _sample_with_source_offset(warped, learned_flow)
                values["learned_msi_l1"] = _masked_l1(
                    learned_aligned_msi, hr_msi, valid, float(valid_threshold)
                )

                for key, value in values.items():
                    accum[key].append(float(value))

        summary = {key: float(np.mean(values)) for key, values in accum.items()}
        summary["learned_gain_db"] = summary["learned_psnr"] - summary["noalign_psnr"]
        summary["oracle_gain_db"] = summary["oracle_psnr"] - summary["noalign_psnr"]
        summary["oracle_gap_db"] = summary["oracle_psnr"] - summary["learned_psnr"]
        summary["registered_gap_db"] = (
            summary["registered_common_psnr"] - summary["oracle_psnr"]
        )
        summary["oracle_gap_recovery"] = _safe_recovery(
            summary["noalign_psnr"],
            summary["learned_psnr"],
            summary["oracle_psnr"],
        )
        results[float(local_max)] = summary

    return results


def parse_oracle_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--oracle_checkpoint", type=str, required=True)
    parser.add_argument("--oracle_eval_severities", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--oracle_control_grid", type=int, default=5)
    parser.add_argument("--oracle_eval_trials", type=int, default=3)
    parser.add_argument("--oracle_valid_threshold", type=float, default=0.999)
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

    if str(cfg.predictor_version).lower() != "v4":
        raise ValueError("oracle upper-bound diagnosis requires predictor_version=v4")
    if args.oracle_control_grid < 2:
        raise ValueError("oracle_control_grid must be >= 2")
    if args.oracle_eval_trials < 1:
        raise ValueError("oracle_eval_trials must be >= 1")
    if not 0.0 < args.oracle_valid_threshold <= 1.0:
        raise ValueError("oracle_valid_threshold must lie in (0,1]")
    if any(v <= 0.0 for v in args.oracle_eval_severities):
        raise ValueError("all oracle eval severities must be > 0")

    for name in (
        "recurrent_hidden_channels", "recurrent_correlation_channels",
        "recurrent_iterations_scale4", "recurrent_iterations_scale2", "recurrent_iterations_scale1",
        "recurrent_max_update_scale4", "recurrent_max_update_scale2", "recurrent_max_update_scale1",
    ):
        setattr(cfg, name, getattr(args, name))
    return cfg, args


def main():
    cfg, args = parse_oracle_args()
    set_seed(cfg.seed)
    _, test_loader, info = build_loaders(cfg)
    device = get_device(cfg.device)
    process = build_progressive_process(cfg)
    model = build_recurrent_model(cfg, info, device, process)
    epoch, metric = load_checkpoint(
        model,
        args.oracle_checkpoint,
        optimizer=None,
        strict=True,
        map_location=str(device),
        load_optimizer=False,
    )
    print(
        f"Oracle upper-bound checkpoint: {args.oracle_checkpoint} "
        f"(epoch={epoch}, stored_metric={metric:.6f})"
    )
    results = evaluate_upper_bound(
        model,
        test_loader,
        process,
        device,
        severities=[float(v) for v in args.oracle_eval_severities],
        control_grid=int(args.oracle_control_grid),
        trials=int(args.oracle_eval_trials),
        valid_threshold=float(args.oracle_valid_threshold),
        seed=int(cfg.seed) + 15431,
    )

    print("\nOracle local-alignment upper bound")
    for severity, result in results.items():
        recovery = result["oracle_gap_recovery"]
        recovery_text = "nan" if not np.isfinite(recovery) else f"{100.0 * recovery:.1f}%"
        print(f"local<={severity:g}")
        print(
            "  registered: "
            f"full={result['registered_full_psnr']:.3f}/{result['registered_full_sam']:.3f} "
            f"common={result['registered_common_psnr']:.3f}/{result['registered_common_sam']:.3f}"
        )
        print(
            "  no-align : "
            f"PSNR={result['noalign_psnr']:.3f} SAM={result['noalign_sam']:.3f}"
        )
        print(
            "  learned  : "
            f"PSNR={result['learned_psnr']:.3f} SAM={result['learned_sam']:.3f} "
            f"MSI_L1={result['learned_msi_l1']:.6f}"
        )
        print(
            "  oracle   : "
            f"PSNR={result['oracle_psnr']:.3f} SAM={result['oracle_sam']:.3f} "
            f"MSI_L1={result['oracle_msi_l1']:.6f}"
        )
        print(
            "  gaps     : "
            f"learned_gain={result['learned_gain_db']:+.3f}dB "
            f"oracle_gain={result['oracle_gain_db']:+.3f}dB "
            f"oracle-learned={result['oracle_gap_db']:+.3f}dB "
            f"registered-oracle={result['registered_gap_db']:+.3f}dB "
            f"recovery={recovery_text}"
        )


if __name__ == "__main__":
    main()
