"""Stage-3A local reconstruction with geometrically correct inverse-flow targets.

This launcher intentionally preserves ``train_v4_local_reconstruction.py`` as
an ablation using the original forward/content displacement supervision.

The synthetic local warp is
    Y_warp(q) = Y(q - d(q))
while the V4 local aligner restores Raw MSI by positive source sampling
    Y_align(p) = Y_warp(p + u(p)).
Therefore the geometrically correct target is the inverse sampling field
    u(p) = d(p + u(p)),
not generally the original forward field d(p).

This launcher reuses the complete Stage-3A training loop but replaces only:
1. multiscale flow targets: forward d -> fixed-point inverse u -> physical 4/2/1;
2. validation flow EPE/magnitude: predicted flow is compared with inverse u.

The recurrent architecture, Raw-MSI warp, reconstruction loss, global-identity
protocol and optimizer behavior remain unchanged for a clean ablation.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch

import train_v4_local_reconstruction as base
from degradations.inverse_flow import (
    forward_to_inverse_sampling_field,
    inverse_field_diagnostics,
)
from degradations.misalignment import make_misaligned_msi
from diagnose_misalignment_translation import calc_masked_psnr_sam
from main import _compact_float_tag
from train_v4_multiscale_bootstrap import (
    field_metrics,
    physical_flow_targets as _forward_physical_flow_targets,
)


INVERSE_FIXED_POINT_ITERATIONS = 8


def inverse_physical_flow_targets(
    process,
    stage_t_by_scale,
    forward_target_flow: torch.Tensor,
):
    """Convert synthetic forward field to inverse sampling field before 4/2/1 targets."""
    with torch.no_grad():
        inverse_full = forward_to_inverse_sampling_field(
            forward_target_flow,
            iterations=INVERSE_FIXED_POINT_ITERATIONS,
            padding_mode="border",
        )
    return _forward_physical_flow_targets(
        process,
        stage_t_by_scale,
        inverse_full,
    )


@torch.no_grad()
def evaluate_inverse_local_only_reconstruction(
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
    """Local-only PSNR/SAM and EPE against the true inverse sampling field."""
    model.eval()
    results = {}

    for local_max in severities:
        psnrs: List[float] = []
        sams: List[float] = []
        epes: List[float] = []
        inverse_true_values: List[float] = []
        forward_true_values: List[float] = []
        pred_values: List[float] = []
        ratios: List[float] = []
        inverse_residual_values: List[float] = []

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

                inverse_target = forward_to_inverse_sampling_field(
                    params.local_displacement_px,
                    iterations=INVERSE_FIXED_POINT_ITERATIONS,
                    padding_mode="border",
                )
                forward_mag, inverse_mag, inverse_residual = inverse_field_diagnostics(
                    params.local_displacement_px,
                    inverse_target,
                )

                terminal_lr = process.terminal_observation(gt)
                pred = base.reconstruct_local_only_identity(
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
                    inverse_target,
                )
                epes.append(epe)
                inverse_true_values.append(true_mag)
                forward_true_values.append(forward_mag)
                pred_values.append(pred_mag)
                ratios.append(ratio)
                inverse_residual_values.append(inverse_residual)

        results[float(local_max)] = {
            "PSNR": float(np.mean(psnrs)),
            "SAM": float(np.mean(sams)),
            "EPE": float(np.mean(epes)),
            "true": float(np.mean(inverse_true_values)),
            "pred": float(np.mean(pred_values)),
            "ratio": float(np.mean(ratios)),
            "forward_true": float(np.mean(forward_true_values)),
            "inverse_fp_residual": float(np.mean(inverse_residual_values)),
        }

    return results


def main():
    cfg, args = base.parse_reconstruction_args()

    # Never overwrite the forward-supervision Stage-3A checkpoint/log by default.
    if not args.reconstruction_save_name:
        args.reconstruction_save_name = (
            f"{cfg.dataset}_v4_local_reconstruction_inverse"
            f"_l{_compact_float_tag(args.reconstruction_local_max_px)}"
            f"_flow{_compact_float_tag(args.lambda_flow)}"
            f"_{args.reconstruction_train_scope}"
        )

    # Clean ablation: only target geometry and validation EPE definition change.
    base.physical_flow_targets = inverse_physical_flow_targets
    base.evaluate_local_only_reconstruction = evaluate_inverse_local_only_reconstruction

    print(
        "Inverse-flow supervision enabled: "
        f"u(p)=d(p+u(p)), fixed-point iterations={INVERSE_FIXED_POINT_ITERATIONS}."
    )
    base.run(cfg, args)


if __name__ == "__main__":
    main()
