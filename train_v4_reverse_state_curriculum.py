"""Final Innovation-2 local-severity curriculum fine-tune.

This launcher preserves the validated reverse-state reconstruction architecture
and losses, and changes only the training distribution of synthetic local
non-rigid severity.  One severity is sampled per batch so the flow loss can be
normalized by the exact displacement range used to synthesize that batch.

Default curriculum:
    local_max_px in {0.5, 1.0, 2.0}
    probability   = {0.2, 0.4, 0.4}

Everything else remains identical to train_v4_reverse_state_reconstruction.py:
- local-only synthetic warp;
- global rigid branch fixed to identity;
- reverse-state HSI matching domain;
- geometrically correct inverse-sampling flow targets;
- physical 4->2->1 recurrent flow;
- Raw-MSI reconstruction on reverse states;
- weak oracle-state flow anchor;
- only the recurrent local branch is trainable.

This is intentionally the final structural-neutral test for whether the 2-pixel
Oracle gap is mainly an out-of-distribution severity effect.
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, Optional, Sequence, Tuple

import torch

import train_v4_reverse_state_reconstruction as base
from degradations.misalignment import make_misaligned_msi
from losses import SAMLoss
from reverse_state_local_flow import oracle_states, rollout_reverse_states_identity, run_local_flow_from_states
from train_v4_reverse_state_flow import inverse_targets


_CURRICULUM_SEVERITIES: Tuple[float, ...] = (0.5, 1.0, 2.0)
_CURRICULUM_PROBABILITIES: Tuple[float, ...] = (0.2, 0.4, 0.4)


def validate_curriculum(
    severities: Sequence[float], probabilities: Sequence[float]
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    severities = tuple(float(v) for v in severities)
    probabilities = tuple(float(v) for v in probabilities)
    if len(severities) < 1:
        raise ValueError("curriculum must contain at least one severity")
    if len(severities) != len(probabilities):
        raise ValueError("curriculum severities/probabilities must have equal length")
    if any(v <= 0.0 for v in severities):
        raise ValueError("all curriculum severities must be > 0")
    if any(p < 0.0 for p in probabilities):
        raise ValueError("curriculum probabilities must be >= 0")
    total = float(sum(probabilities))
    if total <= 0.0:
        raise ValueError("curriculum probabilities must have positive sum")
    probabilities = tuple(p / total for p in probabilities)
    return severities, probabilities


def sample_curriculum_severity(
    severities: Sequence[float],
    probabilities: Sequence[float],
    *,
    generator: Optional[torch.Generator],
) -> float:
    sev, prob = validate_curriculum(severities, probabilities)
    weights = torch.tensor(prob, dtype=torch.float32, device="cpu")
    index = int(torch.multinomial(weights, 1, replacement=True, generator=generator).item())
    return float(sev[index])


def curriculum_train_epoch(
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
):
    """Reverse-state reconstruction epoch with one sampled severity per batch."""
    del local_max_px  # replaced by the explicit curriculum distribution
    model.train()
    sam_fn = SAMLoss()
    meters = {
        key: base.AverageMeter()
        for key in (
            "loss", "l1", "sam", "reverse_flow", "oracle_flow",
            "reverse_epe", "reverse_true", "reverse_pred", "reverse_ratio",
            "oracle_epe",
        )
    }

    for batch in loader:
        severity = sample_curriculum_severity(
            _CURRICULUM_SEVERITIES,
            _CURRICULUM_PROBABILITIES,
            generator=generator,
        )
        gt = batch["gt"].to(device, non_blocking=True)
        hr_msi = batch["hr_msi"].to(device, non_blocking=True)
        b = int(gt.shape[0])

        warped, _, params = make_misaligned_msi(
            hr_msi,
            translation_max_px=0.0,
            rotation_max_deg=0.0,
            local_max_displacement_px=float(severity),
            control_grid_size=int(control_grid),
            generator=generator,
        )
        _, targets = inverse_targets(process, model, params.local_displacement_px)

        reverse_states = rollout_reverse_states_identity(model, process, gt, warped)
        reverse_finals, reverse_sequences = run_local_flow_from_states(
            model, reverse_states, warped
        )
        reverse_flow_loss = base.stage_sequence_loss(
            reverse_sequences,
            targets,
            normalization_px=float(severity),
            gamma=float(gamma),
            stage_weights=stage_weights,
        )

        with torch.no_grad():
            oracle = oracle_states(model, process, gt)
        oracle_finals, oracle_sequences = run_local_flow_from_states(model, oracle, warped)
        oracle_flow_loss = base.stage_sequence_loss(
            oracle_sequences,
            targets,
            normalization_px=float(severity),
            gamma=float(gamma),
            stage_weights=stage_weights,
        )

        l1, sam, _ = base.reconstruction_loss_on_reverse_states(
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
            raise FloatingPointError("non-finite reverse-state curriculum loss")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=float(grad_clip) if float(grad_clip) > 0.0 else float("inf"),
            error_if_nonfinite=True,
        )
        optimizer.step()
        base._clear_training_cache(model)

        re, true_mag, pred_mag, ratio = base.field_metrics(
            reverse_finals[1], targets[1]
        )
        oe, _, _, _ = base.field_metrics(oracle_finals[1], targets[1])
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

    return base.ReverseReconstructionStats(
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


def parse_curriculum_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--curriculum_severities",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0],
    )
    parser.add_argument(
        "--curriculum_probabilities",
        type=float,
        nargs="+",
        default=[0.2, 0.4, 0.4],
    )
    curriculum_args, remaining = parser.parse_known_args()

    # Remove curriculum-only options before delegating the rest to the validated
    # reverse-state reconstruction parser.
    original_argv = list(sys.argv)
    try:
        sys.argv = [original_argv[0]] + remaining
        cfg, args = base.parse_reverse_reconstruction_args()
    finally:
        sys.argv = original_argv

    sev, prob = validate_curriculum(
        curriculum_args.curriculum_severities,
        curriculum_args.curriculum_probabilities,
    )
    curriculum_args.curriculum_severities = list(sev)
    curriculum_args.curriculum_probabilities = list(prob)
    return cfg, args, curriculum_args


def main():
    global _CURRICULUM_SEVERITIES, _CURRICULUM_PROBABILITIES

    cfg, args, curriculum = parse_curriculum_args()
    _CURRICULUM_SEVERITIES = tuple(curriculum.curriculum_severities)
    _CURRICULUM_PROBABILITIES = tuple(curriculum.curriculum_probabilities)

    if not args.reverse_recon_save_name:
        severity_tag = "-".join(str(v).replace(".", "p") for v in _CURRICULUM_SEVERITIES)
        args.reverse_recon_save_name = (
            f"{cfg.dataset}_v4_reverse_state_curriculum_{severity_tag}"
        )

    # The base runner keeps the existing evaluation, checkpoint guard, optimizer,
    # and reverse-state reconstruction protocol.  Only the training severity
    # distribution is replaced.
    base.train_epoch = curriculum_train_epoch

    description = ", ".join(
        f"{s:g}px:{100.0*p:.0f}%"
        for s, p in zip(_CURRICULUM_SEVERITIES, _CURRICULUM_PROBABILITIES)
    )
    print(f"Local-severity curriculum enabled -> {description}")
    print(
        "Architecture unchanged: reverse-state + inverse-flow + physical 4->2->1 + "
        "Raw-MSI reconstruction + weak oracle anchor."
    )
    base.run(cfg, args)


if __name__ == "__main__":
    main()
