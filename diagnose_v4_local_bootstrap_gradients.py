"""One-batch gradient-path probe for the scale-1 local bootstrap."""

from __future__ import annotations

import argparse

import torch

from config import parse_args
from data_loader import build_loaders
from degradations.misalignment import make_misaligned_msi
from innovation1 import build_progressive_process
from train_v4_local_bootstrap import (
    freeze_to_local_branch,
    predict_scale1_sequence,
    tiny_initialize_delta_head,
)
from train_v4_recurrent_flow import (
    build_recurrent_model,
    sequence_flow_loss,
    warm_start_compatible,
)
from utils import get_device, set_seed


def grad_mean(parameter):
    grad = parameter.grad
    if grad is None:
        return 0.0
    return float(grad.detach().abs().mean().item())


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--bootstrap_init_checkpoint", type=str, required=True)
    parser.add_argument("--bootstrap_local_max_px", type=float, default=1.0)
    parser.add_argument("--bootstrap_control_grid", type=int, default=5)
    parser.add_argument("--bootstrap_iterations", type=int, default=4)
    parser.add_argument("--bootstrap_max_update_px", type=float, default=0.5)
    parser.add_argument("--bootstrap_delta_init_std", type=float, default=1e-3)
    probe, remaining = parser.parse_known_args()
    cfg = parse_args(remaining)

    if cfg.stage != "train":
        raise ValueError("gradient probe uses --stage train")
    cfg.recurrent_hidden_channels = 64
    cfg.recurrent_correlation_channels = 32
    cfg.recurrent_iterations_scale1 = int(probe.bootstrap_iterations)
    cfg.recurrent_iterations_scale2 = 2
    cfg.recurrent_iterations_scale4 = 3
    cfg.recurrent_max_update_scale1 = float(probe.bootstrap_max_update_px)
    cfg.recurrent_max_update_scale2 = 1.0
    cfg.recurrent_max_update_scale4 = 2.0

    set_seed(cfg.seed)
    train_loader, _, info = build_loaders(cfg)
    device = get_device(cfg.device)
    process = build_progressive_process(cfg)
    model = build_recurrent_model(cfg, info, device, process)
    warm_start_compatible(model, probe.bootstrap_init_checkpoint, device)
    tiny_initialize_delta_head(model, probe.bootstrap_delta_init_std)
    freeze_to_local_branch(model)
    model.train()

    batch = next(iter(train_loader))
    gt = batch["gt"].to(device)
    msi = batch["hr_msi"].to(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(cfg.seed) + 991)
    warped, _, params = make_misaligned_msi(
        msi,
        translation_max_px=0.0,
        rotation_max_deg=0.0,
        local_max_displacement_px=float(probe.bootstrap_local_max_px),
        control_grid_size=int(probe.bootstrap_control_grid),
        generator=generator,
    )
    sequence = predict_scale1_sequence(model, process, gt, warped)
    mask = torch.ones(gt.shape[0], dtype=torch.bool, device=device)
    loss = sequence_flow_loss(
        {1: sequence},
        params.local_displacement_px,
        mask,
        normalization_px=float(probe.bootstrap_local_max_px),
        gamma=0.8,
    )
    loss.backward()

    local = model.geometry_aligner.local_aligner
    values = {
        "descriptor": grad_mean(local.descriptor.net[0].weight),
        "correlation": grad_mean(local.correlation_encoders["1"].net[0].weight),
        "gru_gates": grad_mean(local.gru.gates.weight),
        "gru_candidate": grad_mean(local.gru.candidate.weight),
        "delta_hidden": grad_mean(local.delta_head[0].weight),
        "delta_output": grad_mean(local.delta_head[-1].weight),
    }
    true_mag = torch.linalg.vector_norm(
        params.local_displacement_px.float(), dim=1
    ).mean().item()
    pred_mag = torch.linalg.vector_norm(sequence[-1].float(), dim=1).mean().item()
    print(
        f"loss={loss.item():.6f} true={true_mag:.4f}px "
        f"pred={pred_mag:.4f}px"
    )
    for name, value in values.items():
        print(f"grad_{name}={value:.6e}")
    if any(value <= 0.0 for value in values.values()):
        raise RuntimeError(
            "At least one bootstrap gradient path is zero; do not start full bootstrap."
        )
    print("Gradient path OK: all recurrent local components receive non-zero gradients.")


if __name__ == "__main__":
    main()
