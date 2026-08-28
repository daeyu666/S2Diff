"""Sanity-check progressive HSI degradation before training a network.

Example:
    python check_degradation_trajectory.py \
        --dataset PaviaU \
        --mode physical \
        --lift_mode auto \
        --crop_size 128 \
        --total_steps 12 \
        --scale_ratio 4 \
        --mtf_nyquist 0.2

The script reports per-timestep:
- progressive scale and strength
- state PSNR / SAM against HR-HSI
- mean and standard deviation
- high-frequency power ratio
- L1 change from the previous state
- whether the current step crosses a scale transition
- terminal closure error

Only a center crop is retained by default so Chikusei/PaviaU checks remain cheap.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from typing import Dict, List

import numpy as np
import torch

from config import get_dataset_configs
from data_loader import crop_to_scale, normalize_hsi, read_hsi_mat
from degradations import ProgressiveDegradation, build_degradation


def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    if mse <= eps:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def sam_degrees(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> float:
    dot = torch.sum(pred * target, dim=1)
    pred_norm = torch.sqrt(torch.sum(pred * pred, dim=1) + eps)
    target_norm = torch.sqrt(torch.sum(target * target, dim=1) + eps)
    cosine = dot / (pred_norm * target_norm + eps)
    cosine = torch.clamp(cosine, -1.0 + eps, 1.0 - eps)
    angle = torch.acos(cosine)
    return float(angle.mean().item() * 180.0 / math.pi)


def high_frequency_ratio(
    x: torch.Tensor, cutoff: float = 0.25, eps: float = 1e-12
) -> float:
    _, _, h, w = x.shape
    spectrum = torch.fft.fft2(x, dim=(-2, -1), norm="ortho")
    power = spectrum.abs().square()

    fy = torch.fft.fftfreq(h, device=x.device, dtype=x.dtype)
    fx = torch.fft.fftfreq(w, device=x.device, dtype=x.dtype)
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    radius = torch.sqrt(xx.square() + yy.square())
    mask = radius >= cutoff

    high = power[..., mask].sum()
    total = power.sum().clamp_min(eps)
    return float((high / total).item())


def center_crop_hsi(img: np.ndarray, crop_size: int, scale_ratio: int) -> np.ndarray:
    if crop_size <= 0:
        return crop_to_scale(img, scale_ratio)

    h, w, _ = img.shape
    size = min(int(crop_size), h, w)
    size = max(scale_ratio, size // scale_ratio * scale_ratio)
    top = max((h - size) // 2, 0)
    left = max((w - size) // 2, 0)
    crop = img[top:top + size, left:left + size, :]
    return crop_to_scale(crop, scale_ratio)


def choose_rgb_indices(n_bands: int):
    return [
        min(n_bands - 1, int(round(0.75 * (n_bands - 1)))),
        min(n_bands - 1, int(round(0.50 * (n_bands - 1)))),
        min(n_bands - 1, int(round(0.25 * (n_bands - 1)))),
    ]


def to_rgb(x: torch.Tensor) -> np.ndarray:
    arr = x[0].detach().cpu().numpy()
    ids = choose_rgb_indices(arr.shape[0])
    rgb = np.stack([arr[i] for i in ids], axis=-1)
    lo = np.percentile(rgb, 1.0)
    hi = np.percentile(rgb, 99.0)
    if hi <= lo:
        return np.clip(rgb, 0.0, 1.0)
    return np.clip((rgb - lo) / (hi - lo), 0.0, 1.0)


def resolve_lift(mode: str, lift_mode: str) -> str:
    if lift_mode == "auto":
        return "normalized_adjoint" if mode == "physical" else "bilinear"
    if mode != "physical" and lift_mode in ("adjoint", "normalized_adjoint"):
        raise ValueError(
            "Strict adjoint lift is only defined for physical mode. "
            "Use --lift_mode auto, bilinear, or nearest for ordinary degradation."
        )
    return lift_mode


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="PaviaU")
    parser.add_argument("--data_root", default="./data/raw")
    parser.add_argument(
        "--mode",
        default="physical",
        choices=["physical", "gaussian_bicubic", "bicubic"],
    )
    parser.add_argument(
        "--lift_mode",
        default="auto",
        choices=["auto", "bilinear", "nearest", "adjoint", "normalized_adjoint"],
    )
    parser.add_argument("--scale_ratio", type=int, default=4)
    parser.add_argument("--total_steps", type=int, default=12)
    parser.add_argument("--crop_size", type=int, default=128)
    parser.add_argument("--mtf_nyquist", type=float, default=0.2)
    parser.add_argument("--legacy_sigma", type=float, default=2.0)
    parser.add_argument("--legacy_kernel", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output_dir", default="./outputs/degradation_trajectory"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfgs = get_dataset_configs()
    if args.dataset not in cfgs:
        raise KeyError(
            f"Unknown dataset {args.dataset!r}. "
            f"Available: {sorted(cfgs.keys())}"
        )

    dcfg = cfgs[args.dataset]
    file_path = os.path.join(args.data_root, dcfg.file_name)
    hsi = read_hsi_mat(file_path, dcfg.mat_keys)
    hsi = normalize_hsi(hsi)
    hsi = center_crop_hsi(hsi, args.crop_size, args.scale_ratio)

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    x = torch.from_numpy(hsi).permute(2, 0, 1).unsqueeze(0).to(device)

    kwargs = {}
    if args.mode == "physical":
        kwargs["mtf_nyquist"] = args.mtf_nyquist
    elif args.mode == "gaussian_bicubic":
        kwargs["sigma"] = args.legacy_sigma
        kwargs["kernel_size"] = args.legacy_kernel

    operator = build_degradation(
        args.mode, scale_ratio=args.scale_ratio, **kwargs
    )
    lift_mode = resolve_lift(args.mode, args.lift_mode)
    trajectory = ProgressiveDegradation(
        operator=operator,
        total_steps=args.total_steps,
        default_lift_mode=lift_mode,
    )

    trajectory.assert_terminal_closure(x)
    direct = trajectory.terminal_observation(x)
    end = trajectory.degrade_at(x, args.total_steps)
    closure_error = float((direct - end).abs().max().item())

    rows: List[Dict[str, float]] = []
    states = []
    previous_state = None
    previous_scale = None

    with torch.no_grad():
        for t in range(args.total_steps + 1):
            spec = trajectory.state(t)
            x_t = trajectory.state_at(x, t)
            step_l1 = (
                0.0
                if previous_state is None
                else float((x_t - previous_state).abs().mean().item())
            )
            transition = int(
                previous_scale is not None and spec.scale != previous_scale
            )

            states.append(x_t.detach().cpu())
            rows.append(
                {
                    "t": t,
                    "scale": spec.scale,
                    "strength": spec.strength,
                    "scale_transition": transition,
                    "step_l1": step_l1,
                    "psnr": psnr(x_t, x),
                    "sam_deg": sam_degrees(x_t, x),
                    "mean": float(x_t.mean().item()),
                    "std": float(x_t.std().item()),
                    "hf_ratio": high_frequency_ratio(x_t),
                }
            )
            previous_state = x_t
            previous_scale = spec.scale

    os.makedirs(args.output_dir, exist_ok=True)
    stem = f"{args.dataset}_{args.mode}_{lift_mode}"
    csv_path = os.path.join(args.output_dir, stem + ".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 116)
    print(
        f"dataset={args.dataset} crop={hsi.shape[:2]} mode={args.mode} "
        f"lift={lift_mode} T={args.total_steps} scale={args.scale_ratio}"
    )
    if args.mode == "physical":
        print(
            f"mtf_nyquist={args.mtf_nyquist} "
            f"terminal_sigma={operator.terminal_sigma:.6f}"
        )
    print(f"terminal_closure_max_abs_error={closure_error:.6e}")
    print("-" * 116)
    print(
        f"{'t':>3} {'r':>3} {'str':>7} {'jump':>5} {'stepL1':>10} "
        f"{'PSNR':>10} {'SAM(deg)':>10} {'mean':>10} {'std':>10} {'HF ratio':>10}"
    )
    for row in rows:
        psnr_text = "inf" if math.isinf(row["psnr"]) else f"{row['psnr']:.4f}"
        print(
            f"{row['t']:3d} {row['scale']:3d} {row['strength']:7.4f} "
            f"{row['scale_transition']:5d} {row['step_l1']:10.6f} "
            f"{psnr_text:>10} {row['sam_deg']:10.4f} "
            f"{row['mean']:10.6f} {row['std']:10.6f} "
            f"{row['hf_ratio']:10.6f}"
        )
    print(f"\nCSV saved to: {csv_path}")

    try:
        import matplotlib.pyplot as plt

        cols = 4
        rows_n = int(math.ceil(len(states) / cols))
        fig, axes = plt.subplots(
            rows_n, cols, figsize=(4 * cols, 3.5 * rows_n)
        )
        axes = np.asarray(axes).reshape(-1)
        for idx, (state_tensor, row) in enumerate(zip(states, rows)):
            marker = " *" if row["scale_transition"] else ""
            axes[idx].imshow(to_rgb(state_tensor))
            axes[idx].set_title(
                f"t={row['t']} r={row['scale']}{marker}\n"
                f"PSNR={row['psnr']:.2f} step={row['step_l1']:.4f}"
            )
            axes[idx].axis("off")
        for idx in range(len(states), len(axes)):
            axes[idx].axis("off")
        fig.tight_layout()
        figure_path = os.path.join(args.output_dir, stem + ".png")
        fig.savefig(figure_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"Figure saved to: {figure_path}")
    except ImportError:
        print("matplotlib is not installed; skipping trajectory figure.")


if __name__ == "__main__":
    main()
