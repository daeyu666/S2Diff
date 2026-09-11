"""Standalone flow-only diagnosis for the scale-1 local bootstrap checkpoint."""

from __future__ import annotations

import argparse
import csv
import os

from config import parse_args
from data_loader import build_loaders
from innovation1 import build_progressive_process
from train_v4_local_bootstrap import evaluate_flow_severities
from train_v4_recurrent_flow import build_recurrent_model
from utils import get_device, load_checkpoint, set_seed


def parse_args_bootstrap_diagnostic():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--bootstrap_checkpoint", type=str, required=True)
    parser.add_argument(
        "--bootstrap_eval_severities",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0],
    )
    parser.add_argument("--bootstrap_control_grid", type=int, default=5)
    parser.add_argument("--bootstrap_iterations", type=int, default=4)
    parser.add_argument("--bootstrap_max_update_px", type=float, default=0.5)
    parser.add_argument("--bootstrap_eval_trials", type=int, default=5)
    parser.add_argument("--bootstrap_eval_seed_offset", type=int, default=15431)
    parser.add_argument("--bootstrap_output", type=str, default="")
    diagnostic, remaining = parser.parse_known_args()
    cfg = parse_args(remaining)

    if cfg.stage != "test":
        raise ValueError("bootstrap diagnosis is test-only; use --stage test")
    if str(cfg.predictor_version).lower() != "v4":
        raise ValueError("bootstrap diagnosis requires --predictor_version v4")
    if diagnostic.bootstrap_iterations < 1:
        raise ValueError("--bootstrap_iterations must be >= 1")
    if diagnostic.bootstrap_max_update_px <= 0.0:
        raise ValueError("--bootstrap_max_update_px must be > 0")
    if diagnostic.bootstrap_eval_trials < 1:
        raise ValueError("--bootstrap_eval_trials must be >= 1")
    if any(value <= 0.0 for value in diagnostic.bootstrap_eval_severities):
        raise ValueError("all bootstrap eval severities must be > 0")

    cfg.recurrent_hidden_channels = 64
    cfg.recurrent_correlation_channels = 32
    cfg.recurrent_iterations_scale1 = int(diagnostic.bootstrap_iterations)
    cfg.recurrent_iterations_scale2 = 2
    cfg.recurrent_iterations_scale4 = 3
    cfg.recurrent_max_update_scale1 = float(diagnostic.bootstrap_max_update_px)
    cfg.recurrent_max_update_scale2 = 1.0
    cfg.recurrent_max_update_scale4 = 2.0
    return cfg, diagnostic


def main():
    cfg, diagnostic = parse_args_bootstrap_diagnostic()
    set_seed(cfg.seed)
    _, test_loader, info = build_loaders(cfg)
    device = get_device(cfg.device)
    process = build_progressive_process(cfg)
    model = build_recurrent_model(cfg, info, device, process)

    epoch, stored_metric = load_checkpoint(
        model,
        diagnostic.bootstrap_checkpoint,
        optimizer=None,
        strict=True,
        map_location=str(device),
        load_optimizer=False,
    )
    print(
        f"Loaded bootstrap checkpoint {diagnostic.bootstrap_checkpoint}: "
        f"epoch={epoch}, stored_metric={stored_metric:.6f}"
    )
    print(
        "Protocol: local-only, global identity, physical scale=1 only; "
        "reported values are flow-field metrics, not reconstruction metrics."
    )

    results = evaluate_flow_severities(
        model,
        test_loader,
        process,
        device,
        severities=[float(v) for v in diagnostic.bootstrap_eval_severities],
        control_grid=int(diagnostic.bootstrap_control_grid),
        trials=int(diagnostic.bootstrap_eval_trials),
        seed=int(cfg.seed) + int(diagnostic.bootstrap_eval_seed_offset),
    )

    rows = []
    for severity, result in results.items():
        iter_text = " ".join(
            f"{key}={value:.4f}"
            for key, value in result.items()
            if key.startswith("iter")
        )
        print(
            f"local<={severity:g}px "
            f"true={result['true_mean_px']:.4f}px "
            f"pred={result['pred_mean_px']:.4f}px "
            f"ratio={result['pred_to_true_ratio']:.3f} "
            f"EPE={result['epe_px']:.4f}px {iter_text}"
        )
        rows.append({"local_max_px": severity, **result})

    output = diagnostic.bootstrap_output
    if not output:
        output = os.path.join(
            cfg.output_root,
            "metrics",
            f"{cfg.dataset}_v4_local_bootstrap_flow.csv",
        )
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with open(output, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved flow-only diagnosis -> {output}")


if __name__ == "__main__":
    main()
