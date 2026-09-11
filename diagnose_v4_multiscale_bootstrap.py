"""Flow-only diagnosis for the 4->2->1 physical multiscale bootstrap."""

from __future__ import annotations

import argparse
import csv
import os

from config import parse_args
from data_loader import build_loaders
from innovation1 import build_progressive_process
from train_v4_multiscale_bootstrap import evaluate_severities
from train_v4_recurrent_flow import build_recurrent_model
from utils import get_device, load_checkpoint, set_seed


def parse_diagnostic_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--multiscale_checkpoint", type=str, required=True)
    parser.add_argument(
        "--multiscale_eval_severities",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0],
    )
    parser.add_argument("--multiscale_control_grid", type=int, default=5)
    parser.add_argument("--multiscale_iterations_scale4", type=int, default=3)
    parser.add_argument("--multiscale_iterations_scale2", type=int, default=2)
    parser.add_argument("--multiscale_iterations_scale1", type=int, default=2)
    parser.add_argument("--multiscale_max_update_scale4", type=float, default=2.0)
    parser.add_argument("--multiscale_max_update_scale2", type=float, default=1.0)
    parser.add_argument("--multiscale_max_update_scale1", type=float, default=0.5)
    parser.add_argument("--multiscale_eval_trials", type=int, default=5)
    parser.add_argument("--multiscale_eval_seed_offset", type=int, default=15431)
    parser.add_argument("--multiscale_output", type=str, default="")
    diagnostic, remaining = parser.parse_known_args()
    cfg = parse_args(remaining)

    if cfg.stage != "test":
        raise ValueError("multiscale diagnosis is test-only; use --stage test")
    if str(cfg.predictor_version).lower() != "v4":
        raise ValueError("multiscale diagnosis requires --predictor_version v4")
    if diagnostic.multiscale_control_grid < 2:
        raise ValueError("multiscale control grid must be >= 2")
    if min(
        diagnostic.multiscale_iterations_scale4,
        diagnostic.multiscale_iterations_scale2,
        diagnostic.multiscale_iterations_scale1,
    ) < 1:
        raise ValueError("multiscale iteration counts must be >= 1")
    if min(
        diagnostic.multiscale_max_update_scale4,
        diagnostic.multiscale_max_update_scale2,
        diagnostic.multiscale_max_update_scale1,
    ) <= 0.0:
        raise ValueError("multiscale max updates must be > 0")
    if diagnostic.multiscale_eval_trials < 1:
        raise ValueError("multiscale_eval_trials must be >= 1")
    if any(v <= 0.0 for v in diagnostic.multiscale_eval_severities):
        raise ValueError("all multiscale severities must be > 0")

    cfg.recurrent_hidden_channels = 64
    cfg.recurrent_correlation_channels = 32
    cfg.recurrent_iterations_scale4 = int(diagnostic.multiscale_iterations_scale4)
    cfg.recurrent_iterations_scale2 = int(diagnostic.multiscale_iterations_scale2)
    cfg.recurrent_iterations_scale1 = int(diagnostic.multiscale_iterations_scale1)
    cfg.recurrent_max_update_scale4 = float(diagnostic.multiscale_max_update_scale4)
    cfg.recurrent_max_update_scale2 = float(diagnostic.multiscale_max_update_scale2)
    cfg.recurrent_max_update_scale1 = float(diagnostic.multiscale_max_update_scale1)
    return cfg, diagnostic


def main():
    cfg, diagnostic = parse_diagnostic_args()
    set_seed(cfg.seed)
    _, test_loader, info = build_loaders(cfg)
    device = get_device(cfg.device)
    process = build_progressive_process(cfg)
    model = build_recurrent_model(cfg, info, device, process)

    epoch, best = load_checkpoint(
        model,
        diagnostic.multiscale_checkpoint,
        optimizer=None,
        strict=True,
        map_location=str(device),
        load_optimizer=False,
    )
    print(
        f"Loaded multiscale bootstrap checkpoint: {diagnostic.multiscale_checkpoint} "
        f"(epoch={epoch}, stored_best_EPE={best:.6f})"
    )

    results = evaluate_severities(
        model,
        test_loader,
        process,
        device,
        severities=[float(v) for v in diagnostic.multiscale_eval_severities],
        control_grid=int(diagnostic.multiscale_control_grid),
        trials=int(diagnostic.multiscale_eval_trials),
        seed=int(cfg.seed) + int(diagnostic.multiscale_eval_seed_offset),
    )

    rows = []
    for severity in sorted(results):
        r = results[severity]
        print(
            f"local<={severity:g}: "
            f"s4 pred/true={r['pred4']:.4f}/{r['true4']:.4f}, EPE={r['epe4']:.4f}; "
            f"s2={r['pred2']:.4f}/{r['true2']:.4f}, EPE={r['epe2']:.4f}; "
            f"s1={r['pred1']:.4f}/{r['true1']:.4f}, EPE={r['epe1']:.4f}; "
            f"final_ratio={r['final_ratio']:.4f}"
        )
        rows.append(
            {
                "local_max_px": severity,
                "scale4_true_mean_px": r["true4"],
                "scale4_pred_mean_px": r["pred4"],
                "scale4_epe_px": r["epe4"],
                "scale2_true_mean_px": r["true2"],
                "scale2_pred_mean_px": r["pred2"],
                "scale2_epe_px": r["epe2"],
                "scale1_true_mean_px": r["true1"],
                "scale1_pred_mean_px": r["pred1"],
                "scale1_epe_px": r["epe1"],
                "final_pred_to_true_ratio": r["final_ratio"],
            }
        )

    output = diagnostic.multiscale_output
    if not output:
        output = os.path.join(
            cfg.output_root,
            "metrics",
            f"{cfg.dataset}_v4_multiscale_bootstrap_flow.csv",
        )
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved multiscale bootstrap flow diagnosis -> {output}")


if __name__ == "__main__":
    main()
