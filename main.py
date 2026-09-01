"""S2Diff training / evaluation entry point.

V1 and V2 are HSI-only predictors used to validate Innovation 1. V3 is the
Innovation 2 predictor, which adds spectrally safer HR-MSI high-frequency
spatial guidance while keeping the same progressive physical degradation and
reverse update.
"""

from __future__ import annotations

import os

import torch

from config import parse_args, print_config
from data_loader import build_loaders
from innovation1 import build_progressive_process, evaluate, train_one_epoch
from models import (
    CleanHSIPredictor,
    MSIHighFrequencyGuidedPredictor,
    SpectralSpatialCleanHSIPredictor,
)
from utils import (
    CSVLogger,
    count_parameters,
    ensure_dir,
    get_device,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)


def _predictor_tag(cfg):
    version = str(getattr(cfg, "predictor_version", "v1")).lower()
    base_channels = int(cfg.predictor_base_channels)
    if version == "v1":
        return "" if base_channels == 64 else f"_bc{base_channels}"
    if version == "v2":
        return "_v2" if base_channels == 64 else f"_v2_bc{base_channels}"
    if version == "v3":
        return "_v3" if base_channels == 64 else f"_v3_bc{base_channels}"
    raise ValueError(f"Unsupported predictor_version: {version}")


def _checkpoint_paths(cfg):
    root = os.path.join(cfg.checkpoint_root, "innovation1")
    ensure_dir(root)

    if cfg.save_name:
        filename = cfg.save_name
        if not filename.endswith(".pth"):
            filename += ".pth"
    else:
        filename = (
            f"{cfg.dataset}_innovation1_{cfg.degradation_mode}"
            f"{_predictor_tag(cfg)}.pth"
        )

    best_path = os.path.join(root, filename)
    stem, ext = os.path.splitext(filename)
    last_path = os.path.join(root, f"{stem}_last{ext}")
    return best_path, last_path


def _build_model(cfg, info, device):
    common = dict(
        n_bands=int(info["n_bands"]),
        total_steps=int(cfg.diffusion_steps),
        base_channels=int(cfg.predictor_base_channels),
        time_dim=int(cfg.predictor_time_dim),
        dropout=float(cfg.predictor_dropout),
        residual_prediction=True,
    )
    version = str(getattr(cfg, "predictor_version", "v1")).lower()

    if version == "v1":
        model = CleanHSIPredictor(**common)
    elif version == "v2":
        model = SpectralSpatialCleanHSIPredictor(
            **common,
            spectral_hidden=int(getattr(cfg, "spectral_stem_hidden", 8)),
        )
    elif version == "v3":
        model = MSIHighFrequencyGuidedPredictor(
            **common,
            n_msi_bands=int(info["n_select_bands"]),
            spectral_hidden=int(getattr(cfg, "spectral_stem_hidden", 8)),
            msi_highpass_kernel=int(getattr(cfg, "msi_highpass_kernel", 5)),
            msi_highpass_sigma=float(getattr(cfg, "msi_highpass_sigma", 1.0)),
        )
    else:
        raise ValueError(f"Unsupported predictor_version: {version}")

    model = model.to(device)
    print(
        f"Predictor version={version}, base_channels={cfg.predictor_base_channels}, "
        f"requires_msi={bool(getattr(model, 'requires_msi', False))}, "
        f"trainable params={count_parameters(model):.3f} M"
    )
    return model


def _format_metrics(metrics):
    keys = [
        "PSNR", "SAM", "RMSE", "ERGAS", "SSIM", "CC",
        "INIT_PSNR", "INIT_SAM",
    ]
    return " ".join(
        f"{key}={metrics[key]:.6f}" for key in keys if key in metrics
    )


def run_train(cfg, train_loader, test_loader, info, device):
    process = build_progressive_process(cfg)
    model = _build_model(cfg, info, device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    best_path, last_path = _checkpoint_paths(cfg)
    start_epoch = 1
    best_psnr = float("-inf")

    if cfg.resume:
        loaded_epoch, loaded_best = load_checkpoint(
            model, cfg.resume, optimizer=optimizer, map_location=str(device)
        )
        start_epoch = int(loaded_epoch) + 1
        best_psnr = float(loaded_best)
        print(
            f"Resumed from {cfg.resume}: epoch={loaded_epoch}, "
            f"best_PSNR={best_psnr:.6f}"
        )

    transitions = process.transition_timesteps(radius=cfg.boundary_radius)
    print(
        "Progressive process: "
        f"mode={process.operator.mode}, T={process.total_steps}, "
        f"lift={process.default_lift_mode}, transitions={transitions}"
    )
    if hasattr(process.operator, "extra_repr"):
        print(f"Degradation operator: {process.operator.extra_repr()}")

    log_path = os.path.join(
        cfg.log_root,
        f"{cfg.dataset}_innovation1_{cfg.degradation_mode}"
        f"{_predictor_tag(cfg)}.csv",
    )
    logger = CSVLogger(
        log_path,
        fieldnames=[
            "epoch", "loss", "l1", "sam_loss", "deg_loss",
            "PSNR", "SAM", "RMSE", "ERGAS", "SSIM", "CC",
            "INIT_PSNR", "INIT_SAM", "best_PSNR",
        ],
    )

    for epoch in range(start_epoch, cfg.epochs + 1):
        stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            process,
            device,
            lambda_l1=cfg.lambda_l1,
            lambda_sam=cfg.lambda_sam,
            lambda_deg=cfg.lambda_deg,
            boundary_probability=cfg.boundary_probability,
            boundary_radius=cfg.boundary_radius,
            grad_clip=cfg.grad_clip,
        )

        print(
            f"Epoch {epoch:04d}/{cfg.epochs:04d} "
            f"loss={stats.loss:.6f} l1={stats.l1:.6f} "
            f"sam={stats.sam:.6f} deg={stats.deg:.6f}"
        )

        metrics = {}
        if epoch % cfg.eval_interval == 0 or epoch == cfg.epochs:
            metrics = evaluate(
                model, test_loader, process, device, scale_ratio=cfg.scale_ratio
            )
            print(f"  eval: {_format_metrics(metrics)}")

            psnr = float(metrics["PSNR"])
            if psnr > best_psnr:
                best_psnr = psnr
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    best_psnr,
                    best_path,
                    extra={"config": vars(cfg), "metrics": metrics},
                )
                print(f"  saved best checkpoint -> {best_path}")

        logger.write(
            {
                "epoch": epoch,
                "loss": stats.loss,
                "l1": stats.l1,
                "sam_loss": stats.sam,
                "deg_loss": stats.deg,
                "PSNR": metrics.get("PSNR", ""),
                "SAM": metrics.get("SAM", ""),
                "RMSE": metrics.get("RMSE", ""),
                "ERGAS": metrics.get("ERGAS", ""),
                "SSIM": metrics.get("SSIM", ""),
                "CC": metrics.get("CC", ""),
                "INIT_PSNR": metrics.get("INIT_PSNR", ""),
                "INIT_SAM": metrics.get("INIT_SAM", ""),
                "best_PSNR": best_psnr,
            }
        )

        if epoch % cfg.save_interval == 0 or epoch == cfg.epochs:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best_psnr,
                last_path,
                extra={"config": vars(cfg), "metrics": metrics},
            )

    print(f"Training complete. best_PSNR={best_psnr:.6f}")
    print(f"Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")
    print(f"Training log: {log_path}")


def run_test(cfg, test_loader, info, device):
    process = build_progressive_process(cfg)
    model = _build_model(cfg, info, device)
    best_path, _ = _checkpoint_paths(cfg)
    checkpoint = cfg.resume or best_path

    loaded_epoch, loaded_best = load_checkpoint(
        model,
        checkpoint,
        optimizer=None,
        map_location=str(device),
        load_optimizer=False,
    )
    print(
        f"Loaded checkpoint {checkpoint}: epoch={loaded_epoch}, "
        f"stored_best_PSNR={loaded_best:.6f}"
    )

    metrics = evaluate(
        model, test_loader, process, device, scale_ratio=cfg.scale_ratio
    )
    print(f"Test metrics: {_format_metrics(metrics)}")


def main():
    cfg = parse_args()
    print_config(cfg)
    set_seed(cfg.seed)

    train_loader, test_loader, info = build_loaders(cfg)
    print("\nDataset info:")
    for key, value in info.items():
        if key not in ("srf_weights", "hsi_wavelengths"):
            print(f"  {key}: {value}")

    device = get_device(cfg.device)
    print(f"Device: {device}")

    if cfg.stage == "train":
        run_train(cfg, train_loader, test_loader, info, device)
    elif cfg.stage == "test":
        run_test(cfg, test_loader, info, device)
    else:
        raise ValueError(f"Unsupported stage: {cfg.stage}")


if __name__ == "__main__":
    main()
