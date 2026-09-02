# config.py
import argparse
import os
from dataclasses import dataclass, field
from typing import List, Optional


TIME_FREE_MSI_ABLATIONS = {
    "raw_direct",
    "raw_gate",
    "hf_direct",
    "hf_gate",
}


@dataclass
class DatasetConfig:
    name: str
    file_name: str
    mat_keys: list
    n_select_bands: int = 5


@dataclass
class TrainConfig:
    project_root: str = "."
    data_root: str = "./data/raw"
    cache_root: str = "./data/cache"
    checkpoint_root: str = "./checkpoints"
    log_root: str = "./logs"
    output_root: str = "./outputs"

    stage: str = "train"
    dataset: str = "PaviaU"

    image_size: int = 128
    patch_size: int = 64
    stride: int = 32
    scale_ratio: int = 4
    n_select_bands: int = 4

    # Fixed sensor protocols: PaviaU->IKONOS4; Houston13/Chikusei->WV2 all8.
    msi_mode: str = "srf"
    srf_path: str = ""
    wavelength_root: str = "./data/wavelengths"
    wavelength_path: str = ""
    srf_interp: str = "pchip"
    srf_band_set: str = "auto"

    degradation_mode: str = "physical"
    diffusion_steps: int = 12
    lift_mode: str = "auto"
    mtf_nyquist: float = 0.2
    psf_truncate: float = 3.0
    gaussian_sigma: float = 2.0
    gaussian_kernel_size: int = 5
    boundary_probability: float = 0.2
    boundary_radius: int = 1

    # v1: plain HSI U-Net; v2: spectral-spatial HSI-only;
    # v3: Innovation 2 MSI-guided predictor.
    predictor_version: str = "v1"
    predictor_base_channels: int = 64
    predictor_time_dim: int = 256
    predictor_dropout: float = 0.0
    spectral_stem_hidden: int = 8
    msi_highpass_kernel: int = 5
    msi_highpass_sigma: float = 1.0
    # Legacy modes plus the new fully time-free Raw/HF x Direct/Gate grid.
    msi_ablation: str = "full"

    epochs: int = 300
    batch_size: int = 4
    num_workers: int = 0
    lr: float = 1e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    seed: int = 10
    device: str = "cuda"

    lambda_l1: float = 1.0
    lambda_sam: float = 0.1
    lambda_deg: float = 0.0
    lambda_dc: float = 0.1
    lambda_sgrad: float = 0.05
    lambda_sdir: float = 0.2
    lambda_ns_l1: float = 1.0
    lambda_srf_region: float = 0.3
    lambda_mse: float = 1.0

    save_interval: int = 20
    eval_interval: int = 1
    resume: str = ""
    save_name: str = ""

    datasets: dict = field(default_factory=dict)


def get_dataset_configs():
    return {
        "PaviaU": DatasetConfig(
            name="PaviaU",
            file_name="PaviaU.mat",
            mat_keys=["paviaU", "PaviaU", "img", "data"],
            n_select_bands=4,
        ),
        "Houston13": DatasetConfig(
            name="Houston13",
            file_name="Houston13.mat",
            mat_keys=["Houston13", "Houston_HSI", "data", "img"],
            n_select_bands=8,
        ),
        "Chikusei": DatasetConfig(
            name="Chikusei",
            file_name="Chikusei.mat",
            mat_keys=["chikusei", "Chikusei", "img", "data"],
            n_select_bands=8,
        ),
    }


def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        description="S2Diff progressive physical degradation + MSI guidance"
    )

    parser.add_argument("--stage", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--dataset", type=str, default="PaviaU")

    parser.add_argument("--data_root", type=str, default="./data/raw")
    parser.add_argument("--checkpoint_root", type=str, default="./checkpoints")
    parser.add_argument("--log_root", type=str, default="./logs")
    parser.add_argument("--output_root", type=str, default="./outputs")

    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--scale_ratio", type=int, default=4)
    parser.add_argument("--n_select_bands", type=int, default=0)

    parser.add_argument("--msi_mode", type=str, default="srf", choices=["uniform", "srf"])
    parser.add_argument("--srf_path", type=str, default="")
    parser.add_argument("--wavelength_root", type=str, default="./data/wavelengths")
    parser.add_argument("--wavelength_path", type=str, default="")
    parser.add_argument("--srf_interp", type=str, default="pchip", choices=["pchip", "linear"])
    parser.add_argument(
        "--srf_band_set",
        type=str,
        default="auto",
        choices=["auto", "ikonos4", "wv2_visible5", "wv2_visible6", "wv2_all8"],
    )

    parser.add_argument(
        "--degradation_mode",
        type=str,
        default="physical",
        choices=["physical", "gaussian_bicubic", "bicubic"],
    )
    parser.add_argument("--diffusion_steps", type=int, default=12)
    parser.add_argument(
        "--lift_mode",
        type=str,
        default="auto",
        choices=["auto", "bilinear", "nearest", "adjoint", "normalized_adjoint"],
    )
    parser.add_argument("--mtf_nyquist", type=float, default=0.2)
    parser.add_argument("--psf_truncate", type=float, default=3.0)
    parser.add_argument("--gaussian_sigma", type=float, default=2.0)
    parser.add_argument("--gaussian_kernel_size", type=int, default=5)
    parser.add_argument("--boundary_probability", type=float, default=0.2)
    parser.add_argument("--boundary_radius", type=int, default=1)

    parser.add_argument(
        "--predictor_version", type=str, default="v1", choices=["v1", "v2", "v3"]
    )
    parser.add_argument("--predictor_base_channels", type=int, default=64)
    parser.add_argument("--predictor_time_dim", type=int, default=256)
    parser.add_argument("--predictor_dropout", type=float, default=0.0)
    parser.add_argument("--spectral_stem_hidden", type=int, default=8)
    parser.add_argument("--msi_highpass_kernel", type=int, default=5)
    parser.add_argument("--msi_highpass_sigma", type=float, default=1.0)
    parser.add_argument(
        "--msi_ablation",
        type=str,
        default="full",
        choices=[
            "no_msi",
            "full",
            "raw_msi",
            "hf_nogate",
            "hf_const",
            "raw_direct",
            "raw_gate",
            "hf_direct",
            "hf_gate",
        ],
        help=(
            "Legacy: no_msi/full/raw_msi/hf_nogate/hf_const. "
            "Time-free orthogonal grid: raw_direct/raw_gate/hf_direct/hf_gate; "
            "the new grid uses alpha=1 and removes timestep conditioning from "
            "the MSI transfer gate."
        ),
    )

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--lambda_l1", type=float, default=1.0)
    parser.add_argument("--lambda_sam", type=float, default=0.1)
    parser.add_argument("--lambda_deg", type=float, default=0.0)
    parser.add_argument("--lambda_dc", type=float, default=0.1)
    parser.add_argument("--lambda_sgrad", type=float, default=0.05)
    parser.add_argument("--lambda_sdir", type=float, default=0.2)
    parser.add_argument("--lambda_ns_l1", type=float, default=1.0)
    parser.add_argument("--lambda_srf_region", type=float, default=0.3)
    parser.add_argument("--lambda_mse", type=float, default=1.0)

    parser.add_argument("--save_interval", type=int, default=20)
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--save_name", type=str, default="")

    args = parser.parse_args(argv)
    cfg = TrainConfig()
    cfg.datasets = get_dataset_configs()
    for key, value in vars(args).items():
        setattr(cfg, key, value)

    dataset_cfg = cfg.datasets.get(cfg.dataset)
    if dataset_cfg is None:
        raise ValueError(
            f"Unknown dataset {cfg.dataset!r}; available: {sorted(cfg.datasets)}"
        )
    cfg.n_select_bands = args.n_select_bands or dataset_cfg.n_select_bands

    if not 0.0 <= cfg.boundary_probability <= 1.0:
        raise ValueError("boundary_probability must lie in [0, 1]")
    if cfg.lambda_deg < 0.0:
        raise ValueError("lambda_deg must be >= 0")
    if cfg.msi_highpass_kernel < 3 or cfg.msi_highpass_kernel % 2 == 0:
        raise ValueError("msi_highpass_kernel must be odd and >= 3")
    if cfg.msi_highpass_sigma <= 0.0:
        raise ValueError("msi_highpass_sigma must be > 0")

    # Innovation-2 ablations must never silently fall back to the HSI-only V1/V2
    # predictor. This fail-fast guard prevents wasting long training runs when a
    # CLI argument is dropped by a shell/IDE launch configuration.
    if cfg.msi_ablation in TIME_FREE_MSI_ABLATIONS and cfg.predictor_version != "v3":
        raise ValueError(
            "Innovation 2 ablation "
            f"{cfg.msi_ablation!r} requires --predictor_version v3, but parsed "
            f"predictor_version={cfg.predictor_version!r}. Check the actual command "
            "received by Python before starting training."
        )

    make_dirs(cfg)
    return cfg


def make_dirs(cfg: TrainConfig):
    dirs = [
        cfg.checkpoint_root,
        cfg.log_root,
        cfg.output_root,
        os.path.join(cfg.output_root, "predictions", cfg.dataset),
        os.path.join(cfg.output_root, "metrics"),
        os.path.join(cfg.output_root, "figures"),
    ]
    for path in dirs:
        os.makedirs(path, exist_ok=True)


def get_checkpoint_path(cfg: TrainConfig, stage: str = None, name: str = None):
    stage = stage or cfg.stage
    if not name:
        name = f"{cfg.dataset}_{stage}.pth"
    return os.path.join(cfg.checkpoint_root, stage, name)


def print_config(cfg: TrainConfig):
    print("=" * 60)
    print("S2Diff Config")
    print("=" * 60)
    for key, value in cfg.__dict__.items():
        if key != "datasets":
            print(f"  {key}: {value}")
    print("=" * 60)


if __name__ == "__main__":
    cfg = parse_args()
    print_config(cfg)
