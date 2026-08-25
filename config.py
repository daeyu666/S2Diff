# config.py
import argparse
import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DatasetConfig:
    """单个数据集的配置信息。"""
    name: str
    file_name: str
    mat_keys: list
    n_select_bands: int = 5


@dataclass
class TrainConfig:
    """通用训练配置，不依赖具体模型结构。"""

    # --- 路径 ---
    project_root: str = "."
    data_root: str = "./data/raw"
    cache_root: str = "./data/cache"
    checkpoint_root: str = "./checkpoints"
    log_root: str = "./logs"
    output_root: str = "./outputs"

    # --- 运行阶段 ---
    stage: str = "train"
    dataset: str = "PaviaU"

    # --- 数据 ---
    image_size: int = 128
    patch_size: int = 64
    stride: int = 32
    scale_ratio: int = 4
    n_select_bands: int = 5

    # --- LR-HSI 退化 ---
    # 新实验默认使用物理退化；旧 make_lr_hsi() 函数仍保留 legacy 默认行为。
    degradation_mode: str = "physical"  # bicubic / gaussian_bicubic / physical
    degradation_sigma: float = 2.0       # gaussian_bicubic legacy sigma
    degradation_kernel_size: int = 5     # gaussian_bicubic legacy kernel
    mtf_nyquist: float = 0.2             # physical Gaussian PSF target MTF@LR Nyquist
    psf_truncate: float = 3.0

    # --- 渐进退化 ---
    progressive_steps: int = 12
    progressive_lift: str = "auto"       # auto -> physical: normalized_adjoint, ordinary: bilinear
    boundary_probability: float = 0.2
    boundary_radius: int = 1

    # --- MSI 生成模式 ---
    msi_mode: str = "uniform"          # "uniform" 或 "srf"
    srf_path: str = "./data/srf/wv2_relative_spectral_response_data_for_i.atcorr.csv"
    wavelength_root: str = "./data/wavelengths"
    wavelength_path: str = ""
    srf_interp: str = "pchip"          # "pchip" 或 "linear"
    srf_band_set: str = "wv2_visible6" # "wv2_visible5" / "wv2_visible6" / "wv2_all8"

    # --- 训练 ---
    epochs: int = 300
    batch_size: int = 4
    num_workers: int = 0
    lr: float = 1e-4
    weight_decay: float = 0.0
    seed: int = 10
    device: str = "cuda"

    # --- 损失权重 ---
    lambda_l1: float = 1.0
    lambda_sam: float = 0.1
    lambda_dc: float = 0.1
    lambda_sgrad: float = 0.05
    lambda_sdir: float = 0.2
    lambda_ns_l1: float = 1.0
    lambda_srf_region: float = 0.3
    lambda_mse: float = 1.0

    # --- 保存 / 恢复 ---
    save_interval: int = 20
    eval_interval: int = 1
    resume: str = ""
    save_name: str = ""

    # --- 数据集注册表 ---
    datasets: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 默认数据集配置
# ---------------------------------------------------------------------------
def get_dataset_configs():
    return {
        "PaviaU": DatasetConfig(
            name="PaviaU",
            file_name="PaviaU.mat",
            mat_keys=["paviaU", "PaviaU", "img", "data"],
            n_select_bands=6,
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


def resolve_config_defaults(cfg: TrainConfig):
    """Resolve defaults that depend on another option without hiding explicit errors."""
    if cfg.progressive_lift == "auto":
        cfg.progressive_lift = (
            "normalized_adjoint"
            if cfg.degradation_mode == "physical"
            else "bilinear"
        )


def validate_config(cfg: TrainConfig):
    if cfg.scale_ratio < 1:
        raise ValueError("scale_ratio must be >= 1")
    if cfg.progressive_steps < 1:
        raise ValueError("progressive_steps must be >= 1")
    if cfg.degradation_kernel_size < 1 or cfg.degradation_kernel_size % 2 == 0:
        raise ValueError("degradation_kernel_size must be a positive odd integer")
    if cfg.degradation_sigma < 0:
        raise ValueError("degradation_sigma must be >= 0")
    if not 0.0 < cfg.mtf_nyquist <= 1.0:
        raise ValueError("mtf_nyquist must lie in (0, 1]")
    if cfg.psf_truncate <= 0:
        raise ValueError("psf_truncate must be > 0")
    if not 0.0 <= cfg.boundary_probability <= 1.0:
        raise ValueError("boundary_probability must lie in [0, 1]")
    if cfg.boundary_radius < 0:
        raise ValueError("boundary_radius must be >= 0")

    if cfg.degradation_mode != "physical" and cfg.progressive_lift in (
        "adjoint",
        "normalized_adjoint",
    ):
        raise ValueError(
            "adjoint/normalized_adjoint lift is only defined for physical "
            "degradation; use bilinear or nearest for ordinary degradation"
        )


# ---------------------------------------------------------------------------
# 命令行参数解析
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="HSI Super-Resolution Template")

    # --- 运行控制 ---
    parser.add_argument("--stage", type=str, default="train")
    parser.add_argument("--dataset", type=str, default="PaviaU")

    # --- 路径 ---
    parser.add_argument("--data_root", type=str, default="./data/raw")
    parser.add_argument("--checkpoint_root", type=str, default="./checkpoints")
    parser.add_argument("--log_root", type=str, default="./logs")
    parser.add_argument("--output_root", type=str, default="./outputs")

    # --- 数据 ---
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--scale_ratio", type=int, default=4)
    parser.add_argument("--n_select_bands", type=int, default=5)

    # --- LR-HSI 退化 ---
    parser.add_argument(
        "--degradation_mode",
        type=str,
        default="physical",
        choices=["bicubic", "gaussian_bicubic", "physical"],
    )
    parser.add_argument("--degradation_sigma", type=float, default=2.0)
    parser.add_argument("--degradation_kernel_size", type=int, default=5)
    parser.add_argument("--mtf_nyquist", type=float, default=0.2)
    parser.add_argument("--psf_truncate", type=float, default=3.0)

    # --- 渐进退化 ---
    parser.add_argument("--progressive_steps", type=int, default=12)
    parser.add_argument(
        "--progressive_lift",
        type=str,
        default="auto",
        choices=["auto", "bilinear", "nearest", "adjoint", "normalized_adjoint"],
    )
    parser.add_argument("--boundary_probability", type=float, default=0.2)
    parser.add_argument("--boundary_radius", type=int, default=1)

    # --- MSI 生成 ---
    parser.add_argument(
        "--msi_mode", type=str, default="uniform", choices=["uniform", "srf"]
    )
    parser.add_argument(
        "--srf_path",
        type=str,
        default="./data/srf/wv2_relative_spectral_response_data_for_i.atcorr.csv",
    )
    parser.add_argument("--wavelength_root", type=str, default="./data/wavelengths")
    parser.add_argument("--wavelength_path", type=str, default="")
    parser.add_argument(
        "--srf_interp", type=str, default="pchip", choices=["pchip", "linear"]
    )
    parser.add_argument(
        "--srf_band_set",
        type=str,
        default="wv2_visible6",
        choices=["wv2_visible5", "wv2_visible6", "wv2_all8"],
    )

    # --- 训练 ---
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")

    # --- 损失权重 ---
    parser.add_argument("--lambda_l1", type=float, default=1.0)
    parser.add_argument("--lambda_sam", type=float, default=0.1)
    parser.add_argument("--lambda_dc", type=float, default=0.1)
    parser.add_argument("--lambda_sgrad", type=float, default=0.05)
    parser.add_argument("--lambda_sdir", type=float, default=0.2)
    parser.add_argument("--lambda_ns_l1", type=float, default=1.0)
    parser.add_argument("--lambda_srf_region", type=float, default=0.3)
    parser.add_argument("--lambda_mse", type=float, default=1.0)

    # --- 保存 / 恢复 ---
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
    if dataset_cfg is not None:
        cfg.n_select_bands = args.n_select_bands or dataset_cfg.n_select_bands

    resolve_config_defaults(cfg)
    validate_config(cfg)
    make_dirs(cfg)
    return cfg


# ---------------------------------------------------------------------------
# 目录创建
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def get_checkpoint_path(cfg: TrainConfig, stage: str = None, name: str = None):
    stage = stage or cfg.stage
    if name is None or name == "":
        name = f"{cfg.dataset}_{stage}.pth"
    return os.path.join(cfg.checkpoint_root, stage, name)


def print_config(cfg: TrainConfig):
    print("=" * 60)
    print("HSI Super-Resolution Template  Config")
    print("=" * 60)
    for key, value in cfg.__dict__.items():
        if key != "datasets":
            print(f"  {key}: {value}")
    print("=" * 60)


if __name__ == "__main__":
    cfg = parse_args()
    print_config(cfg)
