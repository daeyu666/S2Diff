"""HSI Super-Resolution project entrypoint.

The current branch provides data/SRF utilities plus switchable LR-HSI degradation
and progressive sensor-degradation states. The neural model and training loop are
intentionally not implemented yet.
"""

from config import parse_args, print_config
from data_loader import build_loaders
from utils import get_device, set_seed


def main():
    cfg = parse_args()
    print_config(cfg)
    set_seed(cfg.seed)

    train_loader, test_loader, info = build_loaders(cfg)

    hidden = {
        "srf_weights",
        "hsi_wavelengths",
        "degradation_operator",
        "progressive_degradation",
    }
    print("\nDataset / degradation info:")
    for key, value in info.items():
        if key not in hidden:
            print(f"  {key}: {value}")

    degradation = info["degradation_operator"]
    trajectory = info["progressive_degradation"]
    print(f"  degradation_operator: {degradation}")
    print(
        "  progressive_schedule: "
        f"T={trajectory.total_steps}, stages={trajectory.stages}, "
        f"lift={trajectory.default_lift_mode}"
    )

    device = get_device(cfg.device)
    print(f"  device: {device}")

    # Neural model/training will be connected only after degradation trajectory
    # sanity checks pass on real HSI data.
    print(
        "\nDegradation/data pipeline setup complete. "
        "Run check_degradation_trajectory.py before adding the diffusion model."
    )


if __name__ == "__main__":
    main()
