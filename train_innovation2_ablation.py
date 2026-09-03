"""Dedicated launcher for Innovation 2 MSI experiments.

This wrapper forces predictor_version=v3 so an MSI experiment cannot silently
fall back to the HSI-only V1 predictor. All other CLI arguments are forwarded
to main.py unchanged.

Examples:
    python train_innovation2_ablation.py --msi_ablation raw_direct --epochs 200
    python train_innovation2_ablation.py --msi_ablation raw_translate --epochs 200
"""

from __future__ import annotations

import sys

from main import main


TIME_FREE_MODES = {
    "raw_direct",
    "raw_gate",
    "hf_direct",
    "hf_gate",
    "raw_translate",
}


def _value_after(args, flag):
    if flag not in args:
        return None
    index = args.index(flag)
    if index + 1 >= len(args):
        raise SystemExit(f"Missing value after {flag}")
    return args[index + 1]


def _prepare_argv():
    args = list(sys.argv[1:])

    requested = _value_after(args, "--msi_ablation")
    if requested is None:
        raise SystemExit(
            "Innovation 2 launcher requires --msi_ablation with one of: "
            + ", ".join(sorted(TIME_FREE_MODES))
        )
    if requested not in TIME_FREE_MODES:
        raise SystemExit(
            f"This launcher is for the time-free MSI experiment set; "
            f"got msi_ablation={requested!r}. Expected one of "
            f"{sorted(TIME_FREE_MODES)}"
        )

    if "--predictor_version" in args:
        index = args.index("--predictor_version")
        if index + 1 >= len(args):
            raise SystemExit("Missing value after --predictor_version")
        args[index + 1] = "v3"
    else:
        args = ["--predictor_version", "v3"] + args

    sys.argv = [sys.argv[0]] + args
    print(
        "[Innovation 2 launcher] forcing predictor_version=v3; "
        f"msi_ablation={requested}"
    )


if __name__ == "__main__":
    _prepare_argv()
    main()
