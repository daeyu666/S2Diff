# prepare_srf_weights.py
import os
import json
import argparse

import numpy as np
import pandas as pd

from srf_utils import load_hsi_wavelengths, build_srf_weights


IKONOS_4_BANDS = [
    "IKONOS Blue",
    "IKONOS Green",
    "IKONOS Red",
    "IKONOS NIR",
]
WV2_VISIBLE_6_BANDS = [
    "WV2 Coastal Blue",
    "WV2 Blue",
    "WV2 Green",
    "WV2 Yellow",
    "WV2 Red",
    "WV2 RedEdge",
]
WV2_VISIBLE_5_BANDS = [
    "WV2 Coastal Blue",
    "WV2 Blue",
    "WV2 Green",
    "WV2 Yellow",
    "WV2 Red",
]
WV2_ALL_8_BANDS = [
    "WV2 Coastal Blue",
    "WV2 Blue",
    "WV2 Green",
    "WV2 Yellow",
    "WV2 Red",
    "WV2 RedEdge",
    "WV2 NIR1",
    "WV2 NIR2",
]

WV2_SRF_PATH = "./data/srf/wv2_relative_spectral_response_data_for_i.atcorr.csv"
IKONOS_SRF_PATH = "./data/srf/ikonos_relative_spectral_response.csv"
PAVIA_NOMINAL_WAVELENGTH_PATH = "./data/wavelengths/PaviaU_nominal_430_860.txt"


def summarize_srf_weights(weights, band_names, wavelengths):
    rows = []
    for i, band in enumerate(band_names):
        weight = weights[i]
        peak_idx = int(np.argmax(weight))
        peak_wavelength = float(wavelengths[peak_idx])
        max_weight = float(weight.max())
        weight_sum = float(weight.sum())
        active_mask = weight > max_weight * 0.01

        if np.any(active_mask):
            active_min = float(wavelengths[active_mask].min())
            active_max = float(wavelengths[active_mask].max())
            active_count = int(active_mask.sum())
        else:
            active_min = peak_wavelength
            active_max = peak_wavelength
            active_count = 1

        rows.append({
            "band": band,
            "peak_wavelength_nm": peak_wavelength,
            "active_min_nm": active_min,
            "active_max_nm": active_max,
            "active_band_count": active_count,
            "max_weight": max_weight,
            "weight_sum": weight_sum,
        })
    return rows


def save_weight_table(save_path, weights, band_names, wavelengths):
    table = {"wavelength_nm": wavelengths.astype(np.float32)}
    for i, band in enumerate(band_names):
        safe_name = (
            band.replace("WV2 ", "")
            .replace("IKONOS ", "")
            .replace(" ", "_")
            .replace("-", "_")
        )
        table[safe_name] = weights[i].astype(np.float32)
    pd.DataFrame(table).to_csv(save_path, index=False)


def resolve_sensor(dataset_name, band_set, explicit_srf_path, wavelength_root):
    resolved = band_set
    if resolved == "auto":
        resolved = "ikonos4" if dataset_name == "PaviaU" else "wv2_all8"

    if resolved == "ikonos4":
        selected_bands = IKONOS_4_BANDS
        default_srf_path = IKONOS_SRF_PATH
    elif resolved == "wv2_visible5":
        selected_bands = WV2_VISIBLE_5_BANDS
        default_srf_path = WV2_SRF_PATH
    elif resolved == "wv2_visible6":
        selected_bands = WV2_VISIBLE_6_BANDS
        default_srf_path = WV2_SRF_PATH
    elif resolved == "wv2_all8":
        selected_bands = WV2_ALL_8_BANDS
        default_srf_path = WV2_SRF_PATH
    else:
        raise ValueError(f"Unsupported band_set: {resolved}")

    srf_path = explicit_srf_path or default_srf_path
    if dataset_name == "PaviaU" and resolved == "ikonos4":
        wavelength_path = PAVIA_NOMINAL_WAVELENGTH_PATH
    else:
        wavelength_path = os.path.join(wavelength_root, f"{dataset_name}.txt")

    return resolved, selected_bands, srf_path, wavelength_path


def prepare_one_dataset(args, dataset_name):
    resolved_band_set, selected_bands, srf_path, wavelength_path = resolve_sensor(
        dataset_name=dataset_name,
        band_set=args.band_set,
        explicit_srf_path=args.srf_path,
        wavelength_root=args.wavelength_root,
    )

    if not os.path.exists(wavelength_path):
        raise FileNotFoundError(
            f"Cannot find wavelength file for {dataset_name}: {wavelength_path}"
        )

    wavelengths = load_hsi_wavelengths(
        wavelength_path=wavelength_path,
        n_bands=len(np.loadtxt(wavelength_path).reshape(-1)),
    )
    n_bands = len(wavelengths)

    weights, band_names = build_srf_weights(
        srf_path=srf_path,
        hsi_wavelengths=wavelengths,
        selected_bands=selected_bands,
        interp_kind=args.interp,
        normalize=True,
    )

    os.makedirs(args.output_root, exist_ok=True)
    prefix = f"{dataset_name}_{resolved_band_set}"

    npy_path = os.path.join(args.output_root, f"{prefix}_weights.npy")
    csv_path = os.path.join(args.output_root, f"{prefix}_weights.csv")
    summary_csv_path = os.path.join(args.output_root, f"{prefix}_summary.csv")
    meta_path = os.path.join(args.output_root, f"{prefix}_meta.json")

    np.save(npy_path, weights.astype(np.float32))
    save_weight_table(csv_path, weights, band_names, wavelengths)

    summary_rows = summarize_srf_weights(weights, band_names, wavelengths)
    pd.DataFrame(summary_rows).to_csv(summary_csv_path, index=False)

    meta = {
        "dataset": dataset_name,
        "n_bands": int(n_bands),
        "wavelength_min_nm": float(wavelengths.min()),
        "wavelength_max_nm": float(wavelengths.max()),
        "band_set": resolved_band_set,
        "srf_path": srf_path,
        "wavelength_path": wavelength_path,
        "selected_bands": band_names,
        "weights_shape": list(weights.shape),
        "weights_npy": npy_path,
        "weights_csv": csv_path,
        "summary_csv": summary_csv_path,
        "normalize": True,
        "interp": args.interp,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print(f"Dataset: {dataset_name}")
    print(f"Sensor profile: {resolved_band_set}")
    print(f"SRF: {srf_path}")
    print(f"Wavelength grid: {wavelength_path}")
    print(f"Wavelength range: {wavelengths.min():.2f} - {wavelengths.max():.2f} nm")
    print(f"HSI bands: {n_bands}")
    print(f"SRF weights shape: {weights.shape}")
    print(f"Saved npy: {npy_path}")
    print(f"Saved csv: {csv_path}")
    print(f"Saved summary: {summary_csv_path}")
    print(f"Saved meta: {meta_path}")
    print("-" * 80)

    for row in summary_rows:
        print(
            f"{row['band']}: "
            f"peak={row['peak_wavelength_nm']:.2f} nm, "
            f"active={row['active_min_nm']:.2f}-{row['active_max_nm']:.2f} nm, "
            f"count={row['active_band_count']}, "
            f"sum={row['weight_sum']:.6f}, "
            f"max={row['max_weight']:.6f}"
        )

    print("=" * 80)
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["PaviaU", "Houston13", "Chikusei"],
    )
    parser.add_argument(
        "--srf_path",
        type=str,
        default="",
        help="Optional explicit SRF CSV; empty uses the fixed dataset protocol.",
    )
    parser.add_argument(
        "--wavelength_root",
        type=str,
        default="./data/wavelengths",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="./data/srf_weights",
    )
    parser.add_argument(
        "--band_set",
        type=str,
        default="auto",
        choices=["auto", "ikonos4", "wv2_visible5", "wv2_visible6", "wv2_all8"],
        help="auto: PaviaU -> IKONOS4; Houston13/Chikusei -> WV2 all8.",
    )
    parser.add_argument(
        "--interp",
        type=str,
        default="pchip",
        choices=["pchip", "linear"],
    )
    args = parser.parse_args()

    all_meta = []
    for dataset_name in args.datasets:
        all_meta.append(prepare_one_dataset(args, dataset_name))

    all_meta_path = os.path.join(args.output_root, "all_auto_sensor_meta.json")
    with open(all_meta_path, "w", encoding="utf-8") as f:
        json.dump(all_meta, f, indent=2, ensure_ascii=False)
    print(f"All metadata saved to: {all_meta_path}")


if __name__ == "__main__":
    main()
