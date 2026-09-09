#!/usr/bin/env bash
set -euo pipefail

# S2Diff Raw-Direct d=2 training under the shared radial translation protocol.
# Train patch remains 64x64; only HR-MSI is warped. GT-HSI and the progressive
# HSI trajectory remain fixed. d is the Euclidean displacement-radius bound:
#   r~U(0,2), theta~U(0,2pi), dx=r*cos(theta), dy=r*sin(theta), |shift|<=2 px.
# Start from random initialization.

python main.py \
  --stage train \
  --dataset PaviaU \
  --degradation_mode physical \
  --predictor_version v3 \
  --msi_ablation raw_direct \
  --train_msi_translation_max_px 2.0 \
  --train_msi_translation_probability 1.0 \
  --epochs 300 \
  --batch_size 4 \
  --eval_interval 20 \
  --save_interval 20 \
  --seed 10 \
  "$@"
