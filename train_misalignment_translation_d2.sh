#!/usr/bin/env bash
set -euo pipefail

# S2Diff Raw-Direct training with HR-MSI-only translation augmentation.
# IMPORTANT: d=2 is the Euclidean displacement-radius upper bound:
#   r~U(0,2), theta~U(0,2pi), dx=r*cos(theta), dy=r*sin(theta), |shift|<=2 px.
# GT-HSI and LR-HSI remain fixed/registered. Start from random initialization.

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
