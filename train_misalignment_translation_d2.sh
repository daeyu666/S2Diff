#!/usr/bin/env bash
set -euo pipefail

# S2Diff Raw-Direct d=2 training with warp-before-crop HR-MSI context.
# d=2 is the Euclidean displacement-radius upper bound:
#   r~U(0,2), theta~U(0,2pi), dx=r*cos(theta), dy=r*sin(theta), |shift|<=2 px.
# GT-HSI and the progressive HSI trajectory remain 64x64/fixed. HR-MSI is
# generated on a 72x72 parent (margin=ceil(d)+2=4), warped, then center-cropped.
# Start from random initialization and save under a new context-specific name.

python train_context_misalignment.py \
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
  --save_name PaviaU_innovation1_physical_v3_raw_direct_traug2_context.pth \
  "$@"
