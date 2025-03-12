#!/usr/bin/env bash

set -e

DATASETS=("mnist")
LABEL_TAMPERING=("none" "random" "reverse" "zero")
WEIGHT_TAMPERING=("none" "large_neg" "reverse" "random")

for dataset in "${DATASETS[@]}"; do
  for label_tamp in "${LABEL_TAMPERING[@]}"; do
    for weight_tamp in "${WEIGHT_TAMPERING[@]}"; do
      echo "Running fedavg.py with --dataset=$dataset --label_tampering=$label_tamp --weight_tampering=$weight_tamp"
      python3 src/fedavg.py \
        --dataset "$dataset" \
        --label_tampering "$label_tamp" \
        --weight_tampering "$weight_tamp" \
        --iid 0
    done
  done
done
