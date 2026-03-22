#!/bin/bash
set -euo pipefail

shopt -s nullglob
configs=(
  configs/dsec/validation/*.json
  configs/mvsec/validation/*.json
)

for cfg in "${configs[@]}"; do
  echo "Evaluating ${cfg}"
  python -m evaluate --config-path "${cfg}" --checkpoint-file epoch_005.pt
done
