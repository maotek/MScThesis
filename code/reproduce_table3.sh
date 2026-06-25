#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

dry_run=0
if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=1
elif [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: bash reproduce_table3.sh [--dry-run]"
  echo
  echo "Runs the validation configs used for Table 3."
  echo "Outputs are written to the csv_path configured in each JSON file."
  exit 0
fi

run_eval() {
  local label="$1"
  local config_path="$2"

  echo
  echo "=== ${label} ==="
  echo "python -m evaluate --config-path ${config_path}"

  if [[ "$dry_run" -eq 0 ]]; then
    python -m evaluate --config-path "$config_path"
  fi
}

run_eval "Table 3 / DSEC / U-Net DAv2 (baseline)" \
  "configs/dsec/validation/unet_dav2_batch10.json"
run_eval "Table 3 / DSEC / U-Net DAv2 (ReLU activation)" \
  "configs/dsec/validation/unet_dav2_batch10_relu.json"
run_eval "Table 3 / DSEC / U-Net DAv2 (grayscale)" \
  "configs/dsec/validation/unet_1c_dav2_batch10.json"
run_eval "Table 3 / DSEC / U-Net DAv2 (16 channels)" \
  "configs/dsec/validation/unet_dav2_batch10_ch16.json"
run_eval "Table 3 / DSEC / U-Net DAv2 (1 encoder, 1 decoder)" \
  "configs/dsec/validation/unet_small3_dav2_batch10.json"
run_eval "Table 3 / DSEC / U-Net DAv2 (learnable constant)" \
  "configs/dsec/validation/newunet_dav2_batch10.json"

run_eval "Table 3 / DSEC / FullyConv DAv2 (baseline)" \
  "configs/dsec/validation/fully_conv_dav2_batch10_RC.json"
run_eval "Table 3 / DSEC / FullyConv DAv2 (grayscale)" \
  "configs/dsec/validation/fully_conv_1c_dav2_batch10.json"
run_eval "Table 3 / DSEC / FullyConv DAv2 (learnable constant)" \
  "configs/dsec/validation/new_fully_conv_dav2_batch10_seed3.json"

run_eval "Table 3 / MVSEC / U-Net DAv2 (baseline)" \
  "configs/mvsec/validation/train_mvsec_unet_dav2_batch10.json"
run_eval "Table 3 / MVSEC / U-Net DAv2 (ReLU activation)" \
  "configs/mvsec/validation/train_mvsec_unet_dav2_batch10_relu.json"
run_eval "Table 3 / MVSEC / U-Net DAv2 (grayscale)" \
  "configs/mvsec/validation/train_mvsec_unet_1c_dav2_batch10.json"
run_eval "Table 3 / MVSEC / U-Net DAv2 (16 channels)" \
  "configs/mvsec/validation/train_mvsec_unet_dav2_batch10_ch16.json"
run_eval "Table 3 / MVSEC / U-Net DAv2 (1 encoder, 1 decoder)" \
  "configs/mvsec/validation/train_mvsec_unet_small3_dav2_batch10.json"
run_eval "Table 3 / MVSEC / U-Net DAv2 (learnable constant)" \
  "configs/mvsec/validation/train_mvsec_newunet_dav2_batch10.json"

run_eval "Table 3 / MVSEC / FullyConv DAv2 (baseline)" \
  "configs/mvsec/validation/train_mvsec_fully_conv_dav2_batch10.json"
run_eval "Table 3 / MVSEC / FullyConv DAv2 (grayscale)" \
  "configs/mvsec/validation/train_mvsec_fully_conv_1c_dav2_batch10.json"
run_eval "Table 3 / MVSEC / FullyConv DAv2 (learnable constant)" \
  "configs/mvsec/validation/train_mvsec_new_fully_conv_dav2_batch10.json"

echo
echo "Table 3 reproduction complete. CSVs are in ${script_dir}/evaluate_results/."
