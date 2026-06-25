#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

dry_run=0
if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=1
elif [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: bash reproduce_table2.sh [--dry-run]"
  echo
  echo "Runs the validation configs used for Table 2."
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

run_eval "Table 2 / DSEC -> MVSEC / Depth AnyEvent (DAE)" \
  "configs/mvsec/validation/dae_tencode_DSEC_checkpoint.json"
run_eval "Table 2 / DSEC -> MVSEC / U-Net DAv2" \
  "configs/mvsec/validation/unet_dav2_batch10.json"
run_eval "Table 2 / DSEC -> MVSEC / FullyConv DAv2" \
  "configs/mvsec/validation/fully_conv_dav2_batch10_RC.json"

run_eval "Table 2 / MVSEC -> DSEC / Depth AnyEvent (DAE)" \
  "configs/dsec/validation/dae_tencode_MVSEC_checkpoint.json"
run_eval "Table 2 / MVSEC -> DSEC / U-Net DAv2" \
  "configs/dsec/validation/train_mvsec_unet_dav2_batch10.json"
run_eval "Table 2 / MVSEC -> DSEC / FullyConv DAv2" \
  "configs/dsec/validation/train_mvsec_fully_conv_dav2_batch10.json"

echo
echo "Table 2 reproduction complete. CSVs are in ${script_dir}/evaluate_results/."
