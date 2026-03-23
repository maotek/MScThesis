#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

shopt -s nullglob
configs=(
  configs/dsec/validation/*.json
  configs/mvsec/validation/*.json
)

for cfg in "${configs[@]}"; do
  echo "Evaluating ${cfg}"
  csv_path="$(python - <<'PY' "$cfg" "$script_dir"
import json
import os
import sys

cfg_path = sys.argv[1]
base_dir = sys.argv[2]
with open(cfg_path, "r") as f:
    cfg = json.load(f)
csv_path = cfg.get("csv_path")
if not csv_path:
    print("")
    sys.exit(0)
if not os.path.isabs(csv_path):
    csv_path = os.path.normpath(os.path.join(base_dir, csv_path))
print(csv_path)
PY
)"

  if [[ -n "$csv_path" && -f "$csv_path" ]]; then
    echo "SKIP ${cfg} (csv exists: ${csv_path})"
    continue
  fi

  if ! python -m evaluate --config-path "${cfg}"; then
    echo "FAILED ${cfg} (continuing)"
  fi
done
