#!/usr/bin/env bash
set -euo pipefail

host="maoshengjiang@login.daic.tudelft.nl"
epoch_raw="epoch001"
remote_base="/tudelft.net/staff-umbrella/ThesisMaosheng/MScThesis/code"
local_base=""
password=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --password)
      password="${2:-}"
      shift 2
      ;;
    *)
      echo "Unexpected argument: $1" >&2
      echo "Usage: $0 [--password <password>]" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$password" ]]; then
  password="${HPC_PASS:-}"
fi

if [[ -z "$local_base" ]]; then
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  local_base="$(cd "$script_dir/.." && pwd)"
fi

epoch_num=""
if [[ "$epoch_raw" =~ ^epoch[_-]?([0-9]{1,4})$ ]]; then
  epoch_num="${BASH_REMATCH[1]}"
elif [[ "$epoch_raw" =~ ^[0-9]{1,4}$ ]]; then
  epoch_num="$epoch_raw"
else
  echo "Invalid epoch format in script: $epoch_raw (use e.g. epoch050 or 50)" >&2
  exit 1
fi

epoch_file="epoch_$(printf "%03d" "$epoch_num").pt"

local_train_output="$local_base/train_output"
remote_train_output="$remote_base/train_output"

if [[ ! -d "$local_train_output" ]]; then
  echo "Local train_output not found: $local_train_output" >&2
  exit 1
fi

ssh_cmd=(ssh)
rsync_cmd=(rsync -av --progress)
if [[ -n "$password" ]]; then
  if ! command -v sshpass >/dev/null 2>&1; then
    echo "sshpass is required for --password. Install it or set up SSH keys." >&2
    exit 1
  fi
  ssh_cmd=(sshpass -p "$password" ssh)
  rsync_cmd=(sshpass -p "$password" rsync -av --progress)
fi

remote_dirs=()
while IFS= read -r line; do
  [[ -n "$line" ]] && remote_dirs+=("$line")
done < <("${ssh_cmd[@]}" "$host" "cd \"$remote_base\" && ls -1 train_output" 2>/dev/null || true)

if [[ ${#remote_dirs[@]} -eq 0 ]]; then
  echo "No subdirectories in remote train_output: $host:$remote_train_output" >&2
  exit 1
fi

echo "Epoch file: $epoch_file"
echo "Remote: $host:$remote_train_output"
echo "Local:  $local_train_output"
echo ""

for run_name in "${remote_dirs[@]}"; do
  [[ -n "$run_name" ]] || continue
  local_dir="$local_train_output/$run_name"
  local_file="$local_dir/$epoch_file"
  remote_file="$remote_train_output/$run_name/$epoch_file"

  if [[ -f "$local_file" ]]; then
    echo "SKIP  $run_name (exists)"
    continue
  fi

  echo "FETCH $run_name/$epoch_file"
  mkdir -p "$local_dir"
  if "${rsync_cmd[@]}" "${host}:${remote_file}" "$local_dir/" >/dev/null 2>&1; then
    echo "OK    $run_name"
  else
    echo "MISS  $run_name (not found on remote)" >&2
  fi
done
