#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( $1 != 201 && $1 != 202 ) ]]; then
  printf 'Usage: %s {201|202}\n' "$0" >&2
  exit 2
fi

machine=$1
repo_root=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
python3 "$repo_root/sync/build_inventory.py" "$machine"
