#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 || ( $1 != 201 && $1 != 202 ) ]]; then
  printf 'Usage: %s {201|202} DESTINATION_ROOT\n' "$0" >&2
  exit 2
fi

machine=$1
destination=$2
repo_root=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
inventory="$repo_root/sync/$machine"

if [[ ! -f $inventory ]]; then
  printf 'Inventory not found: %s (run sync/inventory.sh %s first)\n' "$inventory" "$machine" >&2
  exit 1
fi
mkdir -p "$destination"

cd "$repo_root"
tail -n +6 "$inventory" | while IFS=$'\t' read -r _size path; do
  [[ -n ${path:-} ]] || continue
  if [[ ! -f $path ]]; then
    printf 'Missing source file: %s\n' "$path" >&2
    exit 1
  fi
  rsync -a --relative "./$path" "$destination/"
done

printf 'Copied inventory %s to %s\n' "$machine" "$destination"
