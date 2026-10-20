#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || ( $1 != 201 && $1 != 202 ) ]]; then
  printf 'Usage: %s {201|202} DESTINATION_ROOT [ITEM_ID ...]\n' "$0" >&2
  printf 'Select numbered inventory items, e.g. D007 Y002.\n' >&2
  exit 2
fi

machine=$1
destination=$2
shift 2
selections=("$@")
repo_root=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
inventory="$repo_root/sync/$machine"

if [[ ! -f $inventory ]]; then
  printf 'Inventory not found: %s (run sync/inventory.sh %s first)\n' "$inventory" "$machine" >&2
  exit 1
fi
cd "$repo_root"

files=()
while IFS=$'\t' read -r marker item_id size path; do
  [[ $marker == FILE && $size =~ ^[0-9]+$ && -n ${path:-} ]] || continue
  if (( ${#selections[@]} == 0 )) || [[ " ${selections[*]} " == *" $item_id "* ]]; then
    if [[ ! -f $path ]]; then
      printf 'Missing source file: %s\n' "$path" >&2
      exit 1
    fi
    files+=("$path")
  fi
done < "$inventory"

if (( ${#files[@]} == 0 )); then
  printf 'No inventory files match the selected IDs.\n' >&2
  exit 1
fi

mkdir -p "$destination"
for path in "${files[@]}"; do
  rsync -a --relative "./$path" "$destination/"
done

printf 'Copied %s files from inventory %s to %s\n' "${#files[@]}" "$machine" "$destination"
