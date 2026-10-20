#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || ( $1 != 201 && $1 != 202 ) ]]; then
  printf 'Usage: %s {201|202} DESTINATION_ROOT MODEL_PATH [MODEL_PATH ...]\n' "$0" >&2
  printf 'Select repository-relative files or directories from the inventory.\n' >&2
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
for selection in "${selections[@]}"; do
  selection=${selection#./}
  selection=${selection%/}
  if [[ -z $selection || $selection == /* || /$selection/ == */../* || $selection == . ]]; then
    printf 'Use a repository-relative file or directory: %s\n' "$selection" >&2
    exit 1
  fi
  matched=0
  while IFS=$'\t' read -r size path; do
    [[ $size =~ ^[0-9]+$ && -n ${path:-} ]] || continue
    if [[ $path == "$selection" || $path == "$selection/"* ]]; then
      if [[ ! -f $path ]]; then
        printf 'Missing source file: %s\n' "$path" >&2
        exit 1
      fi
      files+=("$path")
      matched=1
    fi
  done < "$inventory"
  if [[ $matched -eq 0 ]]; then
    printf 'No inventory files match: %s\n' "$selection" >&2
    exit 1
  fi
done

mkdir -p "$destination"
for path in "${files[@]}"; do
  rsync -a --relative "./$path" "$destination/"
done

printf 'Copied selected files from inventory %s to %s\n' "$machine" "$destination"
