#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( $1 != 201 && $1 != 202 ) ]]; then
  printf 'Usage: %s {201|202}\n' "$0" >&2
  exit 2
fi

machine=$1
repo_root=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
output="$repo_root/sync/$machine"
tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT

{
  printf 'Machine %s inventory\n' "$machine"
  printf 'Generated: %s\n' "$(date -Is)"
  printf 'Repository: %s\n' "$repo_root"
  printf 'Scope: Git-ignored data files and selected trained model payloads; paths are repository-relative.\n\n'
  printf 'SIZE_BYTES\tPATH\n'

  cd "$repo_root"
  {
    find data -type f -print 2>/dev/null || true
    find results/unified_training_2026_09 -type f -name 'checkpoint_*.pt' -print 2>/dev/null || true
    find results/recflow -type f -name 'checkpoint.pt' -print 2>/dev/null || true
    find results/query_relay_2026_10 -type f \( -path '*/source_v*/source.pt' -o -path '*/publication/v*.pt' -o -path '*/feature_banks/*.pt' \) -print 2>/dev/null || true
  } | sort -u | while IFS= read -r path; do
    if git check-ignore -q -- "$path"; then
      size=$(stat -c '%s' -- "$path")
      printf '%s\t%s\n' "$size" "$path"
    fi
  done
} > "$tmp"

mv "$tmp" "$output"
trap - EXIT
printf 'Wrote %s\n' "$output"
