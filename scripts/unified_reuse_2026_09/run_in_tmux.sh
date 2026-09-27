#!/usr/bin/env bash
# Run this script inside a detached tmux session only after the user launches it.
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
output_dir="$repo_root/results/unified_reuse_2026_09"
mkdir -p "$output_dir"

python scripts/unified_reuse_2026_09/run.py --run 2>&1 | tee -a "$output_dir/runtime.log"
run_exit_code="${PIPESTATUS[0]}"
printf '%s\n' "$run_exit_code" > "$output_dir/exit_status.txt"
exit "$run_exit_code"
