#!/usr/bin/env bash
# Start only after the user explicitly approves the formal evaluation.
set -uo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
output_dir="$repo_root/results/selective_recompute_2026_09"
mkdir -p "$output_dir"
if [[ "${1:-}" != "--user-approved" || -z "${TMUX:-}" ]]; then
  printf '%s\n' 'Use launch.sh --start after the user says OK; this script requires tmux.' >&2
  exit 2
fi
python scripts/selective_recompute_2026_09/run.py --run --user-approved 2>&1 | tee -a "$output_dir/runtime.log"
run_exit_code="${PIPESTATUS[0]}"
printf '%s\n' "$run_exit_code" > "$output_dir/exit_status.txt"
exit "$run_exit_code"
