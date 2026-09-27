#!/usr/bin/env bash
# Default prints the command; --start is used only after user approval.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
session="evokv_selective_recompute_20260926"
if [[ "${1:-}" != "--start" ]]; then
  printf '%s\n' 'Prepared only. Formal launch awaits user OK.'
  printf 'After approval: bash %s/scripts/selective_recompute_2026_09/launch.sh --start\n' "$repo_root"
  exit 0
fi
cd "$repo_root"
tmux new-session -d -s "$session" -c "$repo_root" \
  'bash scripts/selective_recompute_2026_09/run_in_tmux.sh --user-approved'
printf 'Started tmux session: %s\n' "$session"
