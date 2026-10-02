#!/usr/bin/env bash
set -euo pipefail
cd /home/gkl/work/evokv
task_root=results/read_correction_2026_09/motivation_final
task_session=evokv_motivation_curves
mkdir -p "$task_root"
if [[ "${1:-}" == "--worker" ]]; then
    set +e
    /home/gkl/miniconda3/bin/python scripts/read_correction_motivation/run_population.py \
        --output-root "$task_root" > "$task_root/driver.log" 2>&1
    task_code=$?
    set -e
    printf '%s\n' "$task_code" > "$task_root/driver.exit"
    exit "$task_code"
fi
if tmux has-session -t "$task_session" 2>/dev/null; then
    printf 'Session already running: %s\n' "$task_session"
    exit 0
fi
tmux new-session -d -s "$task_session" \
    'bash /home/gkl/work/evokv/scripts/read_correction_motivation/launch_tmux.sh --worker'
printf 'Started %s\n' "$task_session"
