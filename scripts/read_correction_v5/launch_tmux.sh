#!/usr/bin/env bash
set -u
cd /home/gkl/work/evokv
run_root=/home/gkl/work/evokv/results/read_correction_2026_09/v5/population_run
session=evokv_read_correction_v5

if [[ "${1:-}" == --worker ]]; then
    mkdir -p "$run_root/runtime"
    run_stamp=$(date -u +%Y%m%dT%H%M%S)
    launch_log="$run_root/runtime/launcher_${run_stamp}.log"
    /home/gkl/miniconda3/bin/python scripts/read_correction_v5/run_population.py \
        --output-root "$run_root" > "$launch_log" 2>&1
    launch_status=$?
    /home/gkl/miniconda3/bin/python - "$run_root" "$launch_status" "$launch_log" <<'PY'
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
root, status, logfile = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
record = dict(exit_code=status, finished_at=datetime.now(timezone.utc).isoformat(), log=str(logfile))
for path in [logfile.with_suffix('.exit.json'), root/'last_exit.json']:
    path.write_text(json.dumps(record, indent=2)+'\n')
PY
    exit "$launch_status"
fi

if [[ ! -f "$run_root/plan.json" ]]; then
    echo 'Missing fixed population plan' >&2
    exit 1
fi
if tmux has-session -t "$session" 2>/dev/null; then
    echo "Session $session already exists" >&2
    exit 1
fi
tmux new-session -d -s "$session" \
    'bash /home/gkl/work/evokv/scripts/read_correction_v5/launch_tmux.sh --worker'
echo "Started tmux session: $session"
