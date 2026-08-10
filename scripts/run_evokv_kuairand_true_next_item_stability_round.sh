#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

config_lr100="configs/evokv_root_cause/kuairand_true_next_item_h512_l8_t512_stability_lr100_20260810_v2.json"
config_lr050="configs/evokv_root_cause/kuairand_true_next_item_h512_l8_t512_stability_lr050_20260810_v2.json"
output_lr100="results/root_cause_campaign/kuairand_true_next_item_h512_l8_t512_stability_lr100_20260810_v2"
output_lr050="results/root_cause_campaign/kuairand_true_next_item_h512_l8_t512_stability_lr050_20260810_v2"

test "$(sha256sum "$config_lr100" | awk '{print $1}')" = "32010f9ae81e099167bd69f20bdf46d7e434f1b95fcd55ee5816b190286af073"
test "$(sha256sum "$config_lr050" | awk '{print $1}')" = "32cfa8a6408ec5a0b1c975b430192f58343d3ed94cb241aceb617a1057e6611e"

python - "$config_lr100" "$config_lr050" <<'PY'
import sys
from hstu_kvcache.streaming.kuairand_query_transition import load_config

for path in sys.argv[1:]:
    document = load_config(path)
    if document["evaluation"]["candidate_count"] != 100:
        raise SystemExit("candidate_count must remain 100")
PY

for gpu in 0 1; do
  used="$(nvidia-smi --id="$gpu" --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '{total += $1} END {print total + 0}')"
  test "$used" -le 512
done

free_bytes="$(df --output=avail -B1 "$repo_root" | tail -n 1 | tr -d ' ')"
test "$free_bytes" -ge 85899345920

mkdir -p "$output_lr100" "$output_lr050"
exec 8>"$output_lr100/round.lock"
exec 9>"$output_lr050/round.lock"
flock -n 8
flock -n 9

python - "$output_lr100/orchestration.json" "$output_lr050/orchestration.json" "$config_lr100" "$config_lr050" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

for output, config, gpu in ((sys.argv[1], sys.argv[3], 0), (sys.argv[2], sys.argv[4], 1)):
    config_path = Path(config)
    payload = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "gpu": gpu,
        "config": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    }
    Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

pids=()
cleanup() {
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM EXIT

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python scripts/run_evokv_kuairand_prediction_query.py --config "$config_lr100" >>"$output_lr100/run.log" 2>&1 &
pids+=("$!")
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python scripts/run_evokv_kuairand_prediction_query.py --config "$config_lr050" >>"$output_lr050/run.log" 2>&1 &
pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
pids=()
test "$status" -eq 0

python scripts/validate_evokv_kuairand_prediction_query.py --config "$config_lr100" --result "$output_lr100/summary.json"
python scripts/validate_evokv_kuairand_prediction_query.py --config "$config_lr050" --result "$output_lr050/summary.json"

python - "$output_lr100/orchestration.json" "$output_lr050/orchestration.json" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

for name in sys.argv[1:]:
    path = Path(name)
    payload = json.loads(path.read_text())
    payload["status"] = "complete"
    payload["completed_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

trap - INT TERM EXIT
