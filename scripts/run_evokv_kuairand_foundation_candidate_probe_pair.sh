#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 4 ]]; then
  echo "usage: $0 VERSION ROUND CANDIDATE_GPU0 CANDIDATE_GPU1" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

version="$1"
round="$2"
candidate_gpu0="$3"
candidate_gpu1="$4"
config="${EVOKV_PROBE_CONFIG:-configs/evokv_root_cause/kuairand_foundation_rebuild_medium_theta11_probes_20260809_v5.json}"
probe_root="${EVOKV_PROBE_OUTPUT_ROOT:-results/root_cause_campaign/kuairand_foundation_rebuild_medium_theta11_probes_20260809_v5}"
output_root="$probe_root/theta_${version}/${round}"
mkdir -p "$output_root"

gpu0_log="$output_root/${candidate_gpu0}.log"
gpu1_log="$output_root/${candidate_gpu1}.log"
gpu0_result="$output_root/${candidate_gpu0}.json"
gpu1_result="$output_root/${candidate_gpu1}.json"
gpu0_pid=
gpu1_pid=

cleanup() {
  if [[ -n "$gpu0_pid" ]]; then kill "$gpu0_pid" 2>/dev/null || true; fi
  if [[ -n "$gpu1_pid" ]]; then kill "$gpu1_pid" 2>/dev/null || true; fi
}
trap cleanup INT TERM

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python scripts/probe_evokv_kuairand_candidate.py \
  --config "$config" --version "$version" --candidate "$candidate_gpu0" \
  --output "$gpu0_result" >"$gpu0_log" 2>&1 &
gpu0_pid=$!

CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python scripts/probe_evokv_kuairand_candidate.py \
  --config "$config" --version "$version" --candidate "$candidate_gpu1" \
  --output "$gpu1_result" >"$gpu1_log" 2>&1 &
gpu1_pid=$!

gpu0_status=0
gpu1_status=0
wait "$gpu0_pid" || gpu0_status=$?
gpu0_pid=
wait "$gpu1_pid" || gpu1_status=$?
gpu1_pid=

if [[ "$gpu0_status" -ne 0 || "$gpu1_status" -ne 0 ]]; then
  echo "candidate probe failed: GPU0=$gpu0_status GPU1=$gpu1_status" >&2
  exit 1
fi

jq -e '.status == "complete"' "$gpu0_result" >/dev/null
jq -e '.status == "complete"' "$gpu1_result" >/dev/null
jq '{candidate:.candidate.name,tuning_ndcg:.partition_summaries.tuning.comparisons.recompute_over_reuse.ndcg_at_5.relative_percent,holdout_ndcg:.partition_summaries.holdout.comparisons.recompute_over_reuse.ndcg_at_5.relative_percent,tuning_fresh:.partition_summaries.tuning.endpoints.recompute.ndcg_at_5,holdout_fresh:.partition_summaries.holdout.endpoints.recompute.ndcg_at_5}' "$gpu0_result" "$gpu1_result"
