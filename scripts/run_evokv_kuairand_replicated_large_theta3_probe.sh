#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  printf 'usage: %s CANDIDATE\n' "$0" >&2
  exit 2
fi

candidate="$1"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

config="configs/evokv_root_cause/kuairand_latest_query_large_e512_r8_theta1_theta8_20260811_v34.json"
probe_config="configs/evokv_root_cause/kuairand_replicated_large_theta3_directional_probes_20260811_v35.json"
config_sha256="907034295b508f64e259a92c22832f27188e66902133c01aa79adab687773282"
probe_sha256="02e6d55f928fa0e5388ad07248e40b94602428dd81714c785cd2abae15ff6d86"
root="results/root_cause_campaign/kuairand_replicated_large_theta3_directional_probes_20260811_v35"
output="$root/${candidate}.json"

test "$(sha256sum "$config" | awk '{print $1}')" = "$config_sha256"
test "$(sha256sum "$probe_config" | awk '{print $1}')" = "$probe_sha256"
jq -e --arg candidate "$candidate" '.candidates | any(.name == $candidate)' "$probe_config" >/dev/null
for version in 1 2; do
  test -f "checkpoints/evokv_kuairand_latest_query_large_e512_r8_theta1_theta8_v30/theta_${version}/manifest.json"
done
mkdir -p "$root"
exec 8>"$root/${candidate}.lock"
flock -n 8
if [[ -f "$output" ]]; then
  jq -e --arg candidate "$candidate" '.status == "complete" and .candidate.name == $candidate' "$output" >/dev/null
  exit 0
fi
for gpu in 0 1; do
  used="$(nvidia-smi --id="$gpu" --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '{total += $1} END {print total + 0}')"
  test "$used" -le 512
done
nvidia-smi --id=0,1 --query-gpu=index,uuid,memory.total,memory.free --format=csv,noheader > "$root/${candidate}_gpu_preflight.csv"
CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 torchrun --standalone --nproc-per-node=2 \
  scripts/probe_evokv_kuairand_candidate_lineage.py \
  --config "$config" \
  --version 3 \
  --candidate "$candidate" \
  --candidate-config "$probe_config" \
  --minimum-source-version 1 \
  --output "$output" \
  2>&1 | tee "$root/${candidate}.log"
jq -e --arg candidate "$candidate" '.status == "complete" and .candidate.name == $candidate' "$output" >/dev/null
