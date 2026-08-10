#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

config="configs/evokv_root_cause/kuairand_latest_query_large_e512_r8_theta1_theta8_20260810_v30.json"
probe_config="configs/evokv_root_cause/kuairand_replicated_large_theta2_directional_probes_20260810_v31.json"
config_sha256="554ddac865c5f299a97dc4343f6704cbd0c622fcd367930d07ed33a8b21c4f6b"
probe_sha256="5e1b2b8bd614cd1e9928ace7a40835aea4e66be8bba77c98b05373be1cb4160c"
root="results/root_cause_campaign/kuairand_replicated_large_theta2_directional_probes_20260810_v31"
output="$root/rowrep_theta2_kv4_e4.json"

test "$(sha256sum "$config" | awk '{print $1}')" = "$config_sha256"
test "$(sha256sum "$probe_config" | awk '{print $1}')" = "$probe_sha256"
test -f checkpoints/evokv_kuairand_latest_query_large_e512_r8_theta1_theta8_v30/theta_1/manifest.json
mkdir -p "$root"
exec 8>"$root/rowrep_theta2_kv4_e4.lock"
flock -n 8
if [[ -f "$output" ]]; then
  jq -e '.status == "complete" and .candidate.name == "rowrep_theta2_kv4_e4"' "$output" >/dev/null
  exit 0
fi
for gpu in 0 1; do
  used="$(nvidia-smi --id="$gpu" --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '{total += $1} END {print total + 0}')"
  test "$used" -le 512
done
nvidia-smi --id=0,1 --query-gpu=index,uuid,memory.total,memory.free --format=csv,noheader > "$root/rowrep_theta2_kv4_e4_gpu_preflight.csv"
CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 torchrun --standalone --nproc-per-node=2 \
  scripts/probe_evokv_kuairand_candidate.py \
  --config "$config" \
  --version 2 \
  --candidate rowrep_theta2_kv4_e4 \
  --candidate-config "$probe_config" \
  --output "$output" \
  2>&1 | tee "$root/rowrep_theta2_kv4_e4.log"
jq -e '.status == "complete" and .candidate.name == "rowrep_theta2_kv4_e4" and .summary.sanity.passed == true' "$output" >/dev/null
