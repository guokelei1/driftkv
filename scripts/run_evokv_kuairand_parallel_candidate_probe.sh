#!/usr/bin/env bash
set -euo pipefail
if [[ "$#" -ne 4 ]]; then
  echo "usage: $0 VERSION ROUND CANDIDATE_A CANDIDATE_B" >&2
  exit 2
fi
version=$1
round=$2
candidate_a=$3
candidate_b=$4
config=configs/evokv_root_cause/kuairand_projected_theta1_theta8_seed53117_20260808_v0.json
root=results/opportunity_discovery/evokv_kuairand_projected_theta1_theta8_seed53117_v0/parallel_probes/theta_${version}_${round}
mkdir -p "$root"
pid_a=
pid_b=
cleanup() {
  if [[ -n "$pid_a" ]]; then kill "$pid_a" 2>/dev/null || true; fi
  if [[ -n "$pid_b" ]]; then kill "$pid_b" 2>/dev/null || true; fi
}
trap cleanup INT TERM
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 scripts/probe_evokv_kuairand_candidate.py --config "$config" --version "$version" --candidate "$candidate_a" --output "$root/${candidate_a}.json" >"$root/${candidate_a}.log" 2>&1 &
pid_a=$!
CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node=2 scripts/probe_evokv_kuairand_candidate.py --config "$config" --version "$version" --candidate "$candidate_b" --output "$root/${candidate_b}.json" >"$root/${candidate_b}.log" 2>&1 &
pid_b=$!
status_a=0
status_b=0
wait "$pid_a" || status_a=$?
pid_a=
wait "$pid_b" || status_b=$?
pid_b=
if [[ "$status_a" -ne 0 || "$status_b" -ne 0 ]]; then
  echo "candidate probe failed: A=$status_a B=$status_b" >&2
  exit 1
fi
