#!/usr/bin/env bash
set -euo pipefail

target_checkpoints=$1
target_results=$2
if test "$#" -eq 2; then
  source_checkpoints=checkpoints/evokv_kuairand_projected_theta1_theta8_seed53117_v0
  source_results=results/opportunity_discovery/evokv_kuairand_imported_anchor_kv2_seed53117_v2
  prefix_versions=6
elif test "$#" -eq 5; then
  source_checkpoints=$3
  source_results=$4
  prefix_versions=$5
else
  exit 2
fi

python - "$source_checkpoints" "$source_results" "$prefix_versions" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

checkpoint_root = Path(sys.argv[1])
result_root = Path(sys.argv[2])
prefix_versions = int(sys.argv[3])
expected = {
    1: "48c4e69177cfe17a9681da21e3b9491f49ebbb588f2559eb3d92e10ce718bf6f",
    **{
        version: "d6f1c5c3807c5e9c6e265d1bb8395ff9313522205606156654e2215a3d6d521f"
        for version in range(2, 7)
    },
    7: "4d287e5a796eb0c5a1b0c6bc01dd31c17a2139a794fd09f57dd94197dae7607c",
}
for version in range(1, prefix_versions + 1):
    config_hash = expected[version]
    manifest_path = checkpoint_root / f"theta_{version}" / "manifest.json"
    accepted_path = result_root / "edges" / f"theta_{version}" / "accepted.json"
    manifest = json.loads(manifest_path.read_text())
    accepted = json.loads(accepted_path.read_text())
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if (
        manifest["version"] != version
        or manifest["config_sha256"] != config_hash
        or accepted["version"] != version
        or accepted["checkpoint"]["sha256"] != digest
    ):
        raise RuntimeError(f"KuaiRand source theta{version} differs")
PY

test ! -e "$target_checkpoints"
test ! -e "$target_results"
mkdir -p "$target_checkpoints" "$target_results/edges"
for version in $(seq 1 "$prefix_versions"); do
  cp -al "$source_checkpoints/theta_$version" "$target_checkpoints/theta_$version"
  cp -a "$source_results/edges/theta_$version" "$target_results/edges/theta_$version"
done
