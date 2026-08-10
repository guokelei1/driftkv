#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 6 ]]; then
  echo "usage: $0 SOURCE_CONFIG SOURCE_CHECKPOINTS SOURCE_RESULTS TARGET_CHECKPOINTS TARGET_RESULTS PREFIX_VERSIONS" >&2
  exit 2
fi

source_config="$1"
source_checkpoints="$2"
source_results="$3"
target_checkpoints="$4"
target_results="$5"
prefix_versions="$6"
source_hash="$(sha256sum "$source_config" | cut -d' ' -f1)"

test ! -e "$target_checkpoints"
test ! -e "$target_results"

for version in $(seq 1 "$prefix_versions"); do
  manifest="$source_checkpoints/theta_$version/manifest.json"
  accepted="$source_results/edges/theta_$version/accepted.json"
  manifest_hash="$(sha256sum "$manifest" | cut -d' ' -f1)"
  jq -e --argjson version "$version" --arg config_hash "$source_hash" '.version == $version and .config_sha256 == $config_hash' "$manifest" >/dev/null
  jq -e --argjson version "$version" --arg manifest_hash "$manifest_hash" '.version == $version and .checkpoint.sha256 == $manifest_hash' "$accepted" >/dev/null
done

mkdir -p "$target_checkpoints" "$target_results/edges"
for version in $(seq 1 "$prefix_versions"); do
  cp -al "$source_checkpoints/theta_$version" "$target_checkpoints/theta_$version"
  cp -a "$source_results/edges/theta_$version" "$target_results/edges/theta_$version"
done
