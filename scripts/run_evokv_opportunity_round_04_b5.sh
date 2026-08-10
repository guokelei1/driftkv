#!/usr/bin/env bash
set -euo pipefail

cd /data/gkl/findnew/o1
config=configs/evokv_root_cause/ml1m_positive_opportunity_synthesis_20260807_v0.json
output=results/opportunity_discovery/evokv_opportunity_discovery_20260807_v0/round_04_b5_ml1m_positive_opportunity_synthesis
test ! -f "$output/result.json"
mkdir -p "$output"
python scripts/summarize_evokv_ml1m_positive_opportunity.py --config "$config" 2>&1 | tee "$output/run.log"
python scripts/validate_evokv_ml1m_positive_opportunity.py --config "$config" --result "$output/result.json"
