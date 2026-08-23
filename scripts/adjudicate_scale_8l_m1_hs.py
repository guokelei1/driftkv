#!/usr/bin/env python3
"""Seal then adjudicate all M1 seed17 8L rolling H/S cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pyarrow.parquet as pq

import adjudicate_scale_8l_hs as metrics
import train_p7_theta0 as p7

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/"results/scale_8l_m1_seed17/hs_raw"
SEAL=ROOT/"results/scale_8l_m1_seed17/hs_raw_seal.json"
LINEAGE=ROOT/"results/scale_8l_m1_seed17/lineage_seal.json"
OUTPUT=ROOT/"results/scale_8l_m1_seed17"
RELEASES=("r1_edge1","r1_edge2","r2")
FORBIDDEN={"label","is_organic","prior_30m_same_item","latest_item","long_gap_at_least_3d","feedback_history_stratum_v2"}


def seal():
    if SEAL.exists(): raise FileExistsError(SEAL)
    artifacts=[]; runs=[]
    for release in RELEASES:
        path=RAW/release/"m1_seed17/raw_manifest.json"; value=json.loads(path.read_text())
        if value["status"]!="raw_scores_written_before_metrics" or value["scope"]!="full_development_edge" or value["metrics_computed"] is not False: raise RuntimeError("M1 raw cell not sealable")
        for row in value["artifacts"]:
            raw=ROOT/row["path"]
            if p7.sha256_file(raw)!=row["sha256"]: raise RuntimeError("M1 raw hash mismatch")
            if row["view"]=="fidelity" and set(pq.read_schema(raw).names)&FORBIDDEN: raise RuntimeError("fidelity leak")
            artifacts.append({"release":release,**row})
        runs.append({"release":release,"manifest":str(path.relative_to(ROOT)),"manifest_sha256":p7.sha256_file(path),
            "checkpoint_sha256":value["checkpoint_sha256"],"parent_checkpoint_sha256":value["parent_checkpoint_sha256"],
            "population":value["population"],"lineage":value["lineage"],"base_full_recent_max_abs_delta":value["base_full_recent_max_abs_delta"],
            "request_local_full_companion_max_abs_logit":value["exact_rolling_vs_request_local_full_max_abs_logit_companion"]})
    payload={"status":"sealed_all_scale_8l_M1_seed17_HS_raw_before_metrics","lineage_seal_sha256":p7.sha256_file(LINEAGE),
        "metrics_computed":False,"qualification_or_theta3_read":False,"runs":runs,"artifacts":artifacts}
    SEAL.write_text(json.dumps(payload,indent=2)+"\n"); print(json.dumps({"status":payload["status"],"artifacts":len(artifacts)},indent=2))


def adjudicate(release):
    output=OUTPUT/f"hs_{release}_adjudication.json"
    if output.exists(): raise FileExistsError(output)
    seal_value=json.loads(SEAL.read_text()); metrics.SEAL=SEAL
    comparisons={name:metrics.evaluate_comparison(seal_value,release,name) for name in metrics.COMPARISONS}
    h=comparisons["H_request_local"]; s=comparisons["S_rolling"]; floor=1e-8
    run=next(row for row in seal_value["runs"] if row["release"]==release)
    components={"model_admitted":True,"Base_Full_Recent_identical":run["base_full_recent_max_abs_delta"]==0,
        "current_H_JS_CI_above_floor":h["fidelity"]["output_js_divergence"]["ci95"][0]>floor,
        "current_H_probability_CI_above_floor":h["fidelity"]["absolute_probability_difference"]["ci95"][0]>1e-7,
        "rolling_S_JS_CI_above_floor":s["fidelity"]["output_js_divergence"]["ci95"][0]>floor,
        "rolling_S_minimum_panels":sum(x>floor for x in s["fidelity"]["panel_points_JS"])>=3}
    passed=all(components.values()); hp=h["fidelity"]["output_js_divergence"]["point"]; sp=s["fidelity"]["output_js_divergence"]["point"]
    result={"status":"scale_M1_HS_gate_passed" if passed else "scale_M1_HS_gate_failed_stop","release":release,"model":"m1","seed":17,
        "comparisons":comparisons,"S_over_H_companion":sp/hp if hp>0 else None,"gate_components":components,"HS_gate_passed":passed,
        "raw_seal_sha256":p7.sha256_file(SEAL),"qualification_or_theta3_read":False,
        "authorization":"M1_frozen_actions_may_be_replayed" if passed else "stop_before_actions"}
    output.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result,indent=2))
    if not passed: raise SystemExit(2)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--stage",choices=("seal","adjudicate"),required=True); parser.add_argument("--release",choices=RELEASES)
    args=parser.parse_args()
    if args.stage=="seal": seal()
    elif not args.release: raise SystemExit("--release required")
    else: adjudicate(args.release)

if __name__=="__main__": main()
