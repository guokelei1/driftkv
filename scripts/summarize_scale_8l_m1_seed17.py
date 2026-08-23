#!/usr/bin/env python3
"""Compact M1 seed17 H/S result after all release cells are adjudicated."""

import json
from pathlib import Path
import train_p7_theta0 as p7

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"results/scale_8l_m1_seed17"
OUTPUT=BASE/"summary.json"

def main():
    if OUTPUT.exists(): raise FileExistsError(OUTPUT)
    h=json.loads((BASE/"h_theta0_adjudication.json").read_text()); releases=[]
    for release in ("r1_edge1","r1_edge2","r2"):
        value=json.loads((BASE/f"hs_{release}_adjudication.json").read_text())
        H=value["comparisons"]["H_request_local"]; S=value["comparisons"]["S_rolling"]
        releases.append({"release":release,"H_JS":H["fidelity"]["output_js_divergence"],"S_JS":S["fidelity"]["output_js_divergence"],
            "S_over_H":value["S_over_H_companion"],"reuse_harm_logloss":S["quality"]["log_loss_gain"],
            "reuse_harm_ROC_AUC":S["quality"]["ROC_AUC_gain"],"reuse_harm_dislike_PR_AUC":S["quality"]["dislike_PR_AUC_gain"],
            "reuse_harm_dislike_only_logloss":S["quality"]["dislike_only_log_loss_gain"],"passed":value["HS_gate_passed"]})
    payload={"status":"scale_8l_M1_seed17_HS_chain_complete","evidence_level":"development_scale_pilot_single_seed",
        "theta0_H":h,"releases":releases,"all_HS_edges_passed":all(x["passed"] for x in releases),
        "next":"freeze_and_replay_existing_actions_only_if_scientifically_warranted","paper_qualification":False,"theta3_read":False}
    OUTPUT.write_text(json.dumps(payload,indent=2)+"\n"); print(json.dumps(payload,indent=2))

if __name__=="__main__": main()
