#!/usr/bin/env python3
"""Create the compact final report for the one-command 8L method replay."""

from __future__ import annotations

import json
from pathlib import Path

import train_p7_theta0 as p7

ROOT=Path(__file__).resolve().parents[1]
ACTIONS=ROOT/"results/scale_8l_v1/actions_adjudication_v1.json"
SCHEDULER=ROOT/"results/scale_8l_v1/scheduler/scheduler_result.json"
QUALITY=ROOT/"results/scale_8l_v1/policy_quality_adjudication_v1.json"
OUTPUT=ROOT/"results/scale_8l_v1/method_summary_v1.json"


def main():
    if OUTPUT.exists(): raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    actions=json.loads(ACTIONS.read_text()); scheduler=json.loads(SCHEDULER.read_text()); quality=json.loads(QUALITY.read_text())
    action_rows=[]
    for cell in actions["cells"]:
        action_rows.append({"release":cell["release"],"recovery":{row["action"]:row["recovery_from_population_mean_MSE"] for row in cell["summaries"]},
            "measured_grouped_transition_runtime_seconds":cell["measured_grouped_transition_runtime_seconds"]})
    scheduler_rows=[]
    for cell in scheduler["cells"]:
        if cell["release"]=="r0": continue
        scheduler_rows.append({"release":cell["release"],"primary_1pct":[{key:row[key] for key in
            ("budget_fraction","charged_cost_fraction","risk_recovery_fraction","ridge_minus_strongest_recovery","offline_oracle_recovery_fraction")}
            for row in cell["policies"] if row["probe_mode"]=="rate_1pct"]})
    runtime=[]
    for release in ("r1_edge1","r1_edge2","r2"):
        path=ROOT/"results/scale_8l_v1/policy_runtime"/release/"m0_f_seed17/result.json"; value=json.loads(path.read_text())
        runtime.append({"release":release,"budgets":[{"budget_fraction":row["budget_fraction"],
            "charged_grouped_transition_runtime_seconds":row["charged_grouped_transition_runtime_seconds"]} for row in value["results"]]})
    payload={"status":"scale_8l_frozen_method_full_replay_complete","evidence_level":"development_scale_pilot_single_seed",
        "actions":action_rows,"scheduler":scheduler_rows,"mixed_policy_runtime":runtime,"quality":quality["cells"],
        "inputs":{"actions_sha256":p7.sha256_file(ACTIONS),"scheduler_sha256":p7.sha256_file(SCHEDULER),"quality_sha256":p7.sha256_file(QUALITY)},
        "cross_seed_or_M1_claim":False,"paper_qualification":False,"theta3_read":False}
    OUTPUT.write_text(json.dumps(payload,indent=2)+"\n"); print(json.dumps(payload,indent=2))


if __name__=="__main__": main()
