#!/usr/bin/env python3
"""Run the sealed development-only H pipeline for 8L M1 seed17."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

import adjudicate_scale_8l_h_pilot as adjud
import eval_scale_8l_h_raw as raw
import scale_8l_common as scale
import train_p7_theta0 as p7
from hstu_kvcache.models import HSTU, HSTUConfig

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/"configs/contracts/scale_8l_m1_seed17_v1.yaml"
CHECKPOINT=ROOT/"results/scale_8l_v1/theta0/m1_seed17/theta0_selected.pt"
OUTPUT=ROOT/"results/scale_8l_m1_seed17/h_theta0"
RESULT=ROOT/"results/scale_8l_m1_seed17/h_theta0_adjudication.json"


def validate():
    value=yaml.safe_load(PLAN.read_text())
    if p7.sha256_file(scale.CONTRACT)!=value["inputs"]["scale_8l_contract_sha256"]: raise RuntimeError("scale contract changed")
    if value["scope"]["qualification_or_theta3"]!="prohibited": raise RuntimeError("sealed data access forbidden")
    if not CHECKPOINT.exists(): raise FileNotFoundError(CHECKPOINT)
    payload=torch.load(CHECKPOINT,map_location="cpu",weights_only=False)
    for key,expected in (("model_name","m1"),("seed",17),("history_limit",1024),("qualification_or_theta3_scored",False)):
        if payload.get(key)!=expected: raise RuntimeError(f"M1 checkpoint identity mismatch: {key}")
    return value


def load_model(device):
    validate(); payload=torch.load(CHECKPOINT,map_location="cpu",weights_only=False); state=payload.pop("model_state_dict")
    model=HSTU(HSTUConfig(**payload["config"])); model.load_state_dict(state,strict=True)
    return model.to(device).eval(),payload


def load_contract():
    value=validate()
    return {"gates":{"numeric_floors":{"output_js_divergence":1e-8,"absolute_probability_difference":1e-7}}}


def columns(*args,**kwargs):
    value=raw._M1_original_columns(*args,**kwargs)
    value["model_condition"]=["m1"]*len(value["model_condition"])
    return value


def patch_raw():
    raw.CONTRACT=PLAN; raw.CHECKPOINT=CHECKPOINT; raw.OUTPUT=OUTPUT
    raw.load_contract=load_contract; raw.load_model=load_model
    if not hasattr(raw,"_M1_original_columns"): raw._M1_original_columns=raw.columns
    raw.columns=columns


def patch_adjudication():
    adjud.CONTRACT=PLAN; adjud.RAW_ROOT=OUTPUT; adjud.SEAL=OUTPUT/"raw_score_seal.json"; adjud.OUTPUT=RESULT
    adjud.sha256_file=p7.sha256_file


def adjudicate():
    patch_adjudication()
    # The original metric implementation expects its self hash in the old
    # contract.  Its math is reused here, but the M1 plan is the authority.
    plan=yaml.safe_load(PLAN.read_text()); seal=json.loads(adjud.SEAL.read_text())
    artifacts={row["view"]:row for row in seal["artifacts"]}
    import numpy as np, pyarrow.parquet as pq
    fidelity=pq.read_table(ROOT/artifacts["fidelity"]["path"]).to_pandas()
    rf=fidelity.recent32_deployment_logit.to_numpy(float); ff=fidelity.full1024_deployment_logit.to_numpy(float); uid=fidelity.uid.to_numpy(np.int64)
    js=adjud.bernoulli_js(rf,ff); prob=np.abs(adjud.sigmoid(ff)-adjud.sigmoid(rf)); std=max(float(np.std(ff)),1e-3)
    fmetrics={"output_js_divergence":adjud.user_mean_summary(uid,js,"m1-fidelity-js"),
        "normalized_score_rms":adjud.user_mean_summary(uid,np.abs(ff-rf)/std,"m1-fidelity-rms"),
        "absolute_probability_difference":adjud.user_mean_summary(uid,prob,"m1-fidelity-probability")}
    quality=pq.read_table(ROOT/artifacts["quality"]["path"]).to_pandas(); y=quality.label.to_numpy(np.int64); uq=quality.uid.to_numpy(np.int64)
    rz=quality.recent32_deployment_logit.to_numpy(float); fz=quality.full1024_deployment_logit.to_numpy(float)
    rp,fp=adjud.sigmoid(rz),adjud.sigmoid(fz); rl=np.logaddexp(0,rz)-y*rz; fl=np.logaddexp(0,fz)-y*fz
    qmetrics={"log_loss_gain":adjud.user_mean_summary(uq,rl-fl,"m1-quality-logloss"),
        "Brier_gain":adjud.user_mean_summary(uq,(rp-y)**2-(fp-y)**2,"m1-quality-brier"),
        **adjud.classification_bootstrap(quality)}
    dislike=y==0; qmetrics["dislike_only_log_loss_gain"]=adjud.user_mean_summary(uq[dislike],(rl-fl)[dislike],"m1-dislike-logloss")
    floor=plan["gates"]["theta0_H"]; components={
        "H_JS_CI_above_floor":fmetrics["output_js_divergence"]["ci95"][0]>float(floor["JS_CI_above_floor"]),
        "H_probability_CI_above_floor":fmetrics["absolute_probability_difference"]["ci95"][0]>float(floor["probability_shift_CI_above_floor"]),
        "quality_logloss_CI_positive":qmetrics["log_loss_gain"]["ci95"][0]>0,
        "ROC_AUC_noninferior":qmetrics["ROC_AUC_gain"]["ci95"][0]>=float(floor["ROC_AUC_noninferiority"]),
        "dislike_PR_AUC_noninferior":qmetrics["dislike_PR_AUC_gain"]["ci95"][0]>=float(floor["dislike_PR_AUC_noninferiority"])}
    passed=all(components.values()); result={"status":"scale_M1_H_gate_passed" if passed else "scale_M1_H_gate_failed_stop",
        "model_condition":"m1","seed":17,"evidence_level":"development_scale_pilot_single_seed","fidelity":fmetrics,"quality":qmetrics,
        "gate_components":components,"H_passed":passed,"qualification_or_theta3_read":False,
        "checkpoint_sha256":p7.sha256_file(CHECKPOINT),"raw_seal_sha256":p7.sha256_file(adjud.SEAL),
        "authorization":"M1_release_chain_may_start" if passed else "stop_before_release_training"}
    RESULT.parent.mkdir(parents=True,exist_ok=True); RESULT.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result,indent=2))
    if not passed: raise SystemExit(2)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--stage",choices=("canary","raw","seal","adjudicate"),required=True); parser.add_argument("--device",default="cuda:0"); args=parser.parse_args()
    patch_raw()
    if args.stage=="canary": raw.run_canary(torch.device(args.device))
    elif args.stage=="raw":
        raw.run_raw(torch.device(args.device))
        path=OUTPUT/"raw_run_summary.json"; value=json.loads(path.read_text()); value["model_condition"]="m1"
        path.write_text(json.dumps(value,indent=2)+"\n")
    elif args.stage=="seal": raw.run_seal()
    else: adjudicate()


if __name__=="__main__": main()
