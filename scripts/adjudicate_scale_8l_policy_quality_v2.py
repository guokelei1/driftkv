#!/usr/bin/env python3
"""Protocol-correct user-weighted adjudication of sealed 8L policy logits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, roc_auc_score

import train_p7_theta0 as p7
from hstu_kvcache.data.p7_training import load_p7_requests

ROOT=Path(__file__).resolve().parents[1]
RAW_SEAL=ROOT/"results/scale_8l_v1/policy_quality_raw_seal_v1.json"
MANIFEST=ROOT/"data/manifests/p8_release_v1"
LISTENS=ROOT/"data/raw/yambda/flat/50m/listens.parquet"
OUTPUT=ROOT/"results/scale_8l_v1/policy_quality_adjudication_v2.json"
INVALIDATION=ROOT/"results/scale_8l_v1/policy_quality_adjudication_v1_invalidation.json"
SPLITS={"r1_edge1":"edge1_evaluation","r1_edge2":"edge2_evaluation","r2":"edge1_evaluation"}


def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-50,50)))
def loss(z,y): return np.maximum(z,0)-y*z+np.log1p(np.exp(-np.abs(z)))
def weighted_mean(x,w): return float(np.sum(x*w)/np.sum(w))


def metrics(frame,column):
    y=frame.label.to_numpy(dtype=np.int64); z=frame[column].to_numpy(dtype=np.float64)
    w=frame.request_weight.to_numpy(dtype=np.float64); probability=sigmoid(z)
    disliked=y==0
    return {"log_loss":weighted_mean(loss(z,y),w),"ROC_AUC":float(roc_auc_score(y,probability,sample_weight=w)),
        "dislike_PR_AUC":float(average_precision_score(1-y,1-probability,sample_weight=w)),
        "Brier":weighted_mean((probability-y)**2,w),
        "dislike_only_log_loss":weighted_mean(loss(z[disliked],y[disliked]),w[disliked])}


def clustered_delta(frame,left,right,key,repetitions=2000):
    points=[]
    for _,group in frame.groupby("uid"):
        y=group.label.to_numpy(dtype=np.int64); w=group.request_weight.to_numpy(dtype=np.float64)
        points.append(weighted_mean(loss(group[left].to_numpy(dtype=np.float64),y)-loss(group[right].to_numpy(dtype=np.float64),y),w))
    points=np.asarray(points); rng=np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],"little"))
    draws=[]
    for begin in range(0,repetitions,100):
        width=min(100,repetitions-begin); sample=rng.integers(0,len(points),size=(width,len(points))); draws.extend(points[sample].mean(axis=1))
    return {"point":float(points.mean()),"CI95":[float(np.quantile(draws,.025)),float(np.quantile(draws,.975))],"users":len(points)}


def main():
    if OUTPUT.exists(): raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    seal=json.loads(RAW_SEAL.read_text()); cells=[]
    for artifact in seal["artifacts"]:
        release=artifact["release"]; raw=ROOT/artifact["raw"]
        if p7.sha256_file(raw)!=artifact["raw_sha256"]: raise RuntimeError("sealed policy logits changed")
        frame=pq.read_table(raw).to_pandas()
        requests=load_p7_requests(MANIFEST,LISTENS,SPLITS[release],"F",manifest_kind="quality",history_limit=1024)
        weights={row.request_id:float(row.request_weight) for row in requests}
        frame["request_weight"]=frame.request_id.map(weights)
        if frame.request_weight.isna().any(): raise RuntimeError(f"request-weight join failed: {release}")
        user_weight=frame.groupby("uid").request_weight.sum().to_numpy()
        conditions={name:metrics(frame,column) for name,column in {"Exact":"exact_logit","Noop":"noop_logit",
            "Policy05":"policy_05_logit","Policy10":"policy_10_logit","Policy25":"policy_25_logit"}.items()}
        comparisons={}
        for budget,column in ((.05,"policy_05_logit"),(.10,"policy_10_logit"),(.25,"policy_25_logit")):
            comparisons[str(budget)]={"Policy_minus_Exact_logloss":clustered_delta(frame,column,"exact_logit",f"v2:{release}:{budget}:exact"),
                "Noop_minus_Policy_logloss":clustered_delta(frame,"noop_logit",column,f"v2:{release}:{budget}:noop")}
        cells.append({"release":release,"requests":len(frame),"users":frame.uid.nunique(),
            "per_user_total_weight_max_abs_delta_from_one":float(np.max(np.abs(user_weight-1))),
            "conditions":conditions,"comparisons":comparisons})
    payload={"status":"scale_8l_policy_quality_user_weighted_adjudicated_v2","raw_seal_sha256":p7.sha256_file(RAW_SEAL),
        "weight_source":"sealed_P8_quality_manifest_request_weight","formal_repeat_unit":"uid","bootstrap_repetitions":2000,
        "cells":cells,"v1_unweighted_adjudication_invalidated":True,"paper_qualification":False}
    OUTPUT.write_text(json.dumps(payload,indent=2)+"\n")
    INVALIDATION.write_text(json.dumps({"status":"invalidated_statistical_summary_only","artifact":"results/scale_8l_v1/policy_quality_adjudication_v1.json",
        "reason":"request-unweighted metrics violated frozen per-user equal-total-weight protocol","raw_logits_or_assignments_invalidated":False,
        "replacement":"results/scale_8l_v1/policy_quality_adjudication_v2.json"},indent=2)+"\n")
    print(json.dumps(payload,indent=2))


if __name__=="__main__": main()
