#!/usr/bin/env python3
"""Seal and adjudicate 8L policy quality after target-free assignment freeze."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, roc_auc_score

import train_p7_theta0 as p7

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results/scale_8l_v1/policy_quality_raw"
SEAL = ROOT / "results/scale_8l_v1/policy_quality_raw_seal_v1.json"
OUTPUT = ROOT / "results/scale_8l_v1/policy_quality_adjudication_v1.json"


def logloss(logit, label): return np.maximum(logit, 0)-label*logit+np.log1p(np.exp(-np.abs(logit)))
def sigmoid(x): return 1/(1+np.exp(-np.clip(x, -50, 50)))


def metrics(frame, column):
    y = frame.label.to_numpy(dtype=np.int64); z = frame[column].to_numpy(dtype=np.float64); p = sigmoid(z)
    return {"log_loss": float(logloss(z,y).mean()), "ROC_AUC": float(roc_auc_score(y,p)),
        "dislike_PR_AUC": float(average_precision_score(1-y,1-p)), "Brier": float(np.mean((p-y)**2)),
        "dislike_only_log_loss": float(logloss(z[y==0], y[y==0]).mean())}


def bootstrap_logloss(frame, left, right, key, repetitions=1000):
    grouped = []
    for _, values in frame.groupby("uid"):
        y=values.label.to_numpy(); grouped.append(float(np.mean(logloss(values[left].to_numpy(),y)-logloss(values[right].to_numpy(),y))))
    grouped=np.asarray(grouped); rng=np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],"little"))
    samples=np.asarray([grouped[rng.integers(0,len(grouped),len(grouped))].mean() for _ in range(repetitions)])
    return {"point": float(grouped.mean()), "CI95": [float(np.quantile(samples,.025)),float(np.quantile(samples,.975))], "users":len(grouped)}


def main():
    if SEAL.exists() or OUTPUT.exists(): raise FileExistsError("refusing to overwrite quality adjudication")
    artifacts=[]
    for release in ("r1_edge1","r1_edge2","r2"):
        manifest_path=RAW/release/"m0_f_seed17/raw_manifest.json"; manifest=json.loads(manifest_path.read_text())
        if manifest["status"]!="policy_quality_raw_written_unadjudicated" or manifest["metrics_computed"]: raise RuntimeError("quality raw invalid")
        raw=ROOT/manifest["raw"]
        if p7.sha256_file(raw)!=manifest["raw_sha256"]: raise RuntimeError("quality raw hash mismatch")
        artifacts.append({"release":release,"manifest":str(manifest_path.relative_to(ROOT)),"manifest_sha256":p7.sha256_file(manifest_path),"raw":manifest["raw"],"raw_sha256":manifest["raw_sha256"]})
    SEAL.write_text(json.dumps({"status":"scale_8l_policy_quality_raw_sealed_before_metrics","artifacts":artifacts},indent=2)+"\n")
    cells=[]
    for artifact in artifacts:
        frame=pq.read_table(ROOT/artifact["raw"]).to_pandas(); conditions={name:metrics(frame,column) for name,column in {
            "Exact":"exact_logit","Noop":"noop_logit","Policy05":"policy_05_logit","Policy10":"policy_10_logit","Policy25":"policy_25_logit"}.items()}
        comparisons={}
        for budget,column in ((.05,"policy_05_logit"),(.10,"policy_10_logit"),(.25,"policy_25_logit")):
            comparisons[str(budget)]={"Policy_minus_Exact_logloss":bootstrap_logloss(frame,column,"exact_logit",f"{artifact['release']}:{budget}:exact"),
                "Noop_minus_Policy_logloss":bootstrap_logloss(frame,"noop_logit",column,f"{artifact['release']}:{budget}:noop")}
        cells.append({"release":artifact["release"],"requests":len(frame),"users":frame.uid.nunique(),"conditions":conditions,"comparisons":comparisons})
    payload={"status":"scale_8l_policy_quality_adjudicated","raw_seal_sha256":p7.sha256_file(SEAL),"cells":cells,
        "dislike_only_logloss_mandatory_companion":True,"paper_qualification":False}
    OUTPUT.write_text(json.dumps(payload,indent=2)+"\n"); print(json.dumps(payload,indent=2))


if __name__=="__main__": main()
