#!/usr/bin/env python3
"""Resumable fail-closed four-GPU queue for the 8L M1 seed17 H/S chain."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,os,shlex,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"results/scale_8l_m1_seed17"; LOGS=BASE/"logs"; SCALE=ROOT/"results/scale_8l_v1"
RELEASES=("r1_edge1","r1_edge2","r2")

def exists(path): return (ROOT/path).exists()
def environment(): return {**os.environ,"PYTHONPATH":"src:scripts","OMP_NUM_THREADS":"8","MKL_NUM_THREADS":"8","CUDA_VISIBLE_DEVICES":"0,1,2,3"}
def run(command,log=None):
    print("RUN",shlex.join(command),flush=True)
    if log is None: subprocess.run(command,cwd=ROOT,env=environment(),check=True); return
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open("ab") as stream: result=subprocess.run(command,cwd=ROOT,env=environment(),stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode: raise RuntimeError(f"failed ({result.returncode}); inspect {log}")
def parallel(commands):
    with ThreadPoolExecutor(max_workers=len(commands)) as pool:
        futures=[pool.submit(run,*item) for item in commands]
        for future in as_completed(futures): future.result()
def gate(path,status):
    value=json.loads((ROOT/path).read_text())
    if value["status"]!=status: raise RuntimeError(f"blocking gate failed: {path}: {value['status']}")

def stages():
    return [
        {"name":"M1_FSDP_canary","done":exists("results/scale_8l_m1_seed17/trainer_canary_all_tasks/train_result.json")},
        {"name":"theta0","done":exists("results/scale_8l_v1/theta0/m1_seed17/train_result.json")},
        {"name":"theta0_H","done":exists("results/scale_8l_m1_seed17/h_theta0_adjudication.json")},
        *[{"name":f"train_{r}","done":exists(f"results/scale_8l_v1/releases/{r}/m1_seed17/train_result.json")} for r in ("r0",*RELEASES)],
        {"name":"lineage_seal","done":exists("results/scale_8l_m1_seed17/lineage_seal.json")},
        {"name":"HS_canaries","done":all(exists(f"results/scale_8l_m1_seed17/hs_canary/{r}/raw_manifest.json") for r in RELEASES)},
        {"name":"HS_raw","done":all(exists(f"results/scale_8l_m1_seed17/hs_raw/{r}/m1_seed17/raw_manifest.json") for r in RELEASES)},
        {"name":"HS_adjudication","done":all(exists(f"results/scale_8l_m1_seed17/hs_{r}_adjudication.json") for r in RELEASES)},
        {"name":"summary","done":exists("results/scale_8l_m1_seed17/summary.json")},]

def execute():
    LOGS.mkdir(parents=True,exist_ok=True)
    if not exists("results/scale_8l_m1_seed17/trainer_canary_all_tasks/train_result.json"):
        run(["torchrun","--standalone","--nproc_per_node=4","scripts/train_scale_8l_fsdp_theta0.py","--model","m1","--seed","17","--canary-steps","3",
            "--output","results/scale_8l_m1_seed17/trainer_canary_all_tasks"],LOGS/"trainer_canary_all_tasks.log")
    if not exists("results/scale_8l_v1/theta0/m1_seed17/train_result.json"):
        run(["torchrun","--standalone","--nproc_per_node=4","scripts/train_scale_8l_fsdp_theta0.py","--model","m1","--seed","17"],LOGS/"theta0.log")
    for stage in ("canary","raw","seal","adjudicate"):
        artifact={"canary":"results/scale_8l_m1_seed17/h_theta0/development_identity_canary.json","raw":"results/scale_8l_m1_seed17/h_theta0/raw_run_summary.json",
            "seal":"results/scale_8l_m1_seed17/h_theta0/raw_score_seal.json","adjudicate":"results/scale_8l_m1_seed17/h_theta0_adjudication.json"}[stage]
        if not exists(artifact): run(["python","scripts/eval_scale_8l_m1_h_pipeline.py","--stage",stage,"--device","cuda:0"],LOGS/f"H_{stage}.log")
    gate("results/scale_8l_m1_seed17/h_theta0_adjudication.json","scale_M1_H_gate_passed")
    for release in ("r0",*RELEASES):
        artifact=f"results/scale_8l_v1/releases/{release}/m1_seed17/train_result.json"
        if not exists(artifact): run(["torchrun","--standalone","--nproc_per_node=4","scripts/train_scale_8l_fsdp_release.py","--release",release,"--model","m1","--seed","17"],LOGS/f"train_{release}.log")
    if not exists("results/scale_8l_m1_seed17/lineage_seal.json"): run(["python","scripts/freeze_scale_8l_m1_lineage.py"])
    pending=[]
    for gpu,release in enumerate(RELEASES):
        artifact=f"results/scale_8l_m1_seed17/hs_canary/{release}/raw_manifest.json"
        if not exists(artifact): pending.append((["python","scripts/eval_scale_8l_m1_hs_raw.py","--release",release,"--device",f"cuda:{gpu}","--max-users","8","--output",f"results/scale_8l_m1_seed17/hs_canary/{release}"],LOGS/f"HS_canary_{release}.log"))
    if pending: parallel(pending)
    pending=[]
    for gpu,release in enumerate(RELEASES):
        artifact=f"results/scale_8l_m1_seed17/hs_raw/{release}/m1_seed17/raw_manifest.json"
        if not exists(artifact): pending.append((["python","scripts/eval_scale_8l_m1_hs_raw.py","--release",release,"--device",f"cuda:{gpu}"],LOGS/f"HS_raw_{release}.log"))
    if pending: parallel(pending)
    if not exists("results/scale_8l_m1_seed17/hs_raw_seal.json"): run(["python","scripts/adjudicate_scale_8l_m1_hs.py","--stage","seal"])
    for release in RELEASES:
        artifact=f"results/scale_8l_m1_seed17/hs_{release}_adjudication.json"
        if not exists(artifact): run(["python","scripts/adjudicate_scale_8l_m1_hs.py","--stage","adjudicate","--release",release])
        gate(artifact,"scale_M1_HS_gate_passed")
    if not exists("results/scale_8l_m1_seed17/summary.json"): run(["python","scripts/summarize_scale_8l_m1_seed17.py"])

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--run",action="store_true"); parser.add_argument("--status",action="store_true"); args=parser.parse_args()
    if args.run: execute()
    else: print(json.dumps({"status":"complete" if all(x["done"] for x in stages()) else "ready","stages":stages(),
        "launch_command":"PYTHONPATH=src:scripts python scripts/run_scale_8l_m1_seed17.py --run","GPUs":[0,1,2,3],"theta3_access":False},indent=2))

if __name__=="__main__": main()
