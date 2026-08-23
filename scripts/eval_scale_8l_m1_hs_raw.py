#!/usr/bin/env python3
"""M1 wrapper around the frozen true-rolling 8L H/S executor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch

import eval_scale_8l_hs_raw as core
import train_p7_theta0 as p7

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/"configs/contracts/scale_8l_m1_seed17_v1.yaml"
LINEAGE=ROOT/"results/scale_8l_m1_seed17/lineage_seal.json"
OUTPUT=ROOT/"results/scale_8l_m1_seed17/hs_raw"


def checkpoint(release): return ROOT/"results/scale_8l_v1/releases"/release/"m1_seed17/selected.pt"


def validate():
    value=json.loads(LINEAGE.read_text())
    if value["status"]!="scale_8l_M1_seed17_lineage_sealed_before_HS": raise RuntimeError("M1 lineage not sealed")
    for row in value["releases"]:
        if p7.sha256_file(ROOT/row["checkpoint"])!=row["checkpoint_sha256"] or p7.sha256_file(ROOT/row["parent"])!=row["parent_sha256"]:
            raise RuntimeError("M1 lineage artifact changed")
    return value


def score_rows(*args,**kwargs):
    rows,delta=core._M1_original_score_rows(*args,**kwargs)
    for row in rows: row["model"]="m1"
    return rows,delta


def patch():
    core.CONTRACT=PLAN; core.OUTPUT_ROOT=OUTPUT; core.checkpoint_path=checkpoint; core.load_contract=validate
    if not hasattr(core,"_M1_original_score_rows"): core._M1_original_score_rows=core.score_rows
    core.score_rows=score_rows


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--release",choices=("r1_edge1","r1_edge2","r2"),required=True)
    parser.add_argument("--device",choices=tuple(f"cuda:{i}" for i in range(4)),required=True); parser.add_argument("--max-users",type=int); parser.add_argument("--output",type=Path)
    args=parser.parse_args(); patch(); output=(args.output or OUTPUT/args.release/"m1_seed17").resolve()
    if output.exists(): raise FileExistsError(output)
    core.evaluate(args.release,torch.device(args.device),output,args.max_users)
    manifest=output/"raw_manifest.json"; value=json.loads(manifest.read_text()); value["model"]="m1"; value["seed"]=17
    value["lineage_seal_sha256"]=p7.sha256_file(LINEAGE); manifest.write_text(json.dumps(value,indent=2)+"\n")

if __name__=="__main__": main()
