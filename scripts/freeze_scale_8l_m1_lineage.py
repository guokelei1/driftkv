#!/usr/bin/env python3
"""Seal admitted M1 seed17 checkpoint lineage before any H/S score."""

import json
from pathlib import Path
import torch
import train_p7_theta0 as p7

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/"results/scale_8l_m1_seed17/lineage_seal.json"
RELEASES=("r0","r1_edge1","r1_edge2","r2")

def main():
    if OUTPUT.exists(): raise FileExistsError(OUTPUT)
    rows=[]
    for release in RELEASES:
        root=ROOT/"results/scale_8l_v1/releases"/release/"m1_seed17"; result=json.loads((root/"train_result.json").read_text())
        checkpoint=root/"selected.pt"; payload=torch.load(checkpoint,map_location="cpu",weights_only=False)
        if result["admitted"] is not True or payload["admitted"] is not True: raise RuntimeError(f"non-admitted M1 release: {release}")
        if result["checkpoint_hash"]!=p7.sha256_file(checkpoint): raise RuntimeError("checkpoint hash mismatch")
        parent=ROOT/payload["parent_checkpoint"]
        if p7.sha256_file(parent)!=payload["parent_checkpoint_hash"]: raise RuntimeError("parent hash mismatch")
        rows.append({"release":release,"checkpoint":str(checkpoint.relative_to(ROOT)),"checkpoint_sha256":p7.sha256_file(checkpoint),
            "parent":str(parent.relative_to(ROOT)),"parent_sha256":p7.sha256_file(parent),"admitted":True,
            "r0_cache_parameter_delta":result.get("r0_frozen_parameter_max_abs_delta")})
    value={"status":"scale_8l_M1_seed17_lineage_sealed_before_HS","model":"m1","seed":17,"releases":rows,
        "qualification_or_theta3_read":False}
    OUTPUT.parent.mkdir(parents=True,exist_ok=True); OUTPUT.write_text(json.dumps(value,indent=2)+"\n"); print(json.dumps(value,indent=2))

if __name__=="__main__": main()
