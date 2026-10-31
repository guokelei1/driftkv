"""Run the actual four-GPU Max trainer uninterrupted versus an exact interruption."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .data_common import json_save, sha256
from .runtime import load_checkpoint, source_binding


def assert_tree(left, right, path="root"):
    if torch.is_tensor(left):
        if not torch.equal(left, right):
            raise ValueError(f"Exact tensor resume differs: {path}")
        return left.numel()
    if isinstance(left, np.ndarray):
        if not np.array_equal(left, right):
            raise ValueError(f"Exact RNG array differs: {path}")
        return 0
    if isinstance(left, dict):
        if left.keys() != right.keys():
            raise ValueError(f"Resume keys differ: {path}")
        return sum(assert_tree(left[k], right[k], f"{path}.{k}") for k in left)
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise ValueError(f"Resume list differs: {path}")
        return sum(assert_tree(a,b,f"{path}.{i}") for i,(a,b) in enumerate(zip(left,right,strict=True)))
    if left != right:
        raise ValueError(f"Resume value differs: {path}")
    return 0


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    torch.set_num_threads(2)
    common=[sys.executable,"-m","torch.distributed.run","--standalone","--nproc_per_node=4","-m",
            "experiments.recflow_full_population.code.train","--data",str(args.data),"--phase","diagnostic",
            "--scale","max","--days","24","25","26","27","--max-steps","8","--epochs","1",
            "--micro-batch","16","--global-batch","64","--workers","4","--wall-hours","1",
            "--checkpoint-every","100000","--log-every","4"]
    commands=[("uninterrupted",common+["--output",str(args.output/"uninterrupted")]),
              ("interrupted",common+["--output",str(args.output/"resumed"),"--stop-after-steps","4"]),
              ("resumed",common+["--output",str(args.output/"resumed"),"--resume",str(args.output/"resumed/step_000004.pt")])]
    json_save(args.output/"plan.json",dict(commands=commands,quality_use_permitted=False,
        source_binding=source_binding(args.data,"vocabulary_max"),checker_sha256=sha256(__file__)))
    for name,command in commands:
        print(json.dumps(dict(stage=name,started=time.time())),flush=True)
        with (args.output/f"{name}.log").open("x") as handle:
            subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT,env=os.environ.copy(),check=True)
    left,side_left=load_checkpoint(args.output/"uninterrupted/step_000008.pt")
    right,side_right=load_checkpoint(args.output/"resumed/step_000008.pt")
    elements={key:assert_tree(left[key],right[key],key) for key in
              ["model","optimizer","scheduler","rngs","total_step","next_epoch","next_step"]}
    if left["quality_use_permitted"] or right["quality_use_permitted"]:
        raise ValueError("Readiness weights must remain diagnostic-only")
    summary=dict(status="passed",world_size=4,full_max=True,all_tensor_elements_compared=elements,
        uninterrupted_sha256=side_left["checkpoint_sha256"],resumed_sha256=side_right["checkpoint_sha256"],
        quality_use_permitted=False,wall_seconds=time.monotonic()-started,
        source_binding=source_binding(args.data,"vocabulary_max"),checker_sha256=sha256(__file__))
    json_save(args.output/"summary.json",summary)
    print(json.dumps(summary),flush=True)


if __name__=="__main__":
    main()
