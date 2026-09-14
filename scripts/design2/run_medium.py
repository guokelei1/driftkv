"""Queue disjoint detection chunks over the four authorized GPUs."""

import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def main():
    root=ROOT/"results/design2"
    canary=json.loads((root/"canary16_01/summary.json").read_text())
    geometry=json.loads((root/"geometry_01/summary.json").read_text())
    assert canary["status"]==geometry["status"]=="complete"
    jobs=[("residual_calibration",16,112)]
    jobs += [("residual_calibration",i,128) for i in (128,256,384)]
    jobs += [("development",i,128) for i in range(0,1024,128)]
    jobs += [("historical_diagnostic",0,128),("fitting_check",0,128),("fitting_check",128,128)]
    named=[(f"detection01_{role}_{start:04d}",role,start,n) for role,start,n in jobs]
    launch=root/"medium_01"
    launch.mkdir(exist_ok=False)
    (launch/"configuration.json").write_text(json.dumps(dict(
        jobs=named,existing_canary="canary16_01",geometry="geometry_01",gpus=[0,1,2,3],
        expected_wall_seconds=1200,expected_peak_gpu_mib=32768,
        resource_basis="16 all-lifetime UID probe33.79s/4466MiB; H80.83s/15420MiB; 128-UID chunks, four queues",
        authorization="2026-09-08 user explicitly requested medium detection experiments",
        scope="frozen six-layer C; no refit, confirmation, sensitivity, scheduler or persistent rebuild"),indent=2)+"\n")
    env=os.environ.copy()
    env["PYTHONPATH"]="src:scripts"
    started=time.perf_counter()
    def queue(gpu):
        completed=[]
        for name,role,start,n in named[gpu::4]:
            command=[sys.executable,"-u","scripts/design2/run_detection.py","--mode","evaluate",
                "--role",role,"--start",str(start),"--users",str(n),"--run-id",name,"--device",f"cuda:{gpu}"]
            with (root/f"{name}.log").open("w") as log:
                code=subprocess.call(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            (root/f"{name}.exit").write_text(str(code)+"\n")
            completed.append(dict(run=name,exit=code,gpu=gpu))
            print(json.dumps(completed[-1]),flush=True)
            if code:
                break
        return completed
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records=[entry for result in pool.map(queue,range(4)) for entry in result]
    status="complete" if len(records)==len(named) and all(r["exit"]==0 for r in records) else "failed"
    (launch/"summary.json").write_text(json.dumps(dict(status=status,jobs=records,elapsed_seconds=time.perf_counter()-started),indent=2)+"\n")
    if status!="complete":
        raise RuntimeError("A chunk failed; retained logs/exit and no incomplete aggregate")


if __name__=="__main__":
    main()
