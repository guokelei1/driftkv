"""Queue the original512/1024 UID chunks after the exact-contraction canary."""

import concurrent.futures
import json
import os
import subprocess
import sys
import time

from design2.conditional_check import ROOT, OUT


def main():
    canary=json.loads((OUT/"chunks/residual_calibration_0000/summary.json").read_text())
    assert canary["status"]=="complete" and all(r["mismatches"]==0 for r in canary["results"])
    jobs=[("residual_calibration",16,112)]+[("residual_calibration",i,128) for i in (128,256,384)]
    jobs += [("development",i,128) for i in range(0,1024,128)]
    launch=OUT/"population"; launch.mkdir(exist_ok=False)
    (launch/"configuration.json").write_text(json.dumps(dict(jobs=jobs,gpus=[0,1,2,3],
        canary_seconds=canary["elapsed_seconds"],expected_wall_seconds=600,expected_peak_gpu_mib=24576,
        resource_basis="16 all-lifetime UID probe~32s includes every accurate validation solve; previous1536UID two-GPU tiered queue577.55s. Four GPUs now free, original batches retained.",
        authorization="User requests exact shared-source contraction and medium verification after canary; no new fit/teacher."),indent=2)+"\n")
    started=time.perf_counter()
    def queue(gpu):
        results=[]
        for role,start,n in jobs[gpu::4]:
            name=f"{role}_{start:04d}"
            cmd=[sys.executable,"-u","scripts/design2/conditional_check.py","replay","--role",role,
                 "--start",str(start),"--users",str(n),"--device",f"cuda:{gpu}"]
            with (OUT/f"{name}.log").open("x") as log:
                code=subprocess.call(cmd,cwd=ROOT,env=dict(os.environ,PYTHONPATH="src:scripts"),stdout=log,stderr=subprocess.STDOUT)
            (OUT/f"{name}.exit").write_text(str(code)+"\n")
            results.append(dict(name=name,gpu=gpu,exit=code)); print(json.dumps(results[-1]),flush=True)
            if code: break
        return results
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records=[r for group in pool.map(queue,range(4)) for r in group]
    status="complete" if len(records)==len(jobs) and all(r["exit"]==0 for r in records) else "failed"
    (launch/"summary.json").write_text(json.dumps(dict(status=status,jobs=records,elapsed_seconds=time.perf_counter()-started),indent=2)+"\n")
    assert status=="complete", "Preserve failed evidence; no incomplete success aggregate."


if __name__=="__main__": main()
