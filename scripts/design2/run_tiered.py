"""Queue fixed tiered32 replay chunks on free GPUs0/1 after the16UID canary."""

import concurrent.futures
import json
import os
import subprocess
import sys
import time

from design2.tiered_check import OUT, ROOT


def main():
    canary = json.loads((OUT/"chunks/residual_calibration_0000/summary.json").read_text())
    assert canary["status"] == "complete" and all(r["mismatches"] == 0 for r in canary["results"])
    jobs = [("residual_calibration", 16, 112)]
    jobs += [("residual_calibration", i, 128) for i in (128, 256, 384)]
    jobs += [("development", i, 128) for i in range(0, 1024, 128)]
    launch = OUT/"population"
    launch.mkdir(exist_ok=False)
    configuration = dict(jobs=jobs, gpus=[0, 1], expected_wall_seconds=900, expected_peak_gpu_mib=24576,
        canary_seconds=canary["elapsed_seconds"],
        resource_basis="16 all-lifetime canary30.82s; remaining1520UID less lifecycle-heavy, two queues; previous full detection1920UID482.87s on four GPUs included teacher. Estimate<15min, no long training.",
        authorization="2026-09-08 user authorizes next fixed32 equivalent-check round",
        change_after_canary="Sort pandas query index before lookup to avoid performance warning; numerical detector/input batching unchanged.")
    (launch/"configuration.json").write_text(json.dumps(configuration, indent=2)+"\n")
    started = time.perf_counter()
    def queue(gpu):
        done = []
        for role, start, n in jobs[gpu::2]:
            name = f"{role}_{start:04d}"
            command = [sys.executable, "-u", "scripts/design2/tiered_check.py", "replay", "--role", role,
                       "--start", str(start), "--users", str(n), "--device", f"cuda:{gpu}"]
            with (OUT/f"{name}.log").open("x") as log:
                code = subprocess.call(command, cwd=ROOT, env=dict(os.environ, PYTHONPATH="src:scripts"), stdout=log, stderr=subprocess.STDOUT)
            (OUT/f"{name}.exit").write_text(str(code)+"\n")
            done.append(dict(name=name, gpu=gpu, exit=code))
            print(json.dumps(done[-1]), flush=True)
            if code:
                break
        return done
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        records = [r for group in pool.map(queue, (0, 1)) for r in group]
    status = "complete" if len(records) == len(jobs) and all(r["exit"] == 0 for r in records) else "failed"
    (launch/"summary.json").write_text(json.dumps(dict(status=status, jobs=records,
        elapsed_seconds=time.perf_counter()-started), indent=2)+"\n")
    assert status == "complete", "Retain failed logs; no incomplete success aggregate"


if __name__ == "__main__":
    main()
