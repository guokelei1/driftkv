"""Run exactly one declared experiment, then its CPU analysis; no successor queue."""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from experiments.recflow_full_population.code.data_common import json_save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    run = parser.parse_args().run.resolve()
    if (run / "active_process.json").exists():
        raise ValueError("This experiment was already dispatched")
    repo = Path(__file__).resolve().parents[3]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0,1,2,3", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               PYTHONPATH=str(repo / "src") + os.pathsep + str(repo), PYTHONUNBUFFERED="1",
               EVOKV_ATTENTION_BACKEND="auto")
    commands = [("scoring", [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
                 "-m", "experiments.recflow_release_design.code.interval_evaluate", "--run", str(run)]),
                ("analysis", [sys.executable, "-m", "experiments.recflow_release_design.code.interval_analyze", "--run", str(run)])]
    for stage, command in commands:
        with (run / f"{stage}.log").open("x") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=repo, env=env)
            record = dict(stage=stage, status="running", pid=child.pid, command=command,
                          started_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat())
            json_save(run / "active_process.json", record)
            result = child.wait()
        record.update(status="completed" if result == 0 else "failed", returncode=result,
                      finished_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat())
        json_save(run / f"{stage}_process.json", record)
        json_save(run / "active_process.json", record)
        print(json.dumps(record), flush=True)
        if result:
            sys.exit(result)


if __name__ == "__main__":
    main()
