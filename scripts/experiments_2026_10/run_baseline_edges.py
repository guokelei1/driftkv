#!/usr/bin/env python3
"""Run the four authorized independent Yambda-Large edges sequentially on GPU 1.

The manifest freezes configs and execution sources before launch. Restarting this
queue skips finished edges and resumes committed user blocks of the current edge.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle,"sha256").hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path,value):
    staged=path.with_suffix(path.suffix+".writing")
    staged.write_text(json.dumps(value,indent=2)+"\n")
    staged.replace(path)


def run(manifest_path):
    manifest=json.loads(manifest_path.read_text())
    folder=manifest_path.parent
    with (folder/"queue.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=dict(status="running",pid=os.getpid(),machine=socket.gethostname(),
            physical_gpu=manifest["physical_gpu"],started_utc=utc(),jobs=[])
        state_path=folder/"progress.json"
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(manifest["physical_gpu"]),PYTHONUNBUFFERED="1")
        for job in manifest["jobs"]:
            for relative,expected in manifest["execution_source_sha256"].items():
                assert sha256(ROOT/relative)==expected,f"execution source changed: {relative}"
            config=ROOT/job["config"]
            assert sha256(config)==job["config_sha256"],"queued config changed"
            output=ROOT/job["output"]
            summary_path=output/"summary.json"
            if summary_path.exists():
                summary=json.loads(summary_path.read_text())
                if summary.get("status")=="complete" and summary.get("completed_utc"):
                    assert summary["edge"]==job["edge"]
                    binding=json.loads((output/"configuration.json").read_text())
                    assert binding["launch_config_sha256"]==job["config_sha256"]
                    state["jobs"].append(dict(edge=job["edge"],status="complete",resumed=True,
                        summary=str(summary_path),completed_utc=summary["completed_utc"]))
                    write_json(state_path,state)
                    continue
            command=[sys.executable,"-u",str(ROOT/"scripts/experiments_2026_10/baselines.py"),
                "--config",str(config),"--output",str(output)]
            if output.exists():command.append("--resume")
            output.parent.mkdir(parents=True,exist_ok=True)
            log=ROOT/job["log"]
            with log.open("a") as handle:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
                entry=dict(edge=job["edge"],status="running",pid=child.pid,started_utc=utc(),
                    output=str(output),log=str(log),command=command,
                    resume_command=command if "--resume" in command else command+["--resume"])
                state["jobs"].append(entry)
                write_json(output.with_name(output.name+".launch.json"),entry)
                write_json(state_path,state)
                print(json.dumps(entry),flush=True)
                code=child.wait()
            entry.update(status="complete" if code==0 else "failed",exit_code=code,finished_utc=utc())
            write_json(state_path,state)
            print(json.dumps(entry),flush=True)
            if code:
                state.update(status="failed",finished_utc=utc())
                write_json(state_path,state)
                return code
        state.update(status="complete",finished_utc=utc())
        write_json(state_path,state)
        write_json(folder/"summary.json",state)
    return 0


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,required=True)
    raise SystemExit(run(parser.parse_args().manifest.resolve()))
