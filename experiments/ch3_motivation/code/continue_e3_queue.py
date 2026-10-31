#!/usr/bin/env python3
"""Durable, sequential GPU0 continuation after the fresh anchor queue exits."""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[3]

from experiments.ch3_motivation.code.common import write_json
from hstu_kvcache.common.artifacts import sha256


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-dir",type=Path,required=True)
    args=parser.parse_args()
    folder=args.queue_dir.resolve()
    config=json.loads((folder/"queue_config.json").read_text())
    launch_path=folder/"launch.json"
    launch=json.loads(launch_path.read_text())
    if sha256(folder/"queue_config.json")!=launch["config_sha256"]:
        raise ValueError("continuation queue configuration changed")
    launch.update(orchestrator_pid=os.getpid(),started=now(),state="waiting_for_original_qualification",jobs=[])
    write_json(launch_path,launch)
    def status(state,**fields):
        value=dict(time=now(),state=state,orchestrator_pid=os.getpid(),**fields)
        write_json(folder/"queue_status.json",value)
        print(json.dumps(value),flush=True)
    def verify_sources():
        changed=[path for path,digest in config["source_sha256"].items() if sha256(ROOT/path)!=digest]
        changed.extend(path for path,digest in config["frozen_inputs"].items() if sha256(Path(path))!=digest)
        if changed:
            raise ValueError("continuation sources changed: "+", ".join(changed))
    try:
        pid=config["wait_for_python_pid"]
        while True:
            try:
                command=Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0",b" ").decode()
            except FileNotFoundError:
                break
            if not command:
                time.sleep(1)
                continue
            if "qualify_scenario.py" not in command:
                raise ValueError("waited PID now identifies another process")
            status("waiting_for_original_qualification",wait_for_python_pid=pid)
            time.sleep(30)
        original=Path(config["qualification_root"])/"queue_status.json"
        for _ in range(12):
            if original.exists():
                break
            time.sleep(1)
        completed=json.loads(original.read_text())
        if completed["state"]!="complete" or completed["exit_code"]!=0:
            raise ValueError("original qualification did not exit successfully")
        verify_sources()
        for index,job in enumerate(config["jobs"]):
            verify_sources()
            command=job["command"]
            log=Path(job["log"])
            log.parent.mkdir(parents=True,exist_ok=True)
            with log.open("a") as stream:
                process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
                record=dict(index=index,name=job["name"],command=command,pid=process.pid,log=str(log),started=now(),state="running")
                launch["jobs"].append(record)
                launch.update(state="running",active_child_pid=process.pid,active_job=job["name"])
                write_json(launch_path,launch)
                status("running",job=job["name"],child_pid=process.pid,log=str(log))
                code=process.wait()
            record.update(exit_code=code,finished=now(),state="complete" if code==0 else "failed")
            write_json(launch_path,launch)
            if code:
                raise RuntimeError(f"{job['name']} exited {code}; see {log}")
            if job["name"]=="wrapper_check_8":
                check=json.loads(Path(job["check_path"]).read_text())
                if not check["passed"] or check["quality_metrics_computed"] or check["users"]!=8:
                    raise ValueError("new manifest wrapper numerical check failed")
                launch["wrapper_check_passed"]=True
                write_json(launch_path,launch)
        verify_sources()
        launch.update(state="complete",finished=now(),active_child_pid=None)
        write_json(launch_path,launch)
        status("complete",jobs=len(config["jobs"]),exit_code=0)
    except BaseException as error:
        launch.update(state="failed",finished=now(),error=str(error))
        write_json(launch_path,launch)
        status("failed",exit_code=1,error=str(error))
        raise


if __name__=="__main__":
    main()
