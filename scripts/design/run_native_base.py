#!/usr/bin/env python3
"""Run fixed independent UID shards and merge real feedback before quality metrics."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from design.data import ROOT
from design.evaluate_native_base import PATHS, PROTOCOL
from design.run import write_json

from hstu_kvcache.evaluation.binary_metrics import binary_metrics


def main(cli):
    out=ROOT / "results/design" / cli.run_id
    out.mkdir(parents=True,exist_ok=False)
    protocol=json.loads(PROTOCOL.read_text())
    uids=protocol["development_uids"]
    shards=[dict(offset=i,users=min(cli.chunk_users,len(uids)-i),run_id=f"{cli.run_id}/shard_{i:05d}") for i in range(0,len(uids),cli.chunk_users)]
    write_json(out / "configuration.json",dict(vars(cli),protocol=protocol,shards=shards,development_uids=uids,
        authorization="User explicitly requested this mature single-migration quality evaluation and two ablations; focused canary passed.",
        protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),confirmation_read=False))
    start=time.perf_counter()
    def worker(gpu,assigned):
        for shard in assigned:
            log=ROOT / "results/design" / (shard["run_id"]+".log")
            command=[sys.executable,"-u","scripts/design/evaluate_native_base.py","--run-id",shard["run_id"],
                     "--offset",str(shard["offset"]),"--users",str(shard["users"]),"--estimate-seconds",str(cli.estimate_seconds)]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH="src:scripts",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="4")
            with log.open("w") as stream:
                result=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            log.with_suffix(".exit").write_text(str(result.returncode)+"\n")
            if result.returncode:
                raise RuntimeError(f"{shard['run_id']} failed; see {log}")
            print(json.dumps(dict(status="shard_complete",gpu=gpu,**shard,elapsed_seconds=time.perf_counter()-start)),flush=True)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(cli.gpus)) as pool:
            futures=[pool.submit(worker,g,shards[i::len(cli.gpus)]) for i,g in enumerate(cli.gpus)]
            for f in concurrent.futures.as_completed(futures):
                f.result()
        frames,coverage,states=[],[],[]
        ledgers,empty_ledgers=defaultdict(lambda:defaultdict(float)),defaultdict(lambda:defaultdict(float))
        source_hashes=weights_hashes=None
        peak=worker_seconds=0.
        for shard in shards:
            path=ROOT / "results/design" / shard["run_id"]
            cfg=json.loads((path / "configuration.json").read_text())
            assert cfg["development_uids"]==uids[shard["offset"]:shard["offset"]+shard["users"]]
            if source_hashes is None:
                source_hashes,weights_hashes=cfg["source_sha256"],cfg["weights_sha256"]
            assert cfg["source_sha256"]==source_hashes and cfg["weights_sha256"]==weights_hashes
            summary=json.loads((path / "summary.json").read_text())
            assert summary["status"]=="development_complete"
            frames.append(pd.read_parquet(path / "quality_raw.parquet"))
            coverage.extend(summary["coverage"])
            states.extend(json.loads((path / "state_counts.json").read_text()))
            peak=max(peak,summary["peak_allocated_mib"])
            worker_seconds+=summary["elapsed_seconds"]
            for key,destination in (("ledger_seconds",ledgers),("no_feedback_ledger_seconds",empty_ledgers)):
                for t,ledger in summary[key].items():
                    for name,value in ledger.items():
                        destination[t][name]+=value
        frame=pd.concat(frames,ignore_index=True)
        assert not frame.duplicated(["target","request_id"]).any()
        assert np.isfinite(frame[list(PATHS)]).all().all()
        assert set(frame.uid)<=set(uids)
        state_frame=pd.DataFrame(states)
        assert not state_frame.duplicated(["target","uid"]).any()
        assert len(state_frame)==4*len(uids)
        frame.to_parquet(out / "quality_raw.parquet",index=False)
        write_json(out / "state_counts.json",states)
        quality=[]
        for target,f in frame.groupby("target"):
            metrics={name:binary_metrics(f.label.to_numpy(),f[name].to_numpy()) for name in PATHS}
            quality.append(dict(target=int(target),requests=len(f),feedback_users=f.uid.nunique(),selected_users=len(uids),
                                no_feedback_users=len(uids)-f.uid.nunique(),metrics=metrics))
        main_effect={m:float(np.mean([r["metrics"][m]["ROC_AUC"]-r["metrics"]["reuse"]["ROC_AUC"] for r in quality])) for m in METHODS}
        write_json(out / "summary.json",dict(status="development_complete",quality=quality,equal_edge_auc_improvement=main_effect,
            ledger_seconds=dict(ledgers),no_feedback_ledger_seconds=dict(empty_ledgers),elapsed_seconds=time.perf_counter()-start,
            worker_seconds=worker_seconds,peak_allocated_mib=peak,selected_users=len(uids),confirmation_read=False,
            source_sha256=source_hashes,weights_sha256=weights_hashes,raw_sha256=hashlib.sha256((out / "quality_raw.parquet").read_bytes()).hexdigest(),
            scope="post-development mature-user base domain; independent adjacent migrations; one seed; M5 E14_partial"))
        (out / "coordinator.exit").write_text("0\n")
        print(json.dumps(dict(status="development_complete",requests=len(frame),equal_edge_auc_improvement=main_effect,elapsed_seconds=time.perf_counter()-start)),flush=True)
    except Exception as exc:
        write_json(out / "summary.json",dict(status="failed",error=repr(exc),elapsed_seconds=time.perf_counter()-start))
        (out / "coordinator.exit").write_text("1\n")
        raise


if __name__=="__main__":
    from design.native_service import METHODS
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id",required=True)
    parser.add_argument("--gpus",type=int,nargs="+",required=True)
    parser.add_argument("--chunk-users",type=int,default=512)
    parser.add_argument("--estimate-seconds",type=int,required=True)
    main(parser.parse_args())
