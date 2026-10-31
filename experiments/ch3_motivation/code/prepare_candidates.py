#!/usr/bin/env python3
"""Known pre-release feedback items for fair E5 calibration candidate mixes.

Labels are never selected or returned. These are observed feedback-item IDs,
distinct from the older helper's recent-listen proxy.
"""

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]
import argparse
import json
from pathlib import Path
import sys

from experiments.ch3_motivation.code.common import ROOT, DATASET, DAY, quoted, write_json
from hstu_kvcache.common.artifacts import sha256
import duckdb
import numpy as np
import pyarrow as pa


def candidates_for(uid,recent,known,*,count=16,recent_fraction=.5,seed=17):
    if not 0<=recent_fraction<=1:raise ValueError("recent fraction outside [0,1]")
    target=int(count*recent_fraction)
    chosen=[]
    for item in recent:
        if 1<=int(item)<=known and int(item) not in chosen:
            chosen.append(int(item))
            if len(chosen)>=target:break
    if target==0:chosen=[]
    seen=set(chosen)
    rng=np.random.default_rng(np.random.SeedSequence([seed,int(uid)]))
    while len(chosen)<count:
        item=int(rng.integers(1,known+1))
        if item not in seen:chosen.append(item);seen.add(item)
    return chosen


def prepare(input_root,edge,threads):
    population=json.loads((input_root/"population.json").read_text())
    uids=population["fit"]+population["validation"]
    ds=json.loads(DATASET.read_text())
    release=231+(edge-1)*14
    directory=input_root/f"edge_{edge}"
    path=directory/"calibration_feedback_candidates.json"
    if path.exists():return json.loads(path.read_text())
    con=duckdb.connect();con.execute(f"PRAGMA threads={threads}")
    con.register("selected",pa.table({"uid":pa.array(uids,type=pa.uint64())}))
    feedback=quoted((DATASET.parent/ds["shared_feedback_glob"]).resolve())
    mapping=quoted((DATASET.parent/ds["item_mapping_path"]).resolve())
    rows=con.execute(f"""
        SELECT uid,m.item_idx,max(timestamp) AS latest
        FROM read_parquet('{feedback}',hive_partitioning=true) f
        SEMI JOIN selected USING(uid)
        JOIN read_parquet('{mapping}') m USING(raw_item_id)
        WHERE timestamp>=? AND timestamp<?
        GROUP BY uid,m.item_idx ORDER BY uid,latest DESC,m.item_idx DESC
    """,[(release-14)*DAY,release*DAY]).fetchall()
    con.close()
    recent={str(uid):[] for uid in uids}
    for uid,item,_ in rows:
        if len(recent[str(uid)])<64:recent[str(uid)].append(int(item))
    value=dict(edge=edge,release_day=release,lookback_days=14,known_items=ds["foundation_items"],
        population_sha256=sha256(input_root/"population.json"),dataset_sha256=sha256(DATASET),
        recent_items=recent,users_with_recent_feedback=sum(bool(x) for x in recent.values()),
        selection="latest distinct known feedback-item IDs before release; feedback labels neither used nor returned",
        catalog="inclusive 1..foundation_items; legacy model/data numbering unchanged")
    write_json(path,value)
    return value


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root",type=Path,default=ROOT/"experiments/ch3_motivation/inputs")
    p.add_argument("--edges",nargs="+",type=int,default=[1,2,3,4,5])
    p.add_argument("--threads",type=int,default=12)
    a=p.parse_args()
    for edge in a.edges:
        v=prepare(a.input_root,edge,a.threads)
        print(json.dumps(dict(edge=edge,status="calibration_candidates_ready",users_with_recent_feedback=v["users_with_recent_feedback"])),flush=True)
