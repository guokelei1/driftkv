#!/usr/bin/env python3
"""CPU-only replay of retained append accounting, excluding tensor operations."""
from collections import Counter
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"scripts"),str(ROOT/"src")]
import numpy as np
import torch
from hstu_kvcache.design_one.lightweight import LightweightValueState
from hstu_kvcache.models import HSTUKVCache
from query_relay_2026_10.session import RelaySession
from system_study_2026_10.batched_session import append_pair_counts,zero_tail_execution
from unified_reuse_2026_09.common import sha256,write_json


def main():
    source=ROOT/"results/system_study_2026_10/service/batched_session32_graph_full/attempt1"
    output=ROOT/"results/system_study_2026_10/service/scatter_accounting_cpu/attempt1"
    output.mkdir(parents=True,exist_ok=False)
    calls={m:[r for r in json.loads((source/(m+"_calls.json")).read_text())
              if r["operation"]=="append"] for m in ("relay","rebuild")}
    # CPU views provide real numel/element_size metadata for _record_memory;
    # neither source cache contents nor GPU kernel/copy costs are simulated.
    scalar=torch.zeros(1)
    caches={n:HSTUKVCache(scalar.expand(6,1,n,192),scalar.expand(6,1,n,192),n)
            for n in {min(1024,r["history_length"]+r["append_width"]) for r in calls["relay"]}}
    states={n:LightweightValueState(scalar.expand(6,1,n,6,8),n) for n in caches}

    def replay(method,mode):
        sessions={u:SimpleNamespace(trajectory=SimpleNamespace(cursor=0,version=2,
                    producers=np.zeros(1024,dtype=np.int64),ledger=Counter(),appended_tokens=0,evicted_tokens=0),
                    costs=Counter(),read_ledger=Counter(),coded_read_ledger=Counter(),memory=Counter())
                  for r in calls[method] for u in r["uids"]}
        metadata_rows=[]
        for r in calls[method]:
            n,w=r["history_length"],r["append_width"]
            total,new_n=n+w,min(1024,n+w)
            v=int(r["version"][1:])
            if mode!="semantic_only":
                metadata=dict(operation="append",uids=list(r["uids"]),batch=r["batch"],
                    version=r["version"],history_length=n,coded_rows=r["coded_rows"],
                    logical_times=list(r["logical_times"]),trace_event_ids=list(r["trace_event_ids"]),
                    query_width=w,append_width=w)
                metadata.update(zero_tail_execution(r["session_coded_rows"],w))
                if metadata["coded_rows"]:
                    padded=w if w<=4 else ((w+15)//16)*16
                    rank=8 if w<=4 else 16
                    metadata.update(projected_query_useful_flops=2*6*6*r["batch"]*w*32*6,
                        projected_query_explicit_tile_flops=2*6*6*r["batch"]*padded*32*rank)
                metadata_rows.append(metadata)
            if mode=="dense_accounting":
                positions=np.arange(n,total); keys=np.arange(total)
                keep=(keys[None]<=positions[:,None]-1)&(keys[None]>positions[:,None]-1024)
            for uid,coded in zip(r["uids"],r["session_coded_rows"],strict=True):
                s=sessions[uid]; t=s.trajectory
                t.cache=s.cache=caches[new_n]
                t.cursor+=w; t.version=v
                # Retain this common per-row producer bookkeeping even in the
                # minimum variant, so measured savings are not due to deleting it.
                t.producers=np.concatenate((t.producers,np.full(w,v,dtype=np.int64)))[-1024:]
                if method=="relay":
                    s.state=states[new_n]
                    s.coded_rows=max(0,coded-max(0,total-new_n))
                if mode=="semantic_only":
                    continue
                t.ledger["append",v,n,w]+=1
                t.appended_tokens+=w; t.evicted_tokens+=max(0,total-1024)
                if method=="relay":
                    pairs=(int(keep.sum()),int(keep[:,:coded].sum())) if mode=="dense_accounting" else append_pair_counts(n,w,coded,1024,-1)
                    s.costs["writer_gate_item_lookup_bytes"]+=w*192*4
                    s.read_ledger["writer",v,n,w,total]+=1
                    s.coded_read_ledger["writer",v,n,w,coded,total]+=1
                    s.costs.update(dict(writer_read_calls=1,writer_queries=w,
                        writer_dense_pairs=w*total,writer_dense_code_pairs=w*coded,
                        writer_valid_pairs=pairs[0],writer_old_code_valid_pairs=pairs[1]))
                    s.costs["zero_code_appended_rows"]+=w
                    RelaySession._record_memory(s)
        semantic={u:(s.trajectory.cursor,s.trajectory.version,s.cache.seq_len,getattr(s,"coded_rows",0))
                  for u,s in sessions.items()}
        ledger={u:dict(costs=dict(s.costs),native=dict(s.trajectory.ledger),
                      read=dict(s.read_ledger),coded=dict(s.coded_read_ledger),memory=dict(s.memory))
                for u,s in sessions.items()}
        return semantic,ledger,metadata_rows

    modes=("dense_accounting","closed_accounting","semantic_only")
    for method in calls:
        reference=replay(method,modes[0])
        closed=replay(method,modes[1]); minimum=replay(method,modes[2])
        assert reference==closed
        assert reference[0]==minimum[0]
    observations=[]
    for repeat in range(7):
        order=[(m,mode) for m in calls for mode in modes]
        if repeat%2: order.reverse()
        for method,mode in order:
            start=time.perf_counter(); replay(method,mode)
            observations.append(dict(repeat=repeat,method=method,mode=mode,seconds=time.perf_counter()-start))
    aggregate={}
    for method in calls:
        aggregate[method]={}
        for mode in modes:
            values=[r["seconds"] for r in observations if r["method"]==method and r["mode"]==mode]
            aggregate[method][mode]=dict(median_seconds=statistics.median(values),min_seconds=min(values),max_seconds=max(values))
    result=dict(status="complete_cpu_only",source=str(source),append_batches={m:len(c) for m,c in calls.items()},
        user_appends={m:sum(r["batch"] for r in c) for m,c in calls.items()},
        aggregate=aggregate,observations=observations,closed_ledger_and_metadata_exact=True,
        minimum_scalar_state_equal=True,
        source_hashes={str(p.relative_to(ROOT)):sha256(p) for p in (Path(__file__),
            ROOT/"scripts/system_study_2026_10/batched_session.py",source/"relay_calls.json",source/"rebuild_calls.json")},
        scope="CPU accounting/metadata proxy on exact32 append geometries, seven interleaved repetitions. Includes allocation of mock session containers and common producer arrays; excludes all actual tensor packing, cache/code copies, CUDA synchronization, scheduling, query, conversion and I/O. No deduction from measured service or projected30k seconds.")
    write_json(output/"summary.json",result)
    print(json.dumps(aggregate))


if __name__=="__main__":
    main()
