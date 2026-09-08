#!/usr/bin/env python3
"""Five frozen paths on mature users and independent adjacent-release states."""

# timed() executes callbacks synchronously inside their defining loop iteration.
# ruff: noqa: B023

import argparse
import hashlib
import json
import tarfile
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from design.data import DAY, ROOT, diagnostic_admissions, frozen_model, histories, quality_requests
from design.diagnose_native_input import read_input
from design.native_service import METHODS, NativeRelease
from design.run import (
    append_events,
    cache_at_many,
    event_range,
    new_state,
    tensor,
    timed,
    write_json,
)
from insight_two.common import CUTOVER_DAYS

from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.evaluation.binary_metrics import binary_metrics

PROTOCOL = ROOT / "results/design/native_base_protocol_01/configuration.json"
FULL = ROOT / "results/design/native_coverage384_01"
ABLATIONS = ROOT / "results/design/native_base_ablation256_01"
PATHS = ("reuse","exact",*METHODS)


def replay_user(current,state,exact,history,uid,requests,start,stop,release,ledger,canary):
    """Existing query-before-same-timestamp-write and rolling128 semantics."""
    events = event_range(history,uid,start,stop)
    grouped_events,grouped_requests = defaultdict(list),defaultdict(list)
    for event in events:
        grouped_events[event[0]].append(event)
    for row in requests:
        grouped_requests[int(row["query_timestamp"])].append(row)
    last = int(state.writer.events[-1][3])
    prepared = release.prepare(state,ledger,"publication")
    pending,raw = [],[]
    independent = state.cache if canary else None
    checks = dict(native_kv_comparisons=0,unchanged_scoring_states=0,serving_reference_comparisons=0)
    dirty = False
    def flush():
        nonlocal last,exact,independent,dirty
        for start_index in range(0,len(pending),128):
            chunk = pending[start_index:start_index+128]
            ordinary = timed(lambda:append_events(current,state.cache,chunk,last,128),ledger,"service_reuse_append")
            timed(lambda:state.install_native_chunk(ordinary,[e[0] for e in chunk]),ledger,"service_summary_maintenance")
            exact = timed(lambda:append_events(current,exact,chunk,last,128),ledger,"service_exact_append")
            if canary:
                independent = append_events(current,independent,chunk,last,128)
                torch.testing.assert_close(state.cache.k,independent.k,atol=1e-6,rtol=1e-6)
                torch.testing.assert_close(state.cache.v,independent.v,atol=1e-6,rtol=1e-6)
                checks["native_kv_comparisons"] += 1
            last = chunk[-1][0]
            dirty = True
        pending.clear()
    for timestamp in sorted(set(grouped_events)|set(grouped_requests)):
        rows = grouped_requests[timestamp]
        if rows:
            flush()
            assert last<timestamp,"same-timestamp writes leaked into query prefix"
            if dirty:
                prepared = release.prepare(state,ledger,"refresh")
                dirty = False
            candidates = tensor([[int(r["item_idx"]) for r in rows]],state.cache.k.device)
            delta = tensor([timestamp-last],state.cache.k.device,floating=True)
            old_k,old_v,old_ordinal = state.cache.k,state.cache.v,state.writer.next_ordinal
            values = {}
            for name in PATHS:
                if name in METHODS:
                    fn = lambda name=name:release.score(current,state,candidates,delta,prepared,name)
                else:
                    cache = state.cache if name=="reuse" else exact
                    fn = lambda cache=cache:score(current,cache,candidates,delta)[0]
                values[name] = timed(fn,ledger,"service_"+name+"_read").cpu().numpy()[0]
            assert state.cache.k is old_k and state.cache.v is old_v and state.writer.next_ordinal==old_ordinal
            checks["unchanged_scoring_states"] += 1
            if canary and checks["serving_reference_comparisons"]==0:
                # FP32-cached serving input weights reproduce the existing frozen prototype reader.
                p = torch.load(FULL / f"translator_C_m{release.target}.pt",map_location=state.cache.k.device,weights_only=True)
                b,a = prepared["views"]["native"]
                reference = read_input(current,state.cache,(candidates,delta),b,a,p,prepared["counts"],prepared["active"],capture=False)[0]
                np.testing.assert_allclose(values["native"],reference.cpu().numpy()[0],atol=2e-6,rtol=1e-5)
                checks["serving_reference_comparisons"] += 1
            for j,row in enumerate(rows):
                raw.append(dict(uid=uid,target=release.target,timestamp=timestamp,request_id=row["request_id"],label=int(row["label"]),
                    item_idx=int(row["item_idx"]),count=state.cache.seq_len,writes_since_release=state.writes_since_release,
                    **{name:float(value[j]) for name,value in values.items()}))
        pending.extend(grouped_events[timestamp])
    flush()
    return raw,dict(uid=uid,target=release.target,requests=len(requests),events=len(events),
                    count=state.cache.seq_len,counts=dict(state.counts),checks=checks)


@torch.no_grad()
def main(cli):
    out = ROOT / "results/design" / cli.run_id
    out.mkdir(parents=True,exist_ok=False)
    protocol = json.loads(PROTOCOL.read_text())
    uids = protocol["development_uids"][cli.offset:cli.offset+cli.users]
    assert len(uids)==cli.users
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(17)
    device = torch.device("cuda:0")
    source_files = [Path(__file__),ROOT / "scripts/design/native_service.py",ROOT / "scripts/design/run.py",
        ROOT / "scripts/design/data.py",ROOT / "scripts/design/diagnose_native_input.py",
        ROOT / "scripts/design/diagnose_summary_objective.py",PROTOCOL,
        ROOT / "docs/design/expert_route_2026-09-07.md",*sorted((ROOT / "src/hstu_kvcache/adaptation").glob("*.py"))]
    weights = [*sorted(FULL.glob("translator_C_m*.pt")),*sorted(ABLATIONS.glob("translator_*_m*.pt")),
               *sorted((ROOT / "results/design/mechanism_aggregate_factorial192_01").glob("source_projection_mean_m*.pt"))]
    cfg = dict(vars(cli),development_uids=uids,targets=protocol["targets"],parents=protocol["parents"],paths=PATHS,
        protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),confirmation_read=False,backbone_seed=17,
        scope="independent adjacent Parent initialization; mature development users; M5 E14_partial; all paths eager",
        admissions=diagnostic_admissions(5),
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        weights_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in weights})
    write_json(out / "configuration.json",cfg)
    with tarfile.open(out / "source.tar.gz","w:gz") as archive:
        for p in source_files:
            archive.add(p,arcname=str(p.relative_to(ROOT)))
    start = time.perf_counter()
    try:
        history = histories(uids,CUTOVER_DAYS[-1]+14)
        args = SimpleNamespace(representation="ridge",write_mode="reuse",history_scope="all")
        raw,counts,ledgers,empty_ledgers,construction = [],[],{}, {},[]
        for target in protocol["targets"]:
            cutover,stop = CUTOVER_DAYS[target-1]*DAY,(CUTOVER_DAYS[target-1]+14)*DAY
            parent,current = frozen_model(target-1,device),frozen_model(target,device)
            release = NativeRelease(target,device,FULL,ABLATIONS)
            ledger,no_feedback = defaultdict(float),defaultdict(float)
            parents = timed(lambda parent=parent:cache_at_many(parent,history,[(u,cutover) for u in uids],16),ledger,"parent_initial_condition")
            exacts = timed(lambda current=current:cache_at_many(current,history,[(u,cutover) for u in uids],16),ledger,"exact_release_rebuild")
            del parent
            requests = quality_requests(uids,cutover,stop)
            for index,(uid,(parent_cache,times),(exact,_)) in enumerate(zip(uids,parents,exacts,strict=True)):
                assert parent_cache.seq_len==exact.seq_len==1024
                per_user = defaultdict(float)
                state = timed(lambda:new_state(parent_cache,times,target-1,args),per_user,"initial_summary")
                assert set(state.writer.segments[s]["producer"] for s in state.writer.segments)=={target-1}
                timed(lambda state=state:state.release(target,None),per_user,"publication_state_metadata")
                user_raw,record = replay_user(current,state,exact,history,uid,requests.get(uid,[]),cutover,stop,release,per_user,cli.canary and index==0)
                assert len(user_raw)==len(requests.get(uid,[]))
                raw.extend(user_raw)
                counts.append(record)
                for key,value in per_user.items():
                    ledger[key] += value
                    if not requests.get(uid):
                        no_feedback[key] += value
                del state
            expected=sum(map(len,requests.values()))
            actual=sum(r["target"]==target for r in raw)
            assert actual==expected
            construction.append(dict(target=target,selected_users=len(uids),feedback_users=len(requests),no_feedback_users=len(uids)-len(requests),requests=expected))
            ledgers[str(target)],empty_ledgers[str(target)] = dict(ledger),dict(no_feedback)
            pd.DataFrame([r for r in raw if r["target"]==target]).to_parquet(out / f"quality_m{target}.parquet",index=False)
            print(json.dumps(dict(target=target,status="edge_complete",users=len(uids),requests=expected,elapsed_seconds=time.perf_counter()-start)),flush=True)
            del release,current,parents,exacts
        frame = pd.DataFrame(raw)
        assert not frame.duplicated(["target","request_id"]).any()
        assert np.isfinite(frame[list(PATHS)]).all().all()
        frame.to_parquet(out / "quality_raw.parquet",index=False)
        write_json(out / "state_counts.json",counts)
        metrics = [dict(target=int(t),metrics={name:binary_metrics(f.label.to_numpy(),f[name].to_numpy()) for name in PATHS}) for t,f in frame.groupby("target")]
        write_json(out / "summary.json",dict(status="development_complete",quality=metrics,coverage=construction,
            ledger_seconds=ledgers,no_feedback_ledger_seconds=empty_ledgers,elapsed_seconds=time.perf_counter()-start,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/(1<<20),confirmation_read=False,
            no_feedback_cost_scope="per-user initialization/maintenance/read costs; batched Parent/Exact prefix costs reported for full cohort separately"))
    except Exception as exc:
        write_json(out / "summary.json",dict(status="failed",error=repr(exc),elapsed_seconds=time.perf_counter()-start))
        raise
    print((out / "summary.json").read_text(),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id",required=True)
    parser.add_argument("--offset",type=int,default=0)
    parser.add_argument("--users",type=int,required=True)
    parser.add_argument("--canary",action="store_true")
    parser.add_argument("--estimate-seconds",type=int,required=True)
    main(parser.parse_args())
