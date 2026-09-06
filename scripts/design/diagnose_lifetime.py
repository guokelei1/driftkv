#!/usr/bin/env python3
"""Locate old-state and inherited-current error on real development lifetimes.

Exact-KV replacements below are diagnostic interventions only. They never
change the method's persisted cache or supply calibration examples.
"""

# The observation closure is called synchronously and intentionally reads the
# current state of this user's loop, including its newly appended caches.
# ruff: noqa: B023

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts"))

from design.data import DAY, fixed_split, frozen_model, histories  # noqa: E402
from design.run import (  # noqa: E402
    append_event,
    append_events,
    cache_at,
    event_range,
    new_state,
    new_translator,
    response_targets,
    tensor,
    write_json,
)
from insight.candidate_shared_causal import signed_head_intervention  # noqa: E402
from insight_two.common import load_frozen_inputs, metrics_row, score_metrics  # noqa: E402

from hstu_kvcache.adaptation.reader import score  # noqa: E402
from hstu_kvcache.models import HSTUKVCache  # noqa: E402


@torch.no_grad()
def main(run_id, reference):
    out = ROOT/"results/design"/run_id
    out.mkdir(parents=True,exist_ok=False)
    trained = ROOT/"results/design"/reference
    config = json.loads((trained/"configuration.json").read_text())
    args = SimpleNamespace(**config)
    if getattr(args,"history_scope","old") != "old":
        raise ValueError("this splice decomposition requires old-only correction; a full-history map also corrects the replaced Current positions")
    uids = fixed_split()["development"][:8]
    native = getattr(args,"write_mode","shared") == "reuse"
    cfg = dict(run=run_id,users=uids,target=1,range_days=[231,245],
        snapshots_after_events=[0,64,256,512,1024,1536,"end"],
        frozen_translator_reference=str(trained.relative_to(ROOT)),
        frozen_translator_sha256=hashlib.sha256((trained/"translator_v1.pt").read_bytes()).hexdigest(),
        probes="frozen cutover64 candidates at causal post-group snapshots; no task labels",
        scope="diagnostic exact-KV interventions; never executable actions or method inputs",
        confirmation_read=False,estimated_seconds=[25,110],native_write=native,
        estimate_basis="15s data + 6s models + sparse probes; scalar writes at 11ms/event or equivalent native band128",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(out/"configuration.json",cfg)
    (out/"source.py").write_bytes(Path(__file__).read_bytes())
    start=time.perf_counter()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device("cuda:0")
    h=histories(uids,245)
    _,panels,_=load_frozen_inputs()
    previous,current=frozen_model(0,device),frozen_model(1,device)
    translator=new_translator(args,1,device)
    saved=torch.load(trained/"translator_v1.pt",map_location=device,weights_only=False)
    translator.load_state_dict(saved["state_dict"])
    translator.supported_producers=set(saved["supported_producers"])
    rows,raw=[],[]
    for index,uid in enumerate(uids):
        cache,times=cache_at(previous,h,uid,231*DAY)
        exact,_=cache_at(current,h,uid,231*DAY)
        state=new_state(cache,times,0,args)
        state.release(1,translator)
        reuse=cache
        last=int(times[-1])
        candidates=tensor(panels[0,index][None],device)

        def observe(ordinal,timestamp):
            if state.view_dirty:
                state.refresh()
            delta=tensor([timestamp-last],device,floating=True)
            old=state.writer.old_mask(1)
            reference=score(current,exact,candidates,delta)[0]
            baseline=score(current,reuse,candidates,delta)[0]
            learned,trace=score(current,state.cache,candidates,delta,state.source,state.translated,trace=True)
            # Copy diagnostic values; ordinary persistent state is untouched.
            hybrid=HSTUKVCache(torch.where(old[None,None,:,None],exact.k,state.cache.k),
                torch.where(old[None,None,:,None],exact.v,state.cache.v),state.cache.seq_len)
            old_oracle=score(current,hybrid,candidates,delta)[0]
            old_shared=signed_head_intervention(current,hybrid,state.cache,candidates,delta,mode="shared_only").scores
            full_shared=signed_head_intervention(current,exact,state.cache,candidates,delta,mode="shared_only").scores
            current_hybrid=HSTUKVCache(torch.where(old[None,None,:,None],state.cache.k,exact.k),
                torch.where(old[None,None,:,None],state.cache.v,exact.v),state.cache.seq_len)
            current_splice=score(current,current_hybrid,candidates,delta,state.source,state.translated)[0]
            outputs=dict(learned=learned,exact_old_intervention=old_oracle,
                shared_old_response_oracle=old_shared,exact_current_with_learned_old_intervention=current_splice)
            outputs["shared_full_response_oracle"] = full_shared
            for name,value in outputs.items():
                rows.append(dict(uid=uid,appends=ordinal,old_events=int(old.sum()),method=name,
                                 **metrics_row(score_metrics(reference,baseline,value))))
            record=dict(uid=uid,appends=ordinal,old_events=int(old.sum()),timestamp=timestamp,
                exact=reference.cpu().tolist(),reuse=baseline.cpu().tolist(),
                outputs={key:value.cpu().tolist() for key,value in outputs.items()})
            if bool(old.any()):
                changes=response_targets(current,trace,state.cache,exact,old)
                record["old_response_relative_mse_by_layer"]=[float((got-wanted).square().sum()/wanted.square().sum().clamp_min(1e-20))
                    for got,wanted in zip(trace.corrections,changes,strict=True)]
            raw.append(record)

        observe(0,231*DAY)
        groups=defaultdict(list)
        for event in event_range(h,uid,231*DAY,245*DAY):
            groups[event[0]].append(event)
        thresholds=[64,256,512,1024,1536]
        ordinal=0
        pending=[]
        for timestamp,events in sorted(groups.items()):
            ordinal+=len(events)
            if native:
                pending.extend(events)
                if thresholds and ordinal>=thresholds[0]:
                    state=append_events(current,state,pending,last,128)
                    exact=append_events(current,exact,pending,last,128)
                    reuse=append_events(current,reuse,pending,last,128)
                    last=timestamp
                    pending=[]
            else:
                for event in events:
                    state=append_event(current,state,event,last)
                    exact=append_event(current,exact,event,last)
                    reuse=append_event(current,reuse,event,last)
                    last=timestamp
            if thresholds and ordinal>=thresholds[0]:
                observe(ordinal,timestamp+1)
                thresholds=[value for value in thresholds if value>ordinal]
        if pending:
            state=append_events(current,state,pending,last,128)
            exact=append_events(current,exact,pending,last,128)
            reuse=append_events(current,reuse,pending,last,128)
            last=pending[-1][0]
        observe(ordinal,245*DAY)
        print(json.dumps(dict(phase="lifetime",uid=uid,events=ordinal,observations=len(raw))),flush=True)
    write_json(out/"raw.json",raw)
    write_json(out/"summary.json",dict(status="diagnostic_complete",rows=rows,
        elapsed_seconds=time.perf_counter()-start,peak_allocated_mib=torch.cuda.max_memory_allocated()/(1<<20),
        configuration_sha256=hashlib.sha256((out/"configuration.json").read_bytes()).hexdigest()))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id",required=True)
    parser.add_argument("--reference",required=True)
    cli=parser.parse_args()
    main(cli.run_id,cli.reference)
