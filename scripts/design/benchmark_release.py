#!/usr/bin/env python3
"""Publication costs on real mixed-producer development states.

Includes installed views and an explicitly extrapolated history-weighted
Exact denominator. No feedback or confirmation model outputs are used.
"""

import argparse
import hashlib
import json
import os
import sys
import time
import tracemalloc
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts"))

from design.data import DAY, fixed_split, frozen_model, histories, prefix  # noqa: E402
from design.run import (  # noqa: E402
    append_events,
    cache_at,
    event_range,
    new_state,
    new_translator,
    tensor,
    write_json,
)
from insight_two.common import CUTOVER_DAYS, load_frozen_inputs  # noqa: E402

from hstu_kvcache.adaptation.inference import exact_tensors  # noqa: E402
from hstu_kvcache.adaptation.reader import score  # noqa: E402
from hstu_kvcache.adaptation.state import release_many  # noqa: E402
from hstu_kvcache.models import HSTUKVCache  # noqa: E402


@torch.no_grad()
def main(cli):
    out=ROOT/"results/design"/cli.run_id
    out.mkdir(parents=True,exist_ok=False)
    trained=ROOT/"results/design"/cli.reference
    config=json.loads((trained/"configuration.json").read_text())
    args=SimpleNamespace(**config)
    uids=fixed_split()["development"][:32]
    sources=[Path(__file__),ROOT/"scripts/design/run.py",*[ROOT/"src/hstu_kvcache/adaptation"/name
        for name in ("translator.py","reader.py","state.py","summary.py","inference.py")]]
    write_json(out/"configuration.json",dict(reference=str(trained.relative_to(ROOT)),users=uids,
        targets=[1,2,3,4,5],measurement_target=5,batch_sizes=[1,16],repetitions=5,
        publication_repetitions=100,source_backfill_repetitions=20,
        expected_seconds=[90,600] if cli.compile_exact else [30,120],
        compiled_exact=cli.compile_exact,torch_version=torch.__version__,cxx=os.environ.get("CXX","g++"),
        compilation_scope="first compiled shape calls and all measurement warmup retained separately; disk kernel cache may be reused",
        scope="installed publication and history-weighted compute extrapolation; no population or online-latency qualification",
        confirmation_read=False,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}))
    for p in sources:
        (out/p.name).write_bytes(p.read_bytes())
    start=time.perf_counter()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device("cuda:0")
    h=histories(uids,288)
    current=frozen_model(0,device)
    states,initial_sources=[],[]
    for uid in uids:
        cache,times=cache_at(current,h,uid,231*DAY)
        initial_sources.append((cache,times))
        states.append(new_state(cache,times,0,args))
    backfill_times=[]
    for _ in range(20):
        torch.cuda.synchronize()
        begin=time.perf_counter()
        temporary=[new_state(cache,times,0,args) for cache,times in initial_sources]
        torch.cuda.synchronize()
        backfill_times.append(time.perf_counter()-begin)
        del temporary
    initial_backfill=sum(backfill_times)/len(backfill_times)
    # Inspect actual Python metadata separately from the uninstrumented timing.
    # GPU tensor payload and population RSS are outside tracemalloc's scope.
    tracemalloc.start()
    allocated_before=tracemalloc.get_traced_memory()[0]
    temporary=[new_state(cache,times,0,args) for cache,times in initial_sources]
    allocated_now,allocated_peak=tracemalloc.get_traced_memory()
    python_allocation=dict(users=len(states),retained_bytes=allocated_now-allocated_before,
        peak_bytes=allocated_peak-allocated_before,
        scope="actual Python allocations for32 resident initial states; GPU payload excluded; not population RSS")
    tracemalloc.stop()
    del temporary
    del initial_sources
    for target in range(1,5):
        current=frozen_model(target,device)
        for uid,state in zip(uids,states,strict=True):
            state.release(target,None)
            events=event_range(h,uid,CUTOVER_DAYS[target-1]*DAY,CUTOVER_DAYS[target]*DAY)
            if events:
                append_events(current,state,events,int(state.writer.events[-1][3]),128)
    current=frozen_model(5,device)
    translator=new_translator(args,5,device)
    saved=torch.load(trained/"translator_v5.pt",map_location=device,weights_only=False)
    translator.load_state_dict(saved["state_dict"])
    translator.supported_producers=set(saved["supported_producers"])
    for state in states:
        state.release(5,translator)
    features=torch.stack([translator.features(state.source) for state in states])
    packed,counts=translator.writer_features([state.writer for state in states],[0]*len(states))
    torch.testing.assert_close(packed,features,atol=2e-5,rtol=2e-5)
    changes=translator.rates(packed)*counts[:,None,None]
    time_changes=translator.time_coefficients(packed)
    if time_changes is not None:
        time_changes=time_changes*counts[:,None,None,None]
    _,panels,_=load_frozen_inputs()
    max_logit=0.
    for index,state in enumerate(states):
        ids=tensor(panels[4,index][None],device)
        delta=tensor([287*DAY-state.writer.events[-1][3]],device,floating=True)
        ordinary=state.score(current,ids,delta)
        fused=score(current,state.cache,ids,delta,response_delta=changes[index:index+1],
                    response_time_delta=None if time_changes is None else time_changes[index:index+1])[0]
        torch.testing.assert_close(fused,ordinary,atol=2e-4,rtol=2e-5)
        max_logit=max(max_logit,float((fused-ordinary).abs().max()))
    inputs=[prefix(h,uid,287*DAY,device) for uid in uids]
    assert all(value[0].shape[1]==1024 for value in inputs)
    compiled=(torch.compile(exact_tensors,mode="reduce-overhead",dynamic=True,fullgraph=True)
              if cli.compile_exact else None)
    compilation_checks=[]
    warmup_seconds=0.

    def exact_call(items,behaviors,deltas,lengths=None):
        if compiled is None:
            return current.compute_kv(items,behaviors,deltas,lengths=lengths)
        torch.compiler.cudagraph_mark_step_begin()
        values=compiled(current,items,behaviors,deltas,lengths)
        # Exact state must survive the next user's graph invocation as well.
        k,v=(value.clone() for value in values)
        return HSTUKVCache(k,v,items.shape[1])

    def check_exact(values,lengths=None):
        if compiled is None:
            return
        expected=current.compute_kv(*values,lengths=lengths)
        torch.cuda.synchronize()
        begin=time.perf_counter()
        actual=exact_call(*values,lengths=lengths)
        torch.cuda.synchronize()
        seconds=time.perf_counter()-begin
        differences={}
        for field in ("k","v"):
            left,right=getattr(actual,field),getattr(expected,field)
            torch.testing.assert_close(left,right,atol=2e-5,rtol=2e-5)
            differences[field]=float((left-right).abs().max())
        row=dict(batch=values[0].shape[0],length=values[0].shape[1],
                 padded_mask=lengths is not None,first_call_seconds=seconds,max_difference=differences)
        compilation_checks.append(row)
        print(json.dumps(dict(phase="exact_compilation_check",**row)),flush=True)

    def measure(fn,repetitions=5):
        nonlocal warmup_seconds
        begin=time.perf_counter()
        fn()
        torch.cuda.synchronize()
        warmup_seconds+=time.perf_counter()-begin
        begin=time.perf_counter()
        for _ in range(repetitions):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter()-begin)/repetitions

    scalar=measure(lambda:[state.release(5,translator) for state in states])
    rows=[]
    for batch in (1,16):
        def install_views(batch=batch):
            for offset in range(0,len(states),batch):
                part=states[offset:offset+batch]
                release_many(part,5,translator)

        def exact_all(batch=batch):
            values=[]
            for offset in range(0,len(inputs),batch):
                part=inputs[offset:offset+batch]
                values.append(exact_call(*[torch.cat([p[i] for p in part]) for i in range(3)]))
            return values
        check_exact([torch.cat([p[i] for p in inputs[:batch]]) for i in range(3)])
        rows.append(dict(batch=batch,users=len(states),installed_publication_seconds=measure(install_views,100),exact_all_seconds=measure(exact_all)))
    # Only cost fixtures use shorter prefixes: these are untouched real input
    # tokens, never perturbed K/V or quality examples. Dense native compute with
    # padding masks has the same shape cost as these equal-length batches.
    # A fixed 32-event bucket scheme limits padding; include each final partial
    # batch conservatively as a full batch. This is an extrapolation, not a
    # measurement of population I/O or a new population execution result.
    population_path=ROOT/"results/design/analysis/population_history_cost_probe.json"
    population=json.loads(population_path.read_text())
    curve=[]
    for length in range(32,1025,32):
        batch_inputs=[torch.cat([p[i][:,:length] for p in inputs[:16]]) for i in range(3)]
        lengths=torch.full((16,),length,device=device,dtype=torch.long)
        check_exact(batch_inputs,lengths)
        seconds=measure(lambda batch_inputs=batch_inputs,lengths=lengths:exact_call(*batch_inputs,lengths=lengths))
        curve.append(dict(padded_length=length,batch=16,seconds=seconds))
    curve_by_length={r["padded_length"]:r["seconds"] for r in curve}
    estimates=[]
    for row in population["rows"]:
        buckets={length:0 for length in curve_by_length}
        for length,count in row["exact_length_counts"].items():
            buckets[((int(length)+31)//32)*32]+=count
        estimate=sum(((count+15)//16)*curve_by_length[length] for length,count in buckets.items())
        estimates.append(dict(target=row["target"],users=row["users"],
            actual_retained_events=row["total_retained_events"],
            padded_batch_events=sum(((count+15)//16)*16*length for length,count in buckets.items()),
            exact_compute_seconds=estimate))
    calibration=json.loads((trained/"summary.json").read_text())["ledger_seconds"]
    preparation_keys=("teacher_build","teacher_query","teacher_replay","translator_fit",
        "calibration_source_backfill","calibration_adjacent_source","calibration_lifetime_source",
        "calibration_lineage_replay","calibration_lifetime_replay","calibration_query_selection","release_selection")
    preparation={key:calibration.get(key,0) for key in preparation_keys}
    publication_per_user=rows[-1]["installed_publication_seconds"]/len(states)
    population_publication=publication_per_user*30000*len(estimates)
    denominator=sum(row["exact_compute_seconds"] for row in estimates)
    write_json(out/"summary.json",dict(status="diagnostic_complete",rows=rows,
        original_scalar_state_release_seconds=scalar,max_feature_difference=float((packed-features).abs().max()),
        max_logit_difference=max_logit,elapsed_seconds=time.perf_counter()-start,
        source_backfill=dict(users=len(uids),retained_events_per_user=1024,
            measured_seconds=backfill_times,mean_seconds=initial_backfill,
            python_allocation_probe=python_allocation,
            fixed_population_full_window_proxy_seconds=initial_backfill/len(uids)*30000,
            scope="resident-GPU full-window source plus Python event metadata; fixed-population extrapolation, no population I/O or allocator scaling measurement"),
        compiled_exact=cli.compile_exact,exact_compilation_checks=compilation_checks,
        exact_first_compiled_calls_seconds=sum(row["first_call_seconds"] for row in compilation_checks),
        measurement_warmup_seconds=warmup_seconds,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/(1<<20),
        history_weighted_compute=dict(population_metadata_sha256=hashlib.sha256(population_path.read_bytes()).hexdigest(),
            curve=curve,release_estimates=estimates,calibration_preparation_seconds=preparation,
            five_release_publication_seconds=population_publication,
            five_release_exact_seconds=denominator,
            preparation_plus_publication_ratio=(sum(preparation.values())+population_publication)/denominator,
            preparation_publication_and_full_window_backfill_proxy_ratio=(sum(preparation.values())+population_publication
                +initial_backfill/len(uids)*30000)/denominator),
        scope="measured M5 resident-GPU kernels plus explicit fixed-population extrapolation; calibration/source/replay fully retained",
        limitations=["M5 shape timing reused across identical frozen backbone architecture; no five-model timing replication",
            "publication rate from 32 real mixed-state users at M5; not measured 30000-user publication",
            "all final partial Exact batches charged as full batch16; padding up to31 events/user",
            "source backfill reported separately as a full-window proxy; population I/O, storage and service overhead unmeasured; no 20-percent qualification"],
        confirmation_read=False))
    print((out/"summary.json").read_text())


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id",required=True)
    parser.add_argument("--reference",required=True)
    parser.add_argument("--compile-exact",action="store_true")
    main(parser.parse_args())
