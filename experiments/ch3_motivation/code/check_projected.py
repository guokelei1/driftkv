#!/usr/bin/env python3
"""Real full-window equivalence and isolated-path timing of cached h row maps.

Loads bounded diagnostic weights; never fits a model or computes quality AUC.
Run only after other jobs release the selected GPU. Original and projected
paths use identical frozen weights, inputs, caches and candidate self reads.
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback


import numpy as np
import torch

from experiments.ch3_motivation.code.common import ROOT, load_edge, load_models, write_json
from experiments.ch3_motivation.code.check_probes import differences, check_prediction, measure
from experiments.ch3_motivation.code.check_probe_variants import h_output_status
from experiments.ch3_motivation.code.prepare_candidates import candidates_for
from experiments.ch3_motivation.code.probe_fast import GraphProbePool, FastProbeTrajectory, pad_features
from experiments.ch3_motivation.code.probe_projected_fast import (
    project_row_features, ProjectedGraphPool, ProjectedFastProbeTrajectory,
    projected_row_preparation_flops, projected_read_extra_flops,
)
from experiments.ch3_motivation.code.probe_projected_cost import (
    full_projected_preparation, deployment_projected_graph_setup,
)
from experiments.ch3_motivation.code.probe_ledger import full_probe_ledger
from experiments.ch3_motivation.code.probe_variants import load_variant
from hstu_kvcache.adaptation.reader import score as native_score
from hstu_kvcache.evaluation.trajectory import NativeTrajectory
from hstu_kvcache.models import HSTUKVCache
from hstu_kvcache.common.artifacts import sha256


def compare(actual,wanted,*,logits=False):
    result=differences(actual,wanted)
    if (result["max_abs"]>3e-5 if logits else result["relative_rms"]>2e-5):
        raise AssertionError(result)
    return result


def paired_measure(original,projected,repeats):
    """Alternate ordering and include synchronization/input copies in each call."""
    functions=dict(original=original,projected=projected)
    observations={name:[] for name in functions}
    memory={name:[] for name in functions}
    for function in functions.values():
        value=function();del value
    torch.cuda.synchronize()
    for repeat in range(repeats):
        order=("original","projected") if repeat%2==0 else ("projected","original")
        for name in order:
            torch.cuda.synchronize()
            baseline=torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            started=time.perf_counter()
            value=functions[name]()
            torch.cuda.synchronize()
            observations[name].append(time.perf_counter()-started)
            memory[name].append(dict(baseline_allocated_bytes=baseline,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved()))
            del value
    result={name:dict(seconds=values,median_seconds=float(np.median(values)),memory=memory[name])
            for name,values in observations.items()}
    result["original_over_projected_median_ratio"]=result["original"]["median_seconds"]/result["projected"]["median_seconds"]
    result["scope"]="alternating isolated calls; both pools/backbones resident; input copies included; not serving throughput"
    return result


def to_device(cache,device):
    return HSTUKVCache(cache.k.to(device),cache.v.to(device),cache.seq_len)


def sample_from_events(parent,events,cfg,candidates,device):
    initial=NativeTrajectory(parent,cfg["edge"]-1,events,cfg["release_day"]*86400,
        max_length=cfg["history_length"],append_band_size=cfg["append_band_size"])
    return dict(parent_cache=to_device(initial.cache,"cpu"),
        item_ids=torch.tensor(initial.retained_events[1].copy(),dtype=torch.long)[None],
        candidates=torch.tensor([candidates],dtype=torch.long),
        query_delta=torch.tensor([cfg["release_day"]*86400-initial.last_timestamp],dtype=torch.float32))


@torch.no_grad()
def check_one(parent,model,probe,samples,uids,histories,by_user,cfg,args):
    device=next(model.parameters()).device
    cap=cfg["history_length"]
    full=[uid for uid in uids if samples[uid]["parent_cache"].seq_len==cap]
    active=[uid for uid in full if len(histories.rows[uid][0])-int(np.searchsorted(
        histories.rows[uid][0],cfg["release_day"]*86400,side="left"))>=3]
    if not active or len(full)<2:
        raise ValueError("bounded users require two full windows, one with at least three future behaviors")
    uid=active[0]
    cpu=samples[uid]
    source=to_device(cpu["parent_cache"],device)
    ids=cpu["item_ids"].to(device)
    candidates=cpu["candidates"].to(device)
    delta=cpu["query_delta"].to(device)
    features=probe.prepare_read_features(model,source,ids)
    cache=probe.prepare_cache(model,source,ids,read_features=features)
    raw_rows=pad_features(features,cap)
    projected_rows=pad_features(project_row_features(probe,features),cap)
    original=GraphProbePool(model,probe,window=cap)
    projected=ProjectedGraphPool(model,probe,window=cap)
    graphs={q:(original.get("score",q),projected.get("score",q)) for q in (1,16)}
    old_append,new_append=original.get("append",1),projected.get("append",1)
    events=histories.rows[uid]
    cursor=int(np.searchsorted(events[0],cfg["release_day"]*86400,side="left"))
    item=torch.tensor([[int(events[1][cursor])]],device=device)
    behavior=torch.tensor([[int(events[2][cursor])]],device=device)
    behavior_delta=torch.ones(1,1,device=device)
    count_cases=[]
    for count in sorted(set((0,1,2,16,cap//2,cap-1,cap))):
        for q,(old_graph,new_graph) in graphs.items():
            wanted,=old_graph.run(cache,raw_rows,count,candidates[:,:q],delta)
            actual,=new_graph.run(cache,projected_rows,count,candidates[:,:q],delta)
            count_cases.append(dict(inherited_rows=count,queries=q,logits=compare(actual,wanted,logits=True)))
        old_k,old_v,_=old_append.run(cache,raw_rows,count,item,behavior_delta,behavior)
        new_k,new_v,_=new_append.run(cache,projected_rows,count,item,behavior_delta,behavior)
        count_cases[-1]["append"]=dict(k=compare(new_k,old_k),v=compare(new_v,old_v))
    # Every prefix count uses the same already-captured Q1 append graph. Only
    # the seven declared counts above run both old/new numerical oracles.
    all_counts_started=time.perf_counter()
    for count in range(1,cap+1):
        new_append.run(cache,projected_rows,count,item,behavior_delta,behavior)
    torch.cuda.synchronize()
    all_counts_seconds=time.perf_counter()-all_counts_started
    if len(projected.graphs)!=3 or any(graph.capture_forwards!=1 for graph in projected.graphs.values()):
        raise AssertionError("inherited count triggered a projected graph recapture")

    old_score,new_score=graphs[16]
    old_score.run(cache,raw_rows,cap,candidates,delta)
    new_score.run(cache,projected_rows,cap,candidates,delta)
    old_trace=native_score(model,cache,candidates,delta,history_read_fn=old_score._read,trace=True)
    new_trace=native_score(model,cache,candidates,delta,history_read_fn=new_score._read,trace=True)
    traced=check_prediction(new_trace,old_trace)
    del old_trace,new_trace

    # A strict three-write FIFO check on the same three real behavior events.
    old_cache=HSTUKVCache(cache.k.clone(),cache.v.clone(),cap)
    new_cache=HSTUKVCache(cache.k.clone(),cache.v.clone(),cap)
    old_rows,new_rows=raw_rows.clone(),projected_rows.clone()
    remaining=cap
    last=int(events[0][cursor-1])
    fifo=[]
    for position in range(cursor,cursor+3):
        item=torch.tensor([[int(events[1][position])]],device=device)
        behavior=torch.tensor([[int(events[2][position])]],device=device)
        gap=min(7*86400,max(0,int(events[0][position])-last));last=int(events[0][position])
        behavior_delta=torch.tensor([[gap]],device=device,dtype=torch.float32)
        old_k,old_v,old_rows=old_append.run(old_cache,old_rows,remaining,item,behavior_delta,behavior)
        new_k,new_v,new_rows=new_append.run(new_cache,new_rows,remaining,item,behavior_delta,behavior)
        old_cache=HSTUKVCache(old_k,old_v,cap);new_cache=HSTUKVCache(new_k,new_v,cap)
        remaining-=1
        a,=old_score.run(old_cache,old_rows,remaining,candidates,delta)
        b,=new_score.run(new_cache,new_rows,remaining,candidates,delta)
        fifo.append(dict(timestamp=last,k=compare(new_cache.k,old_cache.k),v=compare(new_cache.v,old_cache.v),
                         logits=compare(b,a,logits=True)))

    # Whole bounded real-time prefixes exercise publication, short-history
    # fallback, repeated own-cache writes, and graph reuse across all users.
    prefixes=[]
    for current_uid in uids:
        current_events=histories.rows[current_uid]
        initial=NativeTrajectory(parent,cfg["edge"]-1,current_events,cfg["release_day"]*86400,
            max_length=cap,append_band_size=cfg["append_band_size"])
        old=FastProbeTrajectory(parent,model,probe,current_events,cfg,pool=original,initial=initial)
        new=ProjectedFastProbeTrajectory(parent,model,probe,current_events,cfg,pool=projected,initial=initial)
        old._prepare();new._prepare()
        stop=min(len(current_events[0]),initial.cursor+args.prefix_behaviors)
        timestamps=sorted(set([cfg["release_day"]*86400]+[int(t)+1 for t in current_events[0][initial.cursor:stop]]))
        user_checks=[]
        for until in timestamps:
            old.advance(until);new.advance(until)
            previous=new.cache.k.clone()
            a=old.score(candidates,delta);b=new.score(candidates,delta)
            user_checks.append(dict(until=until,history_length=new.cache.seq_len,inherited_rows=new.source.seq_len,
                logits=compare(b,a,logits=True),k=compare(new.cache.k,old.cache.k),v=compare(new.cache.v,old.cache.v)))
            if not torch.equal(previous,new.cache.k):
                raise AssertionError("candidate read changed prepared persistent cache")
        prefixes.append(dict(uid=current_uid,checks=user_checks,original_execution=old.report(),projected_execution=new.report()))
        del old,new,initial

    # Publish/preparation time is separate from already prepared steady calls.
    preparation=measure(lambda:project_row_features(probe,features),args.repeats)
    raw_cpu=raw_rows.cpu();projected_cpu=projected_rows.cpu()
    cache_cpu=to_device(cache,"cpu")
    timings={}
    for q,(old_graph,new_graph) in graphs.items():
        def resident_old():
            return old_graph.run(cache,raw_rows,cap,candidates[:,:q],delta)[0].cpu()
        def resident_new():
            return new_graph.run(cache,projected_rows,cap,candidates[:,:q],delta)[0].cpu()
        def cpu_old():
            return old_graph.run(to_device(cache_cpu,device),raw_cpu.to(device),cap,
                cpu["candidates"][:,:q].to(device),cpu["query_delta"].to(device))[0].cpu()
        def cpu_new():
            return new_graph.run(to_device(cache_cpu,device),projected_cpu.to(device),cap,
                cpu["candidates"][:,:q].to(device),cpu["query_delta"].to(device))[0].cpu()
        timings[f"score_q{q}"]=dict(resident_including_graph_input_copy_and_score_export=paired_measure(resident_old,resident_new,args.repeats),
            cpu_state_transfer_graph_input_copy_and_score_export=paired_measure(cpu_old,cpu_new,args.repeats))
    timings["append_q1"]=paired_measure(
        lambda:old_append.run(cache,raw_rows,cap,item,behavior_delta,behavior),
        lambda:new_append.run(cache,projected_rows,cap,item,behavior_delta,behavior),args.repeats)

    def first_score(project):
        moved=to_device(cpu["parent_cache"],device)
        moved_ids=cpu["item_ids"].to(device)
        rows=probe.prepare_read_features(model,moved,moved_ids)
        prepared=probe.prepare_cache(model,moved,moved_ids,read_features=rows)
        row_inputs=pad_features(project_row_features(probe,rows) if project else rows,cap)
        graph=new_score if project else old_score
        return graph.run(prepared,row_inputs,cap,candidates,delta)[0].cpu()
    timings["cpu_source_publication_prepare_first_score"]=paired_measure(lambda:first_score(False),lambda:first_score(True),args.repeats)
    complete=full_probe_ledger(histories,by_user,uids,cfg,stage=probe.stage,padded=True)
    return dict(benchmark_uid=uid,stage=probe.stage,config=asdict(probe.cfg),
        h_output=h_output_status(probe,model,dict(parent_cache=source,item_ids=ids,candidates=candidates,query_delta=delta)),
        dynamic_mask_cases=count_cases,all_1_to_window_projected_append_seconds=all_counts_seconds,
        dynamic_counts_do_not_recapture=True,full_read_trace=traced,three_real_fifo_appends=fifo,
        bounded_real_behavior_prefixes=prefixes,row_projection_preparation=preparation,timings=timings,
        row_projection_preparation_flops=projected_row_preparation_flops(probe,cap),
        projected_score16_extra_flops=projected_read_extra_flops(probe,cap,16),
        original_graphs=original.report(),projected_graphs=projected.report(),
        complete_e14_projection_preparation=full_projected_preparation(probe,histories,by_user,uids,cfg),
        complete_e14_projected_graph_setup=deployment_projected_graph_setup(probe,complete,model,window=cap),
        auxiliary_state_scope="one active GPU user during quality replay; row table not low rank; include raw source/features and projected table, plus shared graph buffers; logical bytes are distinct from measured co-resident allocator peak")


def run(args):
    if not 1<=args.users<=16 or not 1<=args.prefix_behaviors<=16:
        raise ValueError("bounded diagnostic permits 1–16 users and prefix behaviors")
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    torch.cuda.set_device(0)
    device=torch.device("cuda:0")
    started=time.perf_counter()
    args.output.mkdir(parents=True,exist_ok=True)
    files=[Path(__file__),*(Path(__file__).with_name(name) for name in (
        "probe_projected_fast.py","probe_projected_cost.py","probe_variants.py","probe_fast.py",
        "probe_replay.py","probe_cost.py","probe_ledger.py","probes.py","common.py",
        "check_probes.py","check_probe_variants.py","prepare_candidates.py")),ROOT/"src/hstu_kvcache/adaptation/reader.py"]
    sources={str(path.relative_to(ROOT)):sha256(path) for path in files}
    initial_probe,metadata=load_variant(args.checkpoint[0])
    edge=metadata["identity"]["edge"]
    parent,model,models=load_models(edge,device,backend="torch")
    binding,histories,by_user=load_edge(args.input_root/f"edge_{edge}")
    cfg=binding["settings"]
    population_path=args.input_root/"population.json"
    if binding["population_sha256"]!=sha256(population_path):
        raise ValueError("population changed")
    for name in ("history","requests"):
        if sha256(args.input_root/f"edge_{edge}"/binding[name]["path"])!=binding[name]["sha256"]:
            raise ValueError("frozen input pack changed")
    candidate_path=args.input_root/f"edge_{edge}/calibration_feedback_candidates.json"
    candidate_data=json.loads(candidate_path.read_text())
    if candidate_data["population_sha256"]!=sha256(population_path):
        raise ValueError("candidate roles changed")
    uids=(metadata["identity"]["fit_uids"]+metadata["identity"]["validation_users"])[:args.users]
    if len(uids)!=args.users or len(set(uids))!=len(uids):
        raise ValueError("checkpoint roles do not supply the requested distinct bounded diagnostic users")
    samples={}
    for uid in uids:
        values=candidates_for(uid,candidate_data["recent_items"][str(uid)],model.cfg.num_items-256,
            count=16,recent_fraction=.5,seed=17)
        samples[uid]=sample_from_events(parent,histories.rows[uid],cfg,values,device)
    identity=dict(edge=edge,users=uids,physical_gpu=args.gpu,hardware=torch.cuda.get_device_name(0),models=models,
        checkpoints={str(path):sha256(path) for path in args.checkpoint},sources=sources,
        input_binding_sha256=sha256(args.input_root/f"edge_{edge}/inputs.json"),
        purpose="same-weight execution equivalence and isolated-path performance only; no fitting or quality AUC")
    write_json(args.output/"configuration.json",identity)
    result=dict(identity=identity,checks=[])
    for index,path in enumerate(args.checkpoint):
        probe,fit=(initial_probe,metadata) if index==0 else load_variant(path)
        if index==0:
            initial_probe=None
        if fit["identity"]["edge"]!=edge or fit["identity"]["models"]!=models or probe.stage!="fgh":
            raise ValueError("all diagnostic checkpoints must be fgh for the same edge/backbones")
        probe=probe.to(device).eval()
        checks=check_one(parent,model,probe,samples,uids,histories,by_user,cfg,args)
        result["checks"].append(dict(checkpoint=str(path),checkpoint_sha256=sha256(path),
            checkpoint_diagnostic_only=fit.get("diagnostic_only",False),quality_use_permitted=False,**checks))
        write_json(args.output/"progress.json",result)
        print(json.dumps(dict(status="projected_checkpoint_passed",checkpoint=str(path),
            score16_ratio=checks["timings"]["score_q16"]["resident_including_graph_input_copy_and_score_export"]["original_over_projected_median_ratio"],
            append_ratio=checks["timings"]["append_q1"]["original_over_projected_median_ratio"])),flush=True)
        del probe
        gc.collect();torch.cuda.empty_cache()
    for path in files:
        if sha256(path)!=sources[str(path.relative_to(ROOT))]:
            raise ValueError("numerical source changed during projected diagnostic")
    result.update(status="passed",quality_auc_computed=False,seconds=time.perf_counter()-started,
        scope="bounded real fit/validation users; numerical execution check and path microbenchmarks; no AUC or online throughput")
    write_json(args.output/"summary.json",result)
    print(json.dumps(dict(status="passed",output=str(args.output),seconds=result["seconds"])),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root",type=Path,default=ROOT/"experiments/ch3_motivation/inputs")
    p.add_argument("--checkpoint",type=Path,nargs="+",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--gpu",type=int,default=1)
    p.add_argument("--users",type=int,default=8)
    p.add_argument("--prefix-behaviors",type=int,default=8)
    p.add_argument("--repeats",type=int,default=6)
    args=p.parse_args()
    try:
        run(args)
    except Exception as error:
        write_json(args.output/"failure.json",dict(status="failed",error=repr(error),traceback=traceback.format_exc(),
            quality_auc_computed=False,scope="retain failed numerical/performance diagnostic; do not relax tolerances from outcomes"))
        raise
