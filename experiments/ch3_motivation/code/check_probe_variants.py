#!/usr/bin/env python3
"""Bounded full-window numerical/performance check of stronger E5 probes.

This entry never computes AUC or selects an evaluation population. It uses
only the first frozen fit/validation users and records both variants' actual
matrix geometry, rather than treating equal hidden width as equal capacity.
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
import copy
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import sys
import time


import numpy as np
import torch

from experiments.ch3_motivation.code.common import ROOT, capture_samples, load_edge, load_models, write_json
from experiments.ch3_motivation.code.check_probes import differences, check_prediction, measure
from experiments.ch3_motivation.code.prepare_candidates import candidates_for
from experiments.ch3_motivation.code.probe_cost import (
    cached_preparation_flops, padded_read_extra_flops, deployment_graph_setup,
)
from experiments.ch3_motivation.code.probe_fast import GraphProbePool, FastProbeTrajectory, pad_features
from experiments.ch3_motivation.code.probe_ledger import full_probe_ledger
from experiments.ch3_motivation.code.probe_variants import build_variant, save_variant
from experiments.ch3_motivation.code.probes import bind_fit_scales, fit_probe, sample_to_device
from hstu_kvcache.adaptation.reader import score as native_score
from hstu_kvcache.models import HSTUKVCache
from hstu_kvcache.common.artifacts import sha256


def nonzero_outputs(probe,seed=17022):
    torch.manual_seed(seed)
    with torch.no_grad():
        for term in (probe.f,probe.g,probe.h):
            if term is not None:
                for layer in term:
                    weight=(layer.weights[-1] if hasattr(layer,"weights") else layer.network[-1].weight)
                    weight.normal_(std=2e-4)


def assert_close(actual,wanted,*,logits=False):
    result=differences(actual,wanted)
    if result["relative_rms"]>3e-5 or (logits and result["max_abs"]>3e-5):
        raise AssertionError(result)
    return result


@torch.no_grad()
def graph_checks(model,probe,sample,second,events,cfg,repeats):
    window=cfg["history_length"]
    features=probe.prepare_read_features(model,sample["parent_cache"],sample["item_ids"])
    cache=probe.prepare_cache(model,sample["parent_cache"],sample["item_ids"],read_features=features)
    padded=pad_features(features,window)
    pool=GraphProbePool(model,probe,window=window)
    score_graph=pool.get("score",sample["candidates"].shape[1])
    append_graph=pool.get("append",1)
    at=int(np.searchsorted(events[0],cfg["release_day"]*86400,side="left"))
    if len(events[0])-at<3:
        raise ValueError("the selected fitting diagnostic user needs at least three post-release behaviors")
    device=cache.k.device
    item=torch.tensor([[int(events[1][at])]],device=device)
    behavior=torch.tensor([[int(events[2][at])]],device=device)
    behavior_delta=torch.ones(1,1,device=device)
    cases=[]
    for count in sorted(set((0,1,2,16,window//2,window-1,window))):
        if count>window:
            continue
        source=HSTUKVCache(sample["parent_cache"].k[:,:,:count],sample["parent_cache"].v[:,:,:count],count)
        ids=sample["item_ids"][:,:count]
        fs=tuple(row[:,:,:count] for row in features)
        actual,=score_graph.run(cache,padded,count,sample["candidates"],sample["query_delta"])
        wanted,_=probe.score(model,source,ids,sample["candidates"],sample["query_delta"],
                             prepared_cache=cache,read_features=fs)
        logits=assert_close(actual,wanted,logits=True)
        k,v,_=append_graph.run(cache,padded,count,item,behavior_delta,behavior)
        expected,_,_,_=probe.append(model,cache,source,ids,item,behavior,behavior_delta,
                                    max_length=window,read_features=fs)
        cases.append(dict(inherited_rows=count,logits=logits,
            append_k=assert_close(k,expected.k),append_v=assert_close(v,expected.v)))

    # Sweep every count on the Q=1 behavior graph, whose feature/mask path is
    # identical to score Q=16. The selected score cases above use the oracle.
    dynamic_started=time.perf_counter()
    for count in range(1,window+1):
        append_graph.run(cache,padded,count,item,behavior_delta,behavior)
    torch.cuda.synchronize()
    dynamic_seconds=time.perf_counter()-dynamic_started
    if len(pool.graphs)!=2 or score_graph.capture_forwards!=1 or append_graph.capture_forwards!=1:
        raise AssertionError("changing inherited count recaptured a CUDA graph")

    # Reuse one graph for another real user, then return to the first user.
    second_features=probe.prepare_read_features(model,second["parent_cache"],second["item_ids"])
    second_cache=probe.prepare_cache(model,second["parent_cache"],second["item_ids"],read_features=second_features)
    actual,=score_graph.run(second_cache,pad_features(second_features,window),window,
                            second["candidates"],second["query_delta"])
    wanted,_=probe.score(model,second["parent_cache"],second["item_ids"],second["candidates"],
        second["query_delta"],prepared_cache=second_cache,read_features=second_features)
    cross_user=assert_close(actual,wanted,logits=True)
    other=copy.deepcopy(probe)
    try:
        FastProbeTrajectory(model,model,other,events,cfg,pool=pool)
    except ValueError:
        weight_guard=True
    else:
        raise AssertionError("pool accepted another probe's weights")
    del other,second_features,second_cache

    score_graph.run(cache,padded,window,sample["candidates"],sample["query_delta"])
    callback=native_score(model,cache,sample["candidates"],sample["query_delta"],trace=True,
                          history_read_fn=score_graph._read)
    reference=probe.score(model,sample["parent_cache"],sample["item_ids"],sample["candidates"],
                           sample["query_delta"],trace=True,prepared_cache=cache,read_features=features)
    full_read=check_prediction(callback,reference)
    del callback,reference

    # Consume the first three actual timestamped behavior events one by one,
    # using each method's OWN preceding persistent state and feature slice.
    actual_cache=HSTUKVCache(cache.k.clone(),cache.v.clone(),window)
    actual_rows=padded.clone()
    expected_cache=cache
    source,ids,fs=sample["parent_cache"],sample["item_ids"],features
    previous=int(events[0][at-1])
    fifo=[]
    for position in range(at,at+3):
        item=torch.tensor([[int(events[1][position])]],device=device)
        behavior=torch.tensor([[int(events[2][position])]],device=device)
        delta=torch.tensor([[min(7*86400,max(0,int(events[0][position])-previous))]],device=device,dtype=torch.float32)
        previous=int(events[0][position])
        k,v,actual_rows=append_graph.run(actual_cache,actual_rows,source.seq_len,item,delta,behavior)
        actual_cache=HSTUKVCache(k,v,window)
        expected_cache,source,ids,_=probe.append(model,expected_cache,source,ids,item,behavior,delta,
                                                 max_length=window,read_features=fs)
        fs=tuple(row[:,:,1:] for row in fs)
        predicted,=score_graph.run(actual_cache,actual_rows,source.seq_len,sample["candidates"],sample["query_delta"])
        wanted,_=probe.score(model,source,ids,sample["candidates"],sample["query_delta"],
                             prepared_cache=expected_cache,read_features=fs)
        fifo.append(dict(timestamp=previous,inherited_rows=source.seq_len,
            k=assert_close(actual_cache.k,expected_cache.k),v=assert_close(actual_cache.v,expected_cache.v),
            logits=assert_close(predicted,wanted,logits=True)))

    # Graph calls include every cache/feature input copy. Resident reference
    # already retains its fixed features; it does not repeat preparation.
    reference_score=measure(lambda:probe.score(model,sample["parent_cache"],sample["item_ids"],
        sample["candidates"],sample["query_delta"],prepared_cache=cache,read_features=features),repeats)
    graph_score=measure(lambda:score_graph.run(cache,padded,window,sample["candidates"],sample["query_delta"]),repeats)
    reference_append=measure(lambda:probe.append(model,cache,sample["parent_cache"],sample["item_ids"],
        item,behavior,delta,max_length=window,read_features=features),repeats)
    graph_append=measure(lambda:append_graph.run(cache,padded,window,item,delta,behavior),repeats)
    return dict(dynamic_mask_cases=cases,all_behavior_counts_1_to_window_seconds=dynamic_seconds,
        no_count_recapture=True,cross_user_logits=cross_user,weight_owner_guard=weight_guard,
        full_history_read_trace=full_read,three_causal_fifo_appends=fifo,
        reference_resident_score=reference_score,graph_score_including_input_copy=graph_score,
        reference_resident_append=reference_append,graph_append_including_input_copy=graph_append,
        graph_pool=pool.report(),f_append_graph_scope="diagnostic equivalence only; deployment f appends use native band")


def geometry(probe,model):
    c=probe.cfg
    joint=getattr(probe,"variant_name",None)=="layerwise_kv"
    return dict(config=asdict(c),row_input_width=c.row_dim,query_head_width=c.head_dim,
        f_output_width=2*(model.cfg.hidden_size if joint else c.head_dim),
        f_maps_per_layer=1 if joint else c.num_heads,
        hidden_width_scope="whole-layer joint f width; g/h width per head" if joint else "width per head for f/g/h",
        parameters={name:sum(p.numel() for p in term.parameters()) if term is not None else 0
                    for name,term in (("f",probe.f),("g",probe.g),("h",probe.h))},
        auxiliary_buffer_bytes=sum(x.numel()*x.element_size() for x in probe.buffers()))


@torch.no_grad()
def h_output_status(probe,model,sample):
    """Check actual student-query h outputs on a declared bounded input tile."""
    candidates=sample["candidates"][:,:2]
    features=probe.prepare_read_features(model,sample["parent_cache"],sample["item_ids"])
    _,trace=probe.score(model,sample["parent_cache"],sample["item_ids"],candidates,
        sample["query_delta"],read_features=features,trace=True)
    maxima=[float(probe.interaction_correction(layer,q,features[layer][:,:,:64]).abs().max())
            for layer,q in enumerate(trace.queries)]
    return dict(nonzero=any(value>0 for value in maxima),max_abs_by_layer=maxima,
        queries=candidates.shape[1],rows=min(64,sample["parent_cache"].seq_len),
        scope="all layers/heads; first two actual student candidates and first 64 inherited rows; numeric diagnostic, not quality")


def save_diagnostic_probe(output,variant,probe,model,sample,identity,stages):
    """Save bounded fitted weights and, when needed, a separate nonzero stress.

    Neither artifact is a quality-trained point. E6 may use them to verify its
    full-shape decomposition/replay implementation and measure execution only.
    """
    best_steps={row["stage"]:row["bounded_training"]["best_step"] for row in stages}
    metadata=dict(identity=dict(identity,validation_users=identity["validation_uids"]),
        variant=variant,method_label=f"{variant}_diagnostic_fgh",seed=17,
        training_users=len(identity["fit_uids"]),validation_users=len(identity["validation_uids"]),
        diagnostic_only=True,quality_auc_computed=False,quality_use_permitted=False,
        best_step=best_steps["fgh"],best_steps=best_steps,
        completed_steps={row["stage"]:row["bounded_training"]["completed_steps"] for row in stages},
        h_output=h_output_status(probe,model,sample),
        purpose="bounded fit/validation numerical and performance diagnostic only; never use for quality results")
    path=output/f"{variant}_diagnostic_fgh.pt"
    save_variant(path,probe,metadata)
    result=dict(trained=dict(path=str(path),sha256=sha256(path),metadata=metadata))
    if not metadata["h_output"]["nonzero"]:
        pressure=copy.deepcopy(probe)
        torch.manual_seed(17023)
        with torch.no_grad():
            for layer in pressure.h:
                layer.weights[-1].normal_(std=2e-4)
        pressure_metadata=dict(metadata,method_label=f"{variant}_nonzero_pressure_fgh",
            pressure_control=True,trained_quality=False,
            h_output=h_output_status(pressure,model,sample),
            purpose="nonzero h pressure control for numerical/decomposition/performance checks; h randomly initialized after bounded fit; never trained quality evidence")
        pressure_path=output/f"{variant}_nonzero_pressure_fgh.pt"
        save_variant(pressure_path,pressure,pressure_metadata)
        result["nonzero_pressure"]=dict(path=str(pressure_path),sha256=sha256(pressure_path),metadata=pressure_metadata)
    return result


def run(args):
    if not 1<=args.steps<=4 or args.fit_users+args.validation_users>16:
        raise ValueError("bounded check permits 1–4 fit steps and at most 16 fit/validation users")
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    torch.cuda.set_device(0)
    device=torch.device("cuda:0")
    started=time.perf_counter()
    args.output.mkdir(parents=True,exist_ok=True)
    paths=[Path(__file__),*(Path(__file__).with_name(name) for name in (
        "probes.py","probe_variants.py","probe_cost.py","probe_fast.py","probe_replay.py",
        "probe_ledger.py","check_probes.py","common.py","prepare_candidates.py")),
        ROOT/"src/hstu_kvcache/adaptation/reader.py"]
    sources={str(p.relative_to(ROOT)):sha256(p) for p in paths}
    population_path=args.input_root/"population.json"
    population=json.loads(population_path.read_text())
    binding,histories,by_user=load_edge(args.input_root/f"edge_{args.edge}")
    if binding["population_sha256"]!=sha256(population_path):
        raise ValueError("population changed")
    for name in ("history","requests"):
        if sha256(args.input_root/f"edge_{args.edge}"/binding[name]["path"])!=binding[name]["sha256"]:
            raise ValueError("input pack changed")
    cfg=dict(binding["settings"],calibration_queries_per_user=16)
    fit_uids=population["fit"][:args.fit_users]
    val_uids=population["validation"][:args.validation_users]
    if not fit_uids or not val_uids or set(fit_uids)&set(val_uids):
        raise ValueError("separate nonempty fit/validation users required")
    parent,model,models=load_models(args.edge,device,backend="torch")
    identity=dict(edge=args.edge,variants=args.variants,fit_uids=fit_uids,validation_uids=val_uids,
        width=args.width,depth=args.depth,steps=args.steps,physical_gpu=args.gpu,
        hardware=torch.cuda.get_device_name(0),models=models,sources=sources,
        input_binding_sha256=sha256(args.input_root/f"edge_{args.edge}/inputs.json"),
        purpose="real-model numeric/performance diagnostic only; no AUC; equal width is not equal parameter capacity")
    write_json(args.output/"configuration.json",identity)
    capture_started=time.perf_counter()
    samples=capture_samples(parent,model,histories,fit_uids+val_uids,cfg,device)
    candidate_path=args.input_root/f"edge_{args.edge}/calibration_feedback_candidates.json"
    candidate_data=json.loads(candidate_path.read_text())
    if candidate_data["population_sha256"]!=sha256(population_path):
        raise ValueError("candidate roles changed")
    identity["calibration_candidate_sha256"]=sha256(candidate_path)
    write_json(args.output/"configuration.json",identity)
    for uid,sample in samples.items():
        sample["candidates"]=torch.tensor([candidates_for(uid,candidate_data["recent_items"][str(uid)],
            model.cfg.num_items-256,count=16,recent_fraction=.5,seed=17)])
    torch.cuda.synchronize()
    capture_seconds=time.perf_counter()-capture_started
    train,validation,scales=bind_fit_scales(model,[samples[u] for u in fit_uids],[samples[u] for u in val_uids])
    full=[uid for uid in samples if samples[uid]["parent_cache"].seq_len==cfg["history_length"]]
    active=[uid for uid in full if len(histories.rows[uid][0])-int(np.searchsorted(
        histories.rows[uid][0],cfg["release_day"]*86400,side="left"))>=3]
    if not active or len(full)<2:
        raise ValueError("bounded diagnostic users need two full windows and one with three later behaviors")
    uid=active[0]
    second_uid=next(value for value in full if value!=uid)
    sample=sample_to_device(samples[uid],device)
    second=sample_to_device(samples[second_uid],device)
    result=dict(identity=identity,benchmark_uid=uid,second_uid=second_uid,capture_seconds=capture_seconds,
        fit_pool_loss_scales=scales,history_lengths={str(u):samples[u]["parent_cache"].seq_len for u in samples},
        benchmark_scope="first full-window fitting/validation diagnostic users; no outcome selection",variants=[])
    for variant in args.variants:
        torch.manual_seed(17)
        base=build_variant(variant,model,hidden_width=args.width,depth=args.depth,row_chunk=1024,query_chunk=16).to(device)
        base.fit_input_statistics(model,train)
        variant_result=dict(variant=variant,stages=[])
        with torch.no_grad():
            native=native_score(model,sample["parent_cache"],sample["candidates"],sample["query_delta"],trace=True)
            for stage in ("f","fg","fgh"):
                probe=base.expanded(stage).eval()
                zero=check_prediction(probe.score(model,sample["parent_cache"],sample["item_ids"],
                    sample["candidates"],sample["query_delta"],trace=True),native)
                nonzero_outputs(probe)
                probe.cfg.row_chunk=128
                small=probe.score(model,sample["parent_cache"],sample["item_ids"],sample["candidates"],sample["query_delta"],trace=True)
                probe.cfg.row_chunk=1024
                large=probe.score(model,sample["parent_cache"],sample["item_ids"],sample["candidates"],sample["query_delta"],trace=True)
                chunks=check_prediction(small,large)
                del small,large
                checks=graph_checks(model,probe,sample,second,histories.rows[uid],cfg,args.repeats)
                complete=full_probe_ledger(histories,by_user,list(samples),cfg,stage=stage,padded=True)
                row=dict(stage=stage,geometry=geometry(probe,model),zero_native=zero,
                    nonzero_chunk128_vs1024=chunks,numerical_and_inference=checks,
                    full_window_prepare_flops=cached_preparation_flops(probe,cfg["history_length"]),
                    full_window_padded_score16_extra_flops=padded_read_extra_flops(probe,cfg["history_length"],16),
                    full_window_padded_append1_extra_flops=padded_read_extra_flops(probe,cfg["history_length"]-1,1),
                    complete_diagnostic_pool_setup=deployment_graph_setup(probe,complete,model,window=cfg["history_length"]))
                variant_result["stages"].append(row)
                print(json.dumps(dict(status="variant_numeric_passed",variant=variant,stage=stage,
                    score_ms=1000*checks["graph_score_including_input_copy"]["median_seconds"],
                    append_ms=1000*checks["graph_append_including_input_copy"]["median_seconds"])),flush=True)
                del probe
                gc.collect()
                torch.cuda.empty_cache()
            del native
        nested=base
        for row in variant_result["stages"]:
            stage=row["stage"]
            if nested.stage!=stage:
                nested=nested.expanded(stage)
            torch.cuda.synchronize()
            baseline=torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            trained_started=time.perf_counter()
            report=fit_probe(model,nested,train,validation,steps=args.steps,validation_every=args.steps,
                             patience=0,learning_rate=1e-4,seed=17)
            torch.cuda.synchronize()
            elapsed=time.perf_counter()-trained_started
            if any(parameter.grad is not None for parameter in model.parameters()):
                raise AssertionError("frozen backbone accumulated a parameter gradient")
            row["bounded_training"]=dict(report,seconds_including_validation_and_transfer=elapsed,
                steps_per_second_including_validation_and_transfer=report["completed_steps"]/elapsed,
                baseline_allocated_bytes=baseline,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved())
            print(json.dumps(dict(status="variant_training_checked",variant=variant,stage=stage,
                steps=report["completed_steps"],seconds=elapsed,loss=report["best_validation_loss"])),flush=True)
        variant_result["diagnostic_checkpoints"]=save_diagnostic_probe(args.output,variant,nested,
            model,sample,identity,variant_result["stages"])
        result["variants"].append(variant_result)
        write_json(args.output/"progress.json",result)
        del nested,base
        gc.collect()
        torch.cuda.empty_cache()
    for path in paths:
        if sha256(path)!=sources[str(path.relative_to(ROOT))]:
            raise ValueError("numerical source changed during diagnostic")
    result.update(status="passed",quality_auc_computed=False,total_seconds=time.perf_counter()-started,
        scope="at most 16 real fit/validation users; numeric/performance only; measured path time is not serving throughput")
    write_json(args.output/"summary.json",result)
    print(json.dumps(dict(status="complete",output=str(args.output),seconds=result["total_seconds"])),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root",type=Path,default=ROOT/"experiments/ch3_motivation/inputs")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--edge",type=int,default=1,choices=range(1,6))
    p.add_argument("--gpu",type=int,default=1)
    p.add_argument("--variants",nargs="+",choices=["full_item","layerwise_kv"],default=["full_item","layerwise_kv"])
    p.add_argument("--fit-users",type=int,default=4)
    p.add_argument("--validation-users",type=int,default=4)
    p.add_argument("--width",type=int,default=128)
    p.add_argument("--depth",type=int,default=1)
    p.add_argument("--steps",type=int,default=3)
    p.add_argument("--repeats",type=int,default=4)
    run(p.parse_args())
