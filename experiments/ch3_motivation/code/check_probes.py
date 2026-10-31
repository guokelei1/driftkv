#!/usr/bin/env python3
"""Bounded real-model E5 numerical/performance diagnostic, never quality AUC."""
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
import json
import os
from pathlib import Path
import sys
import time


import numpy as np
import torch

from experiments.ch3_motivation.code.common import ROOT, capture_samples, load_edge, load_models, write_json
from experiments.ch3_motivation.code.probe_replay import ProbeTrajectory
from experiments.ch3_motivation.code.probes import (
    ProbeConfig, ProbeLadder, fit_probe, prepare_training_samples, probe_loss, sample_to_device,
)
from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.evaluation.trajectory import NativeTrajectory
from hstu_kvcache.common.artifacts import sha256


def differences(actual, expected):
    delta = (actual-expected).detach().double()
    scale = expected.detach().double()
    return dict(max_abs=float(delta.abs().max()), rms=float(delta.square().mean().sqrt()),
        relative_max=float(delta.abs().max()/scale.abs().max().clamp_min(1e-6)),
        relative_rms=float(delta.norm()/scale.norm().clamp_min(1e-6)))


def check_prediction(actual, expected):
    logits, trace = actual
    wanted, reference = expected
    result = dict(logits=differences(logits,wanted),
        reads=[differences(a,b) for a,b in zip(trace.history_heads,reference.history_heads,strict=True)],
        queries=[differences(a,b) for a,b in zip(trace.queries,reference.queries,strict=True)],
        new_kv=differences(trace.new_kv.k,reference.new_kv.k))
    if result["logits"]["max_abs"] > 3e-5:
        raise AssertionError(f"logits differ: {result['logits']}")
    if max(row["relative_rms"] for row in result["reads"]) > 2e-5:
        raise AssertionError("history read relative RMS exceeds 2e-5")
    return result


def measure(function, repeats, *, warmup=1):
    for _ in range(warmup):
        value=function()
        del value
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples=[]
    for _ in range(repeats):
        torch.cuda.synchronize()
        started=time.perf_counter()
        value=function()
        torch.cuda.synchronize()
        samples.append(time.perf_counter()-started)
        del value
    return dict(seconds=samples, median_seconds=float(np.median(samples)),
        mean_seconds=sum(samples)/len(samples), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved())


def run(args):
    started=time.perf_counter()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    torch.cuda.set_device(0)
    device=torch.device("cuda:0")
    args.output.mkdir(parents=True,exist_ok=True)
    source_paths=[Path(__file__),Path(__file__).with_name("probes.py"),
                  Path(__file__).with_name("probe_replay.py"),Path(__file__).with_name("common.py"),
                  ROOT/"src/hstu_kvcache/adaptation/reader.py"]
    sources={str(path.relative_to(ROOT)):sha256(path) for path in source_paths}
    binding,histories,_=load_edge(args.input_root/f"edge_{args.edge}")
    cfg=dict(binding["settings"],calibration_queries_per_user=16)
    population=json.loads((args.input_root/"population.json").read_text())
    train_uids=population["fit"][:args.fit_users]
    val_uids=population["validation"][:args.validation_users]
    if set(train_uids)&set(val_uids):
        raise ValueError("fit/validation overlap")
    if len(train_uids)+len(val_uids) > 32:
        raise ValueError("bounded diagnostic accepts at most 32 users")
    identity=dict(edge=args.edge,fit_uids=train_uids,validation_uids=val_uids,
        physical_gpu=args.gpu,hardware=torch.cuda.get_device_name(0),numerical_sources=sources,
        input_binding_sha256=sha256(args.input_root/f"edge_{args.edge}/inputs.json"),
        purpose="numerical/performance diagnostic only; no evaluation AUC")
    write_json(args.output/"configuration.json",identity)
    parent,current,models=load_models(args.edge,device,backend="torch")
    identity["model_checkpoints"]=models
    torch.cuda.synchronize()
    capture_started=time.perf_counter()
    captured=capture_samples(parent,current,histories,train_uids+val_uids,cfg,device)
    torch.cuda.synchronize()
    capture_seconds=time.perf_counter()-capture_started
    train=[captured[u] for u in train_uids]
    validation=[captured[u] for u in val_uids]
    target_started=time.perf_counter()
    train=prepare_training_samples(current,train)
    validation=prepare_training_samples(current,validation)
    torch.cuda.synchronize()
    target_seconds=time.perf_counter()-target_started
    benchmark_index=max(range(len(train)),key=lambda index:train[index]["parent_cache"].seq_len)
    sample=sample_to_device(train[benchmark_index],device)
    probe_cfg=ProbeConfig.from_model(current,hidden_width=args.width,depth=args.depth,
                                    query_chunk=16,row_chunk=1024)
    shape=dict(history=sample["parent_cache"].seq_len,queries=sample["candidates"].shape[1],
               head_dim=probe_cfg.head_dim,heads=probe_cfg.num_heads,layers=probe_cfg.num_layers)
    results=dict(identity=identity,benchmark_uid=train_uids[benchmark_index],
        benchmark_selection="largest pre-release history within the fixed fitting diagnostic users",
        model_shape=shape,capture_seconds=capture_seconds,
        cached_target_seconds=target_seconds,fit_history_lengths=[x["parent_cache"].seq_len for x in train],
        validation_history_lengths=[x["parent_cache"].seq_len for x in validation],
        zero={},nonzero_chunks={},timings=[],training=[],replay={})
    torch.manual_seed(17022)
    base=ProbeLadder(probe_cfg,"f").to(device)
    base.fit_input_statistics(current,train)
    with torch.no_grad():
        native=score(current,sample["parent_cache"],sample["candidates"],sample["query_delta"],trace=True)
        for stage in ("f","fg","fgh"):
            probe=base.expanded(stage)
            actual=probe.score(current,sample["parent_cache"],sample["item_ids"],
                sample["candidates"],sample["query_delta"],trace=True)
            results["zero"][stage]=check_prediction(actual,native)
            del actual,probe
        del native
    random_probe=base.expanded("fgh")
    with torch.no_grad():
        for modules in (random_probe.f,random_probe.g,random_probe.h):
            for layer in modules:
                layer.weights[-1].normal_(std=2e-4)
        random_probe.cfg.row_chunk=128
        small=random_probe.score(current,sample["parent_cache"],sample["item_ids"],
            sample["candidates"],sample["query_delta"],trace=True)
        random_probe.cfg.row_chunk=1024
        large=random_probe.score(current,sample["parent_cache"],sample["item_ids"],
            sample["candidates"],sample["query_delta"],trace=True)
        results["nonzero_chunks"]=check_prediction(small,large)
        del small,large
    write_json(args.output/"progress.json",dict(status="numerical_passed",seconds=time.perf_counter()-started))

    for stage in ("f","fg","fgh"):
        probe=base.expanded(stage)
        with torch.no_grad():
            for modules in (probe.f,probe.g,probe.h):
                if modules is not None:
                    for layer in modules:
                        layer.weights[-1].normal_(std=2e-4)
        for chunk in (128,1024):
            probe.cfg.row_chunk=chunk
            with torch.no_grad():
                features=probe.prepare_read_features(current,sample["parent_cache"],sample["item_ids"])
                prepared=probe.prepare_cache(current,sample["parent_cache"],sample["item_ids"],read_features=features)
                resident=measure(lambda:probe.score(current,sample["parent_cache"],sample["item_ids"],
                    sample["candidates"],sample["query_delta"],prepared_cache=prepared,read_features=features),args.repeats)

                def transfer_and_score():
                    moved=sample_to_device(train[benchmark_index],device)
                    return probe.score(current,moved["parent_cache"],moved["item_ids"],
                        moved["candidates"],moved["query_delta"])

                transfer=measure(transfer_and_score,args.repeats)
            optimizer=torch.optim.AdamW(probe.parameters(),lr=1e-4)

            def training_step():
                optimizer.zero_grad(set_to_none=True)
                loss,parts=probe_loss(current,probe,sample)
                loss.backward()
                grad=torch.nn.utils.clip_grad_norm_(probe.parameters(),5.)
                optimizer.step()
                return loss,parts,grad

            training=measure(training_step,max(2,args.repeats//2),warmup=1)
            results["timings"].append(dict(stage=stage,row_chunk=chunk,resident_prepared_scoring=resident,
                cpu_transfer_prepare_and_scoring=transfer,resident_training_step=training))
            print(json.dumps(dict(status="timed",stage=stage,row_chunk=chunk,
                resident_query_ms=1000*resident["median_seconds"],
                transfer_prepare_query_ms=1000*transfer["median_seconds"],
                training_ms=1000*training["median_seconds"])),flush=True)
            del prepared,features,optimizer
        del probe

    nested=None
    active_uids=[uid for uid in train_uids if np.searchsorted(histories.rows[uid][0],
        cfg["release_day"]*86400,side="left") < len(histories.rows[uid][0])]
    if not active_uids:
        raise ValueError("bounded replay diagnostic needs a fitting user with post-release behaviors")
    replay_uid=max(active_uids,key=lambda uid:captured[uid]["history_count"])
    results["replay_uid"]=replay_uid
    for stage in ("f","fg","fgh"):
        nested=base.expanded("f") if nested is None else nested.expanded(stage)
        nested.cfg.row_chunk=1024
        torch.cuda.synchronize()
        fitted_started=time.perf_counter()
        fitted=fit_probe(current,nested,train,validation,steps=args.steps,
            seed=17,validation_every=args.steps,patience=0,learning_rate=.001)
        torch.cuda.synchronize()
        fitted["seconds_including_validation_and_transfer"]=time.perf_counter()-fitted_started
        results["training"].append(fitted)
        print(json.dumps(dict(status="fitted_diagnostic",stage=stage,steps=fitted["completed_steps"],
            before=fitted["history"][0]["validation_loss"],after=fitted["best_validation_loss"],
            seconds=fitted["seconds_including_validation_and_transfer"])),flush=True)

    # Real timestamp ordering, bounded to the first few post-release events.
    # The native oracle must agree when all probe outputs are exactly zero.
    for stage in ("f","fg","fgh"):
        probe=base.expanded(stage)
        probe.cfg.row_chunk=1024
        events=histories.rows[replay_uid]
        trajectory=ProbeTrajectory(parent,current,probe,events,cfg)
        oracle=NativeTrajectory(parent,cfg["edge"]-1,events,cfg["release_day"]*86400,
            max_length=cfg["history_length"],append_band_size=cfg["append_band_size"])
        oracle.switch(current,cfg["edge"],cfg["release_day"]*86400)
        f_calls=[0]

        def called(*_):
            f_calls[0]+=1

        hooks=[layer.register_forward_hook(called) for layer in probe.f]
        remaining=len(events[0])-trajectory.native.cursor
        until=(int(events[0][trajectory.native.cursor+min(3,remaining)-1])+1 if remaining else
               cfg["release_day"]*86400)
        replay_started=time.perf_counter()
        trajectory.advance(until)
        oracle.advance(until)
        delta=torch.tensor([until-trajectory.native.last_timestamp],device=device,dtype=torch.float32)
        with torch.no_grad():
            for _ in range(2):
                previous=trajectory.cache.k.clone()
                actual=trajectory.score(sample["candidates"],delta)
                if not torch.equal(previous,trajectory.cache.k):
                    raise AssertionError("candidate read mutated persistent KV")
            wanted,_=score(current,oracle.cache,sample["candidates"],delta)
            metrics=differences(actual,wanted)
            if metrics["max_abs"]>3e-5:
                raise AssertionError("zero-probe append trajectory differs from native")
        torch.cuda.synchronize()
        for hook in hooks:
            hook.remove()
        if f_calls[0] != probe_cfg.num_layers:
            raise AssertionError("f was not prepared exactly once")
        results["replay"][stage]=dict(trajectory.report(),f_layer_calls=f_calls[0],
            logits=metrics,cache=differences(trajectory.cache.k,oracle.cache.k),
            bounded_replay_seconds=time.perf_counter()-replay_started)
        del trajectory,oracle,probe
    if any(sha256(path)!=sources[str(path.relative_to(ROOT))] for path in source_paths):
        raise ValueError("numerical source changed during diagnostic")
    results.update(status="passed",quality_auc_computed=False,total_seconds=time.perf_counter()-started,
        scope="16-or-fewer real fitting/validation users; numerical and shared-machine performance diagnostic only")
    write_json(args.output/"summary.json",results)
    print(json.dumps(dict(status="complete",output=str(args.output),seconds=results["total_seconds"])),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root",type=Path,default=ROOT/"experiments/ch3_motivation/inputs")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--edge",type=int,choices=range(1,6),default=1)
    parser.add_argument("--gpu",type=int,default=1)
    parser.add_argument("--fit-users",type=int,default=8)
    parser.add_argument("--validation-users",type=int,default=8)
    parser.add_argument("--width",type=int,default=64)
    parser.add_argument("--depth",type=int,default=1)
    parser.add_argument("--steps",type=int,default=2)
    parser.add_argument("--repeats",type=int,default=4)
    run(parser.parse_args())
