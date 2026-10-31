#!/usr/bin/env python3
"""Fit stronger row-input variants without changing the frozen head-input sweep."""
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

from experiments.ch3_motivation.code.common import ROOT, capture_samples, load_edge, load_models, write_json
from experiments.ch3_motivation.code.probes import fit_probe, prepare_training_samples, native_query_flops
from experiments.ch3_motivation.code.probe_variants import build_variant, save_variant, load_variant
from hstu_kvcache.evaluation.cost import CostModel
from hstu_kvcache.common.artifacts import sha256
from experiments.ch3_motivation.code.prepare_candidates import candidates_for
import torch


def run(args, *, preparation_cache=None):
    preparation_started=time.perf_counter()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    torch.cuda.set_device(0)
    device=torch.device("cuda:0")
    population_path=args.input_root/"population.json"
    population=json.loads(population_path.read_text())
    binding,histories,_=load_edge(args.input_root/f"edge_{args.edge}")
    if binding["population_sha256"]!=sha256(population_path):
        raise ValueError("population changed")
    for name in ("history","requests"):
        if sha256(args.input_root/f"edge_{args.edge}"/binding[name]["path"])!=binding[name]["sha256"]:
            raise ValueError("input pack changed")
    cfg=dict(binding["settings"],calibration_queries_per_user=args.queries)
    fit_uids=population["fit"][:args.fit_users]
    val_uids=population["validation"]
    if set(fit_uids)&set(val_uids):raise ValueError("fit/validation overlap")
    paths=[Path(__file__),Path(__file__).with_name("probes.py"),Path(__file__).with_name("probe_variants.py"),
        Path(__file__).with_name("common.py"),Path(__file__).with_name("prepare_candidates.py"),
        ROOT/"src/hstu_kvcache/adaptation/reader.py"]
    candidate_path=args.input_root/f"edge_{args.edge}/calibration_feedback_candidates.json"
    candidate_data=json.loads(candidate_path.read_text()) if args.candidate_source=="feedback" else None
    if candidate_data is not None and candidate_data["population_sha256"]!=sha256(population_path):
        raise ValueError("calibration candidate roles changed")
    sources={str(p.relative_to(ROOT)):sha256(p) for p in paths}
    preparation_key=dict(edge=args.edge,input_binding_sha256=sha256(args.input_root/f"edge_{args.edge}/inputs.json"),
        population_sha256=sha256(population_path),fit_uids=fit_uids,validation_users=val_uids,
        candidate_source=args.candidate_source,recent_fraction=args.recent_fraction,queries=args.queries,
        calibration_candidate_sha256=sha256(candidate_path) if candidate_data is not None else None,
        sources=sources)
    preparation_reused=preparation_cache is not None and preparation_cache.get("key")==preparation_key
    if preparation_reused:
        current,models,train,val,shared=preparation_cache["value"]
    else:
        # Sequential sweeps keep only one edge's frozen backbone/targets.
        if preparation_cache is not None:
            preparation_cache.clear()
        parent,current,models=load_models(args.edge,device)
    identity=dict(edge=args.edge,models=models,variant=args.variant,fit_uids=fit_uids,
        validation_users=val_uids,seed=args.seed,queries=args.queries,candidate_seed=17,
        width=args.width,depth=args.depth,steps=args.steps,learning_rate=args.learning_rate,
        read_weight=args.read_weight,score_weight=args.score_weight,patience=args.patience,
        validation_every=args.validation_every,stages=args.stages,controls=args.controls,
        row_chunk=1024,query_chunk=args.queries,weight_decay=.0001,gradient_clip=5.,
        candidate_source=args.candidate_source,recent_fraction=args.recent_fraction,
        calibration_candidate_sha256=sha256(candidate_path) if candidate_data is not None else None,
        candidate_rule="recent pre-release items plus uniform unique inclusive 1..foundation_items; seed17 fixed across all variants",
        input_binding_sha256=sha256(args.input_root/f"edge_{args.edge}/inputs.json"),
        population_sha256=sha256(population_path),numerical_sources=sources)
    args.output.mkdir(parents=True,exist_ok=True)
    configuration=args.output/"configuration.json"
    if configuration.exists() and json.loads(configuration.read_text())!=identity:
        raise ValueError("variant attempt configuration changed")
    write_json(configuration,identity)
    if not preparation_reused:
        samples=capture_samples(parent,current,histories,fit_uids+val_uids,cfg,device)
        # Include the last known ID without renumbering the frozen dataset.
        known=current.cfg.num_items-256
        for uid,sample in samples.items():
            if candidate_data is not None:
                values=candidates_for(uid,candidate_data["recent_items"][str(uid)],known,
                    count=args.queries,recent_fraction=args.recent_fraction,seed=17)
            else:
                values=candidates_for(uid,list(reversed(sample["item_ids"][0].tolist())),known,
                    count=args.queries,recent_fraction=args.recent_fraction,seed=17)
            sample["candidates"]=torch.as_tensor(values,dtype=torch.long)[None]
        train=prepare_training_samples(current,[samples[u] for u in fit_uids])
        val=prepare_training_samples(current,[samples[u] for u in val_uids])
        cm=CostModel(current.cfg.hidden_size,len(current.blocks),current.cfg.num_heads,cfg["attention_backend"])
        shared=0
        for sample in train+val:
            n=sample["history_count"]
            cost=cm if n==1024 else CostModel(cm.hidden_size,cm.num_layers,cm.num_heads,"torch")
            shared+=2*cost.full_cache(n)+2*native_query_flops(current,history_length=n,queries=args.queries)
            shared+=args.queries*n*cm.num_layers*(4*cm.hidden_size+3*cm.num_heads)
        del parent
        torch.cuda.empty_cache()
        if preparation_cache is not None:
            preparation_cache.update(key=preparation_key,value=(current,models,train,val,shared))
    preparation_seconds=time.perf_counter()-preparation_started
    torch.manual_seed(args.seed)
    probe=build_variant(args.variant,current,stage="f",hidden_width=args.width,depth=args.depth,
        query_chunk=args.queries,row_chunk=1024).to(device)
    total=shared
    summaries=[]

    def fit_one(module,label,steps,previous_flops):
        checkpoint=args.output/f"{label}.pt"
        summary_path=args.output/f"{label}.json"
        if checkpoint.exists() and summary_path.exists():
            saved,metadata=load_variant(checkpoint,device)
            summary=json.loads(summary_path.read_text())
            if summary["checkpoint_sha256"]!=sha256(checkpoint):
                raise ValueError("variant checkpoint changed")
            return saved,summary
        started=time.perf_counter()
        if device.type=="cuda":torch.cuda.reset_peak_memory_stats()
        metadata=fit_probe(current,module,train,val,steps=steps,learning_rate=args.learning_rate,
            read_weight=args.read_weight,score_weight=args.score_weight,seed=args.seed,
            patience=args.patience,validation_every=args.validation_every,
            progress_callback=lambda row:print(json.dumps(dict(status="variant_validation",edge=args.edge,
                variant=args.variant,method_label=label,**row)),flush=True))
        metadata.update(identity=identity,variant=args.variant,method_label=label,
            actual_search_preparation_reused=preparation_reused,
            actual_preparation_seconds_this_process=preparation_seconds,
            source_teacher_and_target_preparation_flops=int(shared),
            cumulative_calibration_flops_estimate=int(previous_flops+metadata["total_calibration_flops_estimate"]),
            seconds=time.perf_counter()-started,quality_auc_computed=False)
        if device.type=="cuda":
            metadata.update(physical_gpu=args.gpu,hardware=torch.cuda.get_device_name(0),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved())
        save_variant(checkpoint,module,metadata)
        summary=dict(metadata,checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint))
        write_json(summary_path,summary)
        return module,summary

    for stage in args.stages.split(","):
        if stage!=probe.stage:probe=probe.expanded(stage)
        probe,summary=fit_one(probe,stage,args.steps,total)
        total=summary["cumulative_calibration_flops_estimate"]
        summaries.append(summary)
        if args.controls and stage in ("f","fg"):
            extra=2*args.steps if stage=="f" else args.steps
            _,control=fit_one(copy.deepcopy(probe),stage+"_extended",extra,total)
            summaries.append(control)
    for name,digest in sources.items():
        if sha256(ROOT/name)!=digest:raise ValueError("variant numerical source changed during fit")
    write_json(args.output/"summary.json",dict(status="fitted",identity=identity,stages=summaries,
        scope="pre-release fit/validation only; quality requires fixed >=512-user population"))
    print(json.dumps(dict(status="complete",output=str(args.output),variant=args.variant)),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root",type=Path,default=ROOT/"experiments/ch3_motivation/inputs")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--edge",type=int,choices=range(1,6),required=True)
    p.add_argument("--variant",choices=["head","full_item","layerwise_kv"],required=True)
    p.add_argument("--gpu",type=int,default=1)
    p.add_argument("--seed",type=int,default=17)
    p.add_argument("--fit-users",type=int,choices=[64,128,256],default=64)
    p.add_argument("--queries",type=int,default=16)
    p.add_argument("--width",type=int,default=128)
    p.add_argument("--depth",type=int,default=1)
    p.add_argument("--steps",type=int,default=256)
    p.add_argument("--learning-rate",type=float,default=.0001)
    p.add_argument("--read-weight",type=float,default=1.)
    p.add_argument("--score-weight",type=float,default=1.)
    p.add_argument("--patience",type=int,default=128)
    p.add_argument("--validation-every",type=int,default=32)
    p.add_argument("--stages",default="f,fg,fgh")
    p.add_argument("--controls",action="store_true")
    p.add_argument("--candidate-source",choices=["feedback","listen"],default="feedback")
    p.add_argument("--recent-fraction",type=float,default=.5)
    run(p.parse_args())
