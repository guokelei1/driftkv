#!/usr/bin/env python3
"""Fit fresh E5 probes on pre-release users; evaluation uses separate fixed panels."""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

from experiments.ch3_motivation.code.common import (
    ROOT, capture_samples, load_edge, load_models, sample_to, write_json,
)
from experiments.ch3_motivation.code.probes import (
    ProbeConfig, ProbeLadder, fit_probe, prepare_training_samples, native_query_flops,
)
from hstu_kvcache.evaluation.cost import CostModel
from hstu_kvcache.common.artifacts import sha256
import torch


def save_probe(path, probe, metadata):
    temporary=path.with_suffix(".pt.writing")
    torch.save(dict(config=asdict(probe.cfg),stage=probe.stage,
        state_dict={k:v.detach().cpu() for k,v in probe.state_dict().items()},metadata=metadata),temporary)
    temporary.replace(path)


def load_probe(path, device):
    value=torch.load(path,map_location="cpu",weights_only=False)
    probe=ProbeLadder(ProbeConfig(**value["config"]),value["stage"]).to(device)
    probe.load_state_dict(value["state_dict"])
    return probe,value["metadata"]


def run(args):
    started=time.perf_counter()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    torch.cuda.set_device(0)
    device=torch.device("cuda:0")
    args.output.mkdir(parents=True,exist_ok=True)
    binding,histories,by_user=load_edge(args.input_root/f"edge_{args.edge}")
    for name in ("history","requests"):
        record=binding[name]
        if sha256(args.input_root/f"edge_{args.edge}"/record["path"])!=record["sha256"]:
            raise ValueError(f"frozen {name} pack changed")
    cfg=binding["settings"]
    cfg["calibration_queries_per_user"]=args.queries
    population=json.loads((args.input_root/"population.json").read_text())
    population_sha=sha256(args.input_root/"population.json")
    if binding["population_sha256"]!=population_sha:
        raise ValueError("frozen population changed")
    parent,current,models=load_models(args.edge,device)
    sources={str(path.relative_to(ROOT)):sha256(path) for path in (
        Path(__file__),Path(__file__).with_name("common.py"),Path(__file__).with_name("probes.py"),
        ROOT/"src/hstu_kvcache/adaptation/reader.py")}
    identity=dict(edge=args.edge,models=models,numerical_sources=sources,
        population_sha256=population_sha,
        input_binding_sha256=sha256(args.input_root/f"edge_{args.edge}/inputs.json"),
        seed=args.seed,fit_users=args.fit_users,validation_users=population["validation"],
        fit_uids=population["fit"][:args.fit_users],candidate_seed=cfg["seed"],
        sample_candidates=args.queries,steps=args.steps,learning_rate=args.learning_rate,
        read_weight=args.read_weight,score_weight=args.score_weight,
        stages=args.stages.split(","),patience=args.patience,validation_every=args.validation_every,
        weight_decay=.0001,gradient_clip=5.,
        width=args.width,depth=args.depth,row_chunk=args.row_chunk,
        include_item=not args.no_item,normalize=not args.no_normalize)
    if (args.output/"configuration.json").exists():
        if json.loads((args.output/"configuration.json").read_text()) != identity:
            raise ValueError("attempt configuration changed; use a fresh output directory")
    else:
        write_json(args.output/"configuration.json",identity)
    # Rebuild both source and teacher for the same aligned prefix. No old fits.
    fit_uids=population["fit"][:args.fit_users]
    validation=population["validation"]
    if set(fit_uids)&set(validation):
        raise ValueError("fit/validation overlap")
    captured=capture_samples(parent,current,histories,fit_uids+validation,cfg,device)
    train=[captured[u] for u in fit_uids]
    val=[captured[u] for u in validation]
    cm=CostModel(current.cfg.hidden_size,len(current.blocks),current.cfg.num_heads,cfg["attention_backend"])
    preparation_flops=0
    for sample in train+val:
        n=sample["history_count"]
        cost=cm if n==1024 else CostModel(cm.hidden_size,cm.num_layers,cm.num_heads,"torch")
        native=native_query_flops(current,history_length=n,queries=args.queries)
        teacher_reads=args.queries*n*cm.num_layers*(4*cm.hidden_size+3*cm.num_heads)
        preparation_flops+=2*cost.full_cache(n)+2*native+teacher_reads
    train=prepare_training_samples(current,train)
    val=prepare_training_samples(current,val)
    del parent
    torch.cuda.empty_cache()
    torch.manual_seed(args.seed)
    probe_cfg=ProbeConfig.from_model(current,hidden_width=args.width,depth=args.depth,
        include_item=not args.no_item,normalize=not args.no_normalize,
        query_chunk=args.queries,row_chunk=args.row_chunk)
    stages=args.stages.split(",")
    probe=None
    summaries=[]
    cumulative_calibration_flops=preparation_flops
    for stage in stages:
        checkpoint=args.output/f"{stage}.pt"
        summary_path=args.output/f"{stage}.json"
        if checkpoint.exists() and summary_path.exists():
            probe,metadata=load_probe(checkpoint,device)
            summary=json.loads(summary_path.read_text())
            if summary["checkpoint_sha256"]!=sha256(checkpoint):
                raise ValueError("saved probe checkpoint changed")
            summaries.append(summary)
            cumulative_calibration_flops=summary["cumulative_calibration_flops_estimate"]
            continue
        if probe is None:
            if stage!="f":
                raise ValueError("nested ladder begins with f")
            probe=ProbeLadder(probe_cfg,stage="f").to(device)
        elif stage!=probe.stage:
            probe=probe.expanded(stage)
        torch.cuda.synchronize()
        stage_started=time.perf_counter()
        metadata=fit_probe(current,probe,train,val,steps=args.steps,
            learning_rate=args.learning_rate,read_weight=args.read_weight,
            score_weight=args.score_weight,seed=args.seed,patience=args.patience,
            validation_every=args.validation_every,
            progress_callback=lambda value: print(json.dumps(dict(status="probe_validation",
                edge=args.edge,stage=stage,seed=args.seed,**value)),flush=True))
        torch.cuda.synchronize()
        cumulative_calibration_flops+=metadata["total_calibration_flops_estimate"]
        metadata.update(seconds=time.perf_counter()-stage_started,
            fitting_uses_future_feedback=False,identity=identity,
            quality_auc_computed=False,scope="pre-release fitting/validation losses; not evaluation quality",
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            source_teacher_and_target_preparation_flops=int(preparation_flops),
            cumulative_calibration_flops_estimate=int(cumulative_calibration_flops))
        save_probe(checkpoint,probe,metadata)
        summary=dict(metadata,checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint))
        write_json(summary_path,summary)
        summaries.append(summary)
        print(json.dumps(dict(status="stage_fitted",edge=args.edge,stage=stage,fit_users=len(train),
            seed=args.seed,best_validation_loss=metadata["best_validation_loss"],
            completed_steps=metadata["completed_steps"],seconds=metadata["seconds"])),flush=True)
    write_json(args.output/"summary.json",dict(status="fitted",identity=identity,
        stages=summaries,total_seconds=time.perf_counter()-started,
        scope="No held-out quality conclusion; subsequent >=512-user evaluation is required"))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root",type=Path,default=ROOT/"experiments/ch3_motivation/inputs")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--edge",type=int,choices=range(1,6),required=True)
    parser.add_argument("--gpu",type=int,default=1)
    parser.add_argument("--seed",type=int,default=17)
    parser.add_argument("--fit-users",type=int,choices=[64,128,256],default=64)
    parser.add_argument("--queries",type=int,default=16)
    parser.add_argument("--width",type=int,default=64)
    parser.add_argument("--depth",type=int,default=1)
    parser.add_argument("--row-chunk",type=int,default=1024)
    parser.add_argument("--no-item",action="store_true")
    parser.add_argument("--no-normalize",action="store_true")
    parser.add_argument("--steps",type=int,default=256)
    parser.add_argument("--learning-rate",type=float,default=.001)
    parser.add_argument("--read-weight",type=float,default=1.)
    parser.add_argument("--score-weight",type=float,default=1.)
    parser.add_argument("--patience",type=int,default=64)
    parser.add_argument("--validation-every",type=int,default=32)
    parser.add_argument("--stages",default="f,fg,fgh")
    run(parser.parse_args())
