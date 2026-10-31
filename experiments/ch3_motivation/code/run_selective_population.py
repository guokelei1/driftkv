#!/usr/bin/env python3
"""Run the audited E3 evaluator with a separately frozen scenario manifest.

The original runner and running three-anchor sources stay untouched. All five
predeclared release edges can be reported, including an edge whose anchor gain
is negative. No methods' scores participate in selecting the scenario.
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]

import torch

from experiments.ch3_motivation.code import common, selective, run_selective
from experiments.ch3_motivation.code.scenario_inputs import load_scenario_population, load_scenario_data
from hstu_kvcache.evaluation.baselines import event_shapes, layer_boundary_accounting
from hstu_kvcache.evaluation import fast_baselines as fast
from hstu_kvcache.evaluation.cost import CostModel
from hstu_kvcache.common.artifacts import sha256


def sources():
    result=run_selective.sources()
    for name in ("experiments/ch3_motivation/code/scenario_inputs.py","experiments/ch3_motivation/code/run_selective_population.py"):
        result[name]=sha256(ROOT/name)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge",type=int,choices=range(1,6),required=True)
    parser.add_argument("--population-file",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--diagnostic-users",type=int,default=8)
    parser.add_argument("--gpu",help="physical CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--resume",action="store_true")
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check",action="store_true")
    mode.add_argument("--run",action="store_true")
    args=parser.parse_args()
    if args.check and not 8<=args.diagnostic_users<=16:
        parser.error("--check requires 8--16 diagnostic users")
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    binding,all_users,population,provenance=load_scenario_population(args.population_file,args.edge,
        require_positive=False)
    users=all_users[:args.diagnostic_users] if args.check else all_users
    common.quality_role(users,diagnostic=args.check)
    histories,by_user=load_scenario_data(provenance)
    cfg=dict(binding["settings"])
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    if any((output/"quality_shards").glob("part_*.json")) and not args.resume:
        raise ValueError("score blocks exist; use explicit --resume with unchanged identity or choose a new attempt")
    code_hashes=sources()
    device=torch.device("cuda:0")
    parent,current,models=common.load_models(args.edge,device)
    if models!=provenance["anchor_models"]:
        raise ValueError("method execution checkpoints differ from the measured three anchors")
    identity=dict(experiment="ch3_selective_scenario_fresh",panel=provenance["selection"]["panel_name"],
        population_sha256=provenance["source_population_sha256"],input_binding_sha256=provenance["input_binding_sha256"],
        input_hashes={name:binding[name]["sha256"] for name in ("history","requests")},checkpoint_sha256=models,
        source_sha256=code_hashes,settings=cfg,evaluation_users_sha256=run_selective._hash_json(users),
        calibration_users_sha256=run_selective._hash_json(population["fit"][:cfg["profile_users"]]),scenario=provenance,
        edge_selection="all prescribed edges retained irrespective of per-edge anchor positivity")
    identity_path=output/"identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text())!=identity:
        raise ValueError("execution sources/inputs/scenario changed; use a new attempt")
    common.write_json(identity_path,identity)
    started=time.perf_counter()
    common.write_json(output/"status.json",dict(state="profiling",pid=os.getpid(),command=sys.argv))
    profile_path=output/"layer_profile.json"
    if args.resume and profile_path.exists():
        profiles=json.loads(profile_path.read_text())
        resources=dict(resumed=True)
    else:
        def profile():
            fit=population["fit"][:cfg["profile_users"]]
            samples=common.capture_samples(parent,current,histories,fit,cfg,device,boundaries=True)
            return selective.profile_intervals(current,samples,histories,fit,cfg,device)
        (profiles,_),resources=run_selective._measure(device,profile)
        common.write_json(profile_path,profiles)
    cm=CostModel(current.cfg.hidden_size,len(current.blocks),current.cfg.num_heads,cfg["attention_backend"])
    shapes=event_shapes(histories,by_user,users,cfg)
    policies,denominator=selective.make_policies(cm,shapes,profiles,cfg)
    common.write_json(output/"policies.json",dict(policies=policies,once_rebuild_flops=denominator,
        layer_profile_resources=resources,layer_boundary_state=layer_boundary_accounting(policies,shapes,cm)))
    if args.check:
        result=run_selective.check(parent,current,histories,by_user,users,policies,cfg,device,output,identity)
        result.update(layer_profile_resources=resources,seconds_total=time.perf_counter()-started)
        if sources()!=code_hashes:
            raise ValueError("execution sources changed during diagnostic")
        common.write_json(output/"check.json",result)
        print(json.dumps(dict(passed=result["passed"],users=len(users),quality_metrics_computed=False,
            seconds=result["seconds_total"])),flush=True)
    else:
        costs=run_selective.cost_ledgers(histories,by_user,users,policies,shapes,cm,cfg)
        common.write_json(output/"costs.json",costs)
        def progress(value):
            print(json.dumps(dict(stage="selective_scenario_quality",**value)),flush=True)
            common.write_json(output/"status.json",dict(state="quality",**value))
        scores,execution=fast.evaluate_checkpointed(parent,current,histories,by_user,users,policies,cfg,device,output,
            identity=dict(identity,layer_profile_sha256=sha256(profile_path)),progress=progress)
        if sources()!=code_hashes:
            raise ValueError("execution sources changed; retain score shards but do not report a mixed-source result")
        result=run_selective.summarize(scores,users,policies,costs,cfg,output,identity,execution)
        print(json.dumps(dict(users=len(users),feedback_events=result["feedback_events"],summary=str(output/"summary.json"))),flush=True)
    common.write_json(output/"status.json",dict(state="complete",mode="check" if args.check else "run",users=len(users),
        seconds=time.perf_counter()-started,source_sha256=code_hashes))


if __name__=="__main__":
    main()
