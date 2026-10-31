#!/usr/bin/env python3
"""Sequential fitting-only sweep with one edge's prepared data shared in RAM."""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
import csv
import json
from pathlib import Path
import sys
import time

from experiments.ch3_motivation.code.common import ROOT, write_json
from experiments.ch3_motivation.code.run_probe_variants import run
from hstu_kvcache.common.artifacts import sha256


def main(args):
    plan=json.loads(args.plan.read_text())
    args.output.mkdir(parents=True,exist_ok=True)
    identity=dict(plan_sha256=sha256(args.plan),physical_gpu=args.gpu,
        sources={str(p.relative_to(ROOT)):sha256(p) for p in (
            Path(__file__),Path(__file__).with_name("run_probe_variants.py"),
            Path(__file__).with_name("probe_variants.py"),Path(__file__).with_name("probes.py"),
            Path(__file__).with_name("common.py"),Path(__file__).with_name("prepare_candidates.py"))})
    config=args.output/"configuration.json"
    if config.exists() and json.loads(config.read_text())!=identity:
        raise ValueError("fitting sweep configuration changed")
    write_json(config,identity)
    defaults=dict(input_root=ROOT/"experiments/ch3_motivation/inputs",gpu=args.gpu,seed=17,
        fit_users=64,queries=16,width=128,depth=1,steps=256,learning_rate=.0001,
        read_weight=1.,score_weight=1.,patience=128,validation_every=32,stages="f",
        controls=False,candidate_source="feedback",recent_fraction=.5)
    allowed=set(defaults)|{"variant","edge","name"}
    preparation_cache={}
    points=[]
    started=time.perf_counter()
    for index,job in enumerate(plan["jobs"]):
        if set(job)-allowed:
            raise ValueError(f"unknown sweep options: {set(job)-allowed}")
        if "gpu" in job and job["gpu"]!=args.gpu:
            raise ValueError("a prepared fitting process cannot change its physical GPU between jobs")
        values=dict(defaults,**job)
        name=values.pop("name")
        values["input_root"]=Path(values["input_root"])
        folder=args.output/name/f"edge_{values['edge']}"/f"seed_{values['seed']}"
        values["output"]=folder
        run(argparse.Namespace(**values),preparation_cache=preparation_cache)
        result=json.loads((folder/"summary.json").read_text())
        for stage in result["stages"]:
            chosen=next(row for row in stage["history"] if row["step"]==stage["best_step"])
            points.append(dict(name=name,edge=values["edge"],seed=values["seed"],
                variant=values["variant"],width=values["width"],depth=values["depth"],
                fit_users=values["fit_users"],method_label=stage["method_label"],
                best_step=stage["best_step"],completed_steps=stage["completed_steps"],
                validation_read_mse=chosen["validation_read_mse"],
                validation_score_mse=chosen["validation_score_mse"],
                common_validation_objective=chosen["validation_read_mse"]+chosen["validation_score_mse"],
                calibration_flops_estimate=stage["cumulative_calibration_flops_estimate"],
                checkpoint_sha256=stage["checkpoint_sha256"],checkpoint=stage["checkpoint"],
                actual_search_preparation_reused=stage.get("actual_search_preparation_reused",False)))
        with (args.output/"points.csv").open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(points[0]))
            writer.writeheader();writer.writerows(points)
        write_json(args.output/"progress.json",dict(completed=index+1,total=len(plan["jobs"]),
            seconds=time.perf_counter()-started,last_output=str(folder)))
    for name,digest in identity["sources"].items():
        if sha256(ROOT/name)!=digest:raise ValueError(f"sweep source changed: {name}")
    write_json(args.output/"summary.json",dict(status="complete",identity=identity,
        jobs=len(plan["jobs"]),points=points,seconds=time.perf_counter()-started,
        scope="pre-release fitting/validation only, all configurations retained; no evaluation AUC",
        cost_scope="physical search reuses preparation; every independent method still includes its complete preparation FLOPs"))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--gpu",type=int,default=1)
    main(parser.parse_args())
