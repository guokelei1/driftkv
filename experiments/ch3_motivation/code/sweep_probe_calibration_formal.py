#!/usr/bin/env python3
"""Fit declared sampling main/curve jobs; legacy fitting keeps its own entry."""
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

from experiments.ch3_motivation.code import (
    run_probe_calibration, run_probe_calibration_init, resident_probe_fitting,
    plan_probe_quality as quality,
)
from experiments.ch3_motivation.code.common import ROOT, write_json
from hstu_kvcache.common.artifacts import sha256

DRIVERS={"run_probe_calibration.py":run_probe_calibration,
         "run_probe_calibration_init.py":run_probe_calibration_init}
SEEDS=(17,23,29)
RUNTIME_FIELDS=("input_root","sampling_manifest","edge","recent_fraction","gap_rule","seed",
    "validation_every","stages","controls","fit_users","width","learning_rate","steps","patience")


def _verify(frozen):
    for name,digest in frozen.items():
        path=Path(name)
        if sha256(path if path.is_absolute() else ROOT/path)!=digest:
            raise ValueError(f"formal fitting frozen source/input changed: {name}")


def _merge_frozen(*snapshots):
    frozen={}
    for snapshot in snapshots:
        for name,digest in snapshot.items():
            path=Path(name)
            path=str((path if path.is_absolute() else ROOT/path).resolve())
            if path in frozen and frozen[path]!=digest:
                raise ValueError(f"formal freeze SHA conflict: {path}")
            frozen[path]=digest
    return frozen


def _folder(root,family,job):
    folder=(Path(root)/family/job["name"]/f"edge_{job['edge']}"/f"seed_{job['seed']}").resolve()
    if "output" in job and Path(job["output"]).resolve()!=folder:
        raise ValueError("formal job output must use the separate main/curve execution root")
    return folder


def _jobs(plan,family,gpu,root):
    if plan["status"]!="prepared_for_root_review":
        raise ValueError("formal fitting waits for an actual final recommendation")
    expected_main={(e,s) for e in range(1,6) for s in SEEDS}
    expected_curve={(e,s,n) for e in range(1,6) for s in SEEDS for n in (64,128,256)}
    if (len(plan["main_jobs"])!=15 or {(j["edge"],j["seed"]) for j in plan["main_jobs"]}!=expected_main or
            len(plan["curve_jobs"])!=45 or {(j["edge"],j["seed"],j["fit_users"]) for j in plan["curve_jobs"]}!=expected_curve):
        raise ValueError("formal sampling requires the declared15 main ladders and45 curve jobs")
    paths=set();drivers=set()
    for selected in ("main","curve"):
        cap=set()
        for job in plan[f"{selected}_jobs"]:
            name=job["fitting_driver"];drivers.add(name)
            if name not in DRIVERS:
                raise ValueError("legacy fitting driver must use its existing entry; no implicit law conversion")
            if (job["fitting_driver_sha256"]!=sha256(Path(DRIVERS[name].__file__)) or
                    job["variant"]!="full_item" or job["depth"]!=1 or job["queries"]!=16 or
                    job["read_weight"]!=1. or job["score_weight"]!=1. or job["candidate_source"]!="feedback" or
                    job["candidate_catalog_source_users"]!=256 or job["time_law_source_users"]!=256 or
                    job["gradient_fitting_users"]!=job["fit_users"] or job["teacher_fitting_users"]!=job["fit_users"] or
                    job["gap_rule"]!=job["query_delta_rule"]):
                raise ValueError("formal sampling driver/target/source population differs")
            if selected=="main":
                if job["stages"]!="f,fg,fgh" or job["controls"] is not True:
                    raise ValueError("formal main needs its own control-enabled identity")
            elif job["stages"]!="f" or job["controls"] is not False or job["patience"]!=0 or job["steps"]<256:
                raise ValueError("formal curve requires f-only, own-N and patience0 fullcap")
            if name=="run_probe_calibration_init.py" and job.get("h_initialization") not in ("standard","balanced"):
                raise ValueError("formal initialization driver needs one explicit mode")
            if name=="run_probe_calibration.py" and any(key in job for key in
                    ("h_initialization","h_initialization_spec","initialization_helper_sha256")):
                raise ValueError("standard sampling driver cannot claim the initialization-driver identity")
            path=_folder(root,selected,job)
            if path in paths:raise ValueError("formal main/curve jobs cannot share an output")
            paths.add(path);cap.add(job["steps"])
        if selected=="curve" and len(cap)!=1:
            raise ValueError("formal population curves must execute one common optimization cap")
    if len(drivers)!=1:raise ValueError("one final recipe cannot mix the two fitting drivers")
    jobs=[j for j in plan[f"{family}_jobs"] if (1 if j["edge"]%2 else 0)==gpu]
    return sorted(jobs,key=lambda j:(j["edge"],-j["fit_users"],j["seed"]))


def _execution_jobs(jobs,resident_inputs):
    expected=dict(binding=quality.resident_execution_binding() if resident_inputs else None)
    result=[]
    for job in jobs:
        if "execution_adapter" in job and job["execution_adapter"]!=expected:
            raise ValueError("formal job execution adapter differs from the explicit runner mode")
        result.append(dict(job,execution_adapter=expected))
    return result


def _execute(args,plan,jobs,identity,frozen):
    root=Path(args.output).resolve();ledger=root/args.family/"execution"/f"gpu_{args.gpu}"
    bindings=plan[f"{args.family}_input_bindings"]
    resident_inputs=getattr(args,"resident_inputs",False)
    jobs=_execution_jobs(jobs,resident_inputs)
    methods=quality.MAIN_METHODS if args.family=="main" else ("f",)
    # Refuse mismatched modes for ALL existing jobs before any model/cache
    # preparation. The frozen drivers compare only their method identities.
    for job in jobs:
        folder=_folder(root,args.family,job)
        for method in methods:
            path=folder/f"{method}.pt"
            if path.exists():quality.audit_weight(path,job,bindings[str(job["edge"])],method)
    points,completed=[],[];cache={};last_edge=None;started=time.perf_counter()
    for index,job in enumerate(jobs):
        _verify(frozen)
        if job["edge"]!=last_edge:cache.clear()
        last_edge=job["edge"];folder=_folder(root,args.family,job)
        binding=bindings[str(job["edge"])]
        values={key:job[key] for key in RUNTIME_FIELDS}
        for key in ("input_root","sampling_manifest"):values[key]=Path(values[key])
        values.update(output=folder,gpu=args.gpu,device=args.device)
        if job["fitting_driver"]=="run_probe_calibration_init.py":values["h_initialization"]=job["h_initialization"]
        write_json(ledger/"progress.json",dict(status="running",index=index,total=len(jobs),family=args.family,
            edge=job["edge"],seed=job["seed"],fit_users=job["fit_users"],completed=completed,
            seconds=time.perf_counter()-started))
        previous_key=cache.get("base_key")
        driver=DRIVERS[job["fitting_driver"]]
        previous_fitter=driver.fit_probe
        try:
            if resident_inputs:driver.fit_probe=resident_probe_fitting.fit_probe_resident
            result=driver.run(argparse.Namespace(**values),preparation_cache=cache)
        finally:
            driver.fit_probe=previous_fitter
        invocation_reused=previous_key is not None and previous_key==cache.get("base_key")
        expected=("f","f_extended","fg","fg_extended","fgh") if args.family=="main" else ("f",)
        if result["status"]!="fitted" or tuple(row["method_label"] for row in result["stages"])!=expected:
            raise ValueError("formal fitting returned a different method branch")
        audit=[]
        for row in result["stages"]:
            method=row["method_label"];path=folder/f"{method}.pt"
            checked=quality.audit_weight(path,job,binding,method)
            if str(path)!=row["checkpoint"] or checked["checkpoint_sha256"]!=row["checkpoint_sha256"]:
                raise ValueError("formal fitted checkpoint does not match the actual audited file")
            if row["quality_auc_computed"]:raise ValueError("formal fitting cannot compute population AUC")
            if row.get("execution_adapter")!=checked.get("execution_adapter"):
                raise ValueError("formal stage summary and checkpoint execution adapter differ")
            audit.append(dict(method_label=method,checkpoint=str(path),**{k:v for k,v in checked.items() if k!="method_label"}))
            points.append(dict(family=args.family,name=job["name"],edge=job["edge"],seed=job["seed"],
                fit_users=job["fit_users"],method_label=method,best_step=row["best_step"],
                completed_steps=row["completed_steps"],fitting_driver=job["fitting_driver"],
                h_initialization=job.get("h_initialization","original_standard"),
                execution_adapter_format=(checked["execution_adapter"]["binding"]["format"]
                    if "execution_adapter" in checked else None),
                calibration_flops_estimate=row["cumulative_calibration_flops_estimate"],
                h_initialization_weight_rescale_flops=row.get("h_initialization_weight_rescale_flops",0),
                checkpoint=str(path),checkpoint_sha256=checked["checkpoint_sha256"],
                checkpoint_search_preparation_reused=row["actual_search_preparation_reused"],
                raw_preparation_reused_this_invocation=invocation_reused,seconds=row["seconds"]))
        completed.append(dict(edge=job["edge"],seed=job["seed"],fit_users=job["fit_users"],output=str(folder),
            expected_execution_adapter=job["execution_adapter"],
            summary_sha256=sha256(folder/"summary.json"),actual_weight_audit=audit,
            raw_preparation_reused_this_invocation=invocation_reused,
            standalone_model_preparation_flops=result["calibration_preparation"]["total_model_preparation_flops"]))
        temporary=ledger/"points.csv.writing"
        with temporary.open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(points[0]));writer.writeheader();writer.writerows(points)
        temporary.replace(ledger/"points.csv")
    _verify(frozen)
    result=dict(status="complete",family=args.family,identity=identity,jobs=len(jobs),points=points,
        completed=completed,seconds=time.perf_counter()-started,quality_auc_computed=False,
        execution_memory_scope=("resident adapter metrics are per-stage process/device peaks; adapter resets peak counters, so original driver peak fields are not whole-run peaks"
            if resident_inputs else "original driver memory reporting"),
        standalone_method_calibration_flops_estimate_sum=sum(row["calibration_flops_estimate"] for row in points),
        standalone_model_preparation_flops_sum=sum(row["standalone_model_preparation_flops"] for row in completed),
        cost_scope="per-weight cumulative standalone cost includes own-N teacher/validation preparation and preceding primary stages; raw search preparation reuse is separate",
        quality_roots=dict(main=str(root/"main"),curve=str(root/"curve")),
        quality_scope_gap="natural long_feedback_active2000 must later reuse these exact formal weights in a separate quality plan; this fitting entry does not add an evaluation population")
    write_json(ledger/"summary.json",result)
    write_json(ledger/"progress.json",dict(status="complete",total=len(jobs),completed=completed))
    return result


def main(args):
    plan=json.loads(args.plan.read_text());jobs=_jobs(plan,args.family,args.gpu,args.output)
    jobs=_execution_jobs(jobs,getattr(args,"resident_inputs",False))
    frozen=_merge_frozen(plan["frozen_sha256"],
        {str(args.plan.resolve()):sha256(args.plan),str(Path(__file__).resolve()):sha256(Path(__file__))},
        quality.sources())
    _verify(frozen)
    identity=dict(formal_plan=str(args.plan.resolve()),formal_plan_sha256=sha256(args.plan),family=args.family,
        physical_gpu=args.gpu,device=args.device,output_root=str(Path(args.output).resolve()),frozen_sha256=frozen,
        resident_inputs=getattr(args,"resident_inputs",False),
        execution_adapter=jobs[0]["execution_adapter"] if jobs else dict(binding=None))
    configuration=Path(args.output)/args.family/"execution"/f"gpu_{args.gpu}"/"configuration.json"
    if configuration.exists() and json.loads(configuration.read_text())!=identity:
        raise ValueError("formal execution identity changed; use a new output root")
    write_json(configuration,identity)
    return _execute(args,plan,jobs,identity,frozen)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--family",choices=("main","curve"),required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--gpu",type=int,choices=(0,1),required=True)
    parser.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    parser.add_argument("--resident-inputs",action="store_true",
        help="explicitly use the reviewed stage-local input placement adapter; default uses the original fitter")
    main(parser.parse_args())
