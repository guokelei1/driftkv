#!/usr/bin/env python3
"""Fit one GPU's declared PhaseB jobs with the actual initialization driver."""
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

import torch
from experiments.ch3_motivation.code.common import ROOT, write_json
from experiments.ch3_motivation.code import run_probe_calibration_init as driver
from experiments.ch3_motivation.code.probe_variants import load_variant
from hstu_kvcache.common.artifacts import sha256

DRIVER="run_probe_calibration_init.py"
ORDER=("reference","balanced_h_init","lr1e5","width256","cap1024","fit64","fit128")
RUNTIME_FIELDS=("input_root","output","sampling_manifest","edge","gpu","recent_fraction","gap_rule",
    "seed","validation_every","stages","controls","device","fit_users","width","learning_rate",
    "steps","patience","h_initialization")


def _verify(frozen):
    for name,digest in frozen.items():
        path=Path(name)
        if sha256(path if path.is_absolute() else ROOT/path)!=digest:
            raise ValueError(f"PhaseB frozen source/input changed: {name}")


def _jobs(plan,gpu):
    if (plan.get("format")!="probe_calibration_followup_v1" or
            plan["status"]!="prepared_for_root_review" or plan["expected_ladders"]!=35 or
            plan["expected_weights"]!=105 or len(plan["jobs"])!=35):
        raise ValueError("PhaseB requires the complete prepared35-ladder plan")
    outputs=set()
    for edge in range(1,6):
        rows=[job for job in plan["jobs"] if job["edge"]==edge]
        if tuple(row["name"] for row in rows)!=ORDER:
            raise ValueError("PhaseB requires the seven declared matched recipes in cache-sharing order")
        for job in rows:
            if (job["gpu"]!=(1 if edge%2 else 0) or job["fitting_driver"]!=DRIVER or
                    job["fitting_driver_sha256"]!=sha256(Path(driver.__file__)) or
                    job["reuse_existing_weights"] is not False or job["stages"]!="f,fg,fgh" or
                    job["controls"] is not False or job["seed"]!=17):
                raise ValueError("PhaseB GPU/driver/branch differs from its concrete plan")
            output=str(Path(job["output"]).resolve())
            if output in outputs:raise ValueError("PhaseB recipes cannot share an output directory")
            outputs.add(output)
            if job["expected_checkpoints"]!={stage:str(Path(output)/f"{stage}.pt") for stage in ("f","fg","fgh")}:
                raise ValueError("PhaseB checkpoint paths differ from the new job output")
    jobs=[job for job in plan["jobs"] if job["gpu"]==gpu]
    if jobs!=plan["gpu_jobs"][str(gpu)]:
        raise ValueError("PhaseB GPU subset differs from the full declared plan")
    return jobs


def _paired_predecessors(reference,balanced,expected):
    checks=[]
    for label in ("f","fg"):
        left_path=Path(reference)/f"{label}.pt";right_path=Path(balanced)/f"{label}.pt"
        with torch.random.fork_rng(devices=[]):
            left,left_meta=load_variant(left_path)
            right,right_meta=load_variant(right_path)
        for metadata,mode in ((left_meta,"standard"),(right_meta,"balanced")):
            if metadata["identity"]!=expected[mode] or metadata["method_label"]!=label:
                raise ValueError("PhaseB matched predecessors have a different actual identity")
        lstate,rstate=left.state_dict(),right.state_dict()
        if lstate.keys()!=rstate.keys() or any(not torch.equal(value,rstate[key]) for key,value in lstate.items()):
            raise ValueError(f"PhaseB standard/balanced changed the shared {label} predecessor")
        checks.append(dict(method_label=label,exact_tensor_equality=True,tensors=len(lstate),
            reference_checkpoint=str(left_path),reference_sha256=sha256(left_path),
            balanced_checkpoint=str(right_path),balanced_sha256=sha256(right_path)))
    return checks


def _execute(args,plan,jobs,identity,frozen):
    """Sequential actual driver calls; tests also use a single bounded CPU edge."""
    output=Path(args.output)
    points,completed,pair_checks=[],[],[]
    cache={};last_edge=None;paired={};started=time.perf_counter()
    for index,job in enumerate(jobs):
        _verify(frozen)
        if job["edge"]!=last_edge:cache.clear()
        last_edge=job["edge"]
        values={key:job[key] for key in RUNTIME_FIELDS}
        for key in ("input_root","output","sampling_manifest"):values[key]=Path(values[key])
        write_json(output/"progress.json",dict(status="running",index=index,total=len(jobs),edge=job["edge"],
            recipe=job["name"],completed=completed,seconds=time.perf_counter()-started))
        previous_key=cache.get("base_key")
        result=driver.run(argparse.Namespace(**values),preparation_cache=cache)
        invocation_reused=previous_key is not None and previous_key==cache.get("base_key")
        if (result["status"]!="fitted" or result["identity"]!=job["expected_identity"] or
                [row["method_label"] for row in result["stages"]]!=["f","fg","fgh"]):
            raise ValueError("PhaseB actual fitting identity or three-stage branch differs")
        for row in result["stages"]:
            checkpoint=Path(row["checkpoint"])
            if (str(checkpoint)!=job["expected_checkpoints"][row["method_label"]] or
                    sha256(checkpoint)!=row["checkpoint_sha256"] or row["quality_auc_computed"] or
                    row["identity"]!=job["expected_identity"]):
                raise ValueError("PhaseB checkpoint identity/SHA or fitting-only scope differs")
            chosen=next(history for history in row["history"] if history["step"]==row["best_step"])
            points.append(dict(name=job["name"],edge=job["edge"],seed=job["seed"],method_label=row["method_label"],
                fit_users=job["fit_users"],width=job["width"],learning_rate=job["learning_rate"],
                steps=job["steps"],patience=job["patience"],h_initialization=job["h_initialization"],
                recent_fraction=job["recent_fraction"],query_delta_rule=job["gap_rule"],
                best_step=row["best_step"],completed_steps=row["completed_steps"],
                training_view_validation_loss=chosen["validation_loss"],
                calibration_flops_estimate=row["cumulative_calibration_flops_estimate"],
                source_teacher_and_target_preparation_flops=row["source_teacher_and_target_preparation_flops"],
                h_initialization_weight_rescale_flops=row["h_initialization_weight_rescale_flops"],
                checkpoint=str(checkpoint),checkpoint_sha256=row["checkpoint_sha256"],
                checkpoint_search_preparation_reused=row["actual_search_preparation_reused"],
                raw_preparation_reused_this_invocation=invocation_reused,seconds=row["seconds"]))
        edge_pair=paired.setdefault(job["edge"],{})
        if job["name"] in ("reference","balanced_h_init"):
            edge_pair[job["name"]]=(values["output"],job["expected_identity"])
            if len(edge_pair)==2:
                reference,standard_identity=edge_pair["reference"]
                balanced,balanced_identity=edge_pair["balanced_h_init"]
                pair_checks.append(dict(edge=job["edge"],checks=_paired_predecessors(reference,balanced,
                    dict(standard=standard_identity,balanced=balanced_identity))))
                write_json(output/"paired_initialization.json",dict(status="checked",pairs=pair_checks,
                    scope="exact trained f/fg tensors; each checkpoint retains its own initialization identity"))
        summary=values["output"]/"summary.json"
        completed.append(dict(name=job["name"],edge=job["edge"],summary=str(summary),summary_sha256=sha256(summary),
            raw_preparation_reused_this_invocation=invocation_reused,
            standalone_model_preparation_flops=result["calibration_preparation"]["total_model_preparation_flops"],
            complete_ladder_calibration_flops_estimate=result["stages"][-1]["cumulative_calibration_flops_estimate"]))
        temporary=output/"points.csv.writing"
        with temporary.open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(points[0]));writer.writeheader();writer.writerows(points)
        temporary.replace(output/"points.csv")
    _verify(frozen)
    result=dict(status="complete",identity=identity,jobs=len(jobs),points=points,completed=completed,
        paired_initialization_checks=pair_checks,seconds=time.perf_counter()-started,quality_auc_computed=False,
        complete_standalone_calibration_flops_estimate_sum=sum(row["complete_ladder_calibration_flops_estimate"] for row in completed),
        standalone_model_preparation_flops_sum=sum(row["standalone_model_preparation_flops"] for row in completed),
        cost_scope="each standalone method includes actual teacherN+validation16 preparation, training, validation and h-rescale; raw cache reuse reduces actual search work only",
        weight_reuse_policy=plan["weight_reuse_policy"])
    write_json(output/"summary.json",result)
    write_json(output/"progress.json",dict(status="complete",completed=completed,total=len(jobs)))
    return result


def main(args):
    plan=json.loads(args.plan.read_text());jobs=_jobs(plan,args.gpu)
    frozen={**plan["frozen_sha256"],str(args.plan.resolve()):sha256(args.plan),
            str(Path(__file__).resolve()):sha256(Path(__file__))}
    _verify(frozen)
    identity=dict(plan=str(args.plan.resolve()),plan_sha256=sha256(args.plan),physical_gpu=args.gpu,
        fitting_driver=DRIVER,fitting_driver_sha256=sha256(Path(driver.__file__)),
        sweep_source_sha256=sha256(Path(__file__)),frozen_sha256=frozen)
    output=Path(args.output);configuration=output/"configuration.json"
    if configuration.exists() and json.loads(configuration.read_text())!=identity:
        raise ValueError("PhaseB sweep identity changed; use a new attempt")
    write_json(configuration,identity)
    return _execute(args,plan,jobs,identity,frozen)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--gpu",type=int,choices=(0,1),required=True)
    main(parser.parse_args())
