#!/usr/bin/env python3
"""Evaluate frozen checkpoints sequentially, sharing matched fresh anchors."""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

from experiments.ch3_motivation.code.common import ROOT, write_json
from experiments.ch3_motivation.code.evaluate_probes import run
from hstu_kvcache.common.artifacts import sha256


def _root_path(value):
    path=Path(value)
    return (path if path.is_absolute() else ROOT/path).resolve()


def _bindings(plan,jobs,plan_path,plan_sha,current_sources):
    """Merge only bindings already present in this actual quality plan."""
    frozen={}

    def add(path,digest):
        path=_root_path(path)
        if path in frozen and frozen[path]!=digest:
            raise ValueError(f"conflicting quality sweep SHA bindings: {path}")
        frozen[path]=digest

    add(plan_path,plan_sha)
    for field in ("source_sha256","frozen_sha256"):
        if field in plan:
            if not isinstance(plan[field],dict):
                raise ValueError(f"quality sweep {field} must contain path/SHA bindings")
            for path,digest in plan[field].items():add(path,digest)
    if "formal_plan" in plan or "formal_plan_sha256" in plan:
        if not plan.get("formal_plan") or not plan.get("formal_plan_sha256"):
            raise ValueError("quality sweep formal plan requires its existing path and SHA")
        add(plan["formal_plan"],plan["formal_plan_sha256"])
    if "actual_weight_audit" in plan:
        if not isinstance(plan["actual_weight_audit"],list):
            raise ValueError("quality sweep actual_weight_audit must list audited checkpoints")
        audited=set()
        for row in plan["actual_weight_audit"]:
            if not row.get("checkpoint") or not row.get("checkpoint_sha256"):
                raise ValueError("quality sweep weight audit lacks checkpoint/SHA")
            path=_root_path(row["checkpoint"])
            audited.add(path);add(path,row["checkpoint_sha256"])
        if not {_root_path(job["checkpoint"]) for job in jobs}<=audited:
            raise ValueError("quality sweep weight audit does not cover every job checkpoint")
    for path,digest in current_sources.items():add(path,digest)
    return frozen


def _verify_bindings(frozen):
    # Each canonical dependency, including large model/weight files, is hashed
    # once per boundary rather than again for every method/population point.
    for path,digest in frozen.items():
        if not path.is_file() or sha256(path)!=digest:
            raise ValueError(f"quality sweep frozen dependency changed or missing: {path}")


def main(args):
    plan_path=Path(args.plan).resolve()
    plan_bytes=plan_path.read_bytes()
    plan=json.loads(plan_bytes)
    plan_sha=hashlib.sha256(plan_bytes).hexdigest()
    jobs=plan.get("jobs",[])
    if (plan.get("status") not in ("prepared_for_root_review","queued_for_fixed512_development_after_numeric_checks")
            or not jobs):
        raise ValueError("quality sweep requires a ready, nonempty plan; waiting plans are not completed experiments")
    names=[job["name"] for job in jobs]
    if len(names)!=len(set(names)):
        raise ValueError("quality sweep job names must be unique")
    for job in jobs:
        for field in ("checkpoint","population_file"):
            if job.get(field) is not None and not _root_path(job[field]).is_file():
                raise ValueError(f"quality sweep dependency is missing before GPU preparation: {field}={job[field]}")
    source_paths=[Path(__file__),Path(__file__).with_name("evaluate_probes.py"),
        Path(__file__).with_name("scenario_inputs.py"),Path(__file__).with_name("probe_cost.py")]
    identity=dict(plan_sha256=plan_sha,physical_gpu=args.gpu,
        sources={str(p.relative_to(ROOT)):sha256(p) for p in source_paths})
    frozen=_bindings(plan,jobs,plan_path,plan_sha,identity["sources"])
    _verify_bindings(frozen)
    args.output.mkdir(parents=True,exist_ok=True)
    configuration=args.output/"configuration.json"
    if configuration.exists() and json.loads(configuration.read_text())!=identity:
        raise ValueError("frozen quality sweep changed")
    write_json(configuration,identity)
    defaults=dict(input_root=ROOT/"experiments/ch3_motivation/inputs",gpu=args.gpu,
        backend="graph",panel="main",population_seed=20261023,population_file=None)
    shared_context={}
    rows=[]
    started=time.perf_counter()
    for index,job in enumerate(jobs):
        if set(job)-(set(defaults)|{"edge","checkpoint","name"}):
            raise ValueError("unknown quality sweep options")
        if "gpu" in job and job["gpu"]!=args.gpu:
            raise ValueError("one prepared process cannot migrate its physical GPU")
        values=dict(defaults,**job)
        name=values.pop("name")
        values["output"]=args.output/name
        for field in ("input_root","checkpoint","population_file"):
            if values[field] is not None:values[field]=_root_path(values[field])
        result=run(argparse.Namespace(**values),shared_context=shared_context)
        rows.append({key:result[key] for key in ("edge","variant","method_label","stage","width","depth",
            "panel","seed","fit_users","quality_users","feedback_events","auc","reuse_auc","rebuild_auc","recovery_percent")})
        rows[-1].update(name=name,summary=str(values["output"]/"summary.json"),
            summary_sha256=sha256(values["output"]/"summary.json"),
            complete_added_flops=result["costs"]["complete_added_flops"],
            normalized_230_percent_of_once_rebuild=result["costs"]["candidate_reads_230"]["percent_of_once_rebuild"])
        with (args.output/"points.csv").open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        write_json(args.output/"progress.json",dict(completed=index+1,total=len(plan["jobs"]),
            last_summary=rows[-1]["summary"],seconds=time.perf_counter()-started))
    _verify_bindings(frozen)
    write_json(args.output/"summary.json",dict(status="complete",jobs=len(rows),identity=identity,
        results=rows,seconds=time.perf_counter()-started,
        scope="new fixed-population scores; all methods matched to freshly computed shared anchors; block hashes and own causal trajectories preserved"))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--gpu",type=int,default=1)
    main(parser.parse_args())
