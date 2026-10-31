#!/usr/bin/env python3
"""Bind the final common-validation winner to the original development512.

Uses the already fitted seed17 f/fg/fgh weights with their original control,
candidate/time and initialization identities. No formal fitting identity is
substituted, no population is screened, and this entry never runs a GPU.
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from experiments.ch3_motivation.code import plan_probe_quality as quality
from experiments.ch3_motivation.code.common import ROOT, write_json
from experiments.ch3_motivation.code.plan_probe_formal import _input_binding
from experiments.ch3_motivation.code.sampling_identity import SAMPLING_DRIVERS
from hstu_kvcache.common.artifacts import sha256

EDGES=(1,2,3,4,5)
METHODS=("f","fg","fgh")
DRIVERS=("run_probe_variants.py","run_probe_temporal.py",*SAMPLING_DRIVERS)
INPUTS=ROOT/"experiments/ch3_motivation/inputs"
SELECTION=ROOT/"experiments/ch3_motivation/runs/e5/calibration_followup_common_selection/attempt_1/selection.json"
EXISTING=ROOT/"experiments/ch3_motivation/runs/e5/development_quality/attempt_1"
EDGE_FIELDS={"edge","models","input_binding_sha256","calibration_candidate_sha256",
    "gap_manifest","gap_manifest_sha256","sampling_manifest","sampling_manifest_sha256","source_cutoff","input_root"}


def resolved(value):
    path=Path(value)
    return (path if path.is_absolute() else ROOT/path).resolve()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def runtime_sources():
    # The exact source identity written by evaluate_probes, also needed to
    # judge whether old scores have the same numerical evaluation context.
    names=("evaluate_probes.py","probes.py","probe_replay.py","common.py","run_probes.py",
        "run_probe_variants.py","probe_variants.py","selective.py","probe_fast.py","probe_ledger.py",
        "probe_cost.py","scenario_inputs.py","probe_projected_fast.py","probe_projected_cost.py")
    paths=[Path(__file__).with_name(name) for name in names]
    paths += [ROOT/name for name in ("src/hstu_kvcache/adaptation/reader.py",
        "src/hstu_kvcache/evaluation/fast_baselines.py","src/hstu_kvcache/evaluation/baselines.py",
        "src/hstu_kvcache/evaluation/statistics.py","src/hstu_kvcache/evaluation/cost.py",
        "src/hstu_kvcache/methods/baselines/protocol/cost.py")]
    return {str(path.relative_to(ROOT)):sha256(path) for path in paths}


def sources():
    code={**quality.sources(),**runtime_sources()}
    for path in (Path(__file__),Path(__file__).with_name("plan_probe_formal.py"),
                 Path(__file__).with_name("sweep_evaluate_probes.py")):
        code[str(path.relative_to(ROOT))]=sha256(path)
    return code


def _identity_job(identity,config,chosen):
    job=dict(identity,variant=chosen["variant"],width=config["hidden_width"],depth=config["depth"],
        fit_users=len(identity["fit_uids"]),fitting_driver=chosen["fitting_driver"],
        fitting_driver_sha256=chosen["fitting_driver_sha256"],query_delta_rule=chosen["query_delta_rule"])
    return job


def _recipe(identity):
    value=dict(identity,fitting_driver=identity.get("fitting_driver","run_probe_variants.py"),
        fitting_driver_sha256=identity.get("fitting_driver_sha256",
            identity.get("numerical_sources",{}).get("experiments/ch3_motivation/code/run_probe_variants.py")),
        query_delta_rule=identity.get("query_delta_rule","cutover"))
    return {key:item for key,item in value.items() if key not in EDGE_FIELDS}


def _reuse(path,job,identity,users,code,remember):
    """Verify committed artifacts by metadata/SHA, without reading score values."""
    if not path.is_file():return dict(status="not_found")
    summary=json.loads(path.read_text());old=summary.get("identity",{})
    wanted=dict(edge=job["edge"],panel="development",users=users,models=identity["models"],
        input_binding_sha256=identity["input_binding_sha256"],backend="projected",scenario=None,sources=code)
    if (summary.get("status")!="complete" or summary.get("experiment")!="E5" or
            summary.get("quality_users")!=512 or
            summary.get("seed")!=17 or summary.get("method_label")!=job["method"] or
            old.get("checkpoint_sha256")!=job["checkpoint_sha256"] or
            any(old.get(key)!=value for key,value in wanted.items())):
        return dict(status="different_weight_or_context",summary=str(path.resolve()))
    config=path.parent/"configuration.json"
    if not config.is_file() or json.loads(config.read_text())!=old:
        raise ValueError("matching development result has changed configuration")
    fingerprint=digest(old);covered=[]
    for marker_path in sorted((path.parent/"probe_blocks").glob("part_*.json")):
        marker=json.loads(marker_path.read_text());offset=int(marker_path.stem.split("_")[-1])
        block=marker["users"]
        if offset!=len(covered) or marker["fingerprint"]!=fingerprint or block!=users[offset:offset+len(block)]:
            raise ValueError("matching development score block UID/fingerprint changed")
        remember(marker_path,sha256(marker_path));remember(marker_path.with_suffix(".parquet"),marker["sha256"])
        covered.extend(block)
    if covered!=users:raise ValueError("matching development probe blocks are incomplete")
    folder=Path(summary["anchor_source"]["path"])
    anchor_config=folder.parent/"configuration.json";anchor=json.loads(anchor_config.read_text())
    if any(anchor.get(key)!=value for key,value in wanted.items()):
        raise ValueError("matching development anchor context differs")
    key=digest({name:old[name] for name in ("edge","models","users","input_binding_sha256","sources")})
    if summary["anchor_source"]["key"]!=key:raise ValueError("matching development anchor key changed")
    covered=[]
    for marker_path in sorted((folder/"quality_shards").glob("part_*.json")):
        marker=json.loads(marker_path.read_text());offset=int(marker_path.stem.split("_")[-1]);n=marker["users"]
        block=users[offset:offset+n]
        if (offset!=len(covered) or marker["offset"]!=offset or marker["fingerprint"]!=digest(anchor) or
                marker["uids_sha256"]!=hashlib.sha256(json.dumps(block).encode()).hexdigest() or len(block)!=n):
            raise ValueError("matching development anchor block UID/fingerprint changed")
        remember(marker_path,sha256(marker_path));remember(marker_path.with_suffix(".parquet"),marker["scores_sha256"])
        covered.extend(block)
    if covered!=users:raise ValueError("matching development anchor blocks are incomplete")
    for file in (path,config,anchor_config):remember(file,sha256(file))
    remember(old["checkpoint"],job["checkpoint_sha256"])
    return dict(status="exact_sha_reuse",summary=str(path.resolve()),summary_sha256=sha256(path),
        checkpoint_sha256=job["checkpoint_sha256"],score_values_read=False,
        scope="same seed17 weights and full development512/model/input/backend/numerical-source context; committed probe and anchor block hashes verified")


def generate(selection_path=SELECTION,*,input_root=INPUTS,existing_root=EXISTING):
    selection_path,input_root=resolved(selection_path),resolved(input_root)
    code=sources()
    result=dict(format="selected_development512_v1",status="waiting_final_selection",jobs=[],
        expected_points=15,users_per_point=512,seeds=[17],methods=list(METHODS),edges=list(EDGES),
        source_selection=str(selection_path),source_sha256=code,gpu_launch_performed=False,
        recipe_selection_performed=False,independent_confirmation=False,formal_training_launched=False,
        scope="one final475 common-validation winner, original fixed development512 only; exploratory teacher-loss to AUC check; no main/confirmation method scores or population selection")
    if not selection_path.is_file():return result
    selection_sha=sha256(selection_path);selection=json.loads(selection_path.read_text())
    result["source_selection_sha256"]=selection_sha
    if (selection.get("status")!="recommendation_for_review" or
            selection.get("all_five_edges_complete") is not True or
            selection.get("phase_b_completed") is not True or not selection.get("recommendation")):
        return result
    chosen=selection["recommendation"]
    if (selection.get("selection_seed")!=17 or chosen.get("seed")!=17 or chosen["method_label"]!="fgh" or
            chosen["edges"]!=list(EDGES) or chosen["fitting_driver"] not in DRIVERS or
            set(chosen["checkpoints_by_edge"])!={str(edge) for edge in EDGES}):
        raise ValueError("final recommendation must bind one supported five-edge seed17 primary recipe")
    population_path=input_root/"population.json";population=json.loads(population_path.read_text())
    users=population["development"]
    if (len(users)!=512 or len(set(users))!=512 or
            set(users)&set(population["fit"]+population["validation"]+population["main"]+population["confirmation"])):
        raise ValueError("retain the original disjoint development512 role without rescreening")
    frozen={};checks={}
    def remember(path,digest):
        name=str(resolved(path))
        if name in frozen and frozen[name]!=digest:raise ValueError(f"selected development SHA conflict: {name}")
        frozen[name]=digest
    remember(selection_path,selection_sha);remember(population_path,sha256(population_path))
    for path,value in code.items():remember(path,value)
    missing=[str(resolved(row["checkpoint"]).parent/f"{method}.pt")
             for row in chosen["checkpoints_by_edge"].values() for method in METHODS
             if not (resolved(row["checkpoint"]).parent/f"{method}.pt").is_file()]
    if missing:return dict(result,status="waiting_selected_weights",missing_weights=missing)
    jobs=[];audits=[];reused=[];runtime=runtime_sources()
    for edge in EDGES:
        reference=chosen["checkpoints_by_edge"][str(edge)];path=resolved(reference["checkpoint"])
        if reference["status"]!="validated" or sha256(path)!=reference["checkpoint_sha256"]:
            raise ValueError("selected fgh checkpoint changed after common validation")
        full=torch.load(path,map_location="cpu",weights_only=False,mmap=True)
        identity,config=full["metadata"]["identity"],full["config"]
        settings=chosen["settings"]
        if (_recipe(identity)!=_recipe(settings) or identity["edge"]!=edge or identity["seed"]!=17 or
                config["hidden_width"]!=chosen["width"] or config["depth"]!=chosen["depth"] or
                len(identity["fit_uids"])!=chosen["fit_users"] or
                full["metadata"]["best_step"]!=reference["best_step"] or
                full["metadata"]["completed_steps"]!=reference["completed_steps"]):
            raise ValueError("selected actual weight/recipe/seed differs from the recommendation")
        calibration_root=resolved(chosen.get("input_bindings_by_edge",{}).get(str(edge),{}).get("input_root",settings["input_root"]))
        binding=_input_binding(calibration_root,edge)
        expected=chosen.get("input_bindings_by_edge",{}).get(str(edge))
        if expected and any(expected[key]!=binding[key] for key in
                ("population_sha256","input_binding_sha256","candidate_file_sha256")):
            raise ValueError("selected calibration input binding changed")
        evaluation_path=input_root/f"edge_{edge}/inputs.json";evaluation=json.loads(evaluation_path.read_text())
        if (sha256(evaluation_path)!=identity["input_binding_sha256"] or
                sha256(population_path)!=identity["population_sha256"]):
            raise ValueError("selected calibration and original development input/population differ")
        for base,record in ((input_root,evaluation),
                            (calibration_root,json.loads((calibration_root/f"edge_{edge}/inputs.json").read_text()))):
            remember(base/f"edge_{edge}/inputs.json",identity["input_binding_sha256"])
            remember(base/"population.json",identity["population_sha256"])
            for kind in ("history","requests"):
                remember(base/f"edge_{edge}"/record[kind]["path"],record[kind]["sha256"])
        remember(binding["candidate_file"],binding["candidate_file_sha256"])
        if chosen["fitting_driver"] in SAMPLING_DRIVERS:
            binding.update(sampling_manifest=identity["sampling_manifest"],sampling_manifest_sha256=identity["sampling_manifest_sha256"])
        for field in ("sampling_provenance_by_edge","gap_provenance_by_edge"):
            provided=chosen.get(field,{}).get(str(edge),{})
            for name in ("sampling_manifest","sampling_manifest_sha256","gap_manifest","gap_manifest_sha256",
                    "query_delta_rule","time_rule","source_cutoff","candidate_order","query_delta_layout",
                    "h_initialization","h_initialization_spec","initialization_helper_sha256"):
                if name in provided and provided[name]!=identity.get(name,"cutover" if name=="query_delta_rule" else None):
                    raise ValueError(f"selected per-edge {name} provenance differs from the actual weight")
        for name in ("sampling_manifest","gap_manifest"):
            if identity.get(name):remember(identity[name],identity[name+"_sha256"])
        for name,value in identity.get("numerical_sources",{}).items():remember(name,value)
        for name,value in identity["models"].items():remember(name,value)
        audit_job=_identity_job(identity,config,chosen)
        checks[str(edge)]=dict(actual_training_identity=identity,audit_job=audit_job,calibration_binding=binding)
        for method in METHODS:
            checkpoint=path.parent/f"{method}.pt"
            payload=torch.load(checkpoint,map_location="cpu",weights_only=False,mmap=True)
            if payload["metadata"]["identity"]!=identity:
                raise ValueError("selected f/fg/fgh do not share their original fitting identity/controls")
            checked=quality.audit_weight(checkpoint,audit_job,binding,method)
            remember(checkpoint,checked["checkpoint_sha256"])
            jobs.append(dict(name=f"edge_{edge}_{method}",edge=edge,checkpoint=str(checkpoint),
                input_root=str(input_root),panel="development",population_file=None,backend="projected"))
            audits.append(dict(checkpoint=str(checkpoint),**checked))
            if existing_root is not None:
                row=_reuse(resolved(existing_root)/f"edge_{edge}_{method}/summary.json",
                    dict(edge=edge,method=method,checkpoint_sha256=checked["checkpoint_sha256"]),
                    identity,users,runtime,remember)
                reused.append(dict(edge=edge,method=method,**row))
    for name,value in frozen.items():
        if sha256(Path(name))!=value:raise ValueError(f"selected development dependency changed: {name}")
    reuse_keys={(row["edge"],row["method"]) for row in reused if row["status"]=="exact_sha_reuse"}
    execution_jobs=[job for job in jobs if (job["edge"],Path(job["checkpoint"]).stem) not in reuse_keys]
    complete_reuse=len(reuse_keys)==15
    result.update(status="reuse_existing_complete" if complete_reuse else "prepared_for_root_review",
        jobs=execution_jobs,planned_jobs=jobs,actual_weight_audit=audits,
        recipe=chosen["recipe"],fitting_driver=chosen["fitting_driver"],query_delta_rule=chosen["query_delta_rule"],
        selected_h_step_zero_edges=chosen.get("step_zero_edges",[]),development_uids=users,
        development_uids_sha256=digest(users),input_root=str(input_root),audited_edges=checks,
        frozen_sha256=frozen,reuse=dict(complete=complete_reuse,points=reused,reused_points=len(reuse_keys),
            execution_points=len(execution_jobs),
            action="reuse committed results; no GPU jobs" if complete_reuse else "execute only jobs without exact committed reuse; planned_jobs plus reuse retain the full15-point comparison"))
    if sources()!=code:raise ValueError("selected development planner/audit/evaluation source changed")
    return result


def main(args):
    result=generate(args.selection,input_root=args.input_root,existing_root=args.existing_root)
    write_json(args.output/"plan.json",result)
    print(json.dumps(dict(status=result["status"],jobs=len(result["jobs"]),gpu_launched=False)))
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection",type=Path,default=SELECTION)
    parser.add_argument("--input-root",type=Path,default=INPUTS)
    parser.add_argument("--existing-root",type=Path,default=EXISTING)
    parser.add_argument("--output",type=Path,required=True)
    main(parser.parse_args())
