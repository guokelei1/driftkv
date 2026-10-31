#!/usr/bin/env python3
"""Plan seed17 formal-weight quality on the fixed natural active2000 cohort."""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
import torch
from experiments.ch3_motivation.code import plan_probe_quality as quality
from experiments.ch3_motivation.code.common import write_json
from experiments.ch3_motivation.code.scenario_inputs import load_scenario_population
from hstu_kvcache.common.artifacts import sha256

DEFINITION=dict(retained_history_rows_at_least=512,pre14_behavior_events_at_least=32,pre14_feedback_events_at_least=4)
DEFAULT_ANCHORS=ROOT/"experiments/ch3_motivation/runs/scenario_qualification/natural_active/attempt_1"
METHODS=("f","fg","fgh")


def _remember(frozen,path,digest):
    path=Path(path);path=str((path if path.is_absolute() else ROOT/path).resolve())
    if path in frozen and frozen[path]!=digest:
        raise ValueError(f"natural quality frozen SHA conflict: {path}")
    frozen[path]=digest


def sources():
    code=quality.sources()
    for path in (Path(__file__),Path(__file__).with_name("scenario_inputs.py"),
            Path(__file__).with_name("evaluate_probes.py"),Path(__file__).with_name("sweep_evaluate_probes.py")):
        code[str(path.relative_to(ROOT))]=sha256(path)
    return code


def _scenario(path,edge,binding,frozen):
    actual,users,population,provenance=load_scenario_population(path,edge,minimum_users=2000,require_positive=False)
    selection=provenance["selection"]
    if (len(users)!=2000 or selection["panel_name"]!="long_feedback_active" or
            selection["definition"]!=DEFINITION or actual["candidate_panels"]["long_feedback_active"]!=users):
        raise ValueError("natural quality must retain the existing fixed active2000 definition and all UIDs")
    strata=Path(selection["strata_manifest_path"])
    if sha256(strata)!=selection["strata_manifest_sha256"]:
        raise ValueError("natural publishing-time strata changed")
    chosen=json.loads(strata.read_text())["subgroups_by_edge"][str(edge)]["long_feedback_active"]["first_2000_uid_hashes"]
    if users!=chosen:
        raise ValueError("natural quality membership differs from its frozen UID selection")
    summary_path=Path(provenance["qualification_summary_path"])
    summary=json.loads(summary_path.read_text())
    if (summary.get("status")!="complete" or summary["edge"]!=edge or summary["panel"]!="long_feedback_active" or
            summary["identity"]["experiment"]!="fixed_natural_three_anchors"):
        raise ValueError("natural quality requires completed fresh three-anchor evidence on this cohort")
    calibration=Path(binding["input_root"])/f"edge_{edge}/inputs.json"
    if (sha256(calibration)!=binding["input_binding_sha256"] or
            provenance["calibration_input_binding_sha256"]!=binding["input_binding_sha256"] or
            provenance["source_population_sha256"]!=binding["population_sha256"] or
            json.loads(calibration.read_text())["settings"]!=actual["settings"]):
        raise ValueError("natural scenario and formal fitting calibration input/model context differ")
    scenario_calibration=Path(provenance["calibration_input_binding_path"])
    original=json.loads(scenario_calibration.read_text())
    # Equivalent copied calibration roots are accepted by SHA, not root name.
    references=((path,provenance["manifest_sha256"]),(strata,selection["strata_manifest_sha256"]),
        (summary_path,provenance["qualification_summary_sha256"]),
        (provenance["input_binding_path"],provenance["input_binding_sha256"]),
        (provenance["history_path"],provenance["history_sha256"]),
        (provenance["requests_path"],provenance["requests_sha256"]),
        (provenance["source_population_path"],provenance["source_population_sha256"]),
        (calibration,binding["input_binding_sha256"]),
        (scenario_calibration,provenance["calibration_input_binding_sha256"]),
        (scenario_calibration.parent/original["history"]["path"],original["history"]["sha256"]))
    for file,digest in references:_remember(frozen,file,digest)
    return dict(users=2000,provenance=provenance,anchor_models=provenance["anchor_models"],
        point_qualifies=provenance["qualification"]["point_qualifies"],
        overlap_with_main=len(set(users)&set(population.get("main",[]))),
        overlap_with_confirmation=len(set(users)&set(population.get("confirmation",[]))))


def generate(formal_path,main_root,*,anchors_root=DEFAULT_ANCHORS):
    formal_path,main_root,anchors_root=map(Path,(formal_path,main_root,anchors_root))
    code=sources();result=dict(status="waiting_selection",family="natural_long_feedback_active",gpu_launch_performed=False,
        formal_plan=str(formal_path.resolve()),formal_plan_sha256=sha256(formal_path),
        main_fitting_root=str(main_root.resolve()),anchors_root=str(anchors_root.resolve()),source_sha256=code,
        expected_points=15,users_per_edge=2000,seeds=[17],methods=list(METHODS),definition=DEFINITION,
        jobs=[],edges={},missing_weights=[],missing_anchor_manifests=[],failed_prerequisites=[],
        all_five_edges_required=True,require_positive=False,reselection_performed=False,
        independent_confirmation=False,main_and_confirmation_plans_modified=False,
        anchor_compatibility="same population, E14/native history, causal writes and model pair; E3 Layer intervals/budgets/configuration are not inputs to this planner",
        scope="supplementary natural-cohort comparison using exact main seed17 f/fg/fgh weights; all signs and all fixed UIDs retained; not an independent or three-seed confirmation")
    formal=json.loads(formal_path.read_text())
    if formal["status"]!="prepared_for_root_review":return result
    main=formal["main_jobs"]
    if len(main)!=15 or {(j["edge"],j["seed"]) for j in main}!={(e,s) for e in range(1,6) for s in (17,23,29)}:
        raise ValueError("natural quality requires the complete formal five-edge/three-seed main plan")
    frozen={}
    for snapshot in (formal.get("frozen_sha256",{}),code,{str(formal_path.resolve()):result["formal_plan_sha256"]}):
        for path,digest in snapshot.items():_remember(frozen,path,digest)
    planned=[];model_hashes={}
    for job in sorted((j for j in main if j["seed"]==17),key=lambda j:j["edge"]):
        edge=job["edge"];binding=formal["main_input_bindings"][str(edge)]
        manifest=(anchors_root/f"edge_{edge}/scenario_manifest.json").resolve()
        entry=dict(status="waiting",anchor_manifest=str(manifest),weights=[])
        result["edges"][str(edge)]=entry;scenario=None
        if not manifest.exists():result["missing_anchor_manifests"].append(str(manifest))
        else:
            try:
                scenario=_scenario(manifest,edge,binding,frozen)
                entry["scenario"]=scenario
            except (ValueError,KeyError,FileNotFoundError) as error:
                result["failed_prerequisites"].append(dict(edge=edge,kind="anchors",error=f"{type(error).__name__}: {error}"))
                entry["status"]="failed"
        for method in METHODS:
            path=quality.checkpoint_path(main_root,job,method).resolve()
            planned.append(dict(name=f"natural_edge_{edge}_seed_17_{method}",edge=edge,panel="natural_active",
                checkpoint=str(path),input_root=binding["input_root"],population_file=str(manifest),backend="projected"))
            if not path.exists():result["missing_weights"].append(str(path));continue
            try:
                checked=quality.audit_weight(path,job,binding,method)
                payload=torch.load(path,map_location="cpu",weights_only=False)
                models=payload["metadata"]["identity"]["models"]
                if scenario is not None and models!=scenario["anchor_models"]:
                    raise ValueError("formal weight backbones differ from the natural three anchors")
                for model,digest in models.items():
                    model_path=Path(model);model_path=(model_path if model_path.is_absolute() else ROOT/model_path).resolve()
                    if str(model_path) not in model_hashes:model_hashes[str(model_path)]=sha256(model_path)
                    if model_hashes[str(model_path)]!=digest:raise ValueError("natural/formal backbone checkpoint changed")
                    _remember(frozen,model_path,digest)
                _remember(frozen,path,checked["checkpoint_sha256"])
                entry["weights"].append(dict(method_label=method,checkpoint=str(path),**{k:v for k,v in checked.items() if k!="method_label"}))
            except (ValueError,KeyError,FileNotFoundError) as error:
                result["failed_prerequisites"].append(dict(edge=edge,kind="weight",checkpoint=str(path),error=f"{type(error).__name__}: {error}"))
                entry["status"]="failed"
        if entry["status"]!="failed" and scenario is not None and len(entry["weights"])==3:entry["status"]="validated"
    ready=all(entry["status"]=="validated" for entry in result["edges"].values())
    result.update(status="prepared_for_root_review" if ready else
        ("failed_prerequisites" if result["failed_prerequisites"] else "waiting_prerequisites"),
        jobs=planned if ready else [],planned_jobs=planned,available_weights=sum(len(e["weights"]) for e in result["edges"].values()),
        validated_edges=[int(e) for e,row in result["edges"].items() if row["status"]=="validated"],frozen_sha256=frozen)
    for path,digest in frozen.items():
        if sha256(Path(path))!=digest:raise ValueError(f"natural quality dependency changed during planning: {path}")
    if sources()!=code:raise ValueError("natural quality planner/audit sources changed")
    return result


def main(args):
    result=generate(args.formal_plan,args.main_fitting_root,anchors_root=args.anchors_root)
    write_json(args.output/"plan.json",result)
    print(json.dumps(dict(status=result["status"],jobs=len(result["jobs"]),gpu_launched=False)))
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-plan",type=Path,required=True)
    parser.add_argument("--main-fitting-root",type=Path,required=True)
    parser.add_argument("--anchors-root",type=Path,default=DEFAULT_ANCHORS)
    parser.add_argument("--output",type=Path,required=True)
    main(parser.parse_args())
