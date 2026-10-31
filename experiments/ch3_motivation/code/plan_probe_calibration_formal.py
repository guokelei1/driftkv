#!/usr/bin/env python3
"""Prepare formal sampling-v1 E5 jobs from an external final recommendation.

This does not select an input recipe, fit weights or launch GPU work. Each
selected edge is rechecked against its actual typed checkpoint and manifest.
The population curve keeps the catalog/law256 view fixed and fits its own N.
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
from pathlib import Path
import sys

import torch
from experiments.ch3_motivation.code import plan_probe_temporal_formal
from experiments.ch3_motivation.code.plan_probe_formal import _input_binding
from experiments.ch3_motivation.code.sampling_identity import (
    SAMPLING_DRIVERS, validate_sampling_identity, validate_sampling_capture, sampling_source_paths,
)
from experiments.ch3_motivation.code.select_probe_common import ROOT, EDGES, FORMAL_SEEDS, sha256, write_json

STEPS=(512,1024,2048)
CURVE_USERS=(64,128,256)
JOB_SETTINGS=("learning_rate","steps","read_weight","score_weight","patience",
    "validation_every","candidate_source","recent_fraction","queries")
SAMPLING_FIELDS=("fitting_driver","fitting_driver_sha256","query_delta_rule","sampling_manifest",
    "sampling_manifest_sha256","gap_manifest","gap_manifest_sha256","time_rule","source_cutoff",
    "candidate_order","query_delta_layout","candidate_catalog_source_users","time_law_source_users",
    "gradient_fitting_users","teacher_fitting_users")
INITIALIZATION_FIELDS=("h_initialization","h_initialization_spec","initialization_helper_sha256")
EDGE_FIELDS={"edge","models","input_binding_sha256","calibration_candidate_sha256",
             "sampling_manifest","sampling_manifest_sha256","gap_manifest","gap_manifest_sha256","source_cutoff"}


def source_manifest(driver=None):
    names=["plan_probe_calibration_formal.py","sampling_identity.py","calibration_sampling.py",
        "calibration_gaps.py","common.py","probes.py","probe_variants.py","prepare_candidates.py",
        "plan_probe_formal.py","plan_probe_temporal_formal.py","run_probe_temporal.py","select_probe_common.py"]
    if driver is None:names.extend(SAMPLING_DRIVERS)
    else:names.append(driver)
    if driver=="run_probe_calibration_init.py":names.append("probe_initialization.py")
    paths=[Path(__file__).with_name(name).resolve() for name in dict.fromkeys(names)]
    paths.extend(sampling_source_paths())
    paths += [ROOT/"src/hstu_kvcache/adaptation/reader.py",ROOT/"src/hstu_kvcache/models/hstu.py",
              ROOT/"src/hstu_kvcache/models/embeddings.py",ROOT/"src/hstu_kvcache/evaluation/cost.py",
              ROOT/"src/hstu_kvcache/methods/baselines/protocol/cost.py",ROOT/"src/hstu_kvcache/common/artifacts.py"]
    return {str(Path(path).resolve()):sha256(path) for path in paths}


def _scope(path):
    scope=json.loads(Path(path).read_text())
    e5=scope["e5"]
    if (scope["edges"]!=EDGES or scope["training_seeds"]!=FORMAL_SEEDS or
            e5["main"]["users"]!=2000 or e5["main"]["seeds"]!=FORMAL_SEEDS or
            e5["main"]["same_config_all_edges"] is not True or
            e5["main"]["stages"]!=["f","fg","fgh"] or
            set(e5["main"]["methods"])!={"f","fg","fgh","f_extended","fg_extended"} or
            e5["confirmation"]["users"]!=2000 or e5["confirmation"]["seeds"]!=[17] or
            e5["confirmation"]["stages"]!=["f","fg","fgh"] or
            e5["confirmation"]["independent_of_main_users"] is not True or
            e5["fitting_population_curve"]["users"]!=list(CURVE_USERS) or
            e5["fitting_population_curve"]["seeds"]!=FORMAL_SEEDS or
            e5["fitting_population_curve"]["method"]!="f" or e5["controls"]["seeds"]!=FORMAL_SEEDS):
        raise ValueError("formal scope differs from five edges, seeds17/23/29 and separate main/confirmation/curve")
    return scope


def _normalized(identity):
    return {key:value for key,value in identity.items() if key not in EDGE_FIELDS and key!="input_root"}


def _supported(settings,chosen):
    if (chosen["variant"]!="full_item" or chosen["width"] not in (128,256) or chosen["depth"]!=1 or
            chosen["fit_users"] not in CURVE_USERS or settings["steps"] not in STEPS or
            settings["learning_rate"] not in (.00003,.00001) or settings["patience"] not in (0,128,512,1024)):
        raise ValueError("selected recipe is not implemented by the concrete sampling fitting driver")
    expected=dict(variant="full_item",width=chosen["width"],depth=1,fit_users=chosen["fit_users"],
        queries=16,validation_every=8,read_weight=1.,score_weight=1.,candidate_source="feedback",
        candidate_seed=17,row_chunk=1024,query_chunk=16,weight_decay=.0001,gradient_clip=5.,
        stages="f,fg,fgh",candidate_catalog_source_users=256,time_law_source_users=256,
        gradient_fitting_users=chosen["fit_users"],teacher_fitting_users=chosen["fit_users"],
        query_delta_layout="[B,16]",fitting_uses_future_feedback=False)
    if (any(settings.get(key)!=value for key,value in expected.items()) or
            settings["recent_fraction"] not in (.5,.75) or
            settings["query_delta_rule"] not in ("cutover","gap_mix50","gap_empirical100") or
            settings["fit_users"]==256 and settings["patience"] and settings["patience"]<512):
        raise ValueError("selected sampling targets, sources, query rules or optimization settings differ")


def _read_selected(chosen,settings,edge,cache):
    reference=chosen["checkpoints_by_edge"][str(edge)]
    path=Path(reference["checkpoint"]).resolve()
    if reference.get("status")!="validated" or sha256(path)!=reference["checkpoint_sha256"]:
        raise ValueError("selected checkpoint SHA changed")
    payload=torch.load(path,map_location="cpu",weights_only=False)
    config,metadata=payload["config"],payload["metadata"]
    identity=metadata["identity"]
    if (payload.get("format")!="ch3_probe_variant_v1" or payload.get("variant")!="full_item" or
            payload["stage"]!="fgh" or metadata.get("method_label")!="fgh" or
            identity["edge"]!=edge or identity["seed"]!=17 or
            metadata["seed"]!=17 or metadata["best_step"]!=reference["best_step"] or
            metadata["completed_steps"]!=reference["completed_steps"] or
            config["hidden_width"]!=chosen["width"] or config["depth"]!=chosen["depth"] or
            not config["include_item"] or not config["normalize"] or
            _normalized(identity)!=_normalized(settings)):
        raise ValueError("selected typed checkpoint architecture/seed/branch/recipe differs")
    if metadata.get("diagnostic_only") or metadata.get("quality_use_permitted") is False:
        raise ValueError("diagnostic-only weights cannot define the final fitting recipe")
    driver=identity["fitting_driver"]
    if (driver!=chosen["fitting_driver"] or identity["fitting_driver_sha256"]!=chosen["fitting_driver_sha256"] or
            sha256(Path(__file__).with_name(driver))!=identity["fitting_driver_sha256"]):
        raise ValueError("selected actual fitting driver/SHA differs")
    for name,digest in identity["numerical_sources"].items():
        if sha256(ROOT/name)!=digest:
            raise ValueError(f"selected numerical source changed: {name}")
    sampling=validate_sampling_identity(identity,sampling_cache=cache,config=config)
    validate_sampling_capture(identity,metadata)
    scales=metadata["fit_pool_loss_scales"]
    if (scales["users"]!=chosen["fit_users"] or scales["source"]!="fitting_users_only" or
            scales["version"]!="fit_pool_equal_user_length_normalized_v1" or
            scales["validation_uses_fitting_scales"] is not True):
        raise ValueError("selected training loss scales do not use its own fitting users")
    provided=chosen["sampling_provenance_by_edge"][str(edge)]
    if any(provided.get(key)!=sampling[key] for key in SAMPLING_FIELDS if key not in ("fitting_driver","fitting_driver_sha256")):
        raise ValueError("selected per-edge sampling provenance differs from the real weight")
    binding=_input_binding(settings["input_root"],edge)
    if (binding["candidate_catalog_source_users"]!=256 or
            binding["input_binding_sha256"]!=identity["input_binding_sha256"] or
            binding["population_sha256"]!=identity["population_sha256"] or
            binding["candidate_file_sha256"]!=identity["calibration_candidate_sha256"]):
        raise ValueError("selected calibration input root/binding differs")
    return identity,config,sampling,dict(binding,**sampling)


def generate(selection_path,scope_path,*,curve_input_root=None):
    selection_path,scope_path=Path(selection_path),Path(scope_path)
    scope=_scope(scope_path)
    result=dict(status="waiting_selection",gpu_launch_performed=False,recipe_selection_performed=False,
        source_scope=str(scope_path.resolve()),source_scope_sha256=sha256(scope_path),
        planner_source_sha256=sha256(Path(__file__)),planner_sources_sha256=source_manifest(),
        edges=EDGES,correction_seeds=FORMAL_SEEDS,backbone_retraining=False,
        main_quality=scope["e5"]["main"],confirmation_quality=scope["e5"]["confirmation"],
        main_jobs=[],curve_jobs=[],main_input_bindings={},curve_input_bindings={},
        scope="only materialize an external final recommendation after PhaseB; no input winner chosen, no fitting/quality launch")
    if not selection_path.exists():return result
    selection=json.loads(selection_path.read_text())
    result.update(source_selection=str(selection_path.resolve()),source_selection_sha256=sha256(selection_path),
                  source_selection_scope=selection.get("selection_scope"))
    if selection["status"]!="recommendation_for_review" or not selection["all_five_edges_complete"]:return result
    chosen=selection["recommendation"];settings=chosen["settings"]
    if (selection["selection_seed"]!=17 or chosen["method_label"]!="fgh" or chosen["edges"]!=EDGES or
            chosen["correction_seeds"]!=FORMAL_SEEDS or set(chosen["checkpoints_by_edge"])!={str(e) for e in EDGES}):
        raise ValueError("final recommendation does not identify one five-edge seed17 primary recipe")
    driver=chosen["fitting_driver"]
    if driver not in SAMPLING_DRIVERS:
        delegated=plan_probe_temporal_formal.generate(selection_path,scope_path,curve_input_root=curve_input_root)
        delegated.update(calibration_planner_source_sha256=sha256(Path(__file__)),
            delegated_planner_source_sha256=sha256(Path(plan_probe_temporal_formal.__file__)),
            planner_sources_sha256=source_manifest(driver),
            recipe_selection_performed=False)
        return delegated
    _supported(settings,chosen)
    if (settings["fitting_driver"]!=driver or settings["fitting_driver_sha256"]!=chosen["fitting_driver_sha256"] or
            settings["query_delta_rule"]!=chosen["query_delta_rule"]):
        raise ValueError("recommended driver/rule differs from its actual settings")
    root=Path(settings["input_root"]).resolve()
    if curve_input_root is not None and Path(curve_input_root).resolve()!=root:
        raise ValueError("sampling curve must retain the selected fixed256 catalog/law input root")
    population=json.loads((root/"population.json").read_text())
    for name in ("main","confirmation"):
        if len(population[name])!=2000 or len(set(population[name]))!=2000:
            raise ValueError("formal quality role must retain its frozen2000 UIDs")
    if (set(population["main"])&set(population["confirmation"]) or
            set(population["main"]+population["confirmation"])&set(population["fit"]+population["validation"])):
        raise ValueError("formal fitting/validation/main/confirmation populations overlap")
    missing=[row["checkpoint"] for row in chosen["checkpoints_by_edge"].values() if not Path(row["checkpoint"]).exists()]
    if missing:return dict(result,status="waiting_selected_weights",missing_selected_weights=missing)
    sources=source_manifest(driver);cache={};prepared={}
    for edge in EDGES:
        prepared[edge]=_read_selected(chosen,settings,edge,cache)
    fixed=dict(variant="full_item",width=chosen["width"],depth=1,fit_users=chosen["fit_users"],
        input_root=str(root),**{key:settings[key] for key in JOB_SETTINGS})
    cap=max(256,int(settings["steps"]))
    for edge in EDGES:
        identity,config,sampling,binding=prepared[edge]
        result["main_input_bindings"][str(edge)]=binding
        result["curve_input_bindings"][str(edge)]=dict(binding)
        temporal={key:identity[key] for key in SAMPLING_FIELDS}
        initialization={key:identity[key] for key in INITIALIZATION_FIELDS} if driver=="run_probe_calibration_init.py" else {}
        for seed in FORMAL_SEEDS:
            result["main_jobs"].append(dict(fixed,**temporal,**initialization,name="selected_main",edge=edge,
                seed=seed,stages="f,fg,fgh",controls=True,gap_rule=identity["query_delta_rule"]))
            for n in CURVE_USERS:
                job={**fixed,**temporal,**initialization,"name":f"selected_f_curve_fit{n}","edge":edge,
                    "seed":seed,"fit_users":n,"gradient_fitting_users":n,"teacher_fitting_users":n,
                    "stages":"f","controls":False,"patience":0,"steps":cap,"gap_rule":identity["query_delta_rule"]}
                # Reuse the real loader for every prospective N: the same
                # immutable full256 manifest must cover each prefix plusval16.
                prospective=dict(identity,edge=edge,seed=seed,fit_users=n,fit_uids=population["fit"][:n],
                    gradient_fitting_users=n,teacher_fitting_users=n,stages="f",controls=False,patience=0,steps=cap)
                validate_sampling_identity(prospective,sampling_cache=cache,config=config)
                result["curve_jobs"].append(job)
    result.update(status="prepared_for_root_review",main_status="prepared_for_root_review",curve_plan_status="prepared_for_root_review",
        planner_sources_sha256=sources,selected_recipe=chosen["recipe"],selected_development_seed=17,
        selected_fitting_driver=driver,selected_fitting_driver_sha256=chosen["fitting_driver_sha256"],
        selected_query_delta_rule=chosen["query_delta_rule"],selected_h_step_zero_edges=chosen["step_zero_edges"],
        e6_uses_main_fgh_without_refitting=True,existing_seed17_fgh_sources=chosen["checkpoints_by_edge"],
        existing_weight_reuse="no implicit reuse: source controls/stages/driver/initialization must match the complete formal identity and SHA; otherwise fit the declared fresh seed",
        numerical_constants={key:settings[key] for key in ("row_chunk","query_chunk","weight_decay","gradient_clip")},
        controls=dict(f_extended="independent best-f copy, extra2x base cap",fg_extended="independent best-fg copy, extra1x base cap",
            branch_rule="controls never replace the primary predecessor; charge actual executed training and validation"),
        population_curve_policy=dict(gradient_fit_users=list(CURVE_USERS),teacher_fit_users=list(CURVE_USERS),
            candidate_catalog_source_users=256,time_law_source_users=256,validation_users=16,
            training_statistics="own-N input statistics and fit-pool loss RMS using the unchanged fit_probe",
            candidate_time_view="same selected fraction/rule/Q16/seed17 and same fixed256 sampling manifest on every N within an edge",
            feedback_fraction_semantics="50/75% is the own-plus-other-reserved-fit-user observed feedback pool quota; it is not the fraction supplied by each user's own feedback",
            optimizer_budget=f"same cap{cap},patience0,full cap executed; still restore best validated checkpoint",
            common_validation_loss_RMS_source_users=64,common_validation_contexts=3,
            original_validation_losses_cross_N_comparable=False,
            interpretation="natural fitting-size curve with own-N normalization; catalog/law256 is shared input information, not teacher256 when N64"),
        population_accounting=[dict(gradient_fit_users=n,teacher_fit_users=n,training_loss_RMS_source_users=n,
            candidate_catalog_source_users=256,time_law_source_users=256,private_validation_users=16,
            common_validation_loss_RMS_source_users=64,common_validation_contexts=3) for n in CURVE_USERS],
        curve_steps_per_job=cap,reuse_main_f_for_curve=False,
        job_counts=dict(main_ladders=15,main_stage_weights_with_controls=75,curve_f_weights=45,
            main_quality_points=75,confirmation_quality_points=15,confirmation_additional_fitting=0),
        evaluation_reuse="main2000 at17/23/29; independentconfirmation2000 reuses exact main seed17 f/fg/fgh; E6 reuses matched main fgh",
        cost_scope=scope["common"]["cost_model_selection_scope"],
        dispatch=dict(driver=driver,accepted_cli_fields=["input_root","output","sampling_manifest","edge","gap_rule",
            "recent_fraction","fit_users","width","learning_rate","steps","patience","validation_every","stages",
            "controls","seed","gpu","device"]+(["h_initialization"] if driver=="run_probe_calibration_init.py" else []),
            policy="pass only accepted CLI fields; remaining fields are exact quality/provenance recipe metadata"),
        seed_claim="configuration selection uses17 only; 23/29 repeat the frozen recipe and do not select it again",
        failure_review=selection.get("needs_failure_review",False))
    frozen={**sources,str(selection_path.resolve()):result["source_selection_sha256"],
        str(scope_path.resolve()):result["source_scope_sha256"]}
    for edge in EDGES:
        binding=result["main_input_bindings"][str(edge)]
        frozen.update({str(Path(binding["input_root"])/"population.json"):binding["population_sha256"],
            str(Path(binding["input_root"])/f"edge_{edge}/inputs.json"):binding["input_binding_sha256"],
            binding["candidate_file"]:binding["candidate_file_sha256"],
            binding["sampling_manifest"]:binding["sampling_manifest_sha256"],
            chosen["checkpoints_by_edge"][str(edge)]["checkpoint"]:chosen["checkpoints_by_edge"][str(edge)]["checkpoint_sha256"]})
    for path,digest in frozen.items():
        if sha256(path)!=digest:raise ValueError(f"formal-plan dependency changed during preparation: {path}")
    result["frozen_sha256"]=frozen
    return result


def main(args):
    result=generate(args.selection,args.scope,curve_input_root=args.curve_input_root)
    write_json(args.output/"plan.json",result)
    if result["status"]=="prepared_for_root_review":
        write_json(args.output/"main_fitting.json",dict(status=result["status"],jobs=result["main_jobs"],scope="plan only; dispatch the actual fitting_driver"))
        if result["curve_jobs"]:
            write_json(args.output/"f_population_curve.json",dict(status=result["status"],jobs=result["curve_jobs"],
                population_accounting=result["population_accounting"],policy=result["population_curve_policy"]))
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection",type=Path,required=True)
    parser.add_argument("--scope",type=Path,default=ROOT/"experiments/ch3_motivation/configs/formal_quality_scope.json")
    parser.add_argument("--curve-input-root",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();value=main(args)
    print(json.dumps(dict(status=value["status"],main_jobs=len(value["main_jobs"]),curve_jobs=len(value["curve_jobs"]),gpu_launched=False)))
