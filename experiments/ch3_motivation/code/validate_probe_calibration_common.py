#!/usr/bin/env python3
"""Validate new fit256-sampling weights on the frozen fit64/three-context task.

The existing 280-weight executor, its binding checks, context identities and
results stay unchanged. ``--check`` audits plans and available weights on CPU;
only an explicit ``--run`` prepares common targets and evaluates losses.
"""
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
import pickle
import sys
import time

import torch
from experiments.ch3_motivation.code import calibration_sampling, validate_probe_temporal_common as temporal
from experiments.ch3_motivation.code.common import ROOT, write_json
from experiments.ch3_motivation.code.probe_variants import load_variant
from experiments.ch3_motivation.code.validate_probe_common import (
    _check_weight_binding, _digest, _provenance, evaluate_common_probe, native_baseline,
)
from hstu_kvcache.common.artifacts import sha256

FORMAT="ch3_sampling_common_pre_release_validation_v1"
RULES=temporal.RULES
FITTING_SOURCES=("experiments/ch3_motivation/code/run_probe_calibration.py",
    "experiments/ch3_motivation/code/calibration_sampling.py","experiments/ch3_motivation/code/calibration_gaps.py",
    "experiments/ch3_motivation/code/common.py","experiments/ch3_motivation/code/probes.py",
    "experiments/ch3_motivation/code/probe_variants.py","experiments/ch3_motivation/code/prepare_candidates.py",
    "src/hstu_kvcache/adaptation/reader.py","src/hstu_kvcache/models/hstu.py",
    "src/hstu_kvcache/models/embeddings.py","src/hstu_kvcache/evaluation/cost.py",
    "src/hstu_kvcache/methods/baselines/protocol/cost.py")


def verify_hashes(records):
    for name,digest in records.items():
        path=Path(name) if Path(name).is_absolute() else ROOT/name
        if sha256(path)!=digest:
            raise ValueError(f"planned sampling-common source/input changed: {name}")


def prepare_training_binding(job,sources):
    """Resolve the actual executed view; training and common inputs stay separate."""
    root=Path(job["input_root"])
    folder=root/f"edge_{job['edge']}"
    population=json.loads((root/"population.json").read_text())
    binding=json.loads((folder/"inputs.json").read_text())
    manifest=Path(job["sampling_manifest"]).resolve()
    _,provenance=calibration_sampling.load_sampling_manifest(manifest,root,job["edge"],
        job["recent_fraction"],job["gap_rule"],fit_users=job["fit_users"])
    expected=dict(variant="full_item",width=job["width"],depth=1,fit_users=job["fit_users"],
        queries=16,learning_rate=job["learning_rate"],steps=job["steps"],patience=job["patience"],
        validation_every=8,read_weight=1.,score_weight=1.,candidate_source="feedback",
        recent_fraction=job["recent_fraction"],candidate_seed=17,row_chunk=1024,query_chunk=16,
        weight_decay=.0001,gradient_clip=5.,edge=job["edge"],seed=job["seed"],
        fit_uids=population["fit"][:job["fit_users"]],validation_users=population["validation"],
        stages=job["stages"],controls=job["controls"],input_binding_sha256=sha256(folder/"inputs.json"),
        population_sha256=sha256(root/"population.json"),
        calibration_candidate_sha256=sha256(folder/"calibration_feedback_candidates.json"),
        numerical_sources={name:sources[name] for name in FITTING_SOURCES},
        fitting_driver="run_probe_calibration.py",fitting_driver_sha256=sources[FITTING_SOURCES[0]],
        query_delta_rule=job["gap_rule"],sampling_manifest=str(manifest),sampling_manifest_sha256=sha256(manifest),
        gap_manifest=str(manifest),gap_manifest_sha256=sha256(manifest),
        time_rule=provenance["time_rule"],source_cutoff=provenance["source_cutoff"],
        candidate_order=provenance["candidate_order"],query_delta_layout="[B,16]",
        candidate_catalog_source_users=256,time_law_source_users=256,
        gradient_fitting_users=job["fit_users"],teacher_fitting_users=job["fit_users"],fitting_uses_future_feedback=False)
    if (binding["settings"]["edge"]!=job["edge"] or binding["population_sha256"]!=expected["population_sha256"]
            or provenance["fit_uids"]!=expected["fit_uids"] or provenance["validation_uids"]!=expected["validation_users"]
            or len(expected["validation_users"])!=16 or expected["source_cutoff"]!=binding["settings"]["release_day"]*86400):
        raise ValueError("sampling training role/input/cutoff binding differs")
    return dict(expected_identity=expected,sampling_provenance=provenance)


def load_plan(fitting_plan_path,common_plan_path,edge):
    """Audit the frozen 30-ladder source plan, without loading a backbone."""
    fitting_plan_path,common_plan_path=Path(fitting_plan_path).resolve(),Path(common_plan_path).resolve()
    fitting=json.loads(fitting_plan_path.read_text())
    common=json.loads(common_plan_path.read_text())
    if (fitting["format"]!="probe_calibration_sampling_v1" or fitting["expected_ladders"]!=30
            or fitting["expected_stage_checkpoints"]!=90 or len(fitting["jobs"])!=30
            or common["format"]!="ch3_temporal_common_plan_v1" or common["rules"]!=RULES
            or len(common["weights"])!=280 or len({row["checkpoint"] for row in common["weights"]})!=280):
        raise ValueError("expected new sampling30/90 and unchanged old common280 plans")
    selection=fitting["selection"]
    if (Path(selection["common_context_plan"]).resolve()!=common_plan_path
            or selection["common_context_plan_sha256"]!=sha256(common_plan_path)
            or selection["contexts"]!=RULES or selection["population_scores_used"] is not False):
        raise ValueError("sampling comparison is not bound to the frozen old common plan")
    frozen=dict(fitting["source_sha256"],**fitting["input_sha256"])
    frozen.update(common["source_fitting_plans_sha256"])
    frozen.update(common["executor_sources_sha256"])
    frozen.update({str(fitting_plan_path):sha256(fitting_plan_path),str(common_plan_path):sha256(common_plan_path),
        str(ROOT/"experiments/ch3_motivation/code/prepare_temporal_common_plan.py"):common["planner_sha256"],
        str(Path(__file__).resolve()):sha256(Path(__file__))})
    for value in common["edge_bindings"].values():
        for path,digest in (("candidate_file","candidate_file_sha256"),("gap_manifest","gap_manifest_sha256")):
            frozen[value[path]]=value[digest]
    verify_hashes(frozen)
    jobs=[dict(fitting["fixed_settings"],**job) for job in fitting["jobs"]]
    combinations={(fraction,rule) for fraction in (.5,.75) for rule in RULES}
    for current_edge in range(1,6):
        selected=[job for job in jobs if job["edge"]==current_edge]
        if len(selected)!=6 or {(job["recent_fraction"],job["gap_rule"]) for job in selected}!=combinations:
            raise ValueError("sampling plan must retain all six candidate/time views on every edge")
    if len({str(Path(job["output"]).resolve()) for job in jobs})!=30:
        raise ValueError("sampling checkpoint directories overlap")
    prepared=[]
    for job in jobs:
        if (job["stages"]!="f,fg,fgh" or job["controls"] is not False or job["fit_users"]!=256
                or job["seed"]!=17 or job["validation_every"]!=8):
            raise ValueError("new common plan requires the declared fit256 primary-stage ladder")
        manifest=Path(job["sampling_manifest"]).resolve()
        if fitting["input_sha256"].get(str(manifest))!=sha256(manifest):
            raise ValueError("sampling manifest is not frozen in the fitting plan")
        if job["edge"]!=edge:
            continue
        training=prepare_training_binding(job,fitting["source_sha256"])
        shared=common["edge_bindings"][str(edge)]
        expected=training["expected_identity"]
        if (expected["input_binding_sha256"]!=shared["input_binding_sha256"]
                or expected["population_sha256"]!=shared["population_sha256"]):
            raise ValueError("sampling training and fixed common histories/roles differ")
        for stage in ("f","fg","fgh"):
            spec=dict(origin="sampling256_90",recipe="calibration_sampling/"+job["name"],edge=edge,seed=job["seed"],
                method_label=stage,checkpoint=str((Path(job["output"])/f"{stage}.pt").resolve()),
                fitting_driver="run_probe_calibration.py",query_delta_rule=job["gap_rule"],
                recent_fraction=job["recent_fraction"],training_sampling_manifest=str(manifest),
                training_sampling_manifest_sha256=sha256(manifest))
            prepared.append((spec,training))
    return fitting,common,prepared,frozen


def check_sampling_weight(probe,metadata,spec,training,models):
    """Exact new-driver identity, followed separately by the old common gate."""
    expected=dict(training["expected_identity"],models=models)
    identity=metadata["identity"]
    if identity!=expected:
        changed=sorted(key for key in set(expected)|set(identity) if expected.get(key)!=identity.get(key))
        raise ValueError("sampling checkpoint training identity differs: "+", ".join(changed))
    if (metadata["method_label"]!=spec["method_label"] or probe.stage!=spec["method_label"]
            or getattr(probe,"variant_name","head")!="full_item" or probe.cfg.hidden_width!=expected["width"]
            or probe.cfg.depth!=1 or not probe.cfg.include_item or not probe.cfg.normalize
            or probe.cfg.query_chunk!=16 or probe.cfg.row_chunk!=1024
            or metadata["sampling_provenance"]!=training["sampling_provenance"]):
        raise ValueError("sampling checkpoint architecture/branch/view provenance differs")
    scales=metadata["fit_pool_loss_scales"]
    if (metadata["training_users"]!=256 or metadata["validation_users"]!=16
            or metadata["actual_teacher_users_captured"]!=256 or metadata["actual_validation_users_captured"]!=16
            or metadata["quality_auc_computed"] is not False or metadata["fitting_uses_future_feedback"] is not False
            or scales["users"]!=256 or scales["source"]!="fitting_users_only"
            or scales["version"]!="fit_pool_equal_user_length_normalized_v1"
            or scales["validation_uses_fitting_scales"] is not True):
        raise ValueError("sampling checkpoint fitting/teacher/scaling scope differs")


def backbone_hashes(edge,cache):
    result={}
    for version in (edge-1,edge):
        name=f"results/unified_training_2026_09/large/seed17/checkpoints/v{version}/checkpoint_100.pt"
        if name not in cache:
            cache[name]=sha256(ROOT/name)
        result[name]=cache[name]
    return result


@torch.no_grad()
def evaluate_sampling_weight(contexts,spec,training,*,check_only=False,models=None):
    started=time.perf_counter()
    path=Path(spec["checkpoint"])
    row=dict(spec,status="pending" if not path.exists() else "failed",contexts={},selection_eligible=False)
    if not path.exists():
        row["error"]="planned sampling checkpoint not available yet"
        return row
    try:
        row["checkpoint_sha256"]=sha256(path)
        device="cpu" if check_only else next(contexts[RULES[0]].model.parameters()).device
        probe,metadata=load_variant(path,device)
        row.update(variant=getattr(probe,"variant_name","head"),stage=probe.stage,config=asdict(probe.cfg),
            fitting_users=len(metadata.get("identity",{}).get("fit_uids",[])),best_step=metadata.get("best_step"),
            completed_steps=metadata.get("completed_steps"),step_zero=metadata.get("best_step")==0,
            original_validation_loss=_provenance(metadata.get("best_validation_loss")),
            original_loss_scales=_provenance(metadata.get("fit_pool_loss_scales")),
            training_identity=_provenance(metadata.get("identity")),
            cumulative_calibration_flops_estimate=metadata.get("cumulative_calibration_flops_estimate"))
        actual_models=models if check_only else contexts[RULES[0]].identity["models"]
        check_sampling_weight(probe,metadata,spec,training,actual_models)
        if check_only:
            row.update(status="identity_checked",numerical_common_validation_performed=False)
        else:
            _check_weight_binding(contexts[RULES[0]],probe,metadata)
            for rule in RULES:
                context=contexts[rule]
                row["contexts"][rule]=dict(status="validated",common_validation_sha256=context.fingerprint,
                                            metrics=evaluate_common_probe(context,probe))
            row.update(status="validated",mean_context_loss=sum(row["contexts"][rule]["metrics"]["loss"] for rule in RULES)/3,
                selection_eligible=not metadata.get("diagnostic_only",False) and metadata.get("quality_use_permitted",True) is not False)
        if sha256(path)!=row["checkpoint_sha256"]:
            raise ValueError("sampling checkpoint changed during common validation")
    except (OSError,ValueError,KeyError,TypeError,RuntimeError,EOFError,pickle.UnpicklingError) as error:
        row.update(status="failed",error_type=type(error).__name__,error=str(error),selection_eligible=False)
        row.pop("mean_context_loss",None)
    row["seconds"]=time.perf_counter()-started
    return row


def run(args):
    fitting_path,common_path=Path(args.fitting_plan),Path(args.common_plan)
    missing=[str(path.resolve()) for path in (fitting_path,common_path) if not path.exists()]
    if missing:
        if (Path(args.output)/"configuration.json").exists():
            raise ValueError("preserve the existing common attempt; missing plans need a separate pending output")
        result=dict(format=FORMAT,status="waiting_plans",pending_plan_paths=missing,points=[],
            quality_auc_computed=False,post_release_labels_loaded=False,fitting_performed=False,gpu_started=False)
        write_json(Path(args.output)/"summary.json",result)
        return result
    fitting,common,prepared,frozen=load_plan(fitting_path,common_path,args.edge)
    output=Path(args.output)
    config_path=output/"configuration.json"
    mode="cpu_identity_check" if args.check else "common_validation"
    if config_path.exists():
        previous=json.loads(config_path.read_text())
        expected=dict(format=FORMAT,fitting_plan_sha256=sha256(fitting_path),common_plan_sha256=sha256(common_path),
                      edge=args.edge,mode=mode,frozen_sha256=frozen)
        if any(previous.get(key)!=value for key,value in expected.items()):
            raise ValueError("sampling common attempt changed; preserve existing results and use a new attempt")
    contexts={}
    reference=dict(status="pending",path=str((Path(args.reference_results)/f"edge_{args.edge}/summary.json").resolve()))
    if not args.check:
        if args.device=="cuda":
            os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
            torch.cuda.set_device(0)
        device=torch.device("cuda:0" if args.device=="cuda" else "cpu")
        binding=common["edge_bindings"][str(args.edge)]
        contexts=temporal.prepare_contexts(binding["input_root"],args.edge,device,binding["gap_manifest"],backend=args.backend)
        reference_path=Path(reference["path"])
        if reference_path.exists():
            previous=json.loads(reference_path.read_text())
            if (previous["format"]!=temporal.FORMAT or previous["plan_sha256"]!=sha256(common_path)
                    or previous["contexts"]!={rule:context.identity for rule,context in contexts.items()}):
                raise ValueError("sampling/old280 common targets or loss-scale contexts differ")
            reference.update(status="same_contexts",sha256=sha256(reference_path))
    configuration=dict(format=FORMAT,fitting_plan=str(fitting_path.resolve()),fitting_plan_sha256=sha256(fitting_path),
        common_plan=str(common_path.resolve()),common_plan_sha256=sha256(common_path),edge=args.edge,
        mode=mode,frozen_sha256=frozen,
        contexts={rule:context.identity for rule,context in contexts.items()})
    if config_path.exists() and json.loads(config_path.read_text())!=configuration:
        raise ValueError("sampling common attempt changed; preserve existing results and use a new attempt")
    write_json(config_path,configuration)
    result=dict(format=FORMAT,status="in_progress",edge=args.edge,configuration_sha256=sha256(config_path),
        fitting_plan_sha256=sha256(fitting_path),common_plan_sha256=sha256(common_path),
        quality_auc_computed=False,post_release_labels_loaded=False,fitting_performed=False,
        independent_fitting_users=64,scale_user_context_observations=192,training_catalog_users=256,training_time_law_users=256,
        contexts=configuration["contexts"],reference_280=reference,points=[],
        scope="same frozen fit64 candidate IDs, validation16 and shared three-context scales as old280; training sampling view remains separate")
    if contexts:
        result["native_reuse"]={rule:native_baseline(context) for rule,context in contexts.items()}
    models={}
    for spec,training in prepared:
        expected_models=backbone_hashes(args.edge,models) if args.check and Path(spec["checkpoint"]).exists() else None
        point=evaluate_sampling_weight(contexts,spec,training,check_only=args.check,models=expected_models)
        result["points"].append(point)
        write_json(output/"summary.json",result)
        print(json.dumps(dict(status="sampling_common_point",checkpoint=spec["checkpoint"],result=point["status"],
            mean_context_loss=point.get("mean_context_loss"))),flush=True)
    verify_hashes(frozen)
    if contexts:
        verify_hashes(contexts[RULES[0]].frozen_paths)
    counts={status:sum(point["status"]==status for point in result["points"])
            for status in ("validated","identity_checked","pending","failed")}
    result.update(counts,status="waiting_weights" if counts["pending"] else "complete_with_failures" if counts["failed"] else
                  "ready_for_common_validation" if args.check else "complete",gpu_started=not args.check and args.device=="cuda")
    write_json(output/"summary.json",result)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fitting-plan",type=Path,required=True)
    parser.add_argument("--common-plan",type=Path,default=ROOT/"experiments/ch3_motivation/configs/probe_common_temporal_validation.json")
    parser.add_argument("--reference-results",type=Path,default=ROOT/"experiments/ch3_motivation/runs/e5/common_temporal_validation/attempt_1")
    parser.add_argument("--edge",type=int,choices=range(1,6),required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    parser.add_argument("--gpu",type=int,default=1)
    parser.add_argument("--backend",choices=("torch","triton"),default="triton")
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check",action="store_true")
    mode.add_argument("--run",action="store_true")
    arguments=parser.parse_args()
    result=run(arguments)
    success=result["status"]=="complete" or arguments.check and result["status"] in ("ready_for_common_validation","waiting_weights","waiting_plans")
    raise SystemExit(0 if success else 2)
