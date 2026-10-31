#!/usr/bin/env python3
"""Check PhaseB's actual N/init weights on the unchanged three-context task."""
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
from experiments.ch3_motivation.code import (
    calibration_sampling, validate_probe_calibration_common as phase_a,
    validate_probe_temporal_common as temporal,
)
from experiments.ch3_motivation.code.common import ROOT, write_json
from experiments.ch3_motivation.code.probe_variants import load_variant
from experiments.ch3_motivation.code.sampling_identity import (
    validate_sampling_capture, validate_sampling_identity,
)
from experiments.ch3_motivation.code.validate_probe_common import (
    _check_weight_binding, _provenance, evaluate_common_probe, native_baseline,
)
from hstu_kvcache.common.artifacts import sha256

FORMAT="ch3_calibration_followup_common_v1"
RULES=temporal.RULES
DRIVER="run_probe_calibration_init.py"
EDGES=[1,2,3,4,5]
STAGES=("f","fg","fgh")
INITIALIZATION_FIELDS={"h_initialization","h_initialization_spec","initialization_helper_sha256"}
FITTING_SOURCES=("experiments/ch3_motivation/code/"+DRIVER,
    "experiments/ch3_motivation/code/calibration_sampling.py","experiments/ch3_motivation/code/probe_initialization.py",
    "experiments/ch3_motivation/code/calibration_gaps.py","experiments/ch3_motivation/code/common.py",
    "experiments/ch3_motivation/code/probes.py","experiments/ch3_motivation/code/probe_variants.py",
    "experiments/ch3_motivation/code/prepare_candidates.py","src/hstu_kvcache/adaptation/reader.py",
    "src/hstu_kvcache/models/hstu.py","src/hstu_kvcache/models/embeddings.py",
    "src/hstu_kvcache/evaluation/cost.py","src/hstu_kvcache/methods/baselines/protocol/cost.py")


def prepare_training_binding(job,cache=None):
    """Recheck the planned identity against current sources and its actual view."""
    identity=job["expected_identity"]
    for field in ("edge","seed","fit_users","width","learning_rate","steps","patience",
                  "validation_every","stages","controls","recent_fraction","h_initialization"):
        if identity[field]!=job[field]:
            raise ValueError("PhaseB job and expected identity differ: "+field)
    expected_sources={name:sha256(ROOT/name) for name in FITTING_SOURCES}
    if (identity["fitting_driver"]!=DRIVER or job["fitting_driver"]!=DRIVER or
            identity["fitting_driver_sha256"]!=expected_sources[FITTING_SOURCES[0]] or
            job["fitting_driver_sha256"]!=identity["fitting_driver_sha256"] or
            identity["numerical_sources"]!=expected_sources or identity["query_delta_rule"]!=job["gap_rule"] or
            Path(identity["sampling_manifest"]).resolve()!=Path(job["sampling_manifest"]).resolve()):
        raise ValueError("PhaseB actual driver/helper/source/input view differs")
    if (job["fit_users"] not in (64,128,256) or job["width"] not in (128,256) or
            job["learning_rate"] not in (3e-5,1e-5) or job["steps"] not in (512,1024,2048) or
            job["patience"] not in (0,128,512,1024) or job["validation_every"]!=8 or
            job["seed"]!=17 or job["stages"]!="f,fg,fgh" or job["controls"] is not False or
            job["reuse_existing_weights"] is not False):
        raise ValueError("PhaseB uses fresh, declared actual-N/width/budget full ladders")
    root=Path(job["input_root"])
    folder=root/f"edge_{job['edge']}"
    population=json.loads((root/"population.json").read_text())
    binding=json.loads((folder/"inputs.json").read_text())
    if (identity["fit_uids"]!=population["fit"][:job["fit_users"]] or
            identity["validation_users"]!=population["validation"] or
            len(identity["fit_uids"])!=job["fit_users"] or len(identity["validation_users"])!=16 or
            identity["population_sha256"]!=sha256(root/"population.json") or
            identity["input_binding_sha256"]!=sha256(folder/"inputs.json") or
            binding["population_sha256"]!=identity["population_sha256"] or
            binding["settings"]["edge"]!=job["edge"]):
        raise ValueError("PhaseB actual fitting prefix/validation/input binding differs")
    cache={} if cache is None else cache
    key=(str(Path(job["sampling_manifest"]).resolve()),identity["sampling_manifest_sha256"],
         job["edge"],job["fit_users"],job["recent_fraction"],job["gap_rule"])
    if key not in cache:
        _,cache[key]=calibration_sampling.load_sampling_manifest(job["sampling_manifest"],root,job["edge"],
            job["recent_fraction"],job["gap_rule"],fit_users=job["fit_users"])
    full_provenance=cache[key]
    h_spec=identity["h_initialization_spec"]
    prospective=dict(head_dim=h_spec["query_dim"],item_dim=h_spec["row_dim"]-2*h_spec["query_dim"],include_item=True)
    verified=validate_sampling_identity(identity,sampling_cache=cache,config=prospective)
    if verified!=job["sampling_provenance"]:
        raise ValueError("PhaseB planned sampling/initialization provenance differs")
    frozen={str(root/"population.json"):identity["population_sha256"],
        str(folder/"inputs.json"):identity["input_binding_sha256"],
        str(folder/"calibration_feedback_candidates.json"):identity["calibration_candidate_sha256"],
        str(Path(job["sampling_manifest"]).resolve()):identity["sampling_manifest_sha256"]}
    for name in ("history","requests"):
        frozen[str((folder/binding[name]["path"]).resolve())]=binding[name]["sha256"]
    phase_a.verify_hashes(frozen)
    return dict(expected_identity=identity,sampling_provenance=full_provenance,
        verified_sampling_identity=verified,frozen_sha256=frozen)


def load_plan(followup_path,common_path,edge):
    followup_path,common_path=Path(followup_path).resolve(),Path(common_path).resolve()
    plan,common=[json.loads(path.read_text()) for path in (followup_path,common_path)]
    if (plan["format"]!="probe_calibration_followup_v1" or
            Path(plan["common_plan_path"]).resolve()!=common_path or plan["common_plan_sha256"]!=sha256(common_path) or
            common["format"]!="ch3_temporal_common_plan_v1" or common["rules"]!=RULES or len(common["weights"])!=280):
        raise ValueError("PhaseB must retain the original frozen three-context common plan")
    frozen=dict(plan["planner_sources_sha256"])
    frozen.update(plan.get("frozen_sha256",{}))
    for path_key,hash_key in (("source_config","source_config_sha256"),("source_selection","source_selection_sha256")):
        if plan.get(hash_key) is not None:
            frozen[plan[path_key]]=plan[hash_key]
    frozen.update(common["source_fitting_plans_sha256"])
    frozen.update(common["executor_sources_sha256"])
    frozen.update({str(followup_path):sha256(followup_path),str(common_path):sha256(common_path),
        str(ROOT/"experiments/ch3_motivation/code/prepare_temporal_common_plan.py"):common["planner_sha256"],
        str(Path(__file__).resolve()):sha256(Path(__file__)),
        str(Path(phase_a.__file__).resolve()):sha256(Path(phase_a.__file__)),
        str(Path(__file__).with_name("sampling_identity.py")):sha256(Path(__file__).with_name("sampling_identity.py"))})
    for binding in common["edge_bindings"].values():
        for name in ("candidate_file","gap_manifest"):
            frozen[binding[name]]=binding[name+"_sha256"]
    phase_a.verify_hashes(frozen)
    jobs=plan["jobs"]
    if not jobs:
        return plan,common,[],frozen
    config=json.loads(Path(plan["source_config"]).read_text())
    recipes={row["name"]:row for row in config["recipes"]}
    if (plan["expected_ladders"]!=len(jobs) or plan["expected_weights"]!=3*len(jobs) or
            {(job["name"],job["edge"]) for job in jobs}!={(name,e) for name in recipes for e in EDGES} or
            len({str(Path(job["output"]).resolve()) for job in jobs})!=len(jobs)):
        raise ValueError("PhaseB plan must retain every declared recipe on all five edges without overlapping outputs")
    if any(any(job[key]!=value for key,value in recipes[job["name"]].items()) for job in jobs):
        raise ValueError("PhaseB job changed a recipe's declared N/width/budget/initialization")
    prepared=[];cache={}
    for job in jobs:
        if job["edge"]!=edge:continue
        training=prepare_training_binding(job,cache)
        shared=common["edge_bindings"][str(edge)]
        if (training["expected_identity"]["population_sha256"]!=shared["population_sha256"] or
                training["expected_identity"]["input_binding_sha256"]!=shared["input_binding_sha256"]):
            raise ValueError("PhaseB training and common roles/history binding differ")
        frozen.update(training["frozen_sha256"])
        for stage in STAGES:
            checkpoint=str((Path(job["output"])/f"{stage}.pt").resolve())
            if checkpoint!=job["expected_checkpoints"][stage]:
                raise ValueError("PhaseB planned stage output differs")
            spec=dict(origin="calibration_followup",recipe=job["name"],edge=edge,seed=job["seed"],
                method_label=stage,checkpoint=checkpoint,fitting_driver=DRIVER,
                h_initialization=job["h_initialization"],query_delta_rule=job["gap_rule"],
                recent_fraction=job["recent_fraction"],fit_users=job["fit_users"],width=job["width"],
                training_sampling_manifest=str(Path(job["sampling_manifest"]).resolve()),
                training_sampling_manifest_sha256=training["expected_identity"]["sampling_manifest_sha256"])
            prepared.append((spec,training))
    return plan,common,prepared,frozen


def check_weight(probe,metadata,spec,training,models):
    expected=training["expected_identity"]
    if metadata["identity"]!=expected or expected["models"]!=models:
        raise ValueError("PhaseB typed identity or actual backbone models differ from its exact job")
    cfg=asdict(probe.cfg)
    verified=validate_sampling_identity(expected,config=cfg)
    if verified!=training["verified_sampling_identity"] or metadata["sampling_provenance"]!=training["sampling_provenance"]:
        raise ValueError("PhaseB actual typed initialization/sampling provenance differs")
    validate_sampling_capture(expected,metadata)
    if (metadata["method_label"]!=spec["method_label"] or probe.stage!=spec["method_label"] or
            getattr(probe,"variant_name","head")!="full_item" or probe.cfg.hidden_width!=expected["width"] or
            probe.cfg.depth!=1 or not probe.cfg.include_item or not probe.cfg.normalize or
            probe.cfg.query_chunk!=16 or probe.cfg.row_chunk!=1024):
        raise ValueError("PhaseB actual typed width/stage/geometry differs")
    scales=metadata["fit_pool_loss_scales"]
    rescale=sum(network.weights[0].numel() for network in probe.h) if probe.stage=="fgh" and expected["h_initialization"]=="balanced" else 0
    if (metadata["quality_auc_computed"] is not False or metadata["fitting_uses_future_feedback"] is not False or
            scales["users"]!=expected["fit_users"] or scales["source"]!="fitting_users_only" or
            scales["version"]!="fit_pool_equal_user_length_normalized_v1" or
            scales["validation_uses_fitting_scales"] is not True or metadata["h_initialization_weight_rescale_flops"]!=rescale):
        raise ValueError("PhaseB own-N teacher/statistics/scaling scope differs")


@torch.no_grad()
def evaluate_weight(contexts,spec,training,*,check_only=False,models=None):
    started=time.perf_counter();path=Path(spec["checkpoint"])
    row=dict(spec,status="pending" if not path.exists() else "failed",contexts={},selection_eligible=False)
    if not path.exists():
        row["error"]="planned PhaseB checkpoint not available yet"
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
            training_identity=_provenance(metadata.get("identity")),sampling_view_key=training["sampling_provenance"]["view_key"],
            sampling_provenance=_provenance(metadata.get("sampling_provenance")),
            verified_sampling_identity=training["verified_sampling_identity"],
            cumulative_calibration_flops_estimate=metadata.get("cumulative_calibration_flops_estimate"))
        check_weight(probe,metadata,spec,training,models if check_only else contexts[RULES[0]].identity["models"])
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
            raise ValueError("PhaseB checkpoint changed during common validation")
    except (OSError,ValueError,KeyError,TypeError,RuntimeError,EOFError,pickle.UnpicklingError) as error:
        row.update(status="failed",error_type=type(error).__name__,error=str(error),selection_eligible=False)
        row.pop("mean_context_loss",None)
    row["seconds"]=time.perf_counter()-started
    return row


def check_paired_predecessors(prepared,points):
    """The matched initialization control must retain identical f/fg tensors."""
    specs={(spec["recipe"],spec["method_label"]):spec for spec,_ in prepared}
    rows={(point["recipe"],point["method_label"]):point for point in points}
    cells=[]
    for stage in ("f","fg"):
        left,right=[rows.get((recipe,stage)) for recipe in ("reference","balanced_h_init")]
        cell=dict(stage=stage,status="pending")
        if left and right and all(row["status"] in ("validated","identity_checked") for row in (left,right)):
            try:
                a,ma=load_variant(specs[("reference",stage)]["checkpoint"],"cpu")
                b,mb=load_variant(specs[("balanced_h_init",stage)]["checkpoint"],"cpu")
                normalized=lambda identity:{key:value for key,value in identity.items() if key not in INITIALIZATION_FIELDS}
                if (normalized(ma["identity"])!=normalized(mb["identity"]) or asdict(a.cfg)!=asdict(b.cfg) or
                        set(a.state_dict())!=set(b.state_dict()) or any(not torch.equal(value,b.state_dict()[name])
                            for name,value in a.state_dict().items())):
                    raise ValueError("reference and balanced f/fg tensors or normalization buffers differ")
                for row in (left,right):
                    if sha256(Path(row["checkpoint"]))!=row["checkpoint_sha256"]:
                        raise ValueError("matched predecessor changed during tensor audit")
                cell.update(status="verified_equal",reference_sha256=left["checkpoint_sha256"],balanced_sha256=right["checkpoint_sha256"])
            except (OSError,ValueError,KeyError,RuntimeError,EOFError,pickle.UnpicklingError) as error:
                cell.update(status="failed",error=str(error))
        elif any(row and row["status"]=="failed" for row in (left,right)):
            cell.update(status="failed",error="a matched predecessor failed its independent typed identity check")
        cells.append(cell)
    status="failed" if any(cell["status"]=="failed" for cell in cells) else "pending" if any(
        cell["status"]=="pending" for cell in cells) else "verified_equal"
    if status=="failed":
        for row in points:
            if row["recipe"] in ("reference","balanced_h_init"):
                row.update(selection_eligible=False,matched_initialization_status="failed")
    return dict(status=status,stages=cells,scope="actual typed state_dict including learned normalization buffers; only h initialization identity differs")


def run(args):
    followup,common_path=Path(args.followup_plan),Path(args.common_plan)
    missing=[str(path.resolve()) for path in (followup,common_path) if not path.exists()]
    if missing:
        if (Path(args.output)/"configuration.json").exists():
            raise ValueError("preserve the existing PhaseB common attempt; missing plans need a separate pending output")
        result=dict(format=FORMAT,status="waiting_plans",pending_plan_paths=missing,points=[],gpu_started=False,
            quality_auc_computed=False,post_release_labels_loaded=False,fitting_performed=False)
        write_json(Path(args.output)/"summary.json",result);return result
    plan,common,prepared,frozen=load_plan(followup,common_path,args.edge)
    output=Path(args.output)
    if not prepared:
        if (output/"configuration.json").exists():
            raise ValueError("preserve the existing PhaseB common attempt; pending plans need a separate output")
        result=dict(format=FORMAT,status="waiting_followup_plan",followup_plan_path=str(followup.resolve()),
            followup_plan_sha256=sha256(followup),expected_weights=plan["expected_weights"],
            expected_context_cells=3*plan["expected_weights"],points=[],gpu_started=False,
            quality_auc_computed=False,post_release_labels_loaded=False,fitting_performed=False)
        write_json(output/"summary.json",result);return result
    mode="cpu_identity_check" if args.check else "common_validation"
    config_path=output/"configuration.json"
    base=dict(format=FORMAT,followup_plan_path=str(followup.resolve()),followup_plan_sha256=sha256(followup),
        common_plan_path=str(common_path.resolve()),common_plan_sha256=sha256(common_path),edge=args.edge,
        mode=mode,frozen_sha256=frozen)
    if config_path.exists() and any(json.loads(config_path.read_text()).get(key)!=value for key,value in base.items()):
        raise ValueError("PhaseB common attempt changed; preserve it and use a new attempt")
    contexts={};references={}
    if not args.check:
        if args.device=="cuda":
            os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu);torch.cuda.set_device(0)
        binding=common["edge_bindings"][str(args.edge)]
        contexts=temporal.prepare_contexts(binding["input_root"],args.edge,torch.device("cuda:0" if args.device=="cuda" else "cpu"),
            binding["gap_manifest"],backend=args.backend)
        for name,root in (("old280",args.reference_results),("sampling90",args.sampling_reference_results)):
            path=Path(root)/f"edge_{args.edge}/summary.json"
            references[name]=dict(status="pending",path=str(path.resolve()))
            if path.exists():
                reference=json.loads(path.read_text())
                key="plan_sha256" if name=="old280" else "common_plan_sha256"
                fmt=temporal.FORMAT if name=="old280" else phase_a.FORMAT
                if (reference["format"]!=fmt or reference[key]!=sha256(common_path) or
                        reference["contexts"]!={rule:context.identity for rule,context in contexts.items()}):
                    raise ValueError("PhaseB and previous common candidates/targets/scales/fingerprints differ")
                references[name].update(status="same_contexts",sha256=sha256(path))
    configuration=dict(base,contexts={rule:context.identity for rule,context in contexts.items()})
    if config_path.exists() and json.loads(config_path.read_text())!=configuration:
        raise ValueError("PhaseB common attempt changed; preserve it and use a new attempt")
    write_json(config_path,configuration)
    result=dict(base,status="in_progress",configuration_sha256=sha256(config_path),points=[],references=references,
        expected_weights=plan["expected_weights"],expected_context_cells=3*plan["expected_weights"],edge_expected_weights=len(prepared),
        quality_auc_computed=False,post_release_labels_loaded=False,fitting_performed=False,
        independent_fitting_users=64,scale_user_context_observations=192,training_catalog_users=256,training_time_law_users=256,
        contexts=configuration["contexts"],scope="same original fit64-ID/validation16 three-context task and RMS; actual own-N trained normalization preserved")
    if contexts:result["native_reuse"]={rule:native_baseline(context) for rule,context in contexts.items()}
    models={}
    for spec,training in prepared:
        actual_models=phase_a.backbone_hashes(args.edge,models) if args.check and Path(spec["checkpoint"]).exists() else None
        point=evaluate_weight(contexts,spec,training,check_only=args.check,models=actual_models)
        result["points"].append(point);write_json(output/"summary.json",result)
        print(json.dumps(dict(status="followup_common_point",recipe=spec["recipe"],method_label=spec["method_label"],
            result=point["status"],mean_context_loss=point.get("mean_context_loss"))),flush=True)
    result["paired_initialization"]=check_paired_predecessors(prepared,result["points"])
    phase_a.verify_hashes(frozen)
    if contexts:phase_a.verify_hashes(contexts[RULES[0]].frozen_paths)
    counts={status:sum(point["status"]==status for point in result["points"]) for status in ("validated","identity_checked","pending","failed")}
    result.update(counts,status="waiting_weights" if counts["pending"] else "complete_with_failures" if counts["failed"] or
        result["paired_initialization"]["status"]=="failed" else "ready_for_common_validation" if args.check else "complete",
        gpu_started=not args.check and args.device=="cuda")
    write_json(output/"summary.json",result);return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--followup-plan",type=Path,required=True)
    parser.add_argument("--common-plan",type=Path,default=ROOT/"experiments/ch3_motivation/configs/probe_common_temporal_validation.json")
    parser.add_argument("--reference-results",type=Path,default=ROOT/"experiments/ch3_motivation/runs/e5/common_temporal_validation/attempt_1")
    parser.add_argument("--sampling-reference-results",type=Path,default=ROOT/"experiments/ch3_motivation/runs/e5/calibration_sampling_common/attempt_1")
    parser.add_argument("--edge",type=int,choices=EDGES,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    parser.add_argument("--gpu",type=int,default=1)
    parser.add_argument("--backend",choices=("torch","triton"),default="triton")
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check",action="store_true");mode.add_argument("--run",action="store_true")
    arguments=parser.parse_args();result=run(arguments)
    print(json.dumps(dict(status=result["status"],gpu_launched=result["gpu_started"])))
    if arguments.run and result["status"]!="complete":raise SystemExit(2)
