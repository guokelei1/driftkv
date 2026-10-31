#!/usr/bin/env python3
"""Local32 Key/Value read alignment, then joint16 final-teacher-logit fitting.

The input, code rank, fixed users and teacher capture are unchanged. Each local
epoch recomputes only its cached layer attention with an explicit length mask.
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]

import torch

from experiments.archive.compact_read.calibrate_lightweight_fullitem import (
    SOURCE_PATHS as READ_SOURCES, save_adapter,
)
from experiments.archive.compact_read.local_kv_fit import capture_layer, fit_layer
from experiments.archive.compact_read.calibrate_lightweight_decoder import prepare_rows
from experiments.archive.compact_read.calibrate_lightweight_fullitem_joint import fit_joint
from hstu_kvcache.evaluation.capture import capture_rows
from hstu_kvcache.evaluation.release_context import RELEASES, load_models, setup_device, sources
from hstu_kvcache.evaluation.panels import PANEL, load_history, load_panel
from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.methods.cachetide.compact.lightweight_fullitem import LightweightFullItemAdapter
from hstu_kvcache.methods.cachetide.compact.lightweight_fullitem_kv import LightweightFullItemKVAdapter
from hstu_kvcache.methods.cachetide.refinement.joint_local_kv_refinement import JointLocalKVRefinement
from hstu_kvcache.models import HSTUKVCache
from hstu_kvcache.evaluation.cost import CostModel, eager_read
from hstu_kvcache.common.artifacts import sha256, write_json

SOURCE_PATHS=READ_SOURCES+("experiments/archive/compact_read/calibrate_lightweight_local_kv.py",
    "experiments/archive/compact_read/local_kv_fit.py",
    "experiments/archive/compact_read/calibrate_lightweight_fullitem_joint.py",
    "src/hstu_kvcache/methods/cachetide/compact/lightweight_fullitem_kv.py",
    "src/hstu_kvcache/methods/cachetide/refinement/joint_value_refinement.py",
    "src/hstu_kvcache/methods/cachetide/refinement/joint_kv_refinement.py",
    "src/hstu_kvcache/methods/cachetide/refinement/joint_local_kv_refinement.py")


def bound_sources():
    # The new inference kernel is developed independently and is never imported
    # by this Torch calibration. Its own probe/evaluator binds its source.
    result=sources(*SOURCE_PATHS)
    result.pop("src/hstu_kvcache/methods/cachetide/compact/lightweight_kv_kernel.py",None)
    result.pop("src/hstu_kvcache/methods/cachetide/compact/lightweight_kv_runtime_kernel.py",None)
    return result


def run(args):
    began=time.perf_counter()
    if args.output.exists():
        raise FileExistsError(args.output)
    panel,groups,_=load_panel(args.panel)
    train,validation=list(groups["fit"]),list(groups["validation"])
    if args.small:
        train,validation=train[:4],validation[:2]
    if set(train)&set(validation) or set(train+validation)&set(groups["evaluation"]):
        raise RuntimeError("fitting/validation/evaluation roles overlap")
    read_epochs,joint_epochs=(4,4) if args.small else (32,16)
    read_checks=[0,1,2,4] if args.small else [0,8,16,32]
    joint_checks=[0,1,2,4] if args.small else [0,8,16]
    binding=bound_sources()
    args.output.mkdir(parents=True)
    settings=dict(target=args.target,rank=8,response_hidden_width=0,seed=17,
        fit_uids=train,validation_uids=validation,queries_per_user=16,
        read_epochs=read_epochs,joint_epochs=joint_epochs,read_checkpoints=read_checks,
        joint_checkpoints=joint_checks,batch_size=args.batch_size,learning_rate=.001,
        identity_weight=.001,gradient_clip=1.,panel_sha256=sha256(args.panel/"binding.json"),
        input="local native K/V and complete Current item embedding",
        local_objective="Same-q Current read-rate residual MSE with fit-only per-head output scaling; masked dynamic Key/Value attention",
        objective="Final Current-teacher logit MSE plus relative encoder/Key/Value coordinate prior",
        teacher_target="Current Full final logits; no historical feedback labels or online quality observations",
        coordinates="Local Key uses fit-only K/init-code RMS; Value uses fit-only read-residual RMS. Both fold into decoders. Joint preserves the selected local E/b/Dk/Dv as origins and uses their per-head RMS, with fit-only zero fallback. Prior averages all coordinates.",
        lineage="Fresh local32 jointly trains E/b/Dk/Dv from the original random encoder and zero decoders, then joint16; no inherited weights or additional training stage.",
        teacher="One Current prefix and one final-query capture per calibration UID, shared across both stages.",
        serving="One full-item rank8 code for factored Key/Value reading; native writes unchanged, lazy first-query code initialization",
        execution_sources=binding)
    write_json(args.output/"configuration.json",settings)
    device=setup_device(args.gpu,args.threads)
    models=load_models(panel,device,int(args.target[1:]))
    captured,capture_costs,capture_records=capture_rows(models,
        load_history(args.panel,requested_uids=train+validation),train+validation,
        args.target,panel["known_vocab_size"])
    model=models[args.target].requires_grad_(False)
    del models
    costs=defaultdict(int,teacher_cache_capture_flops=capture_costs["teacher_cache_capture_flops"])
    torch.manual_seed(17)
    adapter=LightweightFullItemAdapter(len(model.blocks),model.cfg.num_heads,
        model.cfg.hidden_size//model.cfg.num_heads,rank=8,max_length=model.cfg.max_seq_len).to(device).eval()
    rng_cpu,rng_cuda=torch.get_rng_state(),torch.cuda.get_rng_state(device)
    original=adapter
    adapter=LightweightFullItemKVAdapter.from_value(original).eval()
    if not torch.equal(rng_cpu,torch.get_rng_state()) or not torch.equal(rng_cuda,torch.cuda.get_rng_state(device)):
        raise RuntimeError("adding Key changed encoder initialization RNG")
    if any(not torch.equal(t,adapter.state_dict()[name]) for name,t in original.state_dict().items()):
        raise RuntimeError("adding Key changed initial Value encoder tensors")
    del original
    read_diagnostics,timings=[],[]
    for index,layer in enumerate(adapter.layers):
        started=time.perf_counter()
        data=capture_layer(model,adapter,captured,train+validation,index,device,costs)
        diagnostics=fit_layer(layer,data,len(train),epochs=read_epochs,batch_size=args.batch_size,
            checks=read_checks,costs=costs)
        diagnostics["layer"]=index
        read_diagnostics.append(diagnostics)
        del data
        timings.append(dict(layer=index,read_capture_and_fit_seconds=time.perf_counter()-started))
    adapter.eval().requires_grad_(False)
    adapter.zero_grad(set_to_none=True)
    read_costs=dict(costs)
    read_only=args.output/"read_only"
    save_adapter(adapter,read_only/"calibration.pt")
    resource=capture_costs["source_trajectory_flops"]
    write_json(read_only/"calibration.json",dict(status="complete",kind=adapter.kind,target=args.target,
        config=adapter.get_config(),settings=settings,execution_sources=binding,
        fit_uids=train,validation_uids=validation,diagnostics=read_diagnostics,
        capture_records=capture_records,weights_sha256=sha256(read_only/"calibration.pt"),
        cost=dict(components=read_costs,calibration_flops=sum(read_costs.values()),
            development_resource=dict(source_trajectory_flops=resource),
            source_reconstruction_sensitivity=dict(additional_flops=resource,
                calibration_if_source_reconstructed=sum(read_costs.values())+resource))))
    # This existing helper runs the teacher once and retains initial frozen
    # codes for identity diagnostics. Training ignores those codes completely.
    rows,lookup_bytes=prepare_rows(model,adapter,captured,train+validation,device,costs)
    for uid,row in rows.items():
        row["item_features"]=captured[uid]["item_features"].to(device)
    refiner=JointLocalKVRefinement(adapter,[x["output_scale"] for x in read_diagnostics],
        [x["key_scale"] for x in read_diagnostics]).to(device)
    key_statistics=dict(fit_users=len(train),scope="per-layer fit histories only, before local training",
        key_scales=[x["key_scale"] for x in read_diagnostics],
        initial_code_rms=[x["initial_code_rms"] for x in read_diagnostics],
        joint_origin="selected local Key decoder, including nonzero entries",
        joint_key_scale=refiner.key_scale[:,:,0,0].detach().cpu().tolist())
    size=sum(p.numel() for p in refiner.parameters())
    costs["joint_coordinate_statistics_flops"]+=4*size+8*adapter.num_layers*adapter.heads
    probe=rows[validation[0]]
    cache=probe["cache"]
    original_k,original_v=cache.k.clone(),cache.v.clone()
    cm=CostModel(model.cfg.hidden_size,len(model.blocks),model.cfg.num_heads,"torch")

    @torch.no_grad()
    def check_score(refined):
        # Regenerate even for the published check. Reusing initial or pre-fold
        # codes would not check that the encoder has actually been published.
        state=(refiner.encode_cache(adapter,cache,probe["item_features"]) if refined else
               adapter.encode_cache(cache,probe["item_features"]))
        fn=(refiner.make_history_read(adapter,state) if refined else adapter.make_history_read(state))
        result=score(model,cache,probe["candidates"],probe["query_delta"],history_read_fn=fn)[0]
        q=probe["candidates"].shape[1]
        extra=adapter.estimate_flops(tokens=cache.seq_len,candidates=q)
        costs["execution_check_query_flops"]+=eager_read(cm,cache.seq_len,queries=q)+extra["extra_history_read"]
        costs["execution_check_reencoding_flops"]+=extra["token_transform"]
        if refined:
            costs["execution_check_coordinate_flops"]+=2*size
        return result,state.codes

    initial,zero=check_score(False),check_score(True)
    identity_error=float((initial[0]-zero[0]).abs().max())
    identity_code_error=float((initial[1]-zero[1]).abs().max())
    costs["execution_check_comparison_flops"]+=3*(initial[0].numel()+initial[1].numel())
    if max(identity_error,identity_code_error)>2e-5:
        raise RuntimeError("zero joint coordinates changed native adapter outputs")
    started=time.perf_counter()
    joint_diagnostics=fit_joint(model,adapter,refiner,rows,train,validation,
        epochs=joint_epochs,batch_size=args.batch_size,checks=joint_checks,costs=costs)
    joint_diagnostics["key_decoder_parameters"]=refiner.key_delta.numel()
    costs["first_gradient_diagnostic_flops"]+=sum(2*p.numel()+1 for p in refiner.parameters())
    torch.cuda.synchronize(device)
    timings.append(dict(stage="joint",fitting_seconds=time.perf_counter()-started))
    before=check_score(True)
    refiner.fold_into(adapter)
    costs["publication_joint_fold_flops"]+=2*size
    after=check_score(False)
    fold_error=float((before[0]-after[0]).abs().max())
    fold_code_error=float((before[1]-after[1]).abs().max())
    costs["execution_check_comparison_flops"]+=3*(before[0].numel()+before[1].numel())
    if max(fold_error,fold_code_error)>2e-5:
        raise RuntimeError("joint publication changed codes or logits")
    if not torch.equal(cache.k,original_k) or not torch.equal(cache.v,original_v):
        raise RuntimeError("joint fitting mutated native KV")
    if any(p.grad is not None for module in (model,adapter) for p in module.parameters()):
        raise RuntimeError("joint fitting changed the frozen foundation or base parameter graph")
    if bound_sources()!=binding:
        raise RuntimeError("joint calibration sources changed")
    if costs["teacher_cache_capture_flops"]!=read_costs["teacher_cache_capture_flops"]:
        raise RuntimeError("teacher prefix was rebuilt twice")
    save_adapter(adapter,args.output/"calibration.pt")
    joint_costs={k:v-read_costs.get(k,0) for k,v in costs.items() if v!=read_costs.get(k,0)}
    metadata=dict(status="complete",method="local_then_joint_full_item_factored_kv",
        role="development_canary" if args.small else "development_calibration",target=args.target,
        kind=adapter.kind,config=adapter.get_config(),settings=settings,execution_sources=binding,
        weights_sha256=sha256(args.output/"calibration.pt"),fit_uids=train,validation_uids=validation,
        cutover=dict(RELEASES)[args.target],capture_records=capture_records,diagnostics=read_diagnostics,
        joint_diagnostics=joint_diagnostics,key_statistics=key_statistics,timings=timings,elapsed_seconds=time.perf_counter()-began,
        peak_gpu_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
        checks=dict(teacher_prefix_capture_passes=1,teacher_prefix_users=len(train+validation),
            native_cache_unchanged=True,foundation_and_input_statistics_frozen=True,
            encoder_and_both_decoder_coordinates_trained=True,codes_reencoded_after_publication=True,
            initial_encoder_and_rng_preserved=True,local_key_and_value_trained=True,
            local_padding_mask_explicit=True,local_first_decoder_second_encoder_gradients_recorded=True,
            joint_nonzero_key_origin_preserved=True,
            key_decoder_parameters=adapter.num_layers*adapter.heads*adapter.rank*adapter.head_dim,
            key_statistics_fit_only=True,key_fold_has_no_runtime_scale=True,
            zero_coordinate_logit_max_error=identity_error,zero_coordinate_code_max_error=identity_code_error,
            publication_fold_logit_max_error=fold_error,publication_fold_code_max_error=fold_code_error,
            single_code_for_key_and_value=True,no_added_user_state=True,quality_claim=False),
        cost=dict(components=dict(costs),calibration_flops=sum(costs.values()),
            stage_components=dict(read_calibration=read_costs,joint_refinement=joint_costs),
            teacher_prefix_capture_passes=1,teacher_prefix_reused_for_joint=True,
            item_feature_payload_lookup_bytes=lookup_bytes,
            development_resource=dict(source_trajectory_flops=resource),
            source_reconstruction_sensitivity=dict(additional_flops=resource,
                calibration_if_source_reconstructed=sum(costs.values())+resource),
            convention="Multiply-add=2. Local frozen q/S0/V and same-q teacher are captured once; each local epoch recomputes masked attention and native/code reductions with full Key/Value gradients. Initialization-code encoding and fit-only Key statistics are charged. Selected local Key and Value folds are joint origins. Key score work is counted in every query/check. Each joint forward freshly encodes all retained rows. Query and encoder backward each conservatively charged2x forward; all validation, coordinate, loss, prior and optimizer work included. Existing frozen-code preparation retained for initial checks; no free reuse of trainable codes."))
    write_json(args.output/"calibration.json",metadata)
    print(json.dumps(dict(status="complete",target=args.target,elapsed_seconds=metadata["elapsed_seconds"],
        calibration_flops=metadata["cost"]["calibration_flops"],selected_joint_epoch=joint_diagnostics["selected_epoch"],
        peak_gpu_allocated_gib=metadata["peak_gpu_allocated_gib"])),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel",type=Path,default=PANEL)
    parser.add_argument("--target",choices=("v3","v4","v5"),required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--gpu",type=int,choices=range(4),default=2)
    parser.add_argument("--small",action="store_true")
    parser.add_argument("--batch-size",type=int,default=8)
    parser.add_argument("--threads",type=int,default=4)
    run(parser.parse_args())
