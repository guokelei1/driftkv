#!/usr/bin/env python3
"""One disposable Adam step comparing original and reusable same-N execution."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"scripts"),str(ROOT/"src")]

import torch
from design_two.common import RELEASES,setup_device
from design_two.prepare import load_history,load_panel
from evaluate_yambda500m_foundation_raw import load_model
from query_relay_2026_10.adapter import UnifiedRelayAdapter
from query_relay_2026_10.calibrate import attach_rollout_tail,prepare
from query_relay_2026_10.calibration_cost import OperationCounts
from system_study_2026_10.joint_batch import execute as grouped_execute
from system_study_2026_10.joint_batch_probe import execute as reference_execute,flatten_result,difference
from unified_reuse_2026_09.common import sha256,write_json


def run(args):
    start=time.perf_counter()
    config=json.loads(args.config.read_text())
    meta_path=ROOT/config["calibration_metadata"]
    meta=json.loads(meta_path.read_text())
    selected=[meta["fit_uids"][i] for i in torch.randperm(len(meta["fit_uids"]),
        generator=torch.Generator().manual_seed(config["seed"])).tolist()[:8]]
    if selected!=config["selected_uids"]:
        raise ValueError("original optimizer subset changed")
    panel_path=ROOT/config["panel"]
    panel,_,_=load_panel(panel_path)
    source=ROOT/meta["source_pack"]
    checkpoint=meta_path.with_suffix(".pt")
    model_record=panel["models"]["v4"]
    model_path=ROOT/model_record["checkpoint"]
    for path,digest in ((source,meta["source_pack_sha256"]),(checkpoint,meta["weights_sha256"]),
        (model_path,model_record["checkpoint_sha256"]),(panel_path/"binding.json",meta["panel_sha256"])):
        if sha256(path)!=digest:
            raise ValueError(f"original binding differs: {path}")
    settings=dict(config=config,pid=__import__("os").getpid(),source_sha256=meta["source_pack_sha256"],
        adapter_sha256=meta["weights_sha256"],model_sha256=model_record["checkpoint_sha256"],
        sources={str(p.relative_to(ROOT)):sha256(p) for p in (Path(__file__),
            ROOT/"scripts/system_study_2026_10/joint_batch.py",
            ROOT/"scripts/system_study_2026_10/joint_batch_probe.py",
            ROOT/"scripts/query_relay_2026_10/calibrate.py")},
        scope="One Adam step in two disposable adapter copies; original first8 fit users. No new fitted candidate or complete calibration timing.")
    write_json(args.output/"configuration.json",settings)
    device=setup_device(args.gpu,2)
    model,payload=load_model(model_path,device)
    del payload
    model.eval().requires_grad_(False)
    for block in model.blocks:
        block.attn.backend="triton"
    payload=torch.load(checkpoint,map_location="cpu",weights_only=False)
    seed=UnifiedRelayAdapter(**payload["config"]).to(device).eval()
    seed.load_state_dict(payload["state_dict"])
    immutable={n:v.detach().cpu().clone() for n,v in seed.state_dict().items()}
    del payload
    pack=torch.load(source,map_location="cpu",weights_only=False)
    captured={u:pack["rows"][u] for u in selected}
    histories=load_history(panel_path,requested_uids=selected)
    attach_rollout_tail(captured,histories,dict(RELEASES)["v4"],4,seed.max_length,pack["records"])
    del pack,histories
    rows=prepare(model,captured,seed,device)
    del captured
    snapshots={}
    for mode in ("serial_reference","grouped"):
        adapter=copy.deepcopy(seed)
        parameters=[p for p in adapter.parameters() if p.requires_grad]
        optimizer=torch.optim.Adam(parameters,lr=meta["learning_rate"]*.25)
        optimizer.zero_grad(set_to_none=True)
        costs,execution,operations=Counter(),Counter(),OperationCounts()
        if mode=="serial_reference":
            outputs=reference_execute(model,adapter,rows,selected,grouped=False,backward=True)
        else:
            outputs=grouped_execute(model,adapter,rows,selected,backward=True,
                costs=costs,operations=operations,execution_counts=execution)
        values,logits=flatten_result(outputs,selected)
        gradients={n:None if p.grad is None else p.grad.detach().cpu().clone()
            for n,p in adapter.named_parameters() if p.requires_grad}
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
        optimizer.step()
        states={}
        for n,p in adapter.named_parameters():
            if p.requires_grad:
                states[n]={k:v.detach().cpu().clone() if isinstance(v,torch.Tensor) else v
                    for k,v in optimizer.state[p].items()}
        snapshots[mode]=dict(losses=values,logits=logits,gradients=gradients,
            parameters={n:p.detach().cpu().clone() for n,p in adapter.named_parameters()},
            adam=states,clip_input_norm=float(norm),logical_counts=dict(costs),
            execution_counts=dict(execution),
            operation_ledger=operations.report(origin={}) if mode=="grouped" else None)
        del optimizer,adapter,outputs
    ref,got=snapshots["serial_reference"],snapshots["grouped"]
    metrics=dict(losses=difference(ref["losses"],got["losses"]),logits=difference(ref["logits"],got["logits"]),
        serial_losses=ref["losses"].tolist(),grouped_losses=got["losses"].tolist(),
        clip_input_norms=[ref["clip_input_norm"],got["clip_input_norm"]])
    all_pass=True
    for field in ("parameters","gradients"):
        per={}
        for name,value in ref[field].items():
            candidate=got[field][name]
            if value is None:
                if candidate is not None:
                    raise AssertionError(("gradient presence",name))
                per[name]=None
            else:
                per[name]=difference(value,candidate)
                all_pass=all_pass and bool(torch.allclose(value,candidate,
                    atol=config["adam_atol"],rtol=config["adam_rtol"]))
        metrics[field]=per
        rv=torch.cat([v.flatten() for v in ref[field].values() if v is not None])
        gv=torch.cat([v.flatten() for v in got[field].values() if v is not None])
        metrics["all_"+field]=difference(rv,gv)
    adam_metrics={}
    for name,state in ref["adam"].items():
        adam_metrics[name]={}
        for field,value in state.items():
            candidate=got["adam"][name][field]
            if not isinstance(value,torch.Tensor):
                if candidate!=value:
                    raise AssertionError(("Adam scalar",name,field))
            else:
                adam_metrics[name][field]=difference(value,candidate)
                all_pass=all_pass and bool(torch.allclose(value,candidate,
                    atol=config["adam_atol"],rtol=config["adam_rtol"]))
    metrics["adam_state"]=adam_metrics
    metrics["within_declared_tolerance"]=all_pass
    write_json(args.output/"equivalence.json",metrics)
    if not all_pass:
        raise AssertionError("single Adam step differs beyond declared tolerance")
    unchanged=all(torch.equal(immutable[n],v.detach().cpu()) for n,v in seed.state_dict().items())
    if not unchanged or any(p.grad is not None for p in model.parameters()):
        raise AssertionError("original adapter or backbone changed")
    expected=dict(joint_forward_calls=8,joint_rollout_forward_calls=8,
        joint_rollout_behavior_queries=32,joint_backward_calls=8,
        joint_forward_token_queries=8*1024*16,joint_rollout_forward_token_queries=8*1024*16)
    if got["logical_counts"]!=expected:
        raise AssertionError(("logical count changed",got["logical_counts"],expected))
    summary=dict(status="complete",selected_uids=selected,losses=metrics["losses"],
        logits=metrics["logits"],gradients=metrics["all_gradients"],after_step_parameters=metrics["all_parameters"],
        logical_counts=got["logical_counts"],execution_group_calls=got["execution_counts"],
        operation_ledger=got["operation_ledger"],optimizer_steps_per_disposable_copy=1,
        original_weights_unchanged=True,backbone_gradients_absent=True,within_declared_tolerance=True,
        elapsed_whole_probe_seconds=time.perf_counter()-start,scope=settings["scope"],
        timing_scope="Includes loading, checks, CPU copies and ledger instrumentation; not a speed measurement.")
    write_json(args.output/"summary.json",summary)
    (args.output/"summary.md").write_text("# 一次Adam更新核对\n\n"+
        "原seed17第一8名fit用户；使用既有V4参数的两个临时副本，原逐人执行与新same-N模块各反传、clip及Adam一步，未保存候选权重。\n\n"+
        f"参数最大差 {summary['after_step_parameters']['max_abs']:.6g}，梯度相对L2 {summary['gradients']['relative_l2']:.6g}。逐参数与Adam一、二阶moment原始误差见equivalence.json。\n\n"+
        "logical ledger仍为8次static、8次rollout、32个逐时序事件、8份反传；实际执行合为1次group forward/backward与4次批事件。原8人Adam边界未变。\n\n"+
        "这不代表完整训练后的权重逐位一致或最佳epoch必然相同；浮点归约顺序仍可能影响接近的验证值。未测完整S。\n")
    print(json.dumps(dict(status="complete",parameters=summary["after_step_parameters"],
        gradients=summary["gradients"],execution=summary["execution_group_calls"])),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--gpu",type=int,default=0)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    sys.stdout=sys.stderr=(args.output/"run.log").open("a",buffering=1)
    try:
        run(args)
    except BaseException as error:
        write_json(args.output/"failure.json",dict(error=repr(error),traceback=traceback.format_exc()))
        raise
