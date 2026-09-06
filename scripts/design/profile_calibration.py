#!/usr/bin/env python3
"""Time the unchanged scheme5 calibration equations on the first frozen edge."""

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts"))

from design.data import DAY, fixed_split, frozen_model, histories, prefix  # noqa: E402
from design.run import (  # noqa: E402
    fit_translator,
    initialize_calibration,
    make_scenes,
    new_translator,
    timed,
    training_state,
    write_json,
)

from hstu_kvcache.adaptation import RidgeTranslator, inference  # noqa: E402
from hstu_kvcache.adaptation.reader import full_response_rates, history_read, score  # noqa: E402
from hstu_kvcache.models import HSTUKVCache  # noqa: E402


@torch.no_grad()
def response_canary(model, scenes, args, reference, ledger, compile_response, out):
    """Compare the new target kernel with the retained explicit two-read formula."""
    device=next(model.parameters()).device
    mapper=new_translator(args,1,device)
    saved=torch.load(reference/"translator_v1.pt",map_location=device,weights_only=False)
    mapper.load_state_dict(saved["state_dict"])
    selected=[*scenes[:14],*scenes[-2:]]
    states=[training_state(scene,model,mapper) for scene in selected]
    sources=[state.pack_source(1) for state in states]
    if compile_response:
        inference.enable_calibration()
    records=[]
    retained=[]
    for batch in (1,16):
        features=torch.stack([mapper.features(s) for s in sources[:batch]])
        counts=torch.stack([s.count.sum() for s in sources[:batch]])
        cache=HSTUKVCache(torch.cat([s.cache.k for s in states[:batch]],1),
                         torch.cat([s.cache.v for s in states[:batch]],1),states[0].cache.seq_len)
        teacher=HSTUKVCache(torch.cat([s.teacher.k for s in selected[:batch]],1),
                           torch.cat([s.teacher.v for s in selected[:batch]],1),cache.seq_len)
        candidates=torch.cat([s.candidates for s in selected[:batch]])
        delta=torch.cat([s.query_delta for s in selected[:batch]])
        response=mapper.rates(features)*counts[:,None,None]
        temporal=mapper.time_coefficients(features)*counts[:,None,None,None]
        mask=torch.stack([s.response_mask(1) for s in states[:batch]])
        _,trace=score(model,cache,candidates,delta,trace=True,response_delta=response,response_time_delta=temporal)
        expected=[]
        for layer,(block,q) in enumerate(zip(model.blocks,trace.queries,strict=True)):
            wanted=history_read(block.attn,q,teacher.k[layer],teacher.v[layer],count=mask)
            actual=history_read(block.attn,q,cache.k[layer],cache.v[layer],count=mask)
            values=(wanted-actual).reshape(batch,block.attn.num_heads,4,candidates.shape[1]//4,block.attn.head_dim)
            expected.append(values.mean(3).transpose(1,2).flatten(2)/counts[:,None,None])
        expected=torch.stack(expected,dim=2)
        eager=full_response_rates(model,cache,teacher,candidates,delta,counts,response,temporal,4)
        torch.testing.assert_close(eager,expected,atol=1e-9,rtol=1e-5)
        actual=timed(partial(inference.calibration_rates,model,cache,teacher,candidates,delta,counts,response,temporal,4),
                     ledger,"response_compilation_and_warmup")
        difference=(actual-expected).abs()
        relative_rms=float(difference.square().mean().sqrt()/expected.square().mean().sqrt().clamp_min(1e-8))
        layer_rms=(difference.square().mean((0,1,3)).sqrt()/expected.square().mean((0,1,3)).sqrt().clamp_min(1e-8))
        elementwise_misses=int((difference>1e-8+5e-4*expected.abs()).sum())
        row=dict(batch=batch,max_abs=float(difference.max()),relative_rms=relative_rms,
                 layer_relative_rms=layer_rms.tolist(),eager_max_abs=float((eager-expected).abs().max()),
                 original_elementwise_threshold_misses=elementwise_misses)
        records.append(row)
        write_json(out/"response_comparison.json",dict(comparisons=records))
        assert relative_rms<1e-4 and float(layer_rms.max())<1e-4
        retained.append((actual,actual.clone()))
    for actual,expected in retained:
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    return dict(passed=True,comparisons=records,rate_outputs_own_storage=True,
                criterion="eager vs old elementwise formula; compiled relative RMS below1e-4 globally and in every layer; original elementwise misses retained",
                scenarios="Parent source, real early replay, lifetime descendant and Current-Exact zero control")


@torch.no_grad()
def main(cli):
    out=ROOT/"results/design"/cli.run_id
    out.mkdir(parents=True,exist_ok=False)
    reference=ROOT/"results/design/v5_efficient512_01"
    config=json.loads((reference/"configuration.json").read_text())
    args=SimpleNamespace(**config)
    uids=fixed_split()["calibration"][:args.fit_users]
    write_json(out/"configuration.json",dict(reference=reference.name,target=1,
        calibration_uids=uids,lifetime_users=args.lifetime_users,passes=args.steps,
        estimated_seconds=[90,300] if cli.compile_response else ([60,240] if cli.compile_prefixes else [30,150]),
        expected_peak_gib=24 if cli.compile_response else 18,
        compile_prefixes=cli.compile_prefixes,torch_version=torch.__version__,cxx=os.environ.get("CXX","g++"),
        compile_response=cli.compile_response,
        compilation_scope="actual warmup retained; existing Inductor disk cache may be reused; first calls of unseen shapes may remain in preparation",
        purpose="split source/teacher preparation, corrected-response generation, ridge solve and SVD cost; unchanged inputs/equations",
        confirmation_read=False,
        reference_configuration_sha256=hashlib.sha256((reference/"configuration.json").read_bytes()).hexdigest(),
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                      [Path(__file__),ROOT/"scripts/design/run.py",ROOT/"src/hstu_kvcache/adaptation/translator.py",
                       ROOT/"src/hstu_kvcache/adaptation/inference.py",ROOT/"src/hstu_kvcache/adaptation/reader.py"]}))
    for p in [Path(__file__),ROOT/"scripts/design/run.py",ROOT/"src/hstu_kvcache/adaptation/translator.py",
              ROOT/"src/hstu_kvcache/adaptation/inference.py",ROOT/"src/hstu_kvcache/adaptation/reader.py"]:
        (out/p.name).write_bytes(p.read_bytes())
    torch.set_num_threads(4)
    torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device("cuda:0")
    start=time.perf_counter()
    ledger=defaultdict(float)
    history=timed(lambda:histories(uids,232),ledger,"history_io")
    previous=timed(lambda:frozen_model(0,device),ledger,"model_io")
    current=timed(lambda:frozen_model(1,device),ledger,"model_io")
    cutover=231*DAY
    prefix_checks=[]
    if cli.compile_prefixes:
        inference.enable_prefixes()
        real=[prefix(history,uid,cutover,device) for uid in uids[:16]]
        retained=[]
        for name,model in (("parent",previous),("current",current)):
            for batch in (1,16):
                values=[torch.cat([p[i] for p in real[:batch]]) for i in range(3)]
                expected=timed(partial(model.compute_kv,*values),ledger,"diagnostic_prefix_reference")
                actual=timed(partial(inference.compute_prefix,model,*values),ledger,"prefix_compilation_and_warmup")
                differences={}
                for field in ("k","v"):
                    left,right=getattr(actual,field),getattr(expected,field)
                    torch.testing.assert_close(left,right,atol=2e-5,rtol=2e-5)
                    differences[field]=float((left-right).abs().max())
                retained.append((actual,actual.k.clone(),actual.v.clone()))
                prefix_checks.append(dict(model=name,batch=batch,max_difference=differences))
        for actual,k,v in retained:
            torch.testing.assert_close(actual.k,k,atol=0,rtol=0)
            torch.testing.assert_close(actual.v,v,atol=0,rtol=0)
        del retained,actual,expected,k,v,real,values,left,right
        write_json(out/"canary.pass.json",dict(passed=True,prefix_checks=prefix_checks,snapshots_own_graph_outputs=True))
        print(json.dumps(dict(phase="prefix_compilation_canary",passed=True,comparisons=prefix_checks)),flush=True)
    states,early=initialize_calibration(previous,history,uids,cutover,args,ledger)
    scenes=make_scenes(current,history,uids,states,early,cutover,ledger,previous,1,args)
    response_checks=response_canary(current,scenes,args,reference,ledger,cli.compile_response,out)
    write_json(out/"response_canary.pass.json",response_checks)
    print(json.dumps(dict(phase="response_kernel_canary",**response_checks)),flush=True)
    phases=[]

    def instrument(name,fn):
        def wrapped(*values,**kwargs):
            torch.cuda.synchronize()
            begin=time.perf_counter()
            result=fn(*values,**kwargs)
            torch.cuda.synchronize()
            phases.append(dict(phase=name,seconds=time.perf_counter()-begin,
                input_shape=list(values[1].shape if name=="ridge_fit" else values[0].shape)))
            return result
        return wrapped

    original_fit,original_solve,original_svd=RidgeTranslator.fit,torch.linalg.solve,torch.linalg.svd
    RidgeTranslator.fit=instrument("ridge_fit",original_fit)
    torch.linalg.solve=instrument("linear_solve",original_solve)
    torch.linalg.svd=instrument("svd",original_svd)
    try:
        _,records,_=timed(lambda:fit_translator(current,scenes,1,args,ledger),ledger,"translator_fit")
    finally:
        RidgeTranslator.fit,torch.linalg.solve,torch.linalg.svd=original_fit,original_solve,original_svd
    totals={name:sum(r["seconds"] for r in phases if r["phase"]==name)
            for name in ("ridge_fit","linear_solve","svd")}
    totals["response_generation_and_setup"]=ledger["translator_fit"]-totals["ridge_fit"]
    write_json(out/"summary.json",dict(status="diagnostic_complete",elapsed_seconds=time.perf_counter()-start,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/(1<<20),calibration_scenes=len(scenes),
        ledger_seconds=dict(ledger),fitting_breakdown_seconds=totals,phases=phases,fit_records=records,
        prefix_checks=prefix_checks,
        response_checks=response_checks,
        scope="single fixed calibration edge; solve/SVD timers nested within ridge_fit, not additive with it",
        limitation="synchronized instrumentation adds overhead; no new quality or population result",
        confirmation_read=False))
    print((out/"summary.json").read_text(),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id",required=True)
    parser.add_argument("--compile-prefixes",action="store_true")
    parser.add_argument("--compile-response",action="store_true")
    main(parser.parse_args())
