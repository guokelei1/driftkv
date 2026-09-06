#!/usr/bin/env python3
"""Small matched native/paired CC compilation cost probe; no new predictor."""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts"))

from design.data import DAY, fixed_split, frozen_model, histories  # noqa: E402
from design.run import cache_at, new_state, new_translator, tensor, write_json  # noqa: E402
from insight_two.common import load_frozen_inputs  # noqa: E402

from hstu_kvcache.adaptation.reader import score  # noqa: E402
from hstu_kvcache.adaptation.state import refresh_many, release_many  # noqa: E402
from hstu_kvcache.models import HSTUKVCache  # noqa: E402


def logits(model,k,v,candidates,delta,response=None,time_response=None):
    return score(model,HSTUKVCache(k,v,k.shape[2]),candidates,delta,
                 response_delta=response,response_time_delta=time_response)[0]


def source_logits(model,k,v,candidates,delta,sums,metadata,translator):
    counts=metadata[:,1]
    count=counts.sum()
    owners=torch.nn.functional.one_hot(metadata[:,0].long(),translator.producer_count).float().T
    values=owners@sums/count
    masses=owners@counts/count
    features=torch.cat((values,masses[:,None]),1).flatten()
    features=torch.cat((features,metadata[0,2:3].clamp(max=6144)/1024))
    response=translator.rates(features)[None]*count
    temporal=translator.time_coefficients(features)[None]*count
    return logits(model,k,v,candidates,delta,response,temporal)


def writer_inputs(state,zero):
    segments=list(state.writer.segments.values())
    assert len(segments)<=8  # Six frozen releases, 1024-event segments and cache.
    sums=torch.stack([s["sums"][0].flatten() for s in segments]+[zero]*(8-len(segments)))
    metadata=[[s["producer"],s["count"][0],0] for s in segments]+[[0,0,0]]*(8-len(segments))
    metadata[0][2]=state.writer.next_ordinal-state.release_ordinal
    return sums,torch.tensor(metadata,dtype=zero.dtype).to(zero.device,non_blocking=True)


@torch.no_grad()
def main(cli):
    out=ROOT/"results/design"/cli.run_id
    out.mkdir(parents=True,exist_ok=False)
    trained=ROOT/"results/design/v5_efficient512_01"
    args=SimpleNamespace(**json.loads((trained/"configuration.json").read_text()))
    uid=fixed_split()["development"][0]
    write_json(out/"configuration.json",dict(uid=uid,target=1,reference=str(trained.relative_to(ROOT)),
        mode="torch.compile reduce-overhead dynamic=True fullgraph=True",estimated_seconds=[30,180],
        scope="same native and paired scalar CC math; compilation time retained; no quality or population cost claim",
        confirmation_read=False,torch_version=torch.__version__,cxx=os.environ.get("CXX","g++"),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    (out/"source.py").write_bytes(Path(__file__).read_bytes())
    start=time.perf_counter()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device("cuda:0")
    h=histories([uid],232)
    parent,current=frozen_model(0,device),frozen_model(1,device)
    cache,times=cache_at(parent,h,uid,231*DAY)
    state=new_state(cache,times,0,args)
    translator=new_translator(args,1,device)
    saved=torch.load(trained/"translator_v1.pt",map_location=device,weights_only=False)
    translator.load_state_dict(saved["state_dict"])
    translator.supported_producers=set(saved["supported_producers"])
    release_many([state],1,translator)
    _,panels,_=load_frozen_inputs()
    candidates=tensor(panels[0,0,:1][None],device)
    delta=tensor([231*DAY-int(times[-1])],device,floating=True)
    compiled=torch.compile(logits,mode="reduce-overhead",dynamic=True,fullgraph=True)
    rows=[]
    for name in ("native","paired"):
        extra=() if name=="native" else (state.response_delta,state.response_time_delta)
        expected=logits(current,cache.k,cache.v,candidates,delta,*extra).clone()
        begin=time.perf_counter()
        torch.compiler.cudagraph_mark_step_begin()
        actual=compiled(current,cache.k,cache.v,candidates,delta,*extra)
        torch.cuda.synchronize()
        compile_seconds=time.perf_counter()-begin
        torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-5)
        difference=float((actual-expected).abs().max())
        timings={}
        for engine,fn in (("eager",logits),("compiled",compiled)):
            begin=time.perf_counter()
            for _ in range(100):
                torch.compiler.cudagraph_mark_step_begin()
                fn(current,cache.k,cache.v,candidates,delta,*extra)
            torch.cuda.synchronize()
            timings[engine]=(time.perf_counter()-begin)/100
        begin=time.perf_counter()
        if name=="paired":
            for _ in range(100):
                refresh_many([state],translator)
            torch.cuda.synchronize()
        refresh_seconds=(time.perf_counter()-begin)/100 if name=="paired" else 0.
        rows.append(dict(path=name,compile_seconds=compile_seconds,max_logit_difference=difference,
                         read_seconds=timings,dirty_refresh_seconds=refresh_seconds))
        print(json.dumps(rows[-1]),flush=True)
    zero=state.cache.k.new_zeros(6*2*192)
    sums,metadata=writer_inputs(state,zero)
    source_compiled=torch.compile(source_logits,mode="reduce-overhead",dynamic=True,fullgraph=True)
    expected=state.score(current,candidates,delta).clone()
    begin=time.perf_counter()
    torch.compiler.cudagraph_mark_step_begin()
    actual=source_compiled(current,cache.k,cache.v,candidates,delta,sums,metadata,translator)
    torch.cuda.synchronize()
    compile_seconds=time.perf_counter()-begin
    torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-5)
    difference=float((actual-expected).abs().max())
    begin=time.perf_counter()
    for _ in range(100):
        sums,metadata=writer_inputs(state,zero)
        torch.compiler.cudagraph_mark_step_begin()
        source_compiled(current,cache.k,cache.v,candidates,delta,sums,metadata,translator)
    torch.cuda.synchronize()
    rows.append(dict(path="compiled_dirty_read_including_source_assembly",compile_seconds=compile_seconds,
        max_logit_difference=difference,read_seconds=(time.perf_counter()-begin)/100))
    print(json.dumps(rows[-1]),flush=True)
    write_json(out/"summary.json",dict(status="diagnostic_complete",rows=rows,
        elapsed_seconds=time.perf_counter()-start,peak_allocated_mib=torch.cuda.max_memory_allocated()/(1<<20),
        scope="one real scalar CC; both native and paired compiled; no population, multi-shape or end-to-end qualification"))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id",required=True)
    main(parser.parse_args())
