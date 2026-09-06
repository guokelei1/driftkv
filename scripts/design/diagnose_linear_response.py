#!/usr/bin/env python3
"""Diagnostic first-order ELU response at actual corrected queries.

Teacher moment differences are interventions, never executable translations.
No state, labels, backbone parameters or persistent K/V are modified.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts"))

from design.data import DAY, fixed_split, frozen_model, histories  # noqa: E402
from design.run import cache_at, tensor, write_json  # noqa: E402
from insight_two.common import load_frozen_inputs, metrics_row, score_metrics  # noqa: E402

from hstu_kvcache.adaptation.reader import history_read, score  # noqa: E402


def moments(cache, heads=6):
    layers,batch,length,width=cache.k.shape
    k=cache.k.reshape(layers,batch,length,heads,width//heads).transpose(2,3)
    v=cache.v.reshape(layers,batch,length,heads,width//heads).transpose(2,3)
    return v.sum(-2),k.transpose(-2,-1)@v


@torch.no_grad()
def linear_intervention(model,cache,candidates,delta,value_change,cross_change):
    handles=[]
    for layer,block in enumerate(model.blocks):
        query={}

        def capture(_module,_inputs,output,query=query,attention=block.attn):
            query["q"]=output.reshape(output.shape[0],output.shape[1],attention.num_heads,attention.head_dim).transpose(1,2)

        def inject(_module,inputs,query=query,layer=layer,attention=block.attn):
            correction=value_change[layer][:,:,None]+(query["q"]*attention.scale)@cross_change[layer]
            return (inputs[0]+correction.transpose(1,2).flatten(2),)

        handles.append(block.attn.q_proj.register_forward_hook(capture))
        handles.append(block.attn.out_proj.register_forward_pre_hook(inject))
    try:
        return score(model,cache,candidates,delta,trace=True)
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def main():
    out=ROOT/"results/design/v6_linear_response_probe_01"
    out.mkdir(parents=True,exist_ok=False)
    reference=ROOT/"results/design/v4_query_time_probe_01"
    baseline=json.loads((reference/"raw.json").read_text())
    by_key={(r["uid"],r["query"]):r for r in baseline}
    uids=fixed_split()["development"][:128]
    write_json(out/"configuration.json",dict(users=uids,target=5,reference=str(reference.relative_to(ROOT)),
        reference_raw_sha256=hashlib.sha256((reference/"raw.json").read_bytes()).hexdigest(),
        formula="ELU(z)+1 approximately 1+z; paired response = delta(sum V) + scale*actual_Q @ delta(sum K outer V)",
        query_times="same four causal idle-interval offsets and 16 candidates as retained reference",
        scope="teacher-moment intervention only; not executable adaptation or quality evidence",
        estimated_seconds=[25,60],confirmation_read=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    (out/"source.py").write_bytes(Path(__file__).read_bytes())
    start=time.perf_counter()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device("cuda:0")
    h=histories(uids,288)
    parent,current=frozen_model(4,device),frozen_model(5,device)
    _,panels,_=load_frozen_inputs()
    rows=[]
    max_reference_difference=0.
    for index,uid in enumerate(uids):
        cache,_=cache_at(parent,h,uid,287*DAY)
        exact,_=cache_at(current,h,uid,287*DAY)
        old_v,old_kv=moments(cache)
        new_v,new_kv=moments(exact)
        dv,dkv=new_v-old_v,new_kv-old_kv
        candidates=tensor(panels[4,index,:16][None],device)
        for query in ("cutover","one_second","one_minute","one_hour"):
            ref=by_key[uid,query]
            delta=tensor([ref["delta_seconds"]],device,floating=True)
            wanted=tensor(ref["exact"],device,floating=True)
            reuse=tensor(ref["reuse"],device,floating=True)
            if index==0:
                fresh=score(current,cache,candidates,delta)[0]
                torch.testing.assert_close(fresh,reuse,atol=2e-6,rtol=2e-6)
                zero,_=linear_intervention(current,cache,candidates,delta,torch.zeros_like(dv),torch.zeros_like(dkv))
                torch.testing.assert_close(zero,fresh,atol=1e-6,rtol=1e-6)
                max_reference_difference=max(max_reference_difference,float((fresh-reuse).abs().max()))
            logits,trace=linear_intervention(current,cache,candidates,delta,dv,dkv)
            residual=[]
            for layer,(block,q) in enumerate(zip(current.blocks,trace.queries,strict=True)):
                true=history_read(block.attn,q,exact.k[layer],exact.v[layer])-history_read(block.attn,q,cache.k[layer],cache.v[layer])
                approximation=dv[layer][:,:,None]+(q*block.attn.scale)@dkv[layer]
                residual.append(float((true-approximation).square().sum()/true.square().sum().clamp_min(1e-20)))
            rows.append(dict(uid=uid,query=query,logits=logits.cpu().tolist(),response_relative_mse_by_layer=residual,
                **metrics_row(score_metrics(wanted,reuse,logits))))
    write_json(out/"raw.json",rows)
    summary=[]
    for query in ("cutover","one_second","one_minute","one_hour"):
        selected=[r for r in rows if r["query"]==query]
        summary.append(dict(query=query,users=len(selected),
            mean_user_recovery=float(np.mean([r["probability_gap_recovery"] for r in selected])),
            mean_probability_gap=float(np.mean([r["observed_probability_gap"] for r in selected])),
            mean_response_relative_mse_by_layer=np.mean([r["response_relative_mse_by_layer"] for r in selected],0).tolist()))
    write_json(out/"summary.json",dict(status="diagnostic_complete",rows=summary,
        zero_intervention_checked=True,max_reference_difference=max_reference_difference,
        elapsed_seconds=time.perf_counter()-start,peak_allocated_mib=torch.cuda.max_memory_allocated()/(1<<20),
        scope="paired first-order teacher-moment intervention at actual corrected query; no method or quality claim"))
    print((out/"summary.json").read_text())


if __name__=="__main__":
    main()
