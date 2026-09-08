#!/usr/bin/env python3
"""One representation contrast on a common affine-lower-context query panel."""

import argparse
import json
import time
from collections import defaultdict

import pandas as pd
import torch
from design.diagnose_decoder_closure import main
from design.diagnose_query_holdout import (
    batch_cache,
    panels,
    response_change,
    solve_affine,
    view_score,
)
from design.run import training_state, write_json


def experiment(current,scenes,mapper,history,cutover,fit_uids,uid_count,batch_size,out,target,canary):
    del uid_count
    states=[training_state(s,current,mapper) for s in scenes]
    groups=defaultdict(list)
    for i,s in enumerate(states):
        groups[s.cache.seq_len].append(i)
    rows,ordinals=[],defaultdict(int)
    metadata=[]
    for s in scenes:
        o=ordinals[s.uid]; ordinals[s.uid]+=1
        metadata.append(dict(uid=s.uid,state_ordinal=o,group="fitting_uid" if s.uid in fit_uids else "diagnostic_uid",
                             state_group="continuous_cutover" if o==0 else "auxiliary"))
    start=time.perf_counter()
    for indices in groups.values():
        for offset in range(0,len(indices),batch_size):
            ix=indices[offset:offset+batch_size]
            source=batch_cache([states[i].cache for i in ix]); teacher=batch_cache([scenes[i].teacher for i in ix])
            n=source.k.new_tensor([states[i].cache.seq_len for i in ix])
            all_panels=[panels(history,scenes[i].uid,cutover,float(scenes[i].query_delta.max()),n.device)[0] for i in ix]
            panel={name:tuple(torch.cat([p[name][j] for p in all_panels]) for j in range(2)) for name in ("fit64","held64")}
            b=source.k.new_zeros(len(ix),6,6,32); a=source.k.new_zeros(len(ix),6,6,32,32)
            for layer in range(6):
                # Both candidate functions see identical q and target at this layer.
                # The only propagated lower view is the predeclared per-scene affine.
                _,fit=view_score(current,source,panel["fit64"],b,a,n,True)
                _,held=view_score(current,source,panel["held64"],b,a,n,True)
                qfit,qheld=fit.queries[layer],held.queries[layer]
                yfit=response_change(current,source,teacher,fit,layer,n).double()
                yheld=response_change(current,source,teacher,held,layer,n).double()
                ib,ia,coef,center,scale=solve_affine(qfit,yfit)
                constant=yfit.mean(2,keepdim=True)
                xheld=torch.cat((torch.ones_like(qheld[...,:1]),(qheld.double()-center)/scale),-1)
                affine=xheld@coef
                residuals={"reuse":yheld,"constant":yheld-constant,"affine":yheld-affine}
                for j,i in enumerate(ix):
                    energy={k:float(v[j].square().mean()) for k,v in residuals.items()}
                    rows.append(dict(target=target,scene=i,layer=layer,count=int(n[j]),**metadata[i],
                        **{f"{k}_rate_mse":v for k,v in energy.items()},
                        **{f"{k}_aggregate_mse":v*float(n[j].square()) for k,v in energy.items()},
                        constant_ratio=None if energy["reuse"]==0 else energy["constant"]/energy["reuse"],
                        affine_ratio=None if energy["reuse"]==0 else energy["affine"]/energy["reuse"]))
                if canary:
                    # Save complete common tensors only for the tiny canary; formal
                    # evidence is per-state/per-layer energies, with deterministic code.
                    torch.save(dict(qfit=qfit.cpu(),qheld=qheld.cpu(),yfit=yfit.cpu(),yheld=yheld.cpu(),
                                    constant=constant.cpu(),affine=affine.cpu(),coef=coef.cpu(),center=center.cpu(),scale=scale.cpu()),
                               out/f"common_m{target}_scene{ix[0]}_layer{layer}.pt")
                    torch.testing.assert_close(constant,yfit.sum(2,keepdim=True)/64)
                    raw_a=coef[:,:,1:]/scale.transpose(-2,-1)
                    raw_b=coef[:,:,0]-(center@raw_a).squeeze(2)
                    torch.testing.assert_close(raw_b[:,:,None]+qheld.double()@raw_a,affine,rtol=1e-8,atol=1e-9)
                b[:,layer],a[:,layer]=ib,ia
    frame=pd.DataFrame(rows)
    frame.to_parquet(out/f"responses_m{target}.parquet",index=False)
    write_json(out/f"cost_m{target}.json",dict(seconds=time.perf_counter()-start,states=len(states),uids=len(ordinals),
        fit_queries=64,held_queries=64,context="same affine lower path for both representations",new_shared_fits=0))
    print(json.dumps(dict(target=target,states=len(states),seconds=time.perf_counter()-start)),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id",required=True)
    p.add_argument("--fit-users",type=int,default=64)
    p.add_argument("--diagnostic-users",type=int,default=128)
    p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--canary",action="store_true")
    p.add_argument("--estimate-seconds",type=int,required=True)
    main(p.parse_args(),experiment)
