#!/usr/bin/env python3
"""Two predeclared 256x16 ablations: summary response input and no source condition."""

import argparse
import json
import time
from collections import defaultdict

import pandas as pd
import torch
from design.diagnose_decoder_closure import SMALL, main
from design.diagnose_native_coverage import META, frozen_solver_canary
from design.diagnose_native_input import fit_joint, producer_mean_cache, read_input
from design.diagnose_query_holdout import batch_cache, panels
from design.diagnose_summary_objective import install_layer
from design.run import training_state, write_json

from hstu_kvcache.adaptation.reader import history_read


def experiment(current,scenes,mapper,history,cutover,fit_uids,uid_count,batch_size,out,target,canary):
    """Same C fitting scenarios, native-A coordinates and mu; fit two necessary ablations only."""
    del uid_count
    if canary:
        frozen_solver_canary(next(current.parameters()).device)
    states = [training_state(s,current,mapper) for s in scenes]
    sources = [s.pack_source(target) for s in states]
    base = torch.stack([mapper.features(s) for s in sources])
    counts = torch.stack([s.count.sum() for s in sources])
    active = mapper.active_layers(base)
    projection = torch.load(SMALL / f"source_projection_mean_m{target}.pt",map_location=base.device,weights_only=True)
    latent = (base.double()-projection["center"])/projection["scale"]@projection["projection"]
    latent = torch.cat((torch.ones_like(latent[:,:1]),latent),-1)
    fixed = torch.load(out.parent / f"native_coverage384_01/translator_C_m{target}.pt",map_location=base.device,weights_only=True)
    original = pd.read_parquet(META / f"scenes_m{target}.parquet")
    mu = float(original.loc[original.group=="fitting_uid","count"].pow(2).mean())
    groups,all_panels = defaultdict(list),[]
    for i,state in enumerate(states):
        groups[state.cache.seq_len].append(i)
        all_panels.append(panels(history,scenes[i].uid,cutover,float(scenes[i].query_delta.max()),base.device)[0])
    batches = [v[j:j+batch_size] for v in groups.values() for j in range(0,len(v),batch_size)]
    fitting = [i for i,s in enumerate(scenes) if s.uid in fit_uids]
    means,masses = producer_mean_cache(sources,target)
    records,costs = [],[]
    meta = pd.DataFrame(dict(uid=[s.uid for s in scenes],count=counts.cpu().tolist()))
    meta["state_ordinal"] = meta.groupby("uid").cumcount()
    if not canary:
        cmeta = pd.read_parquet(out.parent / f"native_coverage384_01/scenes_m{target}.parquet")
        cmeta = cmeta[cmeta.group.isin(["original_fitting","additional_fitting"])]
        pair = meta.merge(cmeta,on=["uid","state_ordinal"],validate="one_to_one",suffixes=("_new","_C"))
        assert len(pair)==len(meta)==len(cmeta) and (pair["count_new"]==pair["count_C"]).all()
    meta.to_parquet(out / f"scenes_m{target}.parquet",index=False)
    for method in ("summary_input","no_source"):
        x = latent if method=="summary_input" else torch.ones_like(latent[:,:1])
        b,a,params = base.new_zeros(len(scenes),6,6,32),base.new_zeros(len(scenes),6,6,32,32),[]
        acquire,solve = 0.,0.
        for layer in range(6):
            torch.cuda.synchronize()
            start = time.perf_counter()
            q = base.new_zeros(len(scenes),6,16,32)
            wanted,observed = torch.zeros_like(q),base.new_zeros(len(scenes),16,192)
            for batch in batches:
                indices = [i for i in batch if scenes[i].uid in fit_uids]
                if not indices:
                    continue
                panel = tuple(torch.cat([all_panels[i]["fit64"][j] for i in indices])[:,::4] for j in range(2))
                source = batch_cache([states[i].cache for i in indices])
                teacher = batch_cache([scenes[i].teacher for i in indices])
                mean = type(means)(k=means.k[:,indices],v=means.v[:,indices]) if method=="summary_input" else None
                _,trace,obs,_ = read_input(current,source,panel,b[indices],a[indices],params,counts[indices],active[indices],mean,masses[indices])
                q[indices],observed[indices] = trace.queries[layer],obs[layer]
                wanted[indices] = (history_read(current.blocks[layer].attn,trace.queries[layer],teacher.k[layer],teacher.v[layer])-trace.history_heads[layer])/counts[indices,None,None,None]
            torch.cuda.synchronize()
            acquire += time.perf_counter()-start
            start = time.perf_counter()
            p,record = fit_joint(x[fitting],q[fitting],wanted[fitting],observed[fitting],counts[fitting],coordinates=fixed[layer],count_square_mean=mu)
            params.append(p)
            b[:,layer],a[:,layer] = install_layer(x,p["weights"],p["query_center"],p["query_scale"],active[:,layer])
            torch.cuda.synchronize()
            solve += time.perf_counter()-start
            records.append(dict(method=method,target=target,layer=layer,**record))
        torch.save([{k:v.cpu() for k,v in p.items()} for p in params],out / f"translator_{method}_m{target}.pt")
        costs.append(dict(method=method,target=target,query_acquisition_seconds=acquire,shared_fit_seconds=solve,
                          fitting_uids=len(fit_uids),fitting_scenes=len(fitting),queries_per_scene=16))
        print(json.dumps(dict(target=target,method=method,status="frozen")),flush=True)
    write_json(out / f"fits_m{target}.json",records)
    write_json(out / f"cost_m{target}.json",costs)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id",required=True)
    parser.add_argument("--cohort-path",required=True)
    parser.add_argument("--fit-users",type=int,default=64)
    parser.add_argument("--diagnostic-users",type=int,default=128)
    parser.add_argument("--batch-size",type=int,default=4)
    parser.add_argument("--canary",action="store_true")
    parser.add_argument("--estimate-seconds",type=int,required=True)
    main(parser.parse_args(),experiment)
