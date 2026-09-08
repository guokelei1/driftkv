#!/usr/bin/env python3
"""Frozen A/C reads only: residual and target-change energy at each branch's own q."""

import argparse
import json
from collections import defaultdict

import pandas as pd
import torch
from design.diagnose_decoder_closure import SMALL, main
from design.diagnose_native_coverage import A
from design.diagnose_native_input import read_input
from design.diagnose_query_holdout import batch_cache, panels
from design.diagnose_summary_objective import install_layer
from design.run import training_state

from hstu_kvcache.adaptation.reader import history_read


def experiment(current,scenes,mapper,history,cutover,fit_uids,uid_count,batch_size,out,target,canary):
    """No fitting: same-q residual / teacher-change audit of frozen native A and C."""
    del uid_count,canary
    run = out.parent / "native_coverage384_01"
    meta = pd.read_parquet(run / f"scenes_m{target}.parquet")
    assert len(meta)==len(scenes) and meta.uid.tolist()==[s.uid for s in scenes]
    states = [training_state(s,current,mapper) for s in scenes]
    sources = [s.pack_source(target) for s in states]
    base = torch.stack([mapper.features(s) for s in sources])
    counts = torch.stack([s.count.sum() for s in sources])
    active = mapper.active_layers(base)
    projection = torch.load(SMALL / f"source_projection_mean_m{target}.pt",map_location=base.device,weights_only=True)
    latent = (base.double()-projection["center"])/projection["scale"]@projection["projection"]
    latent = torch.cat((torch.ones_like(latent[:,:1]),latent),-1)
    groups,all_panels = defaultdict(list),[]
    for i,s in enumerate(states):
        groups[s.cache.seq_len].append(i)
        all_panels.append(panels(history,scenes[i].uid,cutover,float(scenes[i].query_delta.max()),base.device)[0])
    rows = []
    for method in ("A","C"):
        params = torch.load(A / f"translator_native_input_m{target}.pt" if method=="A" else run / f"translator_C_m{target}.pt",map_location=base.device,weights_only=True)
        b,a = base.new_zeros(len(scenes),6,6,32),base.new_zeros(len(scenes),6,6,32,32)
        for layer,p in enumerate(params):
            b[:,layer],a[:,layer] = install_layer(latent,p["weights"],p["query_center"],p["query_scale"],active[:,layer])
        for indices_all in groups.values():
            for start in range(0,len(indices_all),batch_size):
                batch = indices_all[start:start+batch_size]
                for panel in (("fit64","held64") if method=="A" else ("fit16","held64")):
                    indices = [i for i in batch if meta.iloc[i].group=="original_fitting"] if method=="A" and panel=="fit64" else batch
                    if method=="A" and panel=="held64":
                        indices = [i for i in indices if meta.iloc[i].group!="additional_fitting"]
                    if method=="C" and panel=="fit16":
                        indices = [i for i in indices if scenes[i].uid in fit_uids]
                    if not indices:
                        continue
                    inputs = tuple(torch.cat([all_panels[i]["fit64" if panel=="fit16" else panel][j] for i in indices]) for j in range(2))
                    if panel=="fit16":
                        inputs = tuple(x[:,::4] for x in inputs)
                    cache = batch_cache([states[i].cache for i in indices])
                    teacher = batch_cache([scenes[i].teacher for i in indices])
                    _,trace,_,corrections = read_input(current,cache,inputs,b[indices],a[indices],params,counts[indices],active[indices])
                    for layer,q in enumerate(trace.queries):
                        change = history_read(current.blocks[layer].attn,q,teacher.k[layer],teacher.v[layer])-trace.history_heads[layer]
                        eps = corrections[layer].double()-change.double()
                        for j,i in enumerate(indices):
                            rows.append(dict(target=target,uid=scenes[i].uid,scene=i,method=method,panel=panel,layer=layer,
                                response_mse=float(eps[j].square().mean()),target_change_energy=float(change[j].double().square().mean()),
                                source_response_energy=float(trace.history_heads[layer][j].double().square().mean())))
    result = pd.DataFrame(rows)
    original = pd.read_parquet(run / f"responses_m{target}.parquet")
    pair = result.merge(original,on=["target","uid","scene","method","panel","layer"],validate="one_to_one",suffixes=("_audit","_original"))
    assert len(pair)==len(result)
    torch.testing.assert_close(torch.tensor(pair.response_mse_audit.values),torch.tensor(pair.response_mse_original.values),atol=1e-3,rtol=2e-5)
    result.to_parquet(out / f"response_scale_m{target}.parquet",index=False)
    print(json.dumps(dict(target=target,status="frozen_response_audit_complete",rows=len(result))),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id",default="native_response_scale384_01")
    parser.add_argument("--cohort-path",default="results/design/native_coverage_cohort_01/cohort.json")
    parser.add_argument("--fit-users",type=int,default=64)
    parser.add_argument("--diagnostic-users",type=int,default=128)
    parser.add_argument("--batch-size",type=int,default=4)
    parser.add_argument("--canary",action="store_true")
    parser.add_argument("--estimate-seconds",type=int,default=240)
    main(parser.parse_args(),experiment)
