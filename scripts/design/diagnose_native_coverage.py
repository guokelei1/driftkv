#!/usr/bin/env python3
"""Frozen native architecture: A=64x64, B=64x16, C=256x16; no new model search."""

import argparse
import json
import time
from collections import Counter, defaultdict

import pandas as pd
import torch
from design.data import ROOT
from design.diagnose_decoder_closure import SMALL, main
from design.diagnose_native_input import fit_joint, read_input, solver_canary
from design.diagnose_query_holdout import batch_cache, panels
from design.diagnose_summary_objective import install_layer
from design.run import training_state, write_json

from hstu_kvcache.adaptation.reader import history_read, score

A = ROOT / "results/design/mechanism_native_input192_01"
META = ROOT / "results/design/mechanism_decoder_closure192_01"


def frozen_solver_canary(device):
    solver_canary(device)
    g = torch.Generator(device=device).manual_seed(719)
    def r(*shape):
        return torch.randn(*shape, device=device, dtype=torch.float64, generator=g)
    h, q, y, obs = r(5,3), r(5,2,4,3), r(5,2,4,3), r(5,4,4)
    n = torch.tensor([16,32,256,512,1024], device=device)
    coordinates = dict(query_center=r(1,2,1,3), query_scale=r(1,2,1,3).abs()+1,
                       read_center=r(4), read_scale=r(4).abs()+1)
    mu = 900000.
    p, _ = fit_joint(h,q,y,obs,n,coordinates=coordinates,count_square_mean=mu)
    xq = (q-coordinates["query_center"])/coordinates["query_scale"]
    xq = torch.cat((torch.ones_like(xq[...,:1]),xq),-1)
    xo = (obs-coordinates["read_center"])/coordinates["read_scale"]
    x = torch.cat((torch.einsum("sr,shqj->shqrj",h,xq).flatten(-2),xo[:,None].expand(-1,2,-1,-1)),-1)
    w = n.double().square()/mu
    gram = torch.einsum("shqa,shqb,s->hab",x,x,w)/4
    rhs = torch.einsum("shqa,shqd,s->had",x,y,w)/4
    expected = torch.linalg.solve(gram+.05*torch.eye(gram.shape[-1],device=device,dtype=torch.float64),rhs)
    actual = torch.cat((p["weights"].transpose(0,1).flatten(1,2),p["read_weights"]),1)
    torch.testing.assert_close(actual,expected,atol=1e-9,rtol=1e-8)
    repeated, _ = fit_joint(h,q.repeat(1,1,4,1),y.repeat(1,1,4,1),obs.repeat(1,4,1),n,
                           coordinates=coordinates,count_square_mean=mu)
    for key in p:
        torch.testing.assert_close(p[key],repeated[key],atol=1e-9,rtol=1e-8)


def experiment(current, scenes, mapper, history, cutover, fit_uids, uid_count, batch_size, out, target, canary):
    """Only fit B/C, frozen A coordinates/PCA/mu, own actual lower queries; retain every UID."""
    config = json.loads((out / "configuration.json").read_text())
    cohort = config["cohort"]
    old, new = set(cohort["original_fitting_uids"]), set(cohort["additional_fitting_uids"])
    if canary:
        frozen_solver_canary(next(current.parameters()).device)
    states = [training_state(s,current,mapper) for s in scenes]
    sources = [s.pack_source(target) for s in states]
    base = torch.stack([mapper.features(s) for s in sources])
    counts = torch.stack([s.count.sum() for s in sources])
    active = mapper.active_layers(base)
    projection = torch.load(SMALL / f"source_projection_mean_m{target}.pt", map_location=base.device, weights_only=True)
    latent = (base.double()-projection["center"])/projection["scale"]@projection["projection"]
    latent = torch.cat((torch.ones_like(latent[:,:1]),latent),-1)
    original = pd.read_parquet(META / f"scenes_m{target}.parquet")
    original["state_ordinal"] = original.groupby("uid").cumcount()
    mu = float(original.loc[original.group == "fitting_uid", "count"].pow(2).mean())
    frozen = torch.load(A / f"translator_native_input_m{target}.pt", map_location=base.device, weights_only=True)
    groups, panel_list, metadata, ordinal = defaultdict(list), [], [], Counter()
    for i,(scene,state,source) in enumerate(zip(scenes,states,sources,strict=True)):
        o = ordinal[scene.uid]
        ordinal[scene.uid] += 1
        group = "original_fitting" if scene.uid in old else "additional_fitting" if scene.uid in new else "diagnostic"
        record = dict(target=target,scene=i,uid=scene.uid,state_ordinal=o,group=group,
                      state_group="continuous_cutover" if o == 0 else "auxiliary",
                      count=float(counts[i]),release_age=int(source.release_age))
        for producer in range(target+1):
            record[f"producer_{producer}_fraction"] = float((source.count*(source.producer[:,None]==producer)).sum()/counts[i])
        for layer in range(6):
            record[f"active_{layer}"] = bool(active[i,layer])
        metadata.append(record)
        groups[state.cache.seq_len].append(i)
        panel_list.append(panels(history,scene.uid,cutover,float(scene.query_delta.max()),base.device)[0])
    meta = pd.DataFrame(metadata)
    meta.to_parquet(out / f"scenes_m{target}.parquet",index=False)
    # Preserve original per-UID scenario coverage, independent of pooled scene numbering.
    check = meta[meta.uid.isin(set(original.uid))].merge(original,on=["uid","state_ordinal"],suffixes=("_new","_old"),validate="one_to_one")
    assert (check["count_new"] == check["count_old"]).all()
    assert (check.release_age_new == check.release_age_old).all()
    if not canary:
        assert len(check) == len(original)
        assert float(meta.loc[meta.group == "original_fitting","count"].pow(2).mean()) == mu
    batches = [v[j:j+batch_size] for v in groups.values() for j in range(0,len(v),batch_size)]
    def panel(indices,name):
        source_name = "fit64" if name == "fit16" else name
        result = tuple(torch.cat([panel_list[i][source_name][j] for i in indices]) for j in range(2))
        return tuple(x[:,::4] for x in result) if name == "fit16" else result
    shape = (len(scenes),6,6,32)
    rows, local, risks, fits, costs = [], [], [], [], []
    for method in ("A","B","C"):
        b, a = base.new_zeros(shape), base.new_zeros(*shape,32)
        parameters = frozen if method == "A" else []
        if method == "A":
            for layer,p in enumerate(parameters):
                b[:,layer],a[:,layer] = install_layer(latent,p["weights"],p["query_center"],p["query_scale"],active[:,layer])
        else:
            fitting = [i for i,s in enumerate(scenes) if s.uid in (old if method == "B" else old|new)]
            fitting_set = set(fitting)
            acquisition, solving = 0.,0.
            for layer in range(6):
                torch.cuda.synchronize()
                start = time.perf_counter()
                query = base.new_zeros(len(scenes),6,16,32)
                wanted = torch.zeros_like(query)
                observed = base.new_zeros(len(scenes),16,192)
                for batch in batches:
                    indices = [i for i in batch if i in fitting_set]
                    if not indices:
                        continue
                    cache = batch_cache([states[i].cache for i in indices])
                    teacher = batch_cache([scenes[i].teacher for i in indices])
                    _,trace,obs,_ = read_input(current,cache,panel(indices,"fit16"),b[indices],a[indices],parameters,counts[indices],active[indices])
                    query[indices],observed[indices] = trace.queries[layer],obs[layer]
                    wanted[indices] = (history_read(current.blocks[layer].attn,trace.queries[layer],teacher.k[layer],teacher.v[layer])-trace.history_heads[layer])/counts[indices,None,None,None]
                torch.cuda.synchronize()
                acquisition += time.perf_counter()-start
                start = time.perf_counter()
                p,record = fit_joint(latent[fitting],query[fitting],wanted[fitting],observed[fitting],counts[fitting],coordinates=frozen[layer],count_square_mean=mu)
                parameters.append(p)
                b[:,layer],a[:,layer] = install_layer(latent,p["weights"],p["query_center"],p["query_scale"],active[:,layer])
                torch.cuda.synchronize()
                solving += time.perf_counter()-start
                fits.append(dict(target=target,method=method,layer=layer,**record))
            costs.append(dict(target=target,method=method,query_acquisition_seconds=acquisition,shared_fit_seconds=solving,
                              fitting_scenes=len(fitting),fitting_uids=len(old if method=="B" else old|new),query_pairs_per_layer=len(fitting)*16))
            torch.save([{k:v.cpu() for k,v in p.items()} for p in parameters],out / f"translator_{method}_m{target}.pt")
        for batch_id,batch in enumerate(batches):
            raw = {}
            for name in (("fit64","held64") if method == "A" else ("fit16","held64")):
                indices = [i for i in batch if scenes[i].uid in old] if method == "A" and name == "fit64" else batch
                if not indices:
                    continue
                cache = batch_cache([states[i].cache for i in indices])
                teacher = batch_cache([scenes[i].teacher for i in indices])
                inputs = panel(indices,name)
                exact = score(current,teacher,*inputs)[0]
                z,trace,obs,corrections = read_input(current,cache,inputs,b[indices],a[indices],parameters,counts[indices],active[indices])
                raw[name] = dict(scene_indices=indices,exact=exact.cpu(),prediction=z.cpu())
                def save_errors(label,prediction,exact=exact,indices=indices,name=name):
                    e = prediction.double()-exact.double()
                    bias = e.mean(-1)
                    for j,i in enumerate(indices):
                        rows.append(dict(**metadata[i],method=label,panel=name,logit_mse=float(e[j].square().mean()),
                                         bias_squared=float(bias[j].square()),centered_mse=float((e[j]-bias[j]).square().mean())))
                save_errors(method,z)
                for layer,q in enumerate(trace.queries):
                    wanted_aggregate = history_read(current.blocks[layer].attn,q,teacher.k[layer],teacher.v[layer])-trace.history_heads[layer]
                    eps = corrections[layer].double()-wanted_aggregate.double()
                    # Unmasked FP64 prediction is the actual fitting objective, distinct from serving clearance.
                    p = parameters[layer]
                    xq = (q.double()-p["query_center"])/p["query_scale"]
                    xq = torch.cat((torch.ones_like(xq[...,:1]),xq),-1)
                    xo = (obs[layer].double()-p["read_center"])/p["read_scale"]
                    prediction = torch.einsum("shqj,sr,rhjd->shqd",xq,latent[indices],p["weights"])+torch.einsum("sqa,had->shqd",xo,p["read_weights"])
                    rate_error = prediction-(wanted_aggregate/counts[indices,None,None,None]).double()
                    for j,i in enumerate(indices):
                        local.append(dict(**metadata[i],method=method,panel=name,layer=layer,response_mse=float(eps[j].square().mean()),
                                          rate_mse=float(eps[j].square().mean()/counts[i].square())))
                        risks.append(dict(**metadata[i],method=method,panel=name,layer=layer,
                                          unmasked_data=float(rate_error[j].square().mean()*counts[i].square()/mu)))
                if method == "A" and name == "held64":
                    reuse = score(current,cache,*inputs)[0]
                    sb,sa = mapper.query_view(base[indices])
                    shared = score(current,cache,*inputs,response_delta=sb*counts[indices,None,None],
                                   response_query_delta=sa*counts[indices,None,None,None,None])[0]
                    for label,prediction in (("reuse",reuse),("shared15",shared)):
                        save_errors(label,prediction)
                        raw[name][label] = prediction.cpu()
            torch.save(raw,out / f"raw_{method}_m{target}_batch{batch_id:03d}.pt")
        if method == "A" and not canary:
            # Same frozen A, states, panels and endpoints as the existing result.
            old_results = pd.read_parquet(A / f"outputs_m{target}.parquet")
            old_results = old_results[(old_results.method == "native_input") & (old_results.panel == "held64")].merge(original[["scene","state_ordinal"]],on="scene",validate="many_to_one")
            new_results = pd.DataFrame(rows)
            pair = new_results[(new_results.method=="A")&(new_results.panel=="held64")].merge(old_results,on=["uid","state_ordinal"],suffixes=("_new","_old"),validate="one_to_one")
            assert len(pair) == len(old_results)
            torch.testing.assert_close(torch.tensor(pair.logit_mse_new.values),torch.tensor(pair.logit_mse_old.values),atol=2e-6,rtol=2e-3)
            write_json(out / f"reference_check_m{target}.json",dict(states=len(pair),max_absolute_mse_difference=float((pair.logit_mse_new-pair.logit_mse_old).abs().max())))
        print(json.dumps(dict(target=target,method=method,status="complete",scenes=len(scenes))),flush=True)
    pd.DataFrame(rows).to_parquet(out / f"outputs_m{target}.parquet",index=False)
    pd.DataFrame(local).to_parquet(out / f"responses_m{target}.parquet",index=False)
    pd.DataFrame(risks).to_parquet(out / f"unmasked_risk_m{target}.parquet",index=False)
    write_json(out / f"fits_m{target}.json",fits)
    write_json(out / f"cost_m{target}.json",costs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id",required=True)
    parser.add_argument("--cohort-path",required=True)
    parser.add_argument("--fit-users",type=int,default=64)
    parser.add_argument("--diagnostic-users",type=int,default=128)
    parser.add_argument("--batch-size",type=int,default=4)
    parser.add_argument("--canary",action="store_true")
    parser.add_argument("--estimate-seconds",type=int,required=True)
    main(parser.parse_args(),experiment)
