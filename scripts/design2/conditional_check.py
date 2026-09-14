"""Prepare exact inverse contractions and replay the unchanged detection panel."""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from design.data import ROOT, DAY, frozen_model, histories
from design.diagnose_query_holdout import batch_cache
from design.run import initialize_calibration, replay_calibration, timed, write_json
from design2.common import FrozenC, TARGETS, batches, make_panel, original_args, source_scenes
from design2.run_detection import metadata
from design2.scale_calibration import assign_groups
from design2.tiered_check import old_run, sha
from hstu_kvcache.design2.conditional import prepare_inverse, condition_state, evaluate_state, arithmetic
from hstu_kvcache.design2.geometry import score_layer
from insight_two.common import CUTOVER_DAYS

OUT = ROOT/"results/design2/conditional225_01"
CONFIG = ROOT/"configs/design2/conditional225_01.json"
SOURCE = ROOT/"src/hstu_kvcache/design2/conditional.py"


def prepare():
    torch.set_num_threads(4)
    out=OUT/"inverse"
    out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter(); checks=[]; hashes={}
    for target in TARGETS:
        source=ROOT/f"results/design2/geometry_01/cholesky_m{target}.pt"
        layers=torch.load(source,map_location="cpu",weights_only=True)
        packed,rows=prepare_inverse(layers)
        torch.save(packed,out/f"blocks_m{target}.pt")
        checks.extend(dict(target=target,**r) for r in rows)
        hashes[str(target)]=sha(source)
        print(json.dumps(dict(target=target,seconds=time.perf_counter()-started)),flush=True)
    p=1281; matrices=144
    write_json(out/"summary.json",dict(status="complete",checks=checks,elapsed_seconds=time.perf_counter()-started,
        triangular_inverse_flops=matrices*p**3,inverse_gram_flops=matrices*2*p**3,
        residual_gemm_flops=matrices*2*p**3,scalar_allowance_flops=matrices*64*p*p,
        total_arithmetic_charge=matrices*(5*p**3+64*p*p),
        storage_bytes=sum(p.stat().st_size for p in out.glob("*.pt")),
        configuration_sha256=sha(CONFIG),conditional_source_sha256=sha(SOURCE),geometry_sha256=hashes,
        candidate_costs={str(q):arithmetic(q) for q in (1,2,4,16)}))


def evaluate(current,history,scenes,adapter,cutover,out,target,cli,ledger,fitted):
    p=timed(lambda:adapter.prepare(scenes),ledger,"source_and_view")
    meta=assign_groups(pd.DataFrame(metadata(scenes,p,target,cli.role,cutover)),fitted["groups"])
    params=fitted["final_fit"]; tau=fitted["thresholds"]["calibrated"]["0.8"]
    meta["b"]=meta.group.map(lambda g:params[g]["b"])
    meta["a"]=meta.group.map(lambda g:params[g]["a"])
    b,a=meta.b.to_numpy(),meta.a.to_numpy()
    original=ROOT/"results/design2"/old_run(cli.role,cli.start)
    ref=pd.read_parquet(original/f"states_m{target}.parquet").set_index(["uid","state_ordinal"])
    ref=ref.loc[list(zip(meta.uid,meta.state_ordinal))].reset_index()
    for key in ("uid","state_ordinal","target","kind","count","active_layers"):
        np.testing.assert_array_equal(meta[key],ref[key])
    queries=pd.read_parquet(original/f"queries_m{target}.parquet",
        columns=["uid","state_ordinal","query_index","item","adapt","detection"]).set_index(["uid","state_ordinal","query_index"]).sort_index()
    device=next(current.parameters()).device
    factors=torch.load(ROOT/f"results/design2/geometry_01/cholesky_m{target}.pt",map_location=device,weights_only=True)
    packs=torch.load(OUT/f"inverse/blocks_m{target}.pt",map_location=device,weights_only=True)
    direct=(b>tau+1e-10)|((a==0)&(np.abs(b-tau)>1e-10))
    decision=b<=tau; reference=b+a*ref.detection.to_numpy()<=tau
    stages=np.where(direct,"formula","pending").astype(object)
    point_u=np.full(len(meta),np.nan); lower_u=point_u.copy(); upper_u=point_u.copy(); final_u=point_u.copy()
    invalid=np.zeros(len(meta),dtype=bool); threshold_fallback=invalid.copy()
    max_logit_delta=0.; max_score_rel=0.; max_query_rel=0.; max_bound_relative_width=0.
    query_checks=0; query_rows=[]; read_states=0
    for indices in batches(scenes):
        selected=indices if cli.canary else [i for i in indices if not direct[i]]
        if not selected:
            continue
        read_states+=len(indices)
        cache=batch_cache([scenes[i].state.cache for i in indices])
        panel=make_panel(history,scenes,indices,cutover,device)
        z,trace,obs,_=timed(lambda:adapter.read(current,cache,panel,p,indices),ledger,"frozen_C_input_read")
        expected_queries=[]
        for j,i in enumerate(indices):
            row=queries.loc[(scenes[i].uid,scenes[i].ordinal)].sort_index()
            np.testing.assert_array_equal(panel[0][j].cpu().numpy(),row.item.to_numpy())
            np.testing.assert_allclose(z[j].cpu().numpy(),row.adapt.to_numpy(),atol=2e-5,rtol=2e-5)
            max_logit_delta=max(max_logit_delta,float(np.max(np.abs(z[j].cpu().numpy()-row.adapt.to_numpy()))))
            expected_queries.append(row.detection.to_numpy())
        local=[indices.index(i) for i in selected]
        points=[]; lows=[]; highs=[]; bad=[]
        for layer in range(6):
            state=timed(lambda:condition_state(p["latent"][selected],packs[layer]),ledger,"conditional_state_matrix")
            point,lo,hi,valid=timed(lambda:evaluate_state(state,trace.queries[layer][local],obs[layer][local],
                adapter.parameters[layer],packs[layer]),ledger,"conditional_query_and_guard")
            active=p["active"][selected,layer].bool()[:,None,None]
            count=p["counts"][selected].double()[:,None,None]
            # Nonpositive points have NaN/inf bounds and force fallback when active.
            points.append(torch.where(active,point.sqrt()*count,0).amax(1))
            lows.append(torch.where(active,lo.sqrt()*count,0).amax(1))
            highs.append(torch.where(active,hi.sqrt()*count,0).amax(1))
            bad.append(((~valid)&active).any((1,2)))
            del state
        pq=torch.stack(points).amax(0).cpu().numpy()
        lq=torch.stack(lows).amax(0).cpu().numpy(); hq=torch.stack(highs).amax(0).cpu().numpy()
        nonfinite=torch.stack(bad).any(0).cpu().numpy()
        oldq=np.array([expected_queries[j] for j in local])
        good=~nonfinite
        if good.any():
            np.testing.assert_allclose(pq[good],oldq[good],rtol=1e-8,atol=1e-8)
            assert np.all(lq[good]<=oldq[good]) and np.all(oldq[good]<=hq[good])
            max_query_rel=max(max_query_rel,float(np.max(np.abs(pq[good]-oldq[good])/np.maximum(1,oldq[good]))))
            max_bound_relative_width=max(max_bound_relative_width,float(np.max((hq[good]-lq[good])/np.maximum(1,oldq[good]))))
            query_checks+=int(good.sum())*16
        ps,ls,hs=pq.max(-1),lq.max(-1),hq.max(-1)
        point_u[selected],lower_u[selected],upper_u[selected]=ps,ls,hs
        invalid[selected]=nonfinite
        accept=(b[selected]+a[selected]*hs<=tau-1e-10)&(~nonfinite)
        reject=(b[selected]+a[selected]*ls>tau+1e-10)&(~nonfinite)
        unresolved=[]
        for j,i in enumerate(selected):
            if direct[i]:
                continue
            if accept[j] or reject[j]:
                stages[i]="conditional"
                decision[i]=accept[j]
                final_u[i]=ps[j]
            else:
                stages[i]="accurate_fallback"
                unresolved.append(i)
                threshold_fallback[i]=not nonfinite[j]
        verification=selected if cli.canary else unresolved
        if verification:
            loc=[indices.index(i) for i in verification]
            accurate=[]
            for layer in range(6):
                accurate.append(timed(lambda:score_layer(p["latent"][verification],trace.queries[layer][loc],obs[layer][loc],
                    p["counts"][verification],p["active"][verification,layer],adapter.parameters[layer],factors[layer])[0],ledger,"accurate_and_canary_check"))
            aq=torch.stack(accurate).amax(0).cpu().numpy(); actual=aq.max(-1)
            np.testing.assert_allclose(actual,ref.detection.to_numpy()[verification],rtol=1e-8,atol=1e-8)
            for j,i in enumerate(verification):
                sj=selected.index(i)
                if not invalid[i]:
                    np.testing.assert_allclose(pq[sj],aq[j],rtol=1e-8,atol=1e-8)
                    assert np.all(lq[sj]<=aq[j]) and np.all(aq[j]<=hq[sj])
                if stages[i]=="accurate_fallback":
                    final_u[i]=actual[j]
                    decision[i]=b[i]+a[i]*actual[j]<=tau
                max_score_rel=max(max_score_rel,float(abs(actual[j]-ref.detection.iloc[i])/max(1,ref.detection.iloc[i])))
        for j,i in enumerate(selected):
            for qidx in range(16):
                query_rows.append(dict(uid=scenes[i].uid,target=target,state_ordinal=scenes[i].ordinal,query_index=qidx,
                    point_u=float(pq[j,qidx]),u_lower=float(lq[j,qidx]),u_upper=float(hq[j,qidx]),
                    reference_u=float(oldq[j,qidx]),numerically_valid=not bool(nonfinite[j])))
    assert not np.any(stages=="pending") and np.array_equal(decision,reference)
    meta=meta.assign(stage=stages,accept=decision,reference_accept=reference,point_u=point_u,final_u=final_u,
        u_lower=lower_u,u_upper=upper_u,reference_u=ref.detection.to_numpy(),numerically_invalid=invalid,
        threshold_fallback=threshold_fallback)
    meta.to_parquet(out/f"states_m{target}.parquet",index=False)
    pd.DataFrame(query_rows).to_parquet(out/f"queries_m{target}.parquet",index=False)
    fallback=int((stages=="accurate_fallback").sum())
    return dict(target=target,states=len(meta),stage_counts=meta.stage.value_counts().to_dict(),
        mismatches=int((decision!=reference).sum()),numerical_fallback_states=int((invalid&~direct).sum()),
        threshold_fallback_states=int(threshold_fallback.sum()),max_adapt_logit_delta=max_logit_delta,
        max_original_accurate_relative_delta=max_score_rel,max_conditional_query_relative_delta=max_query_rel,
        max_numerical_interval_relative_width=max_bound_relative_width,query_score_comparisons=query_checks,
        input_read_states=read_states,canary_extra_conditioned_states=int(direct.sum()) if cli.canary else 0,
        canary_extra_accurate_states=len(meta)-fallback if cli.canary else 0)


@torch.no_grad()
def replay(cli):
    out=OUT/"chunks"/f"{cli.role}_{cli.start:04d}"
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False; torch.manual_seed(17)
    device=torch.device(cli.device); torch.cuda.set_device(device)
    protocol=json.loads((ROOT/"configs/design2/detection_01.json").read_text())
    fitted=json.loads((ROOT/"configs/design2/scale_calibration_01_fitted.json").read_text())
    uids=protocol["groups"][cli.role][cli.start:cli.start+cli.users]
    lifetime=set(protocol["lifetime_uids"][cli.role])
    write_json(out/"configuration.json",dict(vars(cli),uids=uids,threshold=fitted["thresholds"]["calibrated"]["0.8"],
        configuration_sha256=sha(CONFIG),fitted_sha256=sha(ROOT/"configs/design2/scale_calibration_01_fitted.json"),
        source_sha256=sha(Path(__file__)),conditional_source_sha256=sha(SOURCE)))
    started=time.perf_counter(); ledger=defaultdict(float); results=[]
    history=timed(lambda:histories(uids,CUTOVER_DAYS[-1]+1),ledger,"history_load")
    previous=frozen_model(0,device)
    states,early=initialize_calibration(previous,history,uids,CUTOVER_DAYS[0]*DAY,original_args(),ledger)
    for target in range(1,6):
        current=frozen_model(target,device); cutover=CUTOVER_DAYS[target-1]*DAY
        if target in TARGETS:
            adapter=FrozenC(target,device)
            scenes=source_scenes(current,previous,history,uids,states,early,cutover,target,lifetime,ledger)
            results.append(evaluate(current,history,scenes,adapter,cutover,out,target,cli,ledger,fitted))
            print(json.dumps(dict(target=target,seconds=time.perf_counter()-started,result=results[-1])),flush=True)
            del scenes,adapter
        for state in states.values(): state.release(target,None)
        if target<5:
            early=replay_calibration(current,history,states,uids,cutover,CUTOVER_DAYS[target]*DAY,ledger,128,16)
        previous=current
    write_json(out/"summary.json",dict(status="complete",users=len(uids),results=results,
        elapsed_seconds=time.perf_counter()-started,ledger_seconds=dict(ledger),
        peak_gpu_mib=torch.cuda.max_memory_allocated(device)/2**20,new_teacher_outputs=0,C_refitted=False))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=["prepare","replay"])
    p.add_argument("--role",default="residual_calibration")
    p.add_argument("--start",type=int,default=0)
    p.add_argument("--users",type=int,default=16)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--canary",action="store_true")
    cli=p.parse_args()
    prepare() if cli.mode=="prepare" else replay(cli)
