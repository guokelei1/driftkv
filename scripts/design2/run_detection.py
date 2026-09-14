"""Recover frozen-C geometry or evaluate a disjoint UID chunk."""

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from design.data import ROOT, DAY, frozen_model, histories
from design.diagnose_query_holdout import batch_cache
from design.run import cache_at_many, initialize_calibration, replay_calibration, timed, write_json
from design2.common import CROOT, COHORT, FrozenC, TARGETS, batches, make_panel, original_args, source_scenes
from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.design2.geometry import factorize, features, gram_sum, score_layer
from insight_two.common import CUTOVER_DAYS


def metadata(scenes, p, target, role, cutover):
    records = []
    for i,(s,source) in enumerate(zip(scenes,p["sources"],strict=True)):
        counts = source.count.sum(1)
        versions = source.producer
        mass = counts/counts.sum()
        stamps = list(s.state.writer.events)
        records.append(dict(uid=s.uid,role=role,target=target,state_ordinal=s.ordinal,kind=s.kind,
            decision_timestamp=cutover,last_timestamp=int(stamps[-1][3]),count=int(p["counts"][i]),
            release_age=int(source.release_age),source_norm=float(p["source_norm"][i]),
            version_age=float((mass*(target-versions)).sum()),
            producer_count=int((mass>0).sum()),active_layers=int(p["active"][i].sum()),
            **{f"producer_{v}_fraction":float(mass[versions==v].sum()) for v in range(target+1)}))
    return records


def recover(current, history, scenes, adapter, cutover, out, target, ledger):
    p = timed(lambda:adapter.prepare(scenes),ledger,"source_and_view")
    meta = metadata(scenes,p,target,"fitting_check",cutover)
    old = pd.read_parquet(CROOT/f"scenes_m{target}.parquet")
    old = old[old.group!="diagnostic"]
    match = pd.DataFrame(meta).merge(old,on=["uid","state_ordinal"],suffixes=("_new","_old"),validate="one_to_one")
    assert len(match)==len(old)==len(scenes)
    assert (match.count_new==match.count_old).all() and (match.release_age_new==match.release_age_old).all()
    for v in range(target+1):
        np.testing.assert_allclose(match[f"producer_{v}_fraction_new"],match[f"producer_{v}_fraction_old"],atol=2e-7)
    fits = json.loads((CROOT/f"fits_m{target}.json").read_text())
    mu = next(f["count_square_mean"] for f in fits if f["method"]=="C")
    query, observed, predictions = [[] for _ in range(6)],[[] for _ in range(6)],[]
    order = []
    for indices in batches(scenes):
        cache = batch_cache([scenes[i].state.cache for i in indices])
        panel = make_panel(history,scenes,indices,cutover,cache.k.device,fitting=True)
        z,trace,obs,_ = timed(lambda:adapter.read(current,cache,panel,p,indices),ledger,"fitting_input_read")
        order.extend(indices)
        predictions.append(z.cpu())
        for l in range(6):
            query[l].append(trace.queries[l])
            observed[l].append(obs[l])
    # Compare frozen predictions, never read old Exact as the new reference.
    old_predictions = {}
    old_meta = pd.read_parquet(CROOT/f"scenes_m{target}.parquet").set_index("scene")
    for path in sorted(CROOT.glob(f"raw_C_m{target}_batch*.pt")):
        raw = torch.load(path,map_location="cpu",weights_only=False)["fit16"]
        for idx,z in zip(raw["scene_indices"],raw["prediction"],strict=True):
            row = old_meta.loc[idx]
            if row["group"]!="diagnostic":
                old_predictions[int(row.uid),int(row.state_ordinal)] = z
    actual = torch.cat(predictions)
    expected = torch.stack([old_predictions[scenes[i].uid,scenes[i].ordinal] for i in order])
    torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-5)
    factors,checks = [],[]
    for l in range(6):
        q,o = torch.cat(query[l]),torch.cat(observed[l])
        h,n = p["latent"][order],p["counts"][order]
        gram = timed(lambda:gram_sum(h,q,o,n,adapter.parameters[l],mu),ledger,"gram")
        chol,residual = timed(lambda:factorize(gram,len(scenes)),ledger,"factor")
        # Large-shape explicit subset checks source-major order and joint block contractions.
        f = features(h[:2],q[:2],o[:2],adapter.parameters[l])
        explicit = torch.einsum("shqa,shqb,s->hab",f,f,n[:2].double().square()/mu)/16
        block = gram_sum(h[:2],q[:2],o[:2],n[:2],adapter.parameters[l],mu)
        torch.testing.assert_close(block,explicit,atol=1e-7,rtol=1e-9)
        factors.append(chol.cpu())
        checks.append(dict(layer=l,cholesky_relative_residual=residual))
        del gram,chol,q,o
    torch.save(factors,out/f"cholesky_m{target}.pt")
    pd.DataFrame(meta).to_parquet(out/f"fitting_scenes_m{target}.parquet",index=False)
    write_json(out/f"checks_m{target}.json",dict(scenes=len(scenes),mu=mu,
        max_frozen_logit_difference=float((actual-expected).abs().max()),layers=checks,
        target_teacher_responses_acquired=0,C_refitted=False))


def evaluate(current,history,scenes,adapter,cutover,out,target,role,geometry,ledger):
    p = timed(lambda:adapter.prepare(scenes),ledger,"source_and_view")
    meta = metadata(scenes,p,target,role,cutover)
    factors = torch.load(geometry/f"cholesky_m{target}.pt",map_location=next(current.parameters()).device,weights_only=True)
    uids = list(dict.fromkeys(s.uid for s in scenes))
    exact_values = timed(lambda:cache_at_many(current,history,[(u,cutover) for u in uids],16),ledger,"evaluation_exact_prefix")
    exacts = dict(zip(uids,exact_values,strict=True))
    records,query_records = [],[]
    for indices in batches(scenes):
        subset = [scenes[i] for i in indices]
        cache = batch_cache([s.state.cache for s in subset])
        panel = make_panel(history,scenes,indices,cutover,cache.k.device)
        before = [s.state.writer.next_ordinal for s in subset]
        z,trace,obs,_ = timed(lambda:adapter.read(current,cache,panel,p,indices),ledger,"adapted_read")
        scores = []
        for l in range(6):
            def check():
                return score_layer(p["latent"][indices],trace.queries[l],obs[l],p["counts"][indices],
                                   p["active"][indices,l],adapter.parameters[l],factors[l])[0]
            scores.append(timed(check,ledger,"geometry_check"))
        detection = torch.stack(scores).amax(0)
        # Teacher is deliberately accessed only after the deployable statistic.
        teacher = batch_cache([exacts[s.uid][0] for s in subset])
        for s in subset:
            times = exacts[s.uid][1]
            actual_times = np.array([e[3] for e in s.state.writer.events])
            assert np.array_equal(times,actual_times) and int(times[-1])<cutover
        exact = timed(lambda:score(current,teacher,*panel)[0],ledger,"evaluation_exact_read")
        reuse = timed(lambda:score(current,cache,*panel)[0],ledger,"evaluation_reuse_read")
        assert before == [s.state.writer.next_ordinal for s in subset]
        eps = z.double()-exact.double()
        for j,i in enumerate(indices):
            values = eps[j]
            records.append(dict(**meta[i],detection=float(detection[j].max()),max_abs_error=float(values.abs().max()),
                mse=float(values.square().mean()),mean_abs_error=float(values.abs().mean()),
                reuse_max_abs_error=float((reuse[j]-exact[j]).abs().max())))
            for k in range(z.shape[1]):
                query_records.append(dict(uid=scenes[i].uid,target=target,state_ordinal=scenes[i].ordinal,
                    item=int(panel[0][j,k]),query_timestamp=cutover,query_index=k,
                    detection=float(detection[j,k]),adapt=float(z[j,k]),exact=float(exact[j,k]),reuse=float(reuse[j,k])))
    pd.DataFrame(records).to_parquet(out/f"states_m{target}.parquet",index=False)
    pd.DataFrame(query_records).to_parquet(out/f"queries_m{target}.parquet",index=False)
    del exacts,exact_values,factors


@torch.no_grad()
def main(cli):
    protocol_path = ROOT/"configs/design2/detection_01.json"
    protocol = json.loads(protocol_path.read_text())
    out = ROOT/"results/design2"/cli.run_id
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(17)
    device = torch.device(cli.device)
    # Shared timed() synchronizes the current CUDA device.
    torch.cuda.set_device(device)
    if cli.mode=="recover":
        uids = protocol["groups"]["fitting_check"]
        lifetime = set(protocol["fitting_lifetime_uids"])
    else:
        uids = protocol["groups"][cli.role][cli.start:cli.start+cli.users]
        lifetime = set(protocol["lifetime_uids"][cli.role])
    assert uids
    files = [Path(__file__),ROOT/"scripts/design2/common.py",ROOT/"src/hstu_kvcache/design2/geometry.py"]
    write_json(out/"configuration.json",dict(vars(cli),uids=uids,protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        confirmation_read=False,C_refitted=False))
    start = time.perf_counter()
    ledger = defaultdict(float)
    try:
        history = timed(lambda:histories(uids,CUTOVER_DAYS[-1]+1),ledger,"history_load")
        previous = frozen_model(0,device)
        states,early = initialize_calibration(previous,history,uids,CUTOVER_DAYS[0]*DAY,original_args(),ledger)
        for target in range(1,6):
            current = frozen_model(target,device)
            cutover = CUTOVER_DAYS[target-1]*DAY
            if target in TARGETS:
                adapter = FrozenC(target,device)
                scenes = source_scenes(current,previous,history,uids,states,early,cutover,target,lifetime,ledger)
                if cli.mode=="recover":
                    recover(current,history,scenes,adapter,cutover,out,target,ledger)
                else:
                    evaluate(current,history,scenes,adapter,cutover,out,target,cli.role,ROOT/"results/design2"/cli.geometry,ledger)
                print(json.dumps(dict(target=target,users=len(uids),scenes=len(scenes),elapsed=time.perf_counter()-start)),flush=True)
                del scenes,adapter
            for state in states.values():
                state.release(target,None)
            if target<5:
                early = replay_calibration(current,history,states,uids,cutover,CUTOVER_DAYS[target]*DAY,ledger,128,16)
            previous=current
        write_json(out/"summary.json",dict(status="complete",users=len(uids),elapsed_seconds=time.perf_counter()-start,
            ledger_seconds=dict(ledger),peak_gpu_mib=torch.cuda.max_memory_allocated(device)/2**20))
    except Exception as exc:
        write_json(out/"summary.json",dict(status="failed",error=repr(exc),elapsed_seconds=time.perf_counter()-start,ledger_seconds=dict(ledger)))
        raise


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id",required=True)
    p.add_argument("--mode",choices=["recover","evaluate"],required=True)
    p.add_argument("--role",default="residual_calibration")
    p.add_argument("--start",type=int,default=0)
    p.add_argument("--users",type=int,default=16)
    p.add_argument("--geometry",default="geometry_01")
    p.add_argument("--device",default="cuda:0")
    main(p.parse_args())
