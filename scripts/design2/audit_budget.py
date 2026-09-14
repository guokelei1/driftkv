"""CPU-only conditional permutation audit and explicit detector FLOPs."""

import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from design.report_native_flops import rebuild
from design2.report_detection import weights

ROOT=Path(__file__).resolve().parents[2]
INPUT=ROOT/"results/design2/analysis/detection_01"
OUTPUT=ROOT/"results/design2/analysis/budget_audit_01"
BUDGETS=np.array([.05,.1,.2,.3,.5])
ERRORS=np.array([.5,.1,1.])


def check_cost():
    layers,heads,p,rank,dim,queries=6,6,1281,33,32,16
    # Ordinary arithmetic: exact triangular solve, both normalizations, joint
    # source×query feature, squared norm/reduction, aggregate scaling.
    solve=layers*heads*p*p
    other=layers*(heads*(2*dim+rank*(dim+1)+(2*p-1)+1)+2*192)
    return dict(solve_per_query=solve,other_per_query=other,
        panel_arithmetic=queries*(solve+other)+layers,
        panel_solve_only=queries*solve,panel_sqrt_calls=queries*layers*heads,
        # Both head maxima are computed by score_layer, even though the caller
        # discards the rate maximum. Then layer maximum and panel maximum.
        panel_comparisons=queries*(2*layers*(heads-1)+layers-1)+queries-1,
        assumptions="counts/latent/native responses already available from frozen C; current implementation computes even masked blocks; casts/memory traffic not FLOPs")


def permutation_groups(frame, blocked=False):
    """Length factor <2^.25, fixed without error labels."""
    band=np.floor(4*np.log2(frame["count"].to_numpy())).astype(int)
    work=frame.assign(length_bin=band)
    if not blocked:
        groups=[g.index.to_numpy() for _,g in work.groupby(["target","length_bin"],sort=True)]
        return groups
    # Preserve each UID's score vector across matching scenario profiles;
    # this also conditions on state kind and number of snapshots.
    buckets=defaultdict(list)
    for (target,b),g in work.groupby(["target","length_bin"],sort=True):
        for _,u in g.groupby("uid",sort=True):
            u=u.sort_values(["kind","state_ordinal"])
            signature=tuple(u.kind.tolist())
            buckets[target,b,signature].append(u.index.to_numpy())
    return [np.stack(v) for v in buckets.values()]


def permute(scores,groups,rng):
    donor=np.arange(len(scores))
    for group in groups:
        if group.ndim==1:
            donor[group]=rng.permutation(group)
        else:
            donor[group.ravel()]=group[rng.permutation(len(group))].ravel()
    return scores[donor],donor


def selected(order,cost,budget):
    """Same frozen ranking-prefix rule as the first experiment."""
    cumulative=np.cumsum(cost[order])
    return order[:np.searchsorted(cumulative,budget,side="right")]


def curve(frame,score,budgets=BUDGETS,multiplicity=None):
    cost=frame.rebuild_flops.to_numpy()
    base=weights(frame)
    if multiplicity is None:
        multiplicity=np.ones(len(frame))
    w=base*multiplicity
    c=cost*multiplicity
    order=np.argsort(-score,kind="stable")
    y=frame.max_abs_error.to_numpy()[:,None]>=ERRORS[None]
    denominator=(w[:,None]*y).sum(0)
    rows=[]
    for budget in budgets:
        idx=selected(order,c,budget*c.sum())
        idx=idx[multiplicity[idx]>0]
        numerator=(w[idx,None]*y[idx]).sum(0)
        recall=np.divide(numerator,denominator,out=np.full(3,np.nan),where=denominator>0)
        rows.append(dict(budget=float(budget),actual_rebuild_fraction=float(c[idx].sum()/c.sum()),
            selected_states=len(idx),selected_uids=int(frame.iloc[idx].uid.nunique()),
            selected_serious_states=[int(y[idx,j].sum()) for j in range(3)],
            selected_serious_uids=[int(frame.iloc[idx[y[idx,j]]].uid.nunique()) for j in range(3)],
            recall=recall.tolist(),
            error_energy_fraction=float((w[idx]*frame.mse.to_numpy()[idx]).sum()/(w*frame.mse.to_numpy()).sum())))
    return rows


def summarize_permutations(records):
    arr=np.array([[row["recall"] for row in draw] for draw in records])
    result=[]
    for j,b in enumerate(BUDGETS):
        result.append(dict(budget=float(b),recall_mean=np.nanmean(arr[:,j],axis=0).tolist(),
            recall_p025=np.nanquantile(arr[:,j],.025,axis=0).tolist(),
            recall_p975=np.nanquantile(arr[:,j],.975,axis=0).tolist(),
            selected_states_mean=float(np.mean([v[j]["selected_states"] for v in records])),
            selected_states_range=np.quantile([v[j]["selected_states"] for v in records],[.025,.975]).tolist(),
            selected_uids_mean=float(np.mean([v[j]["selected_uids"] for v in records])),
            selected_uids_range=np.quantile([v[j]["selected_uids"] for v in records],[.025,.975]).tolist(),
            actual_rebuild_fraction_mean=float(np.mean([v[j]["actual_rebuild_fraction"] for v in records]))))
    return result,arr


def finite_metrics(value):
    """Empty severe strata have undefined recall, represented as JSON null."""
    if isinstance(value,dict):
        return {k:finite_metrics(v) for k,v in value.items()}
    if isinstance(value,list):
        return [finite_metrics(v) for v in value]
    return None if isinstance(value,float) and not np.isfinite(value) else value


def main():
    started=time.perf_counter()
    source=INPUT/"states.parquet"
    frame=pd.read_parquet(source)
    frame=frame[frame.role=="development"].sort_values(["uid","target","state_ordinal"]).reset_index(drop=True)
    protocol=json.loads((ROOT/"configs/design2/detection_01.json").read_text())
    assert set(frame.uid)==set(protocol["groups"]["development"])
    assert not frame.duplicated(["uid","target","state_ordinal"]).any()
    OUTPUT.mkdir(parents=True,exist_ok=False)
    setting=dict(status="frozen_before_new_permutation_results",input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        users=int(frame.uid.nunique()),states=len(frame),targets=[1,3,4,5],budgets=BUDGETS.tolist(),
        error_thresholds=ERRORS.tolist(),primary_budget=.05,primary_error=.5,
        length_group="target × floor(4 log2(actual N)); within-bin max/min<2^.25; N1024 stays separate",
        permutations=1000,seed=170910,
        profile_control="permute UID vectors within target/length_bin/exact ordered kind-profile; keep within-UID dependence",
        bootstrap="500 paired UID draws; original versus cost-only; recompute weighted budget; conditional permutation ranges are not UID confidence intervals",
        signal_rule="original5% primary recall exceeds both conditional-shuffle97.5percentiles AND paired original-minus-cost UID95% lower>0",
        cost_rule="all candidate states pay the complete16-query check before ranking; rebuild-only budget and total arithmetic lower bound shown separately",
        compute_scope="no model forward, no H recomputation/refit, no normalization/threshold search; existing opened development evidence",
        prospective_seconds=120,prospective_memory_mib=1024,
        resource_basis="2000 sorts of15299 states plus500 paired resamples, cached columns; CPU arrays only",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/"configs/design2/budget_audit_01.json").write_text(json.dumps(setting,indent=2)+"\n")
    (OUTPUT/"configuration.json").write_text(json.dumps(setting,indent=2)+"\n")
    cost=frame.rebuild_flops.to_numpy()
    score=frame.detection.to_numpy()
    original=curve(frame,score/cost)
    cheapest=curve(frame,-cost)
    # Verify that the new original curve reproduces retained evidence.
    prior=json.loads((INPUT/"summary.json").read_text())["scores"]["detection"]["budget"]
    np.testing.assert_allclose([r["recall"][0] for r in original],[r["severe_coverage"] for r in prior],atol=1e-12)
    np.testing.assert_equal([r["selected_states"] for r in original],[r["selected_states"] for r in prior])
    references={}; arrays={}
    for name,blocked in (("within_length_shuffle",False),("uid_profile_shuffle",True)):
        groups=permutation_groups(frame,blocked)
        rng=np.random.default_rng(setting["seed"]+int(blocked))
        records=[]; moved=[]; severe_moved=[]
        uids=frame.uid.to_numpy()
        serious=frame.max_abs_error.to_numpy()>=.5
        for _ in range(setting["permutations"]):
            shuffled,donor=permute(score,groups,rng)
            records.append(curve(frame,shuffled/cost))
            moved.append(float(np.mean(uids[donor]!=uids)))
            severe_moved.append(float(np.mean(uids[donor][serious]!=uids[serious])))
        summary,arr=summarize_permutations(records)
        references[name]=dict(curve=summary,mean_cross_uid_donor_fraction=float(np.mean(moved)),
            mean_serious_cross_uid_donor_fraction=float(np.mean(severe_moved)),
            scope="conditional randomization of this reused dataset; not fresh users, backbone seeds or a causal proof")
        arrays[name]=arr
        # Compact per-draw curves are small local evidence; no new model raw.
        np.savez_compressed(OUTPUT/f"{name}.npz",recall=arr,
            selected_uids=np.array([[r["selected_uids"] for r in draw] for draw in records]),
            selected_states=np.array([[r["selected_states"] for r in draw] for draw in records]))
        print(json.dumps(dict(stage=name,elapsed_seconds=time.perf_counter()-started)),flush=True)
    # UID rather than state bootstrap; sampled UID copies contribute costs too.
    uu,ii=np.unique(frame.uid.to_numpy(),return_inverse=True)
    rng=np.random.default_rng(170912)
    diff=[]
    for _ in range(500):
        m=np.bincount(rng.integers(len(uu),size=len(uu)),minlength=len(uu))[ii]
        a=curve(frame,score/cost,multiplicity=m)
        b=curve(frame,-cost,multiplicity=m)
        diff.append(np.array([r["recall"] for r in a])-np.array([r["recall"] for r in b]))
    diff=np.stack(diff)
    paired=dict(point=(np.array([r["recall"] for r in original])-np.array([r["recall"] for r in cheapest])).tolist(),
        p025=np.nanquantile(diff,.025,axis=0).tolist(),p975=np.nanquantile(diff,.975,axis=0).tolist(),
        valid_draws=np.isfinite(diff).sum(0).tolist())
    per_target={}
    for target,g in frame.groupby("target"):
        per_target[str(target)]=dict(original=curve(g,g.detection.to_numpy()/g.rebuild_flops.to_numpy()),
                                    cost_only=curve(g,-g.rebuild_flops.to_numpy()))
    flops=check_cost()
    old_cost=json.loads((INPUT/"cost.json").read_text())["arithmetic"]
    assert flops["solve_per_query"]==old_cost["triangular_solve_flops_per_query"]
    assert flops["other_per_query"]==old_cost["check_other_arithmetic_per_query"]
    exact_total=float(cost.sum())
    check_total=flops["panel_arithmetic"]*len(frame)
    prep_lower=old_cost["gram_main_contraction_flops"]+old_cost["cholesky_leading_flops"]
    flops.update(exact_all_panel_flops=exact_total,check_all_panel_flops=check_total,
        check_all_over_exact=check_total/exact_total,shared_geometry_preparation_leading_flops=prep_lower,
        shared_preparation_over_exact=prep_lower/exact_total,factor_storage_bytes=old_cost["factor_storage_bytes"],
        total_scope="checks complete in ordinary arithmetic, sqrt/comparisons separate. Preparation leading terms only; C preparation/backfill/native query acquisition/summary overhead not assumed zero. Total is a lower bound, not a complete serving ledger.")
    lengths=[]
    for n in (4,16,32,64,128,256,512,1024):
        ops=rebuild(n); exact=ops["gemm"]+ops["scalar"]
        lengths.append(dict(length=n,exact_arithmetic=exact,solve_only_over_exact=flops["panel_solve_only"]/exact,
            check_over_exact=flops["panel_arithmetic"]/exact,exact_special_calls={k:v for k,v in ops.items() if k not in ("gemm","scalar")}))
    flops["length_table"]=lengths
    # Complete checks cover all ranked states, not only the selected rebuilds.
    total_curves=[]
    for name,rows in (("original",original),("cost_only",cheapest)):
        overhead=0 if name=="cost_only" else (check_total+prep_lower)/exact_total
        for row in rows:
            total_curves.append(dict(method=name,**row,total_arithmetic_lower_bound_fraction=row["actual_rebuild_fraction"]+overhead,
                check_fraction=0 if name=="cost_only" else check_total/exact_total,
                shared_preparation_lower_bound_fraction=0 if name=="cost_only" else prep_lower/exact_total))
    fixed_total={}
    for budget in BUDGETS:
        usable=budget-(check_total+prep_lower)/exact_total
        fixed_total[str(budget)]=dict(geometry_check_affordable=bool(usable>=0),
            geometry=None if usable<0 else curve(frame,score/cost,budgets=np.array([usable]))[0],
            cost_only=curve(frame,-cost,budgets=np.array([budget]))[0])
    signal=bool(original[0]["recall"][0]>references["within_length_shuffle"]["curve"][0]["recall_p975"][0]
        and original[0]["recall"][0]>references["uid_profile_shuffle"]["curve"][0]["recall_p975"][0]
        and paired["p025"][0][0]>0)
    summary=dict(status="complete",users=len(uu),states=len(frame),queries=len(frame)*16,
        errors=ERRORS.tolist(),budgets=BUDGETS.tolist(),original=original,cost_only=cheapest,permutations=references,
        paired_original_minus_cost_only=paired,per_target=per_target,cost=flops,total_curves=total_curves,
        fixed_total_budget=fixed_total,incremental_signal_rule_pass=signal,
        elapsed_seconds=time.perf_counter()-started,model_forward_calls=0)
    (OUTPUT/"summary.json").write_text(json.dumps(finite_metrics(summary),indent=2,allow_nan=False)+"\n")
    pd.DataFrame(total_curves).to_csv(OUTPUT/"total_curves.csv",index=False)
    print(json.dumps(dict(status="complete",elapsed_seconds=summary["elapsed_seconds"],incremental_signal_rule_pass=signal,
        original5=original[0],cost5=cheapest[0],shuffle5={k:v["curve"][0] for k,v in references.items()},
        check_fraction=flops["check_all_over_exact"])),flush=True)


if __name__=="__main__":
    main()
