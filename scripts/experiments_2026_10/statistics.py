"""Paired user-cluster confidence intervals for the fixed one-edge comparison."""
from __future__ import annotations

import numpy as np


def weighted_auc(scores, labels, uid_index, user_weights):
    """Exact weighted AUC, half credit for ties; score order is shared by draws."""
    order=np.argsort(scores,kind="stable")
    ordered=np.asarray(scores)[order]
    labels=np.asarray(labels)[order]
    uid_index=np.asarray(uid_index)[order]
    starts=np.r_[0,np.flatnonzero(ordered[1:]!=ordered[:-1])+1]
    result=[]
    for offset in range(0,len(user_weights),8):
        weights=user_weights[offset:offset+8,uid_index].astype(np.float64)
        positive=np.add.reduceat(weights*labels,starts,axis=1)
        negative=np.add.reduceat(weights*(1-labels),starts,axis=1)
        below=np.cumsum(negative,axis=1)-negative*.5
        denominator=positive.sum(axis=1)*negative.sum(axis=1)
        with np.errstate(invalid="ignore",divide="ignore"):
            result.extend((positive*below).sum(axis=1)/denominator)
    return np.asarray(result)


def paired_user_bootstrap(groups,quality_uids,*,replicates,seed,progress=None):
    """Resample the complete eligible UID population, jointly for every method.

    Users with no feedback remain members of the sampling population. All of a
    sampled user's events share a multiplicity. Calibration weights stay fixed;
    these intervals do not quantify variation across correction training seeds.
    """
    base=groups["reuse"]
    keys=[(r["uid"],r["query_timestamp"],r["raw_item_id"],r["label"]) for r in base]
    population=np.asarray(sorted(quality_uids),dtype=np.int64)
    uids=np.asarray([r["uid"] for r in base],dtype=np.int64)
    uid_index=np.searchsorted(population,uids)
    assert np.array_equal(population[uid_index],uids)
    labels=np.asarray([r["label"] for r in base],dtype=np.float64)
    rng=np.random.default_rng(seed)
    weights=np.empty((replicates,len(population)),dtype=np.uint16)
    for draw in range(replicates):
        weights[draw]=np.bincount(rng.integers(len(population),size=len(population)),minlength=len(population))
    draws={}
    for tag,rows in groups.items():
        assert keys==[(r["uid"],r["query_timestamp"],r["raw_item_id"],r["label"]) for r in rows],tag
        draws[tag]=weighted_auc(np.asarray([r["score"] for r in rows]),labels,uid_index,weights)
        if progress is not None:progress(dict(status="paired_uid_bootstrap",method=tag,replicates=replicates))
    gap=draws["rebuild"]-draws["reuse"]
    valid=np.isfinite(gap)&(gap>0)
    def interval(values):
        finite=values[np.isfinite(values)]
        return np.percentile(finite,[2.5,97.5]).tolist() if len(finite) else None
    results={}
    for tag,auc in draws.items():
        gain=auc-draws["reuse"]
        # A denominator crossing zero makes a bounded percentile ratio
        # misleading. Keep absolute AUC/gain intervals and flag that case.
        recovery=interval(100*gain/gap) if valid.all() else None
        results[tag]=dict(auc_ci95=interval(auc),auc_gain_over_reuse_ci95=interval(gain),
            recovery_percent_ci95=recovery)
    return results,dict(method="paired nonparametric UID-cluster percentile bootstrap",replicates=replicates,
        seed=seed,eligible_users=len(population),users_with_feedback=len(set(uids.tolist())),
        reference_gap_ci95=interval(gap),nonpositive_or_invalid_reference_gap_replicates=int((~valid).sum()),
        scope="conditional on these fitted weights and this independent edge; not training-seed variability",
        recovery_rule="unclipped ratio; no bounded recovery CI if any bootstrap reference gap is nonpositive")


def write_tables(summary,output):
    """Export this edge as CSV, LaTeX and a self-contained evidence table."""
    import csv
    names=dict(reuse="Reuse",rebuild="Rebuild",layer="Layer",tail="Tail",deviation="Deviation",
        query="Query",query_only="Query-only",history_conditioned="History-conditioned")
    fields=["method","setting","added_flops_percent","auc","auc_ci95_low","auc_ci95_high",
        "auc_gain_over_reuse","recovery_percent","recovery_ci95_low","recovery_ci95_high"]
    exported=[]
    for row in summary["results"]:
        tag=row["method"]
        kind=next((key for key in sorted(names,key=len,reverse=True) if tag==key or tag.startswith(key+"_")),tag)
        setting=row["setting"]
        if "fit_users" in setting:label=f"{setting['fit_users']} fit users"
        elif "interval" in setting:label=f"layers {setting['interval'][0]+1}-{setting['interval'][1]+1}"
        elif "fraction" in setting:label=f"{100*setting['fraction']:.2f}% tokens"
        else:label="anchor"
        auc_ci=row["auc_ci95"] or [None,None]
        recovery_ci=row["recovery_percent_ci95"] or [None,None]
        exported.append(dict(zip(fields,[names.get(kind,kind),label,row["percent_of_once_rebuild"],row["auc"],
            *auc_ci,row["auc_gain_over_reuse"],row["recovery_percent"],*recovery_ci],strict=True)))
    with (output/"table.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(exported)
    md=[f"# {summary['model_scale']} {summary['edge']} baseline comparison","",
        f"Quality and cost: {summary['quality_users']:,} eligible users, {summary['quality_requests']:,} feedback events. "
        f"Users with feedback: {summary['confidence']['users_with_feedback']:,}. "
        "Independent adjacent edge; frozen backbone and correction seed 17.","",
        "| Method | Setting | Added FLOPs / Rebuild (%) | AUC | Gain over Reuse | Recovery (%) | UID 95% CI (%) |",
        "|---|---|---:|---:|---:|---:|---|" ]
    window=summary.get("evaluation_window")
    if window:
        note=f"Window: days [{window['start_day']}, {window['end_day_exclusive']}); {window.get('name','fixed horizon')}."
        if not window.get("complete_source_horizon",True):
            note+=" Incomplete source horizon: "+window["interpretation"]
        md[2:2]=[note,""]
    tex=[r"\begin{tabular}{llrrrr}",r"\toprule",
        r"Method & Setting & Added FLOPs (\%) & AUC & Recovery (\%) & UID 95\% CI \\",r"\midrule"]
    def number(value,decimals):return "--" if value is None else f"{value:.{decimals}f}"
    for row in exported:
        ci=("unbounded/undefined" if row["recovery_ci95_low"] is None else
            f"[{row['recovery_ci95_low']:.1f}, {row['recovery_ci95_high']:.1f}]")
        md.append(f"| {row['method']} | {row['setting']} | {row['added_flops_percent']:.2f} | {row['auc']:.6f} | "
            f"{row['auc_gain_over_reuse']:.6f} | {number(row['recovery_percent'],1)} | {ci} |")
        tex_setting=row["setting"].replace("%",r"\%")
        tex.append(f"{row['method']} & {tex_setting} & {row['added_flops_percent']:.2f} & "
            f"{row['auc']:.6f} & {number(row['recovery_percent'],1)} & {ci} "+chr(92)*2)
    tex.extend([r"\bottomrule",r"\end{tabular}"])
    prefix_only=all(not p["required_old_boundary_layers"] for p in summary["layer_boundary_state"]["selected_points"])
    boundary_note=("All selected Layer intervals start at layer zero and recompute from raw events; they need no additional persistent old hidden boundary cache. Temporary reconstruction workspace is separate."
        if prefix_only else "Non-prefix Layer points need their actual old input boundary; point-specific persistent bytes are in summary.json. Teacher hidden is not free.")
    md.extend(["","Recovery = 100 × (AUC(method) − AUC(Reuse)) / (AUC(Rebuild) − AUC(Reuse)); values are not clipped.","",
        "Complete added FLOPs include each method's calibration, selection, repair/mapping and query correction. "
        "The denominator is once-per-user/release Rebuild at the first history consumer. "
        "Common native reads/appends and old-cache initialization are reported separately in summary.json. "
        "All population events count toward costs, including users without feedback and events after the last scored request.","",
        boundary_note,"",
        "Intervals resample paired user clusters, conditional on these fitted weights. "
        "This independent-edge comparison alone does not establish cross-edge, cross-dataset or training-seed stability. "
        "Execution progress times are not controlled paper timing results."])
    (output/"report.md").write_text("\n".join(md)+"\n")
    (output/"table.tex").write_text("\n".join(tex)+"\n")
