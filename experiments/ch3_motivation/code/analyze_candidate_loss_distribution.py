#!/usr/bin/env python3
"""Descriptive per-feedback loss gains of one frozen static KV map; CPU only."""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]
import argparse
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
import numpy as np
from experiments.ch3_motivation.code.analyze_candidate_robustness import load_scores
from experiments.ch3_motivation.code.analyze_real_quality import sha, stable_log_loss

SOURCE=ROOT/'experiments/ch3_motivation/runs/candidate_robustness/attempt_1'
THRESHOLD=.001
QUANTILES=[0,.001,.01,.05,.1,.25,.5,.75,.9,.95,.99,.999,1]


def describe(gain,uids):
    unique,index,counts=np.unique(uids,return_inverse=True,return_counts=True)
    masks={'improved_above_threshold':gain>THRESHOLD,'near_zero':np.abs(gain)<=THRESHOLD,'harmed_below_negative_threshold':gain<-THRESHOLD}
    categories={}
    for name,mask in masks.items():
        per_uid=np.bincount(index,weights=mask,minlength=len(unique))/counts
        categories[name]=dict(events=int(mask.sum()),event_fraction=float(mask.mean()),equal_feedback_uid_fraction=float(per_uid.mean()))
    sums=np.bincount(index,weights=gain);squares=np.bincount(index,weights=gain*gain)
    std=np.sqrt(np.maximum(squares/counts-(sums/counts)**2,0))
    lower=np.full(len(unique),np.inf);upper=np.full(len(unique),-np.inf)
    np.minimum.at(lower,index,gain);np.maximum.at(upper,index,gain)
    multi=counts>=2;both=(lower<-THRESHOLD)&(upper>THRESHOLD)
    return dict(events=len(gain),feedback_users=len(unique),mean_delta_log_loss=float(gain.mean()),
        equal_feedback_uid_mean_delta_log_loss=float(np.mean(sums/counts)),
        quantiles={str(q):float(v) for q,v in zip(QUANTILES,np.quantile(gain,QUANTILES),strict=True)},
        categories=categories,within_uid=dict(multiple_event_users=int(multi.sum()),
            users_with_improved_and_harmed_events=int(both.sum()),
            fraction_of_multiple_event_users_with_both=float(both.sum()/multi.sum()) if multi.any() else None,
            median_gain_range_multiple_event_users=float(np.median((upper-lower)[multi])) if multi.any() else None,
            p90_gain_range_multiple_event_users=float(np.quantile((upper-lower)[multi],.9)) if multi.any() else None,
            median_gain_std_multiple_event_users=float(np.median(std[multi])) if multi.any() else None,
            mechanism_support_permitted=bool(multi.sum()>=512)),
        descriptive_feedback_support_at_least512=bool(len(unique)>=512))


def timestamp_coverage(base):
    groups={}
    for r in base:groups.setdefault((r['uid'],r['query_timestamp']),[]).append(r)
    multiple=[rows for rows in groups.values() if len(rows)>=2]
    mixed=[rows for rows in multiple if {r['label'] for r in rows}=={0,1}]
    def count(rows):
        users={r['uid'] for g in rows for r in g}
        return dict(groups=len(rows),events=sum(map(len,rows)),distinct_users=len(users),
            mechanism_support_permitted=len(users)>=512)
    return dict(at_least_two_candidates=count(multiple),at_least_two_candidates_with_both_labels=count(mixed))


def positive_context_control(base,gain,delta_score):
    contexts={}
    for i,row in enumerate(base):
        if row['label']==1:contexts.setdefault((row['uid'],row['query_timestamp']),[]).append(i)
    eligible=[(key,ix) for key,ix in contexts.items() if len(ix)>=2]
    mask=np.zeros(len(base),dtype=bool);loss_ranges=[];score_ranges=[];crossed=0;crossed_users=set()
    for (uid,timestamp),ix in eligible:
        mask[ix]=True;lo,hi=float(gain[ix].min()),float(gain[ix].max())
        loss_ranges.append(hi-lo);score_ranges.append(float(np.ptp(delta_score[ix])))
        if lo<-THRESHOLD and hi>THRESHOLD:crossed+=1;crossed_users.add(uid)
    users={key[0] for key,ix in eligible}
    summary=dict(rule='Keep positive-label events in each fixed(uid,query_timestamp) context having at least2 positive candidates; no selection by delta loss or scores.',
        contexts=len(eligible),events=int(mask.sum()),feedback_users=len(users),mechanism_support_permitted=len(users)>=512,
        median_within_context_delta_log_loss_range=float(np.median(loss_ranges)),
        p90_within_context_delta_log_loss_range=float(np.quantile(loss_ranges,.9)),
        median_within_context_delta_score_range=float(np.median(score_ranges)),
        p90_within_context_delta_score_range=float(np.quantile(score_ranges,.9)),
        contexts_with_loss_min_below_minus001_and_max_above001=crossed,
        fraction_contexts_crossing_both_boundaries=crossed/len(eligible),
        users_with_at_least_one_boundary_crossing_context=len(crossed_users))
    return mask,summary


def run(output):
    output.mkdir(parents=True,exist_ok=True);population=SOURCE/'population.json';data={};results={}
    for edge in (2,3):
        folder=SOURCE/f'quality/edge_{edge}'
        config,_,groups=load_scores(folder,population);base=groups['reuse']
        uids=np.asarray([r['uid'] for r in base],dtype=np.uint64)
        labels=np.asarray([r['label'] for r in base],dtype=np.int8)
        losses={name:stable_log_loss(np.asarray([r['score'] for r in groups[name]],dtype=float),labels)
            for name in ('reuse','linear256_rows','rebuild')}
        gain=losses['reuse']-losses['linear256_rows']
        if not np.isfinite(gain).all():raise ValueError('nonfinite per-event loss gain')
        delta_score=np.asarray([r['score'] for r in groups['linear256_rows']])-np.asarray([r['score'] for r in base])
        control,context_summary=positive_context_control(base,gain,delta_score)
        path=output/f'events_edge{edge}.npz'
        np.savez_compressed(path,uid=uids,label=labels,delta_log_loss=gain,
            map_minus_reuse_logit=delta_score,positive_multi_candidate_context=control,
            query_timestamp=np.asarray([r['query_timestamp'] for r in base],dtype=np.int64),
            item_idx=np.asarray([r['item_idx'] for r in base],dtype=np.int64),
            reuse_log_loss=losses['reuse'],map_log_loss=losses['linear256_rows'],rebuild_log_loss=losses['rebuild'])
        slices={name:describe(gain[mask],uids[mask]) for name,mask in [('all',np.ones(len(gain),bool)),('label0',labels==0),('label1',labels==1),('positive_multi_candidate_context',control)]}
        results[str(edge)]=dict(slices=slices,same_timestamp_coverage=timestamp_coverage(base),
            positive_same_context_control=context_summary,
            events_artifact=dict(path=str(path.resolve()),sha256=sha(path)),
            provenance=dict(source=str(folder.resolve()),configuration_sha256=sha(folder/'configuration.json'),
                source_summary_sha256=sha(folder/'summary.json'),population_sha256=sha(population),
                base_map_name='linear256_rows',method_names=['reuse','linear256_rows','rebuild']))
        data[edge]=(gain,labels,control)
    low=min(float(g.min()) for g,y,c in data.values());high=max(float(g.max()) for g,y,c in data.values())
    bins=np.linspace(low,high if high>low else low+1e-12,201)
    with (output/'cdf.csv').open('w',newline='') as cdf_file,(output/'histogram.csv').open('w',newline='') as hist_file:
        cdf=csv.writer(cdf_file);hist=csv.writer(hist_file)
        cdf.writerow(['edge','slice','delta_log_loss','count_at_value','cumulative_event_fraction'])
        hist.writerow(['edge','slice','left','right','count','event_fraction'])
        for edge,(gain,labels,control) in data.items():
            for name,mask in [('all',np.ones(len(gain),bool)),('label0',labels==0),('label1',labels==1),('positive_multi_candidate_context',control)]:
                values=gain[mask];unique,counts=np.unique(values,return_counts=True)
                for value,n,cumulative in zip(unique,counts,np.cumsum(counts)/len(values),strict=True):cdf.writerow([edge,name,value,n,cumulative])
                h,_=np.histogram(values,bins=bins)
                for left,right,n in zip(bins[:-1],bins[1:],h,strict=True):hist.writerow([edge,name,left,right,n,n/len(values)])
    result=dict(format='fixed_map_candidate_loss_distribution_v1',definition='delta_log_loss=stable_BCE(Reuse_logit,label)−stable_BCE(linear256_rows_logit,label), in nats; positive is lower observed loss after mapping.',
        threshold=THRESHOLD,threshold_rule='Near-zero means absolute delta<=0.001 nat; it is not proof of exact ineffectiveness. Other categories are >0.001 improvement and <−0.001 harm.',
        primary='Complete continuous empirical event CDF and full-range shared-bin histogram, all events and fixed label0/label1 slices.',
        uid_weighting='Equal-feedback-UID category fraction is the mean within-UID fraction; within label slices it averages users having that label. No UID removed or selected by outcome.',
        histogram=dict(bins=200,range=[low,high],range_clipped=False),edges=results,
        limitations=['A single feedback event has a realized log loss, not an AUC. No AUC is calculated after grouping candidates by this outcome.',
            'Label slicing is descriptive and does not eliminate label noise. Same-UID variation can still reflect time, context and label outcomes; it is not proof of stable item identity or candidate-gate necessity.',
            'The positive multi-candidate context control holds user, timestamp, history and label type fixed. A constant logit shift can still yield different loss gains at different native confidence, so context-level delta-score range is also reported. None of this establishes necessity of an extra candidate gate.',
            'All7000 fixed residents and every scored feedback event are retained; the current main metric remains observed-feedback AUC.',
            'The same fixed linear256_rows calibration is used throughout each release. There is no candidate-specific fitting or threshold search.',
            'Same-timestamp subsets below512 distinct users are coverage diagnostics only, not paper-quality mechanism evidence.'],
        analysis_sha256=sha(__file__),metric_helper_sha256=sha(Path(__file__).with_name('analyze_real_quality.py')))
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    rows=[]
    for edge,d in results.items():
        for name,s in d['slices'].items():
            for category,v in s['categories'].items():rows.append(dict(edge=edge,slice=name,category=category,feedback_users=s['feedback_users'],**v))
    with (output/'fractions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Per-feedback effect of one fixed KV calibration','',result['definition'],result['threshold_rule'],'',
        '| Edge | Slice | FeedbackUID | Events | Improved % event / equalUID | Near-zero % event / equalUID | Harmed % event / equalUID |','|---|---|---:|---:|---:|---:|---:|']
    for edge,d in results.items():
        for name,s in d['slices'].items():
            parts=[]
            for category in ['improved_above_threshold','near_zero','harmed_below_negative_threshold']:
                v=s['categories'][category];parts.append(f"{100*v['event_fraction']:.3f} / {100*v['equal_feedback_uid_fraction']:.3f}")
            lines.append('| '+' | '.join([edge,name,str(s['feedback_users']),str(s['events']),*parts])+' |')
    lines+=['','Within-UID counts below use the same fixed ±0.001nat boundary, without any user exclusion.']
    for edge,d in results.items():
        for name,s in d['slices'].items():
            w=s['within_uid'];lines.append(f"- Edge{edge} {name}: {w['multiple_event_users']} users have multiple events; {w['users_with_improved_and_harmed_events']} have both improved and harmed events; median within-UID gain range {w['median_gain_range_multiple_event_users']}nat.")
        lines.append(f"- Edge{edge} same-timestamp coverage: {d['same_timestamp_coverage']}.")
        lines.append(f"- Edge{edge} positive same-context control: {d['positive_same_context_control']}.")
    lines+=['','events_edgeN.npz preserves aligned uid/label/key/delta and all three losses. cdf.csv is the exact untrimmed empirical event CDF; histogram.csv uses200 common full-range bins. fractions.csv includes both event weighting and equal UID weighting.','',*['- '+x for x in result['limitations']]]
    (output/'report.md').write_text('\n'.join(lines)+'\n');(output/'analysis_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(status='complete',output=str(output.resolve()),edges={e:d['slices'] for e,d in results.items()})),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output)
