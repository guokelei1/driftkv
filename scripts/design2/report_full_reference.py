"""Frozen-policy evaluation against Current rebuilt at every real request."""
import json
import numpy as np
import pandas as pd
from design.data import ROOT
from design2.audit_benchmark import save
from design2.report_sparse import load,old_load,failures,KEY,TARGETS

RUN=ROOT/'results/design2/sparse_01'

def tail_intervals(frame):
    x=frame[['uid']].copy()
    for m in ['design1','design2']:
        x[m+'_rate']=(frame['err_'+m]>.5).astype(float)
        x[m+'_excess']=np.maximum(frame['err_'+m]-.5,0)
    x=x.groupby('uid').mean();rng=np.random.default_rng(17);out={}
    samples=rng.integers(len(x),size=(2000,len(x)))
    for metric in ['rate','excess']:
        a=x['design1_'+metric].to_numpy()[samples].mean(1);b=x['design2_'+metric].to_numpy()[samples].mean(1)
        out[metric]=dict(difference=np.quantile(b-a,[.025,.975]).tolist(),relative_reduction=np.quantile((1-b/a)[a>0],[.025,.975]).tolist())
    return out

def main():
    members=pd.read_parquet(ROOT/'results/design2/analysis/scale_followup_01/members.parquet')
    results=[];verification=[];teachers=0
    for cap in [1024,32,128]:
        base=old_load('risk',cap)[0].reset_index(names='row_index')
        ref=pd.concat([pd.read_parquet(RUN/f'full_references/cap{cap}/m{t}.parquet') for t in TARGETS])
        assert len(ref)==len(base[base.target.isin(TARGETS)]) and ref.row_index.is_unique
        old=pd.concat([pd.read_parquet(ROOT/f'results/design2/scale_followup_01/references_cap{cap}/m{t}.parquet') for t in TARGETS])
        overlap=old.merge(ref,on='row_index',suffixes=('_old','_new'),validate='one_to_one')
        delta=(overlap.fresh_current_old-overlap.fresh_current_new).abs()
        np.testing.assert_allclose(overlap.fresh_current_old,overlap.fresh_current_new,rtol=0,atol=2e-5)
        verification.append(dict(cap=cap,rows=len(ref),overlap_rows=len(overlap),overlap_max_abs=float(delta.max())))
        teachers+=sum(json.loads((RUN/f'full_references/cap{cap}/m{t}.json').read_text())['evaluation_teacher_FLOPs'] for t in TARGETS)
        ref=base[['row_index']+KEY].merge(ref,on='row_index',validate='one_to_one')
        modes=['source','observe','readobserve','fullreadobserve','firstusematched','excursion','excursionfull','excursionbg','exactdemand'] if cap==1024 else ['excursion']
        for mode in modes:
            users=2048 if cap==1024 else 512;path=RUN/f'{mode}_cap{cap}_{users}'
            if not (path/'summary.json').exists():continue
            f,c,d=load(path)
            checked=f.merge(ref[KEY+['fresh_current']],on=KEY,validate='one_to_one')
            assert len(checked)==len(ref)
            for m in ['reuse','design1','design2','exact']:checked['err_'+m]=(checked[m]-checked.fresh_current).abs()
            ec=['err_'+m for m in ['reuse','design1','design2','exact']]
            windows=checked.groupby(['uid','target'])[ec].max().reset_index().merge(members[members.cap==cap],on=['uid','target'],validate='one_to_one')
            requests=checked.groupby(['uid','target','timestamp'])[ec].max().reset_index()
            r=dict(mode=mode,cap=cap,users=users,rows=len(checked),request_groups=len(requests),
                windows=len(windows),failures=failures(windows),challenge=failures(windows[windows.challenge]),routine=failures(windows[windows.routine]),
                primary_uid_intervals=tail_intervals(windows),
                request_burden=failures(requests),new_severe_low_states=int(((windows.err_design1<=.1)&(windows.err_design2>.5)).sum()),
                per_target=[dict(target=int(t),failures=failures(g)) for t,g in windows.groupby('target')])
            action=d[d.action=='rebuild'][['uid','target','timestamp']].rename(columns={'timestamp':'renewal_time'})
            anchor=f.merge(action,on=['uid','target'],validate='many_to_one')
            anchor=anchor[anchor.timestamp==anchor.renewal_time].groupby(['uid','target']).writes_since_release.min().reset_index(name='renewal_writes')
            written=checked.merge(anchor,on=['uid','target'],validate='many_to_one')
            written=written[written.writes_since_release>written.renewal_writes]
            inherited=checked[checked.last_rebuild_target.notna() & (checked.last_rebuild_target<checked.target)]
            immediate=checked.merge(action,on=['uid','target'],validate='many_to_one');immediate=immediate[immediate.timestamp==immediate.renewal_time]
            r['immediate_max_abs']=float(immediate.err_design2.max()) if len(immediate) else None
            if len(immediate):assert immediate.err_design2.max()<2e-5
            for name,g in [('after_real_writes',written),('next_release',inherited)]:
                r[name]=dict(rows=len(g),uids=int(g.uid.nunique()),mean_abs={m:float(g['err_'+m].mean()) for m in ['design1','design2']})
            if mode.startswith('excursion'):
                bad=windows[windows.err_design2>.5].merge(d,on=['uid','target'],suffixes=('','_decision'),validate='one_to_one')
                r['remaining_failures']=bad[['uid','target','err_design1','err_design2','reason','screen_estimate','commit_error']].replace({np.nan:None}).to_dict('records')
                assert not d.duplicated(['uid','target']).any()
            windows.to_parquet(RUN/f'analysis/full_{mode}_cap{cap}_windows.parquet',index=False)
            results.append(r)
    save(RUN/'analysis/full_reference.json',dict(results=results,verification=verification,evaluation_teacher_FLOPs=teachers,
        scope='Every real request at M1/M3/M4/M5; policies frozen before new references; reused development UIDs, one training seed. No confirmation access.'))
    lines=['# 全请求 FreshCurrent：冻结策略的最终复核','','|方案/窗口|主失败UID|主超额残余降幅|请求超额降幅|新增低误差→严重窗口|','|---|---:|---:|---:|---:|']
    for r in results:
        a=r['failures'][1];b=r['request_burden'][1]
        gain=lambda x:1-x['design2']['excess']/x['design1']['excess'] if x['design1']['excess'] else float('nan')
        lines.append(f"|{r['mode']}/{r['cap']}|{a['design1']['failed_uids']}→{a['design2']['failed_uids']}|{gain(a):.2%}|{gain(b):.2%}|{r['new_severe_low_states']}|")
    (RUN/'analysis/full_reference.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))

if __name__=='__main__':main()
