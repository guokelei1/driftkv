"""Paper-facing joint acceptance: sparse calls, persistent quality, full cost."""
import json
import numpy as np
import pandas as pd
from design.data import ROOT
from design2.audit_benchmark import save
from design2.report_scale_followup import load as old_load,boot,TARGETS
from hstu_kvcache.evaluation.binary_metrics import binary_metrics

RUN=ROOT/'results/design2/sparse_01';OUT=RUN/'analysis'
KEY=['uid','target','request_id']

def load(path):
    summary=json.loads((path/'summary.json').read_text());assert summary['status']=='complete'
    def table(name):return pd.concat([pd.read_parquet(path/j['name']/f'{name}.parquet') for j in sorted(summary['jobs'],key=lambda x:x['offset'])],ignore_index=True)
    return table('quality_raw'),table('costs').fillna(0),table('decisions')

def failures(frame):
    rows=[]
    for tau in [.1,.5,1.]:
        r=dict(threshold=tau)
        for m in ['design1','design2']:
            e=frame['err_'+m];temp=frame[['uid']].assign(bad=(e>tau).astype(float),excess=np.maximum(e-tau,0))
            r[m]=dict(failed_uids=int(frame.loc[e>tau,'uid'].nunique()),rate=float(temp.groupby('uid').bad.mean().mean()),
                excess=float(temp.groupby('uid').excess.mean().mean()))
        rows.append(r)
    return rows

def main():
    OUT.mkdir(exist_ok=True)
    prep=json.loads((ROOT/'results/design2/lifecycle_01/preparation/summary.json').read_text())
    ids=json.loads((ROOT/'configs/design2/scale_followup_01_uids.json').read_text());allids=ids['original']+ids['extension']
    members=pd.read_parquet(ROOT/'results/design2/analysis/scale_followup_01/members.parquet')
    # Four targets, six layers, six heads; form H_II from existing L rows,
    # Cholesky, conservative scalar allowance, full512 source-score calibration.
    source_prepare=144*(2*1281*33**2+33**3/3+64*33**2)+7613*(36*(33**2+3*33+3)+12)+100_000_000
    observed_prepare=source_prepare+144*(2*1281*225**2+225**3/3+64*225**2)+7613*16*(36*(225**2+3*225+3)+6*192*2+12)
    results=[]
    for path in sorted(RUN.glob('*_cap*_*')):
        if not (path/'summary.json').exists():continue
        mode,cap,users=path.name.split('_');cap=int(cap[3:]);users=int(users)
        f,c,d=load(path);uids=allids[:users]
        baseline,bc,_,_=old_load('risk',cap);a=f.sort_values(KEY);b=baseline[baseline.uid.isin(uids)].sort_values(KEY)
        np.testing.assert_array_equal(a[KEY],b[KEY]);np.testing.assert_array_equal(a[['reuse','design1','exact']],b[['reuse','design1','exact']])
        cols=[x for x in c if x not in ['uid','target','method']]
        totals={m:float(c[c.method==m][cols].sum().sum()) for m in ['design1','design2','exact']}
        if mode=='exactdemand':totals['design2']=float(c[c.method=='design2'].rebuild_tiled.sum())
        extra=0 if mode in ['background','bgobserve','excursionbg'] else (100_000_000 if mode in ['fullreadobserve','excursionfull'] else (observed_prepare if mode in ['readobserve','excursion','firstusematched'] else source_prepare))
        shared=prep['design2_shared']+extra
        if mode in ['direct','observe','bgobserve','readobserve','fullreadobserve','excursion','excursionfull','excursionbg','firstusematched']:shared-=prep['detector_components']['inverse_preparation']
        if mode=='exactdemand':shared=extra=0.
        source=baseline.reset_index(names='row_index')
        ref=pd.concat([pd.read_parquet(ROOT/f'results/design2/scale_followup_01/references_cap{cap}/m{t}.parquet') for t in TARGETS])
        ref=source[['row_index']+KEY].merge(ref,on='row_index',validate='one_to_one')
        checked=f.merge(ref[KEY+['fresh_current']],on=KEY,validate='one_to_one')
        for m in ['design1','design2']:checked['err_'+m]=(checked[m]-checked.fresh_current).abs()
        err=checked.groupby(['uid','target'])[['err_design1','err_design2']].max().reset_index()
        meta=members[members.cap==cap];err=err.merge(meta,on=['uid','target'],validate='one_to_one')
        metrics={str(t):{m:binary_metrics(g.label,g[m]) for m in ['reuse','design1','design2','exact']} for t,g in f.groupby('target')}
        aucdiff=float(np.mean([metrics[str(t)]['design2']['ROC_AUC']-metrics[str(t)]['design1']['ROC_AUC'] for t in TARGETS]))
        routine=f.merge(meta,on=['uid','target'],validate='many_to_one');routine=routine[routine.routine]
        routine_metrics=[]
        intervals=boot(routine,uids) if len(routine) else []
        for t,g in routine.groupby('target'):
            mm={m:binary_metrics(g.label,g[m]) for m in ['design1','design2']}
            routine_metrics.append(dict(target=int(t),uids=int(g.uid.nunique()),requests=len(g),metrics=mm,
                auc_diff=mm['design2']['ROC_AUC']-mm['design1']['ROC_AUC'],
                logloss_relative=mm['design2']['log_loss']/mm['design1']['log_loss']-1))
        dc=d.merge(meta,on=['uid','target'],validate='one_to_one');normal=dc[dc.routine]
        costs=c.merge(meta,on=['uid','target'],validate='many_to_one');costs=costs[costs.routine]
        nc={m:float(costs[costs.method==m][cols].sum().sum()) for m in totals}
        if mode=='exactdemand':nc['design2']=float(costs[costs.method=='design2'].rebuild_tiled.sum())
        active=d.action!='unused';screens=d.screened.eq(True)
        r=dict(mode=mode,cap=cap,users=users,requests=len(f),costs=totals,shared_preparation=shared,
            preparation_note='Original calibration teacher/source replay retained in full; direct excludes unused inverse only; conservative extra source preparation/calibration charged. Experimental replay runtime is separate.',
            source_preparation=extra,source_storage_bytes=0 if mode in ['background','bgobserve','fullreadobserve','excursionfull','excursionbg','exactdemand'] else 144*(33*33+(225*225 if mode in ['readobserve','excursion','firstusematched'] else 0))*8,
            components=c[c.method=='design2'][cols].sum().to_dict(),
            scales=[dict(users=n,total_ratio=(shared+totals['design2']*n/users)/(totals['exact']*n/users)) for n in [30000,100000,1000000]],
            active_states=int(active.sum()),screened_states=int(screens.sum()),screened_active_fraction=float(screens.sum()/active.sum()),
            full_geometry_states=int(d.u.notna().sum()),rebuilds=int((d.action=='rebuild').sum()),
            evidence_request_groups=int(d['checks'].sum()) if 'checks' in d else None,
            evidence_candidates=int(d['evidence_candidates'].sum()) if 'evidence_candidates' in d else None,
            paid_builds=int((c[c.method=='design2'].get('rebuild_tiled',pd.Series(dtype=float))>0).sum()),
            screen_users=int(d.loc[screens,'uid'].nunique()),rebuild_users=int(d.loc[d.action=='rebuild','uid'].nunique()),
            actions=d.groupby(['target','reason']).size().reset_index(name='states').to_dict('records'),
            auc_diff=aucdiff,per_edge=metrics,intervals=boot(f,uids),failures=failures(err),
            challenge_failures=failures(err[err.challenge]),reference_rows=len(checked),reference_users=int(checked.uid.nunique()),
            routine=routine_metrics,routine_intervals=intervals,routine_states=len(normal),routine_rebuilds=int((normal.action=='rebuild').sum()),
            routine_extra_cost_ratio=(nc['design2']-nc['design1'])/nc['exact'] if nc['exact'] else None,
            new_severe_low_states=int(((err.err_design1<=.1)&(err.err_design2>.5)).sum()),
            low_error_states=int((err.err_design1<=.1).sum()))
        if mode=='exactdemand':r['preparation_note']='Cache-only Exact-on-first-use baseline: no adapter or detector; summary counters are evaluator instrumentation and excluded. All actual target constructions charged, native service common.'
        # Persistent benefit: references after at least one real write following
        # the actual action, rather than the immediate rebuilt query.
        action=d[d.action=='rebuild'][['uid','target','timestamp']].rename(columns={'timestamp':'renewal_time'})
        later=checked.merge(action,on=['uid','target'],validate='many_to_one')
        # after_rebuild_writes is recorded directly by the independent state.
        later=later[later.timestamp>later.renewal_time]
        r['later_than_renewal']=dict(rows=len(later),uid_count=int(later.uid.nunique()),scope='later timestamp; not automatically proof of a write',
            mean_abs={m:float(later['err_'+m].mean()) for m in ['design1','design2']})
        anchor=f.merge(action,on=['uid','target'],validate='many_to_one')
        anchor=anchor[anchor.timestamp==anchor.renewal_time].groupby(['uid','target']).writes_since_release.min().reset_index(name='renewal_writes')
        written=checked.merge(anchor,on=['uid','target'],validate='many_to_one')
        written=written[written.writes_since_release>written.renewal_writes]
        r['after_real_writes']=dict(rows=len(written),uids=int(written.uid.nunique()),
            mean_abs={m:float(written['err_'+m].mean()) for m in ['design1','design2']})
        inherited=checked[checked.last_rebuild_target.notna() & (checked.last_rebuild_target<checked.target)]
        r['inherited_after_next_release']=dict(rows=len(inherited),uids=int(inherited.uid.nunique()),
            mean_abs={m:float(inherited['err_'+m].mean()) for m in ['design1','design2']})
        r['cohorts']=[dict(name=name,intervals=boot(f[f.uid.isin(uu)],[u for u in uids if u in set(uu)]))
            for name,uu in [('original',ids['original']),('extension',ids['extension'])]] if users==2048 else []
        err.to_parquet(OUT/f'{path.name}_errors.parquet',index=False);results.append(r)
    save(OUT/'summary.json',dict(results=results,confirmation_read=False,reference='Existing fixed FreshCurrent evaluation checkpoints only; no new teachers for these policies',scope='development, one training seed; failure UID counts are not independent across caps'))
    lines=['# 稀疏状态证据—核验—真实重建：完整候选评价','',
        '本表AUC使用全部真实反馈；主失败UID列仅保留旧检查点参考，**不是全请求结果**。最终保护结论见[全请求复核](full_reference.md)和[联合结论](conclusion.md)。','',
        '| 方案/窗口/UID | 筛中/活跃状态 | 完整几何/付费构造/安装 | AUC差pp | 旧检查点主失败UID | 服务TF | 3万/10万/百万总费用 |','|---|---|---|---|---|---|---|']
    for r in results:
        fail=r['failures'][1]
        lines.append(f"| {r['mode']}/{r['cap']}/{r['users']} | {r['screened_states']}/{r['active_states']} | {r['full_geometry_states']}/{r['paid_builds']}/{r['rebuilds']} | {r['auc_diff']*100:+.4f} | {fail['design1']['failed_uids']}→{fail['design2']['failed_uids']} | {r['costs']['design2']/1e12:.4f} | "+'/'.join(f"{x['total_ratio']:.2%}" for x in r['scales'])+' |')
    lines+=['','完整prepare和普通D1服务均已计费。费用外推对应相同负载分布；实际人口最大2048，短压力512不代表自然比例。',
        '校准分位数10%是状态证据筛查的工作点，不承诺在持续回放仍筛中10%；source原33维证据不包含遗漏方向的安全保证。',
        '背景筛查和直接重建为机制消融，实际支出不完全相同，不能声称严格同预算胜出。全部0.1/0.5/1.0、正常组、两个自然半组和区间见summary.json。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

if __name__=='__main__':main()
