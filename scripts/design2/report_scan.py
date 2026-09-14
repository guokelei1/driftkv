"""Expanded scenario quality, controls, all costs and residuals."""
import json
import numpy as np,pandas as pd
from design.data import ROOT
from design2.report_sparse import load,failures,KEY,TARGETS
from design2.report_scale_followup import boot
from design2.stratum_quality import quality,STRATA
from design2.report_full_reference import tail_intervals
from design2.audit_benchmark import save

RUN=ROOT/'results/design2/scan_30k_01';OUT=RUN/'analysis'
SCENES=STRATA+['stale_no_writes','bursty','long_idle']
POLICIES=None
CONTRASTS=True

def main():
    OUT.mkdir(exist_ok=True)
    ids=json.loads((RUN/'uids.json').read_text());members=pd.read_parquet(RUN/'members.parquet')
    # Secondary mechanism exposures fixed from existing primitives/bounds,
    # before expanded quality outputs; not new sampling or online rules.
    members['stale_no_writes']=members.old_lineage&(members.prior_writes==0)
    members['bursty']=False;members['long_idle']=False
    for b in json.loads((ROOT/'results/design2/analysis/benchmark_01/calibration_bounds.json').read_text()):
        name='bursty' if b['feature']=='prior_writes' else 'long_idle'
        members.loc[(members.target==b['target'])&(members[b['feature']]>b['upper']),name]=True
    shared=next(r['shared_preparation'] for r in json.loads((ROOT/'results/design2/sparse_01/analysis/summary.json').read_text())['results'] if r['mode']=='excursion' and r['users']==2048)
    baseline=load(RUN/'frozen')[0];refpath=RUN/'references/cap1024';reference=None
    if (RUN/'references/summary.json').exists():
        ref=pd.concat([pd.read_parquet(refpath/f'm{t}.parquet') for t in TARGETS]);assert ref.row_index.is_unique
        reference=baseline.reset_index(names='row_index')[['row_index']+KEY].merge(ref,on='row_index',validate='one_to_one')
    results=[]
    policies=POLICIES or (['frozen','witness']+(['margin_gate'] if (RUN/'margin_gate/summary.json').exists() else []))
    for policy in policies:
        f,c,d=load(RUN/policy);ordered=f.sort_values(KEY);other=baseline.sort_values(KEY)
        np.testing.assert_array_equal(ordered[KEY],other[KEY])
        if policy=='margin_gate':np.testing.assert_allclose(ordered[['reuse','design1','exact']],other[['reuse','design1','exact']],rtol=0,atol=2e-6)
        else:np.testing.assert_array_equal(ordered[['reuse','design1','exact']],other[['reuse','design1','exact']])
        cols=c.columns.difference(['uid','target','method'])
        for cohort,uids in [('natural',ids['natural']),('enriched',ids['extension'])]:
            g=f[f.uid.isin(uids)];cc=c[c.uid.isin(uids)];dd=d[d.uid.isin(uids)];meta=members[members.uid.isin(uids)]
            cost={m:float(cc[cc.method==m][cols].sum().sum()) for m in ['design1','design2','exact']}
            q=quality(g,meta,uids,SCENES);edges=[x for x in q if x['stratum']=='all'];auc=float(np.mean([x['auc_difference'] for x in edges]))
            r=dict(policy=policy,cohort=cohort,users=len(uids),rows=len(g),quality=q,auc_difference=auc,auc_intervals=boot(g,uids),
                costs=cost,shared_preparation=shared,components=cc[cc.method=='design2'][cols].sum().to_dict(),
                scales=[dict(users=n,ratio=(shared+cost['design2']*n/len(uids))/(cost['exact']*n/len(uids))) for n in [30000,100000,1000000]],
                active_states=int((dd.action!='unused').sum()),paid_builds=int(dd.screened.eq(True).sum()),installs=int((dd.action=='rebuild').sum()),
                actions=dd.groupby(['target','reason']).size().reset_index(name='states').to_dict('records'))
            sc=cc.merge(meta,on=['uid','target'],validate='many_to_one');sd=dd.merge(meta,on=['uid','target'],validate='one_to_one')
            sc['all']=True;sd['all']=True;r['stratum_costs']=[]
            for s in SCENES:
                x=sc[sc[s]];y=sd[sd[s]];v={m:float(x[x.method==m][cols].sum().sum()) for m in cost}
                r['stratum_costs'].append(dict(stratum=s,states=len(y),active=int((y.action!='unused').sum()),paid=int(y.screened.eq(True).sum()),installs=int((y.action=='rebuild').sum()),
                    service=v,extra_over_own_exact=(v['design2']-v['design1'])/v['exact'] if v['exact'] else None))
            if reference is not None:
                check=g.merge(reference[KEY+['fresh_current']],on=KEY,validate='one_to_one');assert len(check)==len(g[g.target.isin(TARGETS)])
                for m in ['design1','design2']:check['err_'+m]=(check[m]-check.fresh_current).abs()
                win=check.groupby(['uid','target'])[['err_design1','err_design2']].max().reset_index().merge(meta,on=['uid','target'],validate='one_to_one');win['all']=True
                r['residuals']={s:failures(win[win[s]]) for s in SCENES};r['primary_intervals']=tail_intervals(win)
                action=dd[dd.action=='rebuild'][['uid','target','timestamp']].rename(columns={'timestamp':'renewal_time'})
                anchor=g.merge(action,on=['uid','target']);anchor=anchor[anchor.timestamp==anchor.renewal_time].groupby(['uid','target']).writes_since_release.min().reset_index(name='renewal_writes')
                after=check.merge(anchor,on=['uid','target']);after=after[after.writes_since_release>after.renewal_writes]
                r['after_real_writes']=dict(rows=len(after),uids=int(after.uid.nunique()),mean_abs={m:float(after['err_'+m].mean()) for m in ['design1','design2']})
                r['after_real_writes_auc']=boot(after,uids) if len(after) else []
                win.to_parquet(OUT/f'{policy}_{cohort}_windows.parquet',index=False)
            results.append(r)
    contrasts=[]
    left=load(RUN/'witness')[0];right=baseline;left=left.sort_values(KEY).reset_index(drop=True);right=right.sort_values(KEY).reset_index(drop=True);left['design1']=right.design2
    if CONTRASTS:
        for name,uids in [('natural',ids['natural']),('enriched',ids['extension'])]:contrasts.append(dict(cohort=name,quality=quality(left[left.uid.isin(uids)],members[members.uid.isin(uids)],uids,SCENES)))
    save(OUT/'summary.json',dict(results=results,witness_vs_frozen=contrasts,full_reference=reference is not None,confirmation_read=False,
        scope='New-to-Design2 development; natural and actively exposed enrichment separate. No labels used for sampling. Within-UID AUC is secondary.'))
    lines=['# 3万目录扫描后的4082用户扩展实验','','|策略/样本|UID|四边AUC差pp|付费构造/安装|3万/10万/百万费用|','|---|---:|---:|---:|---|']
    for r in results:lines.append(f"|{r['policy']}/{r['cohort']}|{r['users']}|{r['auc_difference']*100:+.4f}|{r['paid_builds']}/{r['installs']}|"+'/'.join(f"{x['ratio']:.2%}" for x in r['scales'])+'|')
    lines+=['','|策略/样本/困难组|目标|反馈UID|AUC差pp|UID95%区间pp|','|---|---|---:|---:|---|']
    for r in results:
        for q in r['quality']:
            if q['stratum']=='all':continue
            diff='NA' if q['auc_difference'] is None else f"{q['auc_difference']*100:+.4f}"
            ci=None if q['auc_interval'] is None else [round(x*100,4) for x in q['auc_interval']]
            lines.append(f"|{r['policy']}/{r['cohort']}/{q['stratum']}|M{q['target']}|{q['uids']}|{diff}|{ci}|")
    (OUT/'report.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines[:9]))

if __name__=='__main__':main()
