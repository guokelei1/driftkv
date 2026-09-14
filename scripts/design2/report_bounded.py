"""Mechanism, reference semantics, residency and whole-method costs."""
import json
import numpy as np,pandas as pd
from design.data import ROOT
from design2 import report_scan as report
from design2.report_sparse import load,KEY,TARGETS,failures
from design2.audit_benchmark import save
from design2.report_scale_followup import boot
from hstu_kvcache.evaluation.binary_metrics import binary_metrics

RUN=ROOT/'results/design2/bounded_01';OLD=ROOT/'results/design2/scan_30k_01'
def table(policy,name):
    jobs=json.loads((RUN/policy/'summary.json').read_text())['jobs']
    return pd.concat([pd.read_parquet(RUN/policy/j['name']/f'{name}.parquet') for j in sorted(jobs,key=lambda x:x['offset'])],ignore_index=True)

def main():
    for name in ['uids.json','members.parquet','references','frozen','witness']:
        path=RUN/name
        if not path.exists():path.symlink_to(OLD/name,target_is_directory=(OLD/name).is_dir())
    report.RUN=RUN;report.OUT=RUN/'analysis';report.POLICIES=['post','bounded']+(['bounded_immediate'] if (RUN/'bounded_immediate/summary.json').exists() else []);report.CONTRASTS=False;report.main()
    results=json.loads((RUN/'analysis/summary.json').read_text())['results'];ids=json.loads((RUN/'uids.json').read_text())
    baseline=load(OLD/'frozen')[0].reset_index(names='row_index');ref=pd.concat([pd.read_parquet(OLD/f'references/cap1024/m{t}.parquet') for t in TARGETS]);ref=baseline[['row_index']+KEY].merge(ref,on='row_index',validate='one_to_one')
    oldf,_,oldd=load(OLD/'witness');evidence=[]
    for policy in ['post','bounded','bounded_immediate']:
        f,c,d=load(RUN/policy);source='bounded' if policy=='bounded_immediate' else policy
        obs=table(source,'witness_reads');memory=table(source,'candidate_memory');life=table(source,'candidate_lifetimes')
        f['candidate']=f.groupby(['uid','target','timestamp']).cumcount()
        observed=f.merge(obs,on=['uid','target','timestamp','candidate','item_idx'],validate='one_to_one');assert len(observed)==len(obs)
        if policy=='bounded_immediate':
            acts=d[d.action=='rebuild'][['uid','target','timestamp']].assign(commit=True)
            observed=observed.merge(acts,on=['uid','target','timestamp'],how='left',validate='many_to_one')
            np.testing.assert_array_equal(observed.design2,observed.witness.where(observed.commit.eq(True),observed.service))
        else:np.testing.assert_array_equal(observed.design2,observed.service)
        check=f.merge(ref[KEY+['fresh_current']],on=KEY,validate='one_to_one')
        observed=observed.merge(ref[KEY+['fresh_current']],on=KEY,validate='one_to_one')
        observed['witness_error']=(observed.witness-observed.fresh_current).abs();observed['service_error']=(observed.service-observed.fresh_current).abs();observed['discrepancy']=(observed.service-observed.witness).abs()
        assert not life.duplicated(['uid','target']).any() and (life.spent<=life.allowance).all()
        charged=float(c[c.method=='design2'][['witness_native_append','witness_extra_read','witness_control']].sum().sum())
        np.testing.assert_allclose(charged,float(life.spent.sum()),rtol=0,atol=1e-4)
        if policy=='post':
            cols=['uid','target','timestamp','reason'];a=d[d.action=='rebuild'][cols].sort_values(['uid','target']);b=oldd[oldd.action=='rebuild'][cols].sort_values(['uid','target']);np.testing.assert_array_equal(a,b)
            paired=f.merge(oldf[KEY+['design2']],on=KEY,suffixes=('_post','_immediate'));change=paired[paired.design2_post!=paired.design2_immediate]
            assert len(change.merge(a[['uid','target','timestamp']],on=['uid','target','timestamp']))==len(change)
            save(RUN/'analysis/timing_identity.json',dict(actions_identical=True,changed_rows=len(change),changes_only_commit_requests=True))
        for cohort,uids in [('natural',ids['natural']),('enriched',ids['extension'])]:
            rr=next(r for r in results if r['policy']==policy and r['cohort']==cohort);g=check[check.uid.isin(uids)].copy();o=observed[observed.uid.isin(uids)];l=life[life.uid.isin(uids)];mm=memory[memory.uid.isin(uids)].copy()
            for m in ['design1','design2']:g['err_'+m]=(g[m]-g.fresh_current).abs()
            # Resource timeline represents persistent uninstalled witnesses.
            mm['order']=np.arange(len(mm));mm=mm.sort_values(['uid','target','timestamp','order']);mm['delta']=mm.groupby(['uid','target']).bytes.diff().fillna(mm.bytes)
            timeline=mm.groupby('timestamp').delta.sum().sort_index();resident=timeline.cumsum();assert resident.min()>=0 and resident.iloc[-1]==0
            byte_seconds=float((resident.iloc[:-1].to_numpy()*np.diff(resident.index.to_numpy())).sum())
            mm['positive']=mm.delta.clip(lower=0);positive=mm.groupby('timestamp').positive.sum().reindex(resident.index)
            upper=float((resident.shift(fill_value=0)+positive).max())
            q=o.groupby(['uid','target','timestamp'])[['witness_error','service_error','discrepancy']].max().reset_index()
            counts=pd.crosstab(q.discrepancy>.5,q.service_error>.5).reindex(index=[False,True],columns=[False,True],fill_value=0)
            act=d[d.uid.isin(uids)&d.action.eq('rebuild')][['uid','target','timestamp']]
            trigger=g.merge(act,on=['uid','target','timestamp']);tmp=g.merge(act.assign(trigger=True),on=['uid','target','timestamp'],how='left');nontrigger=tmp[tmp.trigger.isna()]
            initial=act.groupby('uid').target.min().reset_index(name='first_target');later=g.merge(initial,on='uid');later=later[later.target>later.first_target]
            reads=g.groupby(['uid','target','timestamp'])[['err_design1','err_design2']].max().reset_index()
            burden={m:dict(rate=float(reads.assign(v=reads['err_'+m]>.5).groupby('uid').v.mean().mean()),excess=float(reads.assign(v=(reads['err_'+m]-.5).clip(lower=0)).groupby('uid').v.mean().mean())) for m in ['design1','design2']}
            installed_obs=q.merge(act,on=['uid','target','timestamp'])
            birth=o[o.kind=='construction'].groupby(['uid','target']).agg(birth_writes=('writes','first'),birth_count=('count','first')).reset_index()
            evicted=o.merge(birth,on=['uid','target']);evicted['after_eviction']=evicted.birth_count+evicted.writes-evicted.birth_writes>1024
            routine=[];meta=pd.read_parquet(OLD/'members.parquet');routine_rows=g.merge(meta[['uid','target','routine']],on=['uid','target'],validate='many_to_one');routine_rows=routine_rows[routine_rows.routine]
            for x in rr['quality']:
                if x['stratum']!='routine':continue
                v=routine_rows[routine_rows.target==x['target']].copy()
                for m in ['design1','design2']:v['loss_'+m]=np.logaddexp(0,v[m])-v.label*v[m]
                sums=v.groupby('uid')[['loss_design1','loss_design2']].sum().reindex(uids,fill_value=0).to_numpy();rng=np.random.default_rng(17);idx=rng.integers(len(uids),size=(1000,len(uids)));sample=sums[idx].sum(1);relative=sample[:,1]/sample[:,0]-1;ci=np.quantile(relative,[.025,.975]).tolist()
                routine.append(dict(target=x['target'],uids=x['uids'],auc_delta=x['auc_difference'],interval=x['auc_interval'],auc_noninferior=x['auc_interval'][0]>=-.001,
                    logloss_relative=x['metrics']['design2']['log_loss']/x['metrics']['design1']['log_loss']-1,logloss_interval=ci,logloss_noninferior=ci[1]<=.005))
            record=dict(policy=policy,cohort=cohort,observation_rows=len(o),continued_rows=int(o.kind.eq('continued').sum()),
                witness_error={kind:dict(rows=len(z),uids=int(z.uid.nunique()),mean=float(z.witness_error.mean()),p95=float(z.witness_error.quantile(.95)),max=float(z.witness_error.max())) for kind,z in o.groupby('kind')},
                threshold_table=counts.values.tolist(),threshold_table_axes='rows observed discrepancy false/true; columns actual Fresh service error false/true; actual queried groups only',
                installations_with_worse_witness=int((installed_obs.witness_error>installed_obs.service_error).sum()),
                witness_after_eviction={str(k):dict(rows=len(v),mean=float(v.witness_error.mean()),max=float(v.witness_error.max())) for k,v in evicted.groupby('after_eviction')},
                lifetimes=len(l),closures=l.reason.value_counts().to_dict(),observation_flops=float(l.spent.sum()),
                resident_peak_bytes=float(resident.max()),same_timestamp_conservative_peak_bytes=upper,resident_byte_seconds=byte_seconds,
                residence_seconds=dict(median=float((l.ended-l.created).median()),p95=float((l.ended-l.created).quantile(.95)),max=int((l.ended-l.created).max())),
                storage_scope='Uninstalled persistent witness KV only; timeline after timestamp and conservative simultaneous-update upper value. Excludes construction workspace, shared weights, serving KV and instrumentation.',
                trigger_rows=len(trigger),trigger_uids=int(trigger.uid.nunique()),burden=burden,
                nontrigger=dict(rows=len(nontrigger),mean_abs={m:float(nontrigger['err_'+m].mean()) for m in ['design1','design2']}),
                next_release=dict(rows=len(later),uids=int(later.uid.nunique()),mean_abs={m:float(later['err_'+m].mean()) for m in ['design1','design2']}),
                routine=routine)
            evidence.append(record)
        observed.to_parquet(RUN/f'analysis/{policy}_observations.parquet',index=False)
    save(RUN/'analysis/mechanisms.json',evidence)
    lines=['# 请求后安装与有限观察：同一框架的机制实验','','|策略/队列|主失败UID D1→D2|超额残余减少|AUC差pp|3万/10万/百万费用|','|---|---:|---:|---:|---|']
    for r in results:
        v=r['residuals']['all'][1];a=v['design1'];b=v['design2'];lines.append(f"|{r['policy']}/{r['cohort']}|{a['failed_uids']}→{b['failed_uids']}|{1-b['excess']/a['excess']:.2%}|{r['auc_difference']*100:+.4f}|"+'/'.join(f"{x['ratio']:.2%}" for x in r['scales'])+'|')
    (RUN/'analysis/mechanisms.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
