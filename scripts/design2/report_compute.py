"""Computation-first accounting, coverage losses and unchanged quality benchmark."""
import json
import numpy as np
import pandas as pd
from design.data import ROOT
from design2.report_scale_followup import load as old_load,boot,TARGETS
from design2.audit_benchmark import save
from hstu_kvcache.evaluation.binary_metrics import binary_metrics

RUN=ROOT/'results/design2/compute_01';OUT=ROOT/'results/design2/analysis/compute_01'

def load(path):
    s=json.loads((path/'summary.json').read_text());assert s['status']=='complete'
    raw=[];cost=[];dec=[];wallet=[]
    for j in sorted(s['jobs'],key=lambda j:j['offset']):
        assert j['exit']==0;p=path/j['name']
        raw.append(pd.read_parquet(p/'quality_raw.parquet'));cost.append(pd.read_parquet(p/'costs.parquet'))
        dec.append(pd.read_parquet(p/'decisions.parquet'));wallet.append(json.loads((p/'wallet.json').read_text()))
    return pd.concat(raw,ignore_index=True),pd.concat(cost,ignore_index=True).fillna(0),pd.concat(dec,ignore_index=True),wallet

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    prep=json.loads((ROOT/'results/design2/lifecycle_01/preparation/summary.json').read_text())
    ids=json.loads((ROOT/'configs/design2/scale_followup_01_uids.json').read_text());allids=ids['original']+ids['extension']
    meta=pd.read_parquet(ROOT/'results/design2/analysis/scale_followup_01/members.parquet')
    results=[]
    for path in sorted(RUN.glob('*_cap*_*')):
        if not (path/'summary.json').exists():continue
        mode,cap,users=path.name.split('_');cap=int(cap[3:]);users=int(users)
        f,c,d,w=load(path);uids=allids[:users]
        # Grant computation visits inactive states too. Charge it separately
        # from the runner's128-FLOP first-arrival admission allowance.
        c['release_accounting']=np.where((c.method=='design2')&c.target.isin(TARGETS),128,0)
        baseline,bc,_,_=old_load('risk',cap);baseline=baseline[baseline.uid.isin(uids)];bc=bc[bc.uid.isin(uids)]
        keys=['uid','target','request_id'];a=f.sort_values(keys);b=baseline.sort_values(keys)
        np.testing.assert_array_equal(a[keys],b[keys]);np.testing.assert_array_equal(a[['reuse','design1','exact']],b[['reuse','design1','exact']])
        cols=[k for k in c if k not in ['uid','target','method']]
        total={m:float(c[c.method==m][cols].sum().sum()) for m in ['design1','design2','exact']}
        oldcols=[k for k in bc if k not in ['uid','target','method']];oldcost=float(bc[bc.method=='design2'][oldcols].sum().sum())
        components=c[c.method=='design2'][cols].sum().to_dict()
        paid=sum(s['spent'] for s in w);credit=sum(s['credit'] for s in w)
        np.testing.assert_allclose(credit,.06*total['exact'],rtol=1e-12,atol=.1)
        accounted=sum(components.get(k,0) for k in ['decision_geometry','rebuild_tiled','rebuild_summary','rejected_read_duplicate'])
        assert paid==accounted
        if mode!='readshare':assert all(s['spent']<=s['credit']+1e-3 for s in w)
        shared=prep['design1_shared'] if mode=='fifo' else prep['design2_shared']-prep['detector_components']['inverse_preparation']
        scales=[dict(users=n,total_ratio=(shared+total['design2']*n/users)/(total['exact']*n/users)) for n in [30000,100000,1000000]]
        metrics={str(t):{m:binary_metrics(g.label,g[m]) for m in ['design1','design2','exact']} for t,g in f.groupby('target')}
        aucdiff=float(np.mean([metrics[str(t)]['design2']['ROC_AUC']-metrics[str(t)]['design1']['ROC_AUC'] for t in TARGETS]))
        # Fixed reference index belongs to the old risk run; map by request keys.
        source=old_load('risk',cap)[0].reset_index(names='row_index')
        ref=pd.concat([pd.read_parquet(ROOT/f'results/design2/scale_followup_01/references_cap{cap}/m{t}.parquet') for t in TARGETS])
        ref=source[['row_index']+keys].merge(ref,on='row_index',validate='one_to_one')
        checked=f.merge(ref[keys+['fresh_current']],on=keys,validate='one_to_one')
        for m in ['design1','design2']:checked['err_'+m]=(checked[m]-checked.fresh_current).abs()
        err=checked.groupby(['uid','target'])[['err_design1','err_design2']].max().reset_index()
        err=err.merge(d[['uid','target','reason']],on=['uid','target'],validate='one_to_one')
        failures=[]
        for tau in [.1,.5,1.]:
            row=dict(threshold=tau)
            for m in ['design1','design2']:
                g=err[['uid']].copy();g['bad']=(err['err_'+m]>tau).astype(float);g['excess']=np.maximum(err['err_'+m]-tau,0)
                row[m]=dict(failed_uids=err.loc[g.bad>0,'uid'].nunique(),rate=g.groupby('uid').bad.mean().mean(),excess=g.groupby('uid').excess.mean().mean())
            row['budget_deferred_severe_states']=int(((err.reason=='budget_deferred')&(err.err_design2>tau)).sum())
            failures.append(row)
        normal=[];fm=f.merge(meta[meta.cap==cap],on=['uid','target'],validate='many_to_one')
        for t,g in fm[fm.routine].groupby('target'):
            normal.append(dict(target=int(t),rows=len(g),uids=g.uid.nunique(),auc_diff=binary_metrics(g.label,g.design2)['ROC_AUC']-binary_metrics(g.label,g.design1)['ROC_AUC']))
        grouped=[]
        for cohort,uu in [('original',ids['original']),('extension',ids['extension'])]:
            ff=f[f.uid.isin(uu)]
            if len(ff):grouped.append(dict(cohort=cohort,intervals=boot(ff,[u for u in uids if u in uu])))
        cell_frame=f.copy();cell_frame['uid']=cell_frame.uid.map({u:i//128 for i,u in enumerate(allids)})
        cell_intervals=boot(cell_frame,sorted(cell_frame.uid.unique().tolist()))
        r=dict(mode=mode,cap=cap,users=users,requests=len(f),costs=total,components=components,old_risk_service=oldcost,
            service_reduction=1-total['design2']/oldcost,shared_preparation=shared,scales=scales,wallet_paid=paid,wallet_credit=credit,
            wallet_utilization=paid/credit,all_wallets_within_credit=all(s['spent']<=s['credit']+1e-3 for s in w),
            shared_read_hits=sum(s['checks'].get('first_read_reused',0) for s in w),actions=d.groupby(['target','reason']).size().reset_index(name='states').to_dict('records'),
            geometry_states=int(d.u.notna().sum()),committed_rebuilds=int((d.action=='rebuild').sum()),
            cost_dominant_states=int(d.get('cost_dominant',pd.Series(dtype=bool)).eq(True).sum()),
            first_request_candidate_histogram=d.actual_queries.dropna().value_counts().sort_index().to_dict(),
            auc_diff=aucdiff,per_edge=metrics,intervals=boot(f,uids),cell_intervals=cell_intervals,cell_count=users//128,
            cohorts=grouped,failures=failures,normal=normal,
            reference_rows=len(checked),reference_uids=checked.uid.nunique(),oracle_low_new_severe=int(((err.err_design1<=.1)&(err.err_design2>.5)).sum()))
        results.append(r);err.to_parquet(OUT/f'{path.name}_errors.parquet',index=False)
    save(OUT/'summary.json',dict(results=results,scope='Development; computation mechanisms fixed before output; all original preparation except unused inverse retained; no new calibration or FreshCurrent checkpoints/fit; Exact-All control branch replayed'))
    lines=['# 计算机制实验','', '| 方案/窗口/UID | 服务费用TF | 较risk节省 | 余额使用率 | 检查/重建/真实读复用 | AUC差pp |','| --- | --- | --- | --- | --- | --- |']
    for r in results:lines.append(f"| {r['mode']}/{r['cap']}/{r['users']} | {r['costs']['design2']/1e12:.4f} | {r['service_reduction']:.2%} | {r['wallet_utilization']:.2%} | {r['geometry_states']}/{r['committed_rebuilds']}/{r['shared_read_hits']} | {100*r['auc_diff']:+.4f} |")
    lines+=['','readshare不限制预算，其余额使用率仅作参照；reserve/fifo的每个单元均检查spent≤credit。','',
        '| 方案/窗口/UID | 3万总费用 | 10万 | 百万 | 主0.5失败UID D1→D2 | 预算推迟严重状态 |','| --- | --- | --- | --- | --- | --- |']
    for r in results:
        e=r['failures'][1];lines.append(f"| {r['mode']}/{r['cap']}/{r['users']} | "+' | '.join(f"{x['total_ratio']:.2%}" for x in r['scales'])+f" | {e['design1']['failed_uids']}→{e['design2']['failed_uids']} | {e['budget_deferred_severe_states']} |")
    lines+=['','费用分子含D1、普通维护及完整旧准备，仅移除实际不用的逆准备；FIFO不需要检测准备。短压力外推仅对应全部短窗口的受控负载，不能替代自然人口成本。',
        '完整0.1/0.5/1.0、全部发布与两个半组区间、正常组和预算漏检见summary.json；实际请求检测不宣称保持16候选判定。额度相同不等于支出严格相同。']
    lines+=['共享余额使单元内UID结果相互依赖：cell_intervals按固定128UID单元整体重采样；512仅4单元，2048仅16单元。普通UID区间仅条件于本次已执行轨迹，不能作为独立策略确认，亦不代替训练种子重复。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps([{k:r[k] for k in ['mode','cap','users','service_reduction','aucdiff'] if k in r} for r in results]))

if __name__=='__main__':main()
