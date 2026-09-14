"""Normal-service guards and true-request FreshCurrent errors for follow-up runs."""
import json
import numpy as np
import pandas as pd
from design.data import ROOT
from design2.audit_benchmark import save
from design2.report_scale_followup import OUT,load,boot,TARGETS
from hstu_kvcache.evaluation.binary_metrics import binary_metrics


def loss_intervals(f,users):
    lookup={u:i for i,u in enumerate(users)};rng=np.random.default_rng(17)
    weights=np.array([np.bincount(rng.integers(len(users),size=len(users)),minlength=len(users)) for _ in range(1000)])
    result=[]
    for t,g in f.groupby('target'):
        idx=g.uid.map(lookup).to_numpy();label=g.label.to_numpy()
        a=np.logaddexp(0,g.design1)-label*g.design1;b=np.logaddexp(0,g.design2)-label*g.design2
        old=np.bincount(idx,weights=a,minlength=len(users));delta=np.bincount(idx,weights=b-a,minlength=len(users))
        ratio=(weights@delta)/(weights@old)
        result.append(dict(target=int(t),relative_logloss_ci=np.nanquantile(ratio,[.025,.975]).tolist()))
    return result


def main():
    meta=pd.read_parquet(OUT/'members.parquet');reports=[];normal=[];failures=[];resource=[];subsequent=[]
    for cap in (1024,32,128):
        root=ROOT/f'results/design2/scale_followup_01/references_cap{cap}'
        ref=pd.concat([pd.read_parquet(root/f'm{t}.parquet') for t in TARGETS])
        assert not ref.row_index.duplicated().any()
        source=load('risk',cap)[0].reset_index(names='row_index')
        ref=source[['row_index','uid','target','request_id']].merge(ref,on='row_index',validate='one_to_one').drop(columns='row_index')
        resource.append(dict(cap=cap,teacher_FLOPs=sum(json.loads((root/f'm{t}.json').read_text())['evaluation_teacher_FLOPs'] for t in TARGETS),rows=len(ref)))
        for policy in (('risk','demand','hash30','guard') if cap==1024 else ('risk',)):
            if not (ROOT/f'results/design2/scale_followup_01/{policy}_cap{cap}/summary.json').exists():continue
            f,c,d,_=load(policy,cap)
            m=meta[meta.cap==cap].drop(columns='cap')
            f=f.merge(m,on=['uid','target'],validate='many_to_one')
            # M2 intentionally has no group row and remains in the task report only.
            selected=f.merge(ref,on=['uid','target','request_id'],validate='one_to_one')
            for col in ('design1','design2','exact'):selected['error_'+col]=(selected[col]-selected.fresh_current).abs()
            e=selected.groupby(['uid','target']).agg(d1=('error_design1','max'),d2=('error_design2','max'),exact=('error_exact','max')).reset_index().merge(m,on=['uid','target'],validate='one_to_one')
            for name,mask in [('all',e.uid.notna()),('challenge',e.challenge),('routine',e.routine),('short',e.short)]:
                g=e[mask]
                if g.empty:continue
                for tau in (.1,.5,1.):
                    def stat(col):
                        h=g[['uid']].copy();h['bad']=(g[col]>tau).astype(float);h['excess']=np.maximum(g[col]-tau,0)
                        ordered=g.sort_values(col);weights=1/ordered.groupby('uid')[col].transform('size')/ordered.uid.nunique()
                        quantiles=np.interp([.95,.99],weights.cumsum(),ordered[col])
                        return dict(rate=h.groupby('uid').bad.mean().mean(),excess=h.groupby('uid').excess.mean().mean(),failed_uids=g[g[col]>tau].uid.nunique(),
                                    uid_weighted_p95=quantiles[0],uid_weighted_p99=quantiles[1])
                    reports.append(dict(cap=cap,policy=policy,stratum=name,threshold=tau,uids=g.uid.nunique(),states=len(g),d1=stat('d1'),d2=stat('d2')))
            low=e[e.d1<=.1]
            reports.append(dict(cap=cap,policy=policy,stratum='oracle_low',states=len(low),new_severe=int((low.d2>.5).sum()),
                                mean_absolute_shift=float((selected.design2-selected.design1).abs().mean())))
            failures.extend(e[(e.d1>.5)|(e.d2>.5)].assign(policy=policy,cap=cap).to_dict('records'))
            # Every tested policy acts at the first real request of its window.
            starts=f.sort_values('timestamp').groupby(['uid','target']).first().reset_index()[['uid','target','timestamp','writes_since_release']]
            for t,h in selected.merge(starts,on=['uid','target'],suffixes=('','_first')).groupby('target'):
                h=h[(h.last_rebuild_target==t)&(h.timestamp>h.timestamp_first)&(h.writes_since_release>h.writes_since_release_first)]
                if len(h):
                    subsequent.append(dict(cap=cap,policy=policy,target=int(t),rows=len(h),uids=h.uid.nunique(),
                        d1_mean_abs=float(h.error_design1.mean()),d2_mean_abs=float(h.error_design2.mean()),
                        d1_severe_rows=int((h.error_design1>.5).sum()),d2_severe_rows=int((h.error_design2>.5).sum()),
                        scope='Selected committed-state checkpoints with later real writes; descriptive, not randomized action effects'))
            if cap==1024:
                for cohort in ('all','original','extension'):
                    g=f[f.routine & ((f.cohort==cohort) if cohort!='all' else True)]
                    users=m[m.routine & ((m.cohort==cohort) if cohort!='all' else True)].uid.unique().tolist()
                    if len(g)==0:continue
                    metrics={str(t):{v:binary_metrics(h.label,h[v]) for v in ('design1','design2','exact')} for t,h in g.groupby('target')}
                    cc=c.merge(m,on=['uid','target'],validate='many_to_one')
                    cc=cc[cc.routine & ((cc.cohort==cohort) if cohort!='all' else True)]
                    cols=[k for k in c if k not in ('uid','target','method')]
                    extra=float(cc[cc.method=='design2'][cols].sum().sum()-cc[cc.method=='design1'][cols].sum().sum())
                    denom=float(cc[cc.method=='exact'][cols].sum().sum())
                    dc=cc[cc.method=='design2']
                    attempts=int((dc.get('rebuild_tiled',pd.Series(dtype=float))>0).sum())
                    commits=int((dc.get('rebuild_summary',pd.Series(dtype=float))>0).sum())
                    normal.append(dict(policy=policy,cohort=cohort,requests=len(g),uids=g.uid.nunique(),metrics=metrics,intervals=boot(g,users),
                        logloss_intervals=loss_intervals(g,users),extra_service_ratio=extra/denom,extra_service_FLOPs=extra,
                        windows=len(dc),rebuild_attempts=attempts,committed_rebuilds=commits,commit_window_fraction=commits/len(dc)))
    save(OUT/'benchmark.json',dict(errors=reports,normal=normal,severe_cases=failures,evaluation_resources=resource,
        subsequent_write_checkpoints=subsequent,
        notes='FreshCurrent only on predetermined actual request checkpoints; no all-window safety claim; zero observed new failures is not a probability bound'))
    lines=['# 中规模Benchmark：正常服务与困难状态','',
        '| cap/策略/组/尺度 | UID/状态 | D1失败UID | D1超阈值率 | D2超阈值率 |',
        '| --- | --- | --- | --- | --- |']
    for r in reports:
        if r['stratum'] in ('all','challenge','routine','short'):
            lines.append(f"| {r['cap']}/{r['policy']}/{r['stratum']}/{r['threshold']} | {r['uids']}/{r['states']} | {r['d1']['failed_uids']} | {r['d1']['rate']:.3%} | {r['d2']['rate']:.3%} |")
    lines+=['','超阈值率为UID内窗口平均、再UID等权。0.5是主尺度，0.1/1.0辅助；缺少30个主失败UID时不宣布普遍保护。', '',
        '| 正常组策略/人口 | M1差 | M3差 | M4差 | M5差 | 四边差95%区间 |',
        '| --- | --- | --- | --- | --- | --- |']
    for r in normal:
        values=[100*(r['metrics'][str(t)]['design2']['ROC_AUC']-r['metrics'][str(t)]['design1']['ROC_AUC']) for t in TARGETS]
        ci=next((i['interval'] for i in r['intervals'] if i['target']=='equal_edge'),None)
        lines.append(f"| {r['policy']}/{r['cohort']} | "+' | '.join(f'{v:+.4f}' for v in values)+f" | {None if ci is None else [round(100*v,4) for v in ci]} |")
    lines+=['','AUC差单位为百分点，常规组是预先定义的结构补集，不是根据D2结果挑出的安全用户。窗口32/128全是短历史压力组，没有正常组，不能伪造正常组对照。',
        '完整失败案例、原低误差状态是否新引入严重错误、所有教师费用保存在benchmark.json。评测教师不进入服务决策，也不冒充校准成本节省。']
    lines+=['','正常组逐发布Exact-All的AUC/LogLoss也保存在JSON：它不是每请求FreshCurrent，且更接近Current并不保证该子组的标签指标改善。',
        'LogLoss的逐边UID bootstrap相对变化区间保存在logloss_intervals；不能仅以AUC点估计判断正常组非劣。']
    lines+=['','| 正常组策略/人口 | D2−D1额外服务/本组Exact闭包 | 重算尝试/提交/全部组窗口 |','| --- | --- | --- |']
    for r in normal:lines.append(f"| {r['policy']}/{r['cohort']} | {r['extra_service_ratio']:.2%} | {r['rebuild_attempts']}/{r['committed_rebuilds']}/{r['windows']} |")
    lines+=['','续写检查点要求本次发布已真正提交重建，严格晚于提交且新增至少一次真实写入；其残余见JSON，不能把动作选择后的子集解释为随机化收益。']
    (OUT/'benchmark_report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(normal=normal,resources=resource),default=lambda v:v.item()))

if __name__=='__main__':main()
