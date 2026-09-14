"""Moderate-scale policies, workload strata, UID uncertainty and honest cost curves."""
import json,re
import numpy as np
import pandas as pd
from design.data import ROOT
from design.report_native_base import ranked_auc,weighted_auc
from design.report_native_flops import prefix_literal,value
from design2.audit_benchmark import save
from hstu_kvcache.design2.rebuild import tiled_flops
from hstu_kvcache.evaluation.binary_metrics import binary_metrics

RUN=ROOT/'results/design2/scale_followup_01'
OUT=ROOT/'results/design2/analysis/scale_followup_01'
TARGETS=(1,3,4,5)


def load(policy,cap):
    p=RUN/f'{policy}_cap{cap}';s=json.loads((p/'summary.json').read_text());assert s['status']=='complete'
    raw=[];cost=[];decisions=[];states=[]
    for j in sorted(s['jobs'],key=lambda j:j['offset']):
        assert j['exit']==0
        d=p/j['name'];raw.append(pd.read_parquet(d/'quality_raw.parquet'));cost.append(pd.read_parquet(d/'costs.parquet'))
        decisions.append(pd.read_parquet(d/'decisions.parquet'));states.extend(json.loads((d/'state_counts.json').read_text()))
    return pd.concat(raw,ignore_index=True),pd.concat(cost,ignore_index=True).fillna(0),pd.concat(decisions,ignore_index=True),pd.DataFrame(states)


def preparation():
    old=json.loads((ROOT/'results/design2/lifecycle_01/preparation/summary.json').read_text())
    counts=json.loads((ROOT/'results/design/analysis/native_flops_01/calibration_counts.json').read_text())
    d1=0
    for c in counts.values():
        d1+=sum(v*(value(prefix_literal(int(n)))-tiled_flops(int(n))) for hist in c['builds'].values() for n,v in hist.items())
    d2=0
    for key,count in old['source_replay_histogram'].items():
        m=re.search(r'_prefix_n(\d+)$',key)
        if m:
            n=int(m.group(1));d2+=count*(value(prefix_literal(n))-tiled_flops(n))
    return dict(original_D1=old['design1_shared'],original_D2=old['design2_shared'],
        tiled_D1=old['design1_shared']-d1,tiled_D2=old['design2_shared']-d1-d2,
        D1_prefix_saving=d1,D2_residual_prefix_saving=d2,
        scope='Theoretical reuse of validated same-KV tiled prefix algorithm for existing preparation counts. Calibration fitting/teachers/replay/H unchanged; actual old preparation not rerun, solver coefficients not changed.')


def boot(f,users):
    lookup={u:i for i,u in enumerate(users)};rng=np.random.default_rng(17)
    mult=np.array([np.bincount(rng.integers(len(users),size=len(users)),minlength=len(users)) for _ in range(1000)])
    vals=[];out=[]
    for t in TARGETS:
        g=f[f.target==t];idx=g.uid.map(lookup).to_numpy()
        if len(g)==0 or g.label.nunique()<2:continue
        ranks=[ranked_auc(g.label.to_numpy(),g[m].to_numpy(),idx) for m in ('design1','design2')]
        a=np.array([weighted_auc(ranks[1],w)-weighted_auc(ranks[0],w) for w in mult]);vals.append(a)
        out.append(dict(target=t,interval=np.nanquantile(a,[.025,.975]).tolist()))
    if len(vals)==4:out.append(dict(target='equal_edge',interval=np.nanquantile(np.mean(vals,axis=0),[.025,.975]).tolist()))
    return out


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    ids=json.loads((ROOT/'configs/design2/scale_followup_01_uids.json').read_text())
    prep=preparation();reports=[];scales=[]
    original=None
    for cap in (1024,32,128):
        for policy in (('risk','demand','hash30','guard') if cap==1024 else ('risk',)):
            path=RUN/f'{policy}_cap{cap}'/'summary.json'
            if not path.exists():continue
            f,c,d,states=load(policy,cap)
            if cap==1024:
                keys=['uid','target','request_id']
                ordered=f.sort_values(keys)
                if original is None:original=ordered
                else:
                    np.testing.assert_array_equal(original[keys],ordered[keys])
                    np.testing.assert_allclose(original[['reuse','design1','exact']],ordered[['reuse','design1','exact']],atol=0,rtol=0)
            for cohort,users in ([('all',ids['original']+ids['extension']),('original',ids['original']),('extension',ids['extension'])] if cap==1024 else [('stress',ids['original'][:512])]):
                g=f[f.uid.isin(users)];cc=c[c.uid.isin(users)];dd=d[d.uid.isin(users)]
                metrics={str(t):{m:binary_metrics(v.label.to_numpy(),v[m].to_numpy()) for m in ('reuse','design1','exact','design2')} for t,v in g.groupby('target')}
                means={m:float(np.mean([metrics[str(t)][m]['ROC_AUC'] for t in TARGETS])) for m in ('reuse','design1','exact','design2')}
                cols=[k for k in c if k not in ('uid','target','method')]
                population={m:float(cc[cc.method==m][cols].sum().sum()) for m in ('design1','design2','exact')}
                components=cc[cc.method=='design2'][cols].sum().to_dict()
                r=dict(cap=cap,policy=policy,cohort=cohort,users=len(users),requests=len(g),means=means,per_edge=metrics,
                    uid_intervals=boot(g,users),population=population,components=components,
                    actions=dd.groupby(['target','action','reason']).size().reset_index(name='states').to_dict('records'),
                    rebuild_attempts=int((cc[cc.method=='design2'].get('rebuild_tiled',pd.Series(dtype=float))>0).sum()),
                    committed_rebuilds=int((cc[cc.method=='design2'].get('rebuild_summary',pd.Series(dtype=float))>0).sum()))
                reports.append(r)
                if cohort=='all':
                    for n in (30000,100000,1000000):
                        den=population['exact']*n/len(users)
                        shared=prep['original_D1' if policy=='hash30' else 'original_D2']
                        tiled=prep['tiled_D1' if policy=='hash30' else 'tiled_D2']
                        scales.append(dict(policy=policy,users=n,exact_closed_TF=den/1e12,
                            original_preparation_ratio=(shared+population['design2']*n/len(users))/den,
                            tiled_preparation_ratio=(tiled+population['design2']*n/len(users))/den,
                            service_asymptote=population['design2']/population['exact']))
            f.to_parquet(OUT/f'{policy}_cap{cap}_raw.parquet',index=False)
    contrasts=[]
    for left,right in [('risk','hash30'),('guard','hash30'),('guard','risk')]:
        if not (RUN/f'{left}_cap1024/summary.json').exists() or not (RUN/f'{right}_cap1024/summary.json').exists():continue
        lf=load(left,1024)[0];rf=load(right,1024)[0];keys=['uid','target','request_id']
        lf=lf.sort_values(keys).reset_index(drop=True);rf=rf.sort_values(keys).reset_index(drop=True)
        np.testing.assert_array_equal(lf[keys],rf[keys]);lf['design1']=rf.design2
        contrasts.append(dict(left=left,right=right,intervals=boot(lf,ids['original']+ids['extension'])))
    s=dict(reports=reports,scales=scales,preparation=prep,contrasts=contrasts,confirmation_read=False,
        scope='Development, original versus extension separate; stress caps not natural prevalence; fixed hash30 not automatically exact-budget matched')
    save(OUT/'summary.json',s);render(s)
    print(json.dumps(dict(results=[{k:r[k] for k in ('cap','policy','cohort','means')} for r in reports],scales=scales,prep=prep)))


def render(s):
    lines=['# 中规模闭环、短窗口与费用复核','',
        '冻结检测器；2048自然UID分原组与扩展组，短窗口32/128各原前512UID。首次消费检查、仅质量触发与固定哈希30%对照独立真实回放。','',
        '| cap/方案/组 | UID/请求 | D1 AUC | D2 AUC | 差（百分点） | UID95%区间（百分点） |',
        '| --- | --- | --- | --- | --- | --- |']
    for r in s['reports']:
        ci=next(x['interval'] for x in r['uid_intervals'] if x['target']=='equal_edge')
        lines.append(f"| {r['cap']}/{r['policy']}/{r['cohort']} | {r['users']}/{r['requests']} | {r['means']['design1']:.7f} | {r['means']['design2']:.7f} | {100*(r['means']['design2']-r['means']['design1']):+.4f} | {[round(100*x,4) for x in ci]} |")
    lines+=['','| cap/方案/组 | M1差 | M3差 | M4差 | M5差 |','| --- | --- | --- | --- | --- |']
    for r in s['reports']:
        vals=[100*(r['per_edge'][str(t)]['design2']['ROC_AUC']-r['per_edge'][str(t)]['design1']['ROC_AUC']) for t in TARGETS]
        lines.append(f"| {r['cap']}/{r['policy']}/{r['cohort']} | "+' | '.join(f'{v:+.4f}' for v in vals)+' |')
    lines+=['','单位为AUC百分点；M2单列保存在JSON，不进入四边均值；M5partial保留。扩展队列不是正式封存确认。', '',
        '| 方案/人口 | 原准备＋实际服务/Exact闭包 | 准备也用同KV分块计算（理论） | 服务渐近比例 |',
        '| --- | --- | --- | --- |']
    for r in s['scales']:lines.append(f"| {r['policy']}/{r['users']} | {r['original_preparation_ratio']:.2%} | {r['tiled_preparation_ratio']:.2%} | {r['service_asymptote']:.2%} |")
    lines+=['','准备改写是对已保留真实prefix长度计数重新收费，所用KV算法已canary验证；没有重新拟合，不能将其称为已重跑的完整准备实验。C/H/残余教师/真实回放等依赖仍收费。',
        '新人口的活跃度/长度组成可能与原1024不同；自然混合队列成本外推只对应本队列，不是3万/百万人口实测。固定哈希30%完整费用若不同，就只作费用—质量对照，不能冒充严格同预算。']
    lines+=['','| 方案/人口 | 付费重算尝试 | 实际提交重建 |','| --- | --- | --- |']
    for r in s['reports']:lines.append(f"| {r['policy']}/{r['cohort']}/cap{r['cap']} | {r['rebuild_attempts']} | {r['committed_rebuilds']} |")
    lines+=['','guard为发现正常M4退化后固定的一个额外备选：已付费重算后，只有当前实际请求差异超过原tau才提交，否则保留原KV；所有尝试与两次核对读取均收费，不是免费Exact。',
        '与哈希30%及risk的配对UID区间完整保存在summary.json；不因点估计更好就称显著或同预算优越。', '',
        '[正常组和FreshCurrent Benchmark](benchmark_report.md)']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
