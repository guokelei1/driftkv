"""Score frozen lifecycle outputs against evaluation-only FreshCurrent requests."""

import json
import numpy as np
import pandas as pd

from design.data import ROOT
from design2.audit_benchmark import METHODS, STRATA, save
from design2.benchmark_reference import OUT, RAW, TARGETS

REPORT=ROOT/'results/design2/analysis/benchmark_01'


def table(g, threshold, method):
    error=g[method]
    out=dict(states=len(g),uids=int(g.uid.nunique()),failure_uids=int(g[error>threshold].uid.nunique()),
        severe_uid_equal=float(g.assign(v=error>threshold).groupby('uid').v.mean().mean()),
        excess_uid_equal=float(g.assign(v=np.maximum(error-threshold,0)).groupby('uid').v.mean().mean()),
        p95=float(error.quantile(.95)),p99=float(error.quantile(.99)))
    return out


def main():
    queue=json.loads((OUT/'queue.json').read_text());assert queue['status']=='complete'
    frames=[pd.read_parquet(OUT/f'm{t}.parquet') for t in TARGETS]
    teacher=pd.concat(frames);assert not teacher.row_index.duplicated().any()
    prep=json.loads((OUT/'preparation.json').read_text());assert len(teacher)==prep['rows']
    f=pd.read_parquet(RAW).reset_index(names='row_index').merge(teacher,on='row_index',validate='one_to_one')
    assert set(f.target)==set(TARGETS)
    meta=pd.read_parquet(REPORT/'members.parquet');meta=meta[meta.role=='development']
    f=f.merge(meta,on=['uid','target'],validate='many_to_one')
    assert (f['count']>=f.retained_count).all()
    errors=f[['uid','target']].copy()
    for m in METHODS: errors[m]=(f[m]-f.fresh_current).abs()
    state=errors.groupby(['uid','target'])[list(METHODS)].max().reset_index().merge(meta,on=['uid','target'],validate='one_to_one')
    rows=[]
    for name in STRATA:
        g=state[state[name]]
        for tau in (.1,.5,1.):
            for m in METHODS: rows.append(dict(stratum=name,threshold=tau,method=m,**table(g,tau,m)))
    low=state[state.design1<=.1]
    stable=dict(states=len(low),uids=int(low.uid.nunique()),new_severe_states=int((low.deferred>.5).sum()),
                new_severe_state_rate=float((low.deferred>.5).mean()),
                new_severe_uids=int(low[low.deferred>.5].uid.nunique()))
    # UID bootstrap, same multiplicity for all targets and both methods.
    cohort=prep['users'];lookup={u:i for i,u in enumerate(cohort)}
    rng=np.random.default_rng(17)
    draws=np.array([np.bincount(rng.integers(len(cohort),size=len(cohort)),minlength=len(cohort)) for _ in range(1000)])
    intervals=[]
    for name in ('all','challenge','routine'):
        g=state[state[name]]
        for tau in (.1,.5,1.):
            for metric in ('severe','excess'):
                v=g[['uid']].copy()
                for m in ('design1','deferred'):
                    v[m]=(g[m]>tau).astype(float) if metric=='severe' else np.maximum(g[m]-tau,0)
                v=v.groupby('uid')[['design1','deferred']].mean()
                idx=np.array([lookup[u] for u in v.index]);w=draws[:,idx]
                dif=w@(v.deferred-v.design1).to_numpy()/w.sum(axis=1)
                intervals.append(dict(stratum=name,threshold=tau,metric=metric,delta_ci=np.quantile(dif,[.025,.975]).tolist()))
    after=f[f.deferred_anchored_native & (f.deferred_writes_since_rebuild>0)]
    follow=[]
    for t,g in after.groupby('target'):
        e=pd.DataFrame({'uid':g.uid,'design1':(g.design1-g.fresh_current).abs(),'deferred':(g.deferred-g.fresh_current).abs()}).groupby('uid').max()
        follow.append(dict(target=int(t),uids=len(e),rows=len(g),design1_mean_uid_max=float(e.design1.mean()),deferred_mean_uid_max=float(e.deferred.mean())))
    teacher_cost=sum(json.loads((OUT/f'm{t}.json').read_text())['evaluation_teacher_FLOPs'] for t in TARGETS)
    near={str(tau):int(sum((np.abs(errors[m]-tau)<1e-4).sum() for m in METHODS)) for tau in (.1,.5,1.)}
    result=dict(status='complete',reference='FreshCurrent immediately before each fixed actual request checkpoint; evaluation only',
        checkpoint_rule=prep['protocol'],checkpoints=prep['checkpoints'],rows=len(f),uids=int(f.uid.nunique()),
        uid_target_states=len(state),metrics=rows,intervals=intervals,oracle_low_error=stable,after_rebuild_writes=follow,
        evaluation_teacher_FLOPs=teacher_cost,numerically_near_threshold_rows=near,
        quality_scope='UID then window equal weighting of selected real-request max logit errors; not all-request error or AUC; real policy scores unchanged',
        population_seconds=queue['seconds'],confirmation_read=False,
        adequacy='At least30 primary failureUID required. This development measurement does not establish independent generalization.')
    save(REPORT/'fresh_current_summary.json',result)
    f.to_parquet(REPORT/'fresh_current_requests.parquet',index=False)
    state.to_parquet(REPORT/'fresh_current_states.parquet',index=False)
    lines=['# Benchmark：预定真实请求上的FreshCurrent保护评价','',
        '不改变任何策略。从六条已完成轨迹的相同真实请求取输出，在该时刻用合法保留历史新建Current参考；教师只进入离线评价。',
        f"{prep['checkpoints']}个去重检查点、{len(f)}条真实候选评分、{f.uid.nunique()}名有检查点的UID、{len(state)}个UID×窗口。原1024UID仍是全人口，无请求窗口未伪造质量。",
        '检查点为窗口首请求，以及真实写入首次达到1/128/1024/6144后的首请求。统计单位为每UID×窗口的所选请求最大误差，再对UID内窗口平均、UID等权；不称全窗口逐请求误差。','',
        '| 组/尺度 | D1严重UID | D1严重率 | 延迟D2严重率 | 严重率相对减少 | D1超额质量 | D2超额质量 |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for name in STRATA:
        for tau in (.1,.5,1.):
            a=next(r for r in rows if r['stratum']==name and r['threshold']==tau and r['method']=='design1')
            b=next(r for r in rows if r['stratum']==name and r['threshold']==tau and r['method']=='deferred')
            recovery='无基线失败' if a['severe_uid_equal']==0 else f"{1-b['severe_uid_equal']/a['severe_uid_equal']:.2%}"
            lines.append(f"| {name}/{tau} | {a['failure_uids']} | {a['severe_uid_equal']:.3%} | {b['severe_uid_equal']:.3%} | {recovery} | {a['excess_uid_equal']:.6f} | {b['excess_uid_equal']:.6f} |")
    lines+=['',f"D1低误差≤0.1的{stable['states']}个状态中，新引入>0.5严重误差{stable['new_severe_states']}个（{stable['new_severe_state_rate']:.3%}）。这是评测oracle分组，不可作为线上输入。",'',
        '| 重建后确有新增写入/发布 | UID/评分行 | D1平均UID最大误差 | 延迟D2平均UID最大误差 |',
        '| --- | --- | --- | --- |']
    for r in follow: lines.append(f"| M{r['target']} | {r['uids']}/{r['rows']} | {r['design1_mean_uid_max']:.6f} | {r['deferred_mean_uid_max']:.6f} |")
    lines+=['','该后续子集是描述证据，不是随机化动作效果。完整六方法、三个尺度、UID配对区间和数值近阈值计数保存在JSON/Parquet。',
        f"评测教师{teacher_cost/1e12:.6f} TF，四GPU队列{queue['seconds']:.2f}s；这是新产生的离线参考费用，不扣减也不重复加入方法的服务账本。原校准教师仍完整计入共享准备。",
        '主尺度失败独立UID不足30时，本版保护门槛仍标证据不足，不通过改尺度、扩大小数显著性或从确认集挑失败来过关。']
    (REPORT/'fresh_current_report.md').write_text('\n'.join(lines)+'\n')
    baseline=json.loads((REPORT/'summary.json').read_text())
    a=next(r for r in rows if r['stratum']=='challenge' and r['threshold']==.5 and r['method']=='design1')
    b=next(r for r in rows if r['stratum']=='challenge' and r['threshold']==.5 and r['method']=='deferred')
    gates=dict(
        primary_protection=dict(status='insufficient_evidence',failure_uids=a['failure_uids'],required=30,
            severe_relative_reduction=1-b['severe_uid_equal']/a['severe_uid_equal'],
            excess_relative_reduction=1-b['excess_uid_equal']/a['excess_uid_equal']),
        routine_service_cost=dict(status='fails',observed_per_edge=[r['extra_service_ratio'] for r in baseline['coverage'] if r['stratum']=='routine'],target=.02),
        routine_quality=dict(status='equal_edge_auc_noninferiority_supported_on_reused_development; M5_edge_not_established',
            intervals=[r for r in baseline['auc_intervals'] if r['stratum']=='routine']),
        oracle_low_error=dict(status='zero_observed_new_failures; not_a_probability_guarantee',**stable),
        cost_envelope=dict(status='fails',points=baseline['cost_curves']),
        missing=['controlled32/128 full independent lifetime workload','matched-total-budget rebuild controls','adequate primary failureUID and later independent validation'])
    save(REPORT/'verdict.json',gates)
    print(json.dumps(dict(rows=len(f),states=len(state),oracle_low=stable,primary=[r for r in rows if r['threshold']==.5 and r['stratum']=='all'])))


if __name__=='__main__': main()
