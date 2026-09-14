"""Actual continuous feedback quality and a common lifetime FLOPs ledger."""

import json

import numpy as np
import pandas as pd
from design.data import ROOT
from design.report_native_base import ranked_auc, weighted_auc
from design.run import write_json
from design2.audit_budget import finite_metrics
from design2.run_lifecycle import METHODS, CONFIG, sha
from hstu_kvcache.evaluation.binary_metrics import binary_metrics

RUN=ROOT/'results/design2/lifecycle_01'
OUT=ROOT/'results/design2/analysis/lifecycle_01'
TARGETS=(1,3,4,5)


def quality(frame):
    return {name:binary_metrics(frame.label.to_numpy(),frame[name].to_numpy()) for name in METHODS}


def paired(raw,uids,repetitions=1000):
    lookup={u:i for i,u in enumerate(uids)}
    ranks={}
    for t in TARGETS:
        f=raw[raw.target==t]
        for name in METHODS:
            ranks[t,name]=ranked_auc(f.label.to_numpy(),f[name].to_numpy(),f.uid.map(lookup).to_numpy())
    rng=np.random.default_rng(17); draws=[]
    for _ in range(repetitions):
        m=np.bincount(rng.integers(len(uids),size=len(uids)),minlength=len(uids))
        draws.append([[weighted_auc(ranks[t,name],m) for name in METHODS] for t in TARGETS])
    arr=np.array(draws); valid=np.isfinite(arr).all((1,2)); arr=arr[valid]
    records=[]
    for left,right in [('design2','design1'),('design2','reuse'),('design2','exact'),('design1','reuse'),('exact','reuse')]:
        delta=arr[:,:,METHODS.index(left)]-arr[:,:,METHODS.index(right)]
        for j,t in enumerate(TARGETS):records.append(dict(target=t,left=left,right=right,interval=np.quantile(delta[:,j],[.025,.975]).tolist()))
        records.append(dict(target='equal_edge',left=left,right=right,interval=np.quantile(delta.mean(1),[.025,.975]).tolist()))
    return dict(draws=repetitions,valid=int(valid.sum()),seed=17,users=len(uids),records=records,
        scope='Paired UID resampling across all four edges, includes no-feedbackUID; one model seed and repeatedly opened development set, not confirmation.')


def main():
    assert not OUT.exists(),'Preserve completed analysis.'
    queue=json.loads((RUN/'population/summary.json').read_text()); assert queue['status']=='complete'
    frames=[]; costs=[]; decisions=[]; states=[]; users=[]; summaries=[]
    for j in queue['jobs']:
        assert j['exit']==0
        p=RUN/j['name']; cfg=json.loads((p/'configuration.json').read_text())
        for file,digest in cfg['hashes'].items():assert sha(ROOT/file)==digest,(p,file)
        users.extend(cfg['uids'])
        s=json.loads((p/'summary.json').read_text()); assert s['status']=='complete'; summaries.append(s)
        frames.append(pd.read_parquet(p/'quality_raw.parquet')); costs.append(pd.read_parquet(p/'costs.parquet'))
        decisions.append(pd.read_parquet(p/'decisions.parquet')); states.extend(json.loads((p/'state_counts.json').read_text()))
    protocol=json.loads((ROOT/'configs/design2/detection_01.json').read_text()); uids=protocol['groups']['development']
    assert len(users)==len(set(users))==1024 and set(users)==set(uids)
    raw=pd.concat(frames,ignore_index=True); charges=pd.concat(costs,ignore_index=True).fillna(0)
    actions=pd.concat(decisions,ignore_index=True); counts=pd.DataFrame(states)
    assert len(actions)==4*1024 and not actions.duplicated(['uid','target']).any()
    assert len(counts)==5*1024 and not counts.duplicated(['uid','target']).any()
    assert not raw.duplicated(['target','request_id']).any() and np.isfinite(raw[list(METHODS)]).all().all()
    assert len(raw)==int(counts.requests.sum())
    expected_raw=sum(s['requests'] for s in summaries); assert len(raw)==expected_raw
    q={int(t):dict(users=int(f.uid.nunique()),requests=len(f),**quality(f)) for t,f in raw.groupby('target')}
    means={name:float(np.mean([q[t][name]['ROC_AUC'] for t in TARGETS])) for name in METHODS}
    boot=paired(raw,uids)
    joined=raw[raw.target.isin(TARGETS)].merge(actions[['uid','target','action','reason']],on=['uid','target'],validate='many_to_one')
    followup=[]
    for action in ('rebuild','continue'):
        base=joined[joined.action==action]
        for label,lo,hi in [('zero_writes',0,0),('writes1_127',1,127),('writes128_1023',128,1023),('writes1024plus',1024,10**12)]:
            f=base[base.writes_since_release.between(lo,hi)]
            for t,g in f.groupby('target'):
                followup.append(dict(action=action,band=label,target=int(t),users=int(g.uid.nunique()),requests=len(g),
                    metrics=quality(g),mean_abs_d1_exact=float((g.design1-g.exact).abs().mean()),
                    mean_abs_d2_exact=float((g.design2-g.exact).abs().mean())))
    prep=json.loads((RUN/'preparation/summary.json').read_text()); assert prep['status']=='complete'
    excluded={'uid','target','method','rebuild_dependency_closed','validation_extra_geometry'}
    cost_columns=[c for c in charges if c not in excluded]
    totals={name:dict(charges[charges.method==name][cost_columns].sum()) for name in METHODS}
    pop={name:float(sum(values.values())) for name,values in totals.items()}
    shared=dict(reuse=0.,exact=0.,design1=prep['design1_shared'],design2=prep['design2_shared'])
    closed_exact=float(charges.loc[charges.method=='exact','rebuild_dependency_closed'].sum())
    closed_d2=float(charges.loc[charges.method=='design2','rebuild_dependency_closed'].sum())
    literal_d2=totals['design2'].get('rebuild_literal',0.)
    scales=[]
    for n in (1024,30000,100000,1000000):
        methods={name:shared[name]+pop[name]*n/1024 for name in METHODS}
        den=closed_exact*n/1024
        scales.append(dict(users=n,incremental_FLOPs=methods,exact_dependency_closed_FLOPs=den,
            ratio_actual={name:methods[name]/methods['exact'] for name in ('design1','design2')},
            ratio_conservative={name:methods[name]/den for name in ('design1','design2')},
            note='Fixed this UID mix/event/request intensity across all five windows; shared fit512/256 costs fixed. Extrapolation only; no new population quality evidence.'))
    break_even={}
    for name in ('design1','design2'):
        for label,e in [('actual',pop['exact']/1024),('conservative',closed_exact/1024)]:
            a=pop[name]/1024
            break_even[f'{name}_{label}']=shared[name]/(e-a) if e>a else None
    actions_by_target=[]
    for t,f in actions.groupby('target'):
        actions_by_target.append(dict(target=int(t),users=len(f),rebuilds=int((f.action=='rebuild').sum()),reasons=f.reason.value_counts().to_dict(),
            numerical_fallbacks=int(f.numerical_fallback.sum()),previous_rebuild_users=int(f.previous_rebuild.notna().sum())))
    common={k:sum(s['common_flops'].get(k,0.) for s in states) for k in ('native_append','native_read')}
    result=dict(status='complete',users=1024,real_requests=len(raw),four_edge_requests=int(raw.target.isin(TARGETS).sum()),
        real_events=int(counts.events.sum()),per_edge=q,equal_edge_auc=means,bootstrap=boot,actions=actions_by_target,
        action_followup=followup,cost=dict(shared=shared,population=pop,population_components=totals,scales=scales,
            break_even_users=break_even,common_native_FLOPs_per_branch=common,exact_dependency_closed=closed_exact,
            design2_dependency_closed_rebuild_alternative=closed_d2,design2_literal_rebuild=literal_d2,
            scope='Main actual prefix graph, including native-model overhead; conservative ratio keeps all method charges but replaces Exact denominator by cheaper KV dependency closure. Common M0/read/native append same shapes cancel; four method totals are counterfactual alternatives, never added.'),
        quality_delta_d2_d1=means['design2']-means['design1'],
        positive_d2_vs_d1=means['design2']>means['design1'],
        no_feedback_users={str(t):1024-q[t]['users'] for t in q},
        duration_seconds=queue['elapsed_seconds'],peak_gpu_mib=max(s['peak_gpu_mib'] for s in summaries),
        frozen_detector=True,confirmation_read=False,new_calibration_teacher_outputs=0,
        preparation_summary=str((RUN/'preparation/summary.json').relative_to(ROOT)),
        interpretation='Actual chronological feedback only; M2 native transition diagnostic reported separately. No release-panel residual is counted as quality. Exact means release rebuild then native trajectory, not fresh per-request full history.')
    OUT.mkdir(parents=True)
    write_json(OUT/'summary.json',finite_metrics(result))
    raw.to_parquet(OUT/'quality_raw.parquet',index=False); actions.to_parquet(OUT/'decisions.parquet',index=False)
    charges.to_parquet(OUT/'costs.parquet',index=False)
    pd.DataFrame([dict(target=t,**{m:q[t][m]['ROC_AUC'] for m in METHODS}) for t in q]).to_csv(OUT/'auc.csv',index=False)
    render(result)
    print(json.dumps(dict(auc=means,delta=result['quality_delta_d2_d1'],population_TF={k:v/1e12 for k,v in pop.items()},
        shared_TF={k:v/1e12 for k,v in shared.items()},actions=actions_by_target,scales=scales)))


def render(s):
    q=s['per_edge']; c=s['cost']; means=s['equal_edge_auc']
    def ci(target,left,right):
        return next(r['interval'] for r in s['bootstrap']['records'] if r['target']==target and r['left']==left and r['right']==right)
    lines=['# Design2第一版真实重建与连续服务闭环','',
        '从共同M0起点沿日231/245/259/273/287至301连续回放，原1024开发UID全部保留。M2只作native过渡，四条分支分别持有自己的KV；没有按边重置Parent。',
        '冻结C、H、12组a/b、阈值及225维检测。16候选只在发布决策时使用；以下质量来自真实带反馈请求。M5保留E14_partial，确认集/theta3未读。','',
        f"四边平均AUC：Design2={means['design2']:.8f}，Design1={means['design1']:.8f}，差{100*s['quality_delta_d2_d1']:+.4f}个百分点，配对UID95%区间{[round(100*x,4) for x in ci('equal_edge','design2','design1')]}。",'',
        '## 1. 状态语义与验证','',
        '先用组公式短路。仍需检查时，比对source/视图准备、16候选完整读取及225维计算，与Current完整KV构造及摘要重建的剩余费用；检查更贵就直接重建。数值界无法判定时，按同原则比较剩余原求解与重建。',
        '重建替换真实KV及摘要，旧视图和修订失效；重建锚点与发布年龄分别存储。当前模型继续native追加而不叠加C。下一模型变化（包括M2）取消当前模型锚点。发布通过不是整个窗口的误差保证。',
        '8UID canary完成84项发布/动作语义记录，包含真实追加和淘汰、即时参考一致、下一发布失效。首次仅结果落盘的整数字典键错误，保留失败日志并修复重跑；没有据质量改规则。人口阶段补计源范数与实际padding摘要成本，全部canary成本分支保持。',
        f"完整四GPU队列{s['duration_seconds']:.2f}s，峰值{s['peak_gpu_mib']/1024:.2f}GiB/GPU。真实事件{s['real_events']:,}，请求{s['real_requests']:,}（主四边{s['four_edge_requests']:,}）。",'',
        '## 2. 实际反馈质量','',
        '| 发布 | Reuse AUC | Design1 AUC | Exact-All AUC | Design2 AUC | D2−D1（百分点） |',
        '| --- | --- | --- | --- | --- | --- |']
    for t in (1,2,3,4,5):
        v=q[t] if t in q else q[str(t)]
        lines.append(f"| M{t}{'（过渡）' if t==2 else ''} | {v['reuse']['ROC_AUC']:.7f} | {v['design1']['ROC_AUC']:.7f} | {v['exact']['ROC_AUC']:.7f} | {v['design2']['ROC_AUC']:.7f} | {100*(v['design2']['ROC_AUC']-v['design1']['ROC_AUC']):+.4f} |")
    lines+=['', '四边汇总是逐边AUC等权平均，不能混合不同模型的分数再算AUC。配对bootstrap使用相同UID倍数贯穿四边并保留无反馈用户；只是一个训练seed的开发证据，不是独立确认。', '',
        '## 3. 实际决策及后续服务','',
        '| 发布 | 重建UID/1024 | 原因计数 | 已有以前重建记录UID |', '| --- | --- | --- | --- |']
    for r in s['actions']:lines.append(f"| M{r['target']} | {r['rebuilds']} | {r['reasons']} | {r['previous_rebuild_users']} |")
    lines+=['', '下面仅取本次发布选择重建的用户，观察至少有一次真实写入后的请求。比较的是整条策略分支，不能把这个条件子集的增益当成随机化动作因果效应。Exact-All是同发布重建后持续native的参考分支，距离不是实际标签质量的替代。','',
        '| 发布/写入带 | UID/请求 | D2−D1 AUC（百分点） | D1到Exact平均绝对logit差 | D2到Exact差 |', '| --- | --- | --- | --- | --- |']
    for r in s['action_followup']:
        if r['action']!='rebuild' or r['band']=='zero_writes':continue
        m=r['metrics']; left,right=m['design2']['ROC_AUC'],m['design1']['ROC_AUC']
        delta=f'{100*(left-right):+.4f}' if left is not None and right is not None else '未定义（缺少一类标签）'
        lines.append(f"| M{r['target']}/{r['band']} | {r['users']}/{r['requests']} | {delta} | {r['mean_abs_d1_exact']:.6f} | {r['mean_abs_d2_exact']:.6g} |")
    lines+=['','## 4. 本次生命周期增量费用','',
        'FMA=2，基本算术逐项计数；ELU/SiLU/rsqrt/sin/cos主约定每次折1操作。发布决策及主表采用当前实际完整prefix图；另用只生成KV的依赖闭包作为更严格的Exact分母。源摘要按实际padding到1024计，源范数、全部视图刷新、数值回退与重建后真实省去的修正均按调用收费。',
        '正常native追加/读取有相同调用形状，可抵消其普通计算，但分支的KV仍独立。共同M0也只作共同起点。表中各方法是替代方案，不能把四个方法的费用相加。','',
        '| 方法 | 本1024UID服务增量TFLOPs | 一次共享准备TFLOPs | 合计TFLOPs |', '| --- | --- | --- | --- |']
    for name in METHODS:lines.append(f"| {name} | {c['population'][name]/1e12:.6f} | {c['shared'][name]/1e12:.6f} | {(c['population'][name]+c['shared'][name])/1e12:.6f} |")
    lines+=['', '准备明细见preparation/summary.json：保留原Design1的C256及A64坐标/PCA依赖；追加512残余校准的真实源回放、逐发布Current教师、适配读取/全场景几何、交叉校准标量预算、H分解和逆准备。当前教师KV/输出在同UID/发布的替代场景之间复用。C拟合已经产生相同Gram，因此不重复收费Gram或重新采集C查询。',
        '新增512校准准备只由既有数据/时间戳及固定算法核算，本轮没有重做拟合或获取新校准教师。标量拟合有明确保守预算，8段source准备是上界；不是逐硬件指令测量。旧百分比没有参与本次计算。','',
        '| 外推用户 | Exact实际TF | Exact闭包TF | D1总TF | D2总TF | D2/实际Exact | D2/闭包Exact |', '| --- | --- | --- | --- | --- | --- | --- |']
    for r in c['scales']:
        f=r['incremental_FLOPs'];lines.append(f"| {r['users']} | {f['exact']/1e12:.3f} | {r['exact_dependency_closed_FLOPs']/1e12:.3f} | {f['design1']/1e12:.3f} | {f['design2']/1e12:.3f} | {r['ratio_actual']['design2']:.2%} | {r['ratio_conservative']['design2']:.2%} |")
    lines+=['',f"费用持平用户数：{c['break_even_users']}。外推固定本次用户长度分布、事件/评分强度及动作率，只摊薄共享准备，不是新规模质量结论。",'',
        '## 5. 研究判断','',
        f"本轮四边平均真实AUC是否优于Design1：{s['positive_d2_vs_d1']}。必须结合配对区间、各边方向及费用表判断，不能用即时面板归零替代这个结果。",
        '本轮没有复杂预算排序器或同费用动作对照，因此不能把已观察到的策略收益全归因于几何排序。下一步依据实际质量和费用结果决定；检测器继续冻结。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
