"""Compare two fixed policy alternatives against the completed primary lifetime."""

import json

import numpy as np
import pandas as pd
from design.data import ROOT
from design.report_native_base import ranked_auc,weighted_auc
from design.run import write_json
from design2.audit_budget import finite_metrics
from design2.run_lifecycle import sha
from hstu_kvcache.evaluation.binary_metrics import binary_metrics

OUT=ROOT/'results/design2/analysis/lifecycle_variants_01'
METHODS=('reuse','design1','exact','immediate','cost_only','deferred')
TARGETS=(1,3,4,5)


def load(name):
    run=ROOT/f'results/design2/lifecycle_{name}_01'
    queue=json.loads((run/'population/summary.json').read_text());assert queue['status']=='complete'
    frames=[];costs=[];decisions=[];counts=[];uids=[]
    for j in queue['jobs']:
        assert j['exit']==0
        p=run/j['name']; cfg=json.loads((p/'configuration.json').read_text())
        assert cfg['variant']==name and cfg['variant_source_sha256']==sha(ROOT/f'scripts/design2/lifecycle_{name}.py')
        for file,digest in cfg['hashes'].items():assert sha(ROOT/file)==digest
        uids.extend(cfg['uids']);frames.append(pd.read_parquet(p/'quality_raw.parquet'))
        costs.append(pd.read_parquet(p/'costs.parquet'));decisions.append(pd.read_parquet(p/'decisions.parquet'))
        counts.extend(json.loads((p/'state_counts.json').read_text()))
    assert len(uids)==len(set(uids))==1024
    return pd.concat(frames),pd.concat(costs).fillna(0),pd.concat(decisions),pd.DataFrame(counts),queue['elapsed_seconds']


def bootstrap(f,cohort):
    lookup={u:i for i,u in enumerate(cohort)};ranks={}
    for t in TARGETS:
        g=f[f.target==t]
        for m in METHODS:ranks[t,m]=ranked_auc(g.label.to_numpy(),g[m].to_numpy(),g.uid.map(lookup).to_numpy())
    rng=np.random.default_rng(17);draw=[]
    for _ in range(1000):
        mult=np.bincount(rng.integers(1024,size=1024),minlength=1024)
        draw.append([[weighted_auc(ranks[t,m],mult) for m in METHODS] for t in TARGETS])
    a=np.array(draw);valid=np.isfinite(a).all((1,2));a=a[valid];out=[]
    for left,right in [('immediate','design1'),('immediate','cost_only'),('cost_only','design1'),
                       ('deferred','design1'),('deferred','immediate'),('deferred','cost_only'),('deferred','exact')]:
        d=a[:,:,METHODS.index(left)]-a[:,:,METHODS.index(right)]
        for j,t in enumerate(TARGETS):out.append(dict(target=t,left=left,right=right,interval=np.quantile(d[:,j],[.025,.975]).tolist()))
        out.append(dict(target='equal_edge',left=left,right=right,interval=np.quantile(d.mean(1),[.025,.975]).tolist()))
    return dict(draws=1000,valid=int(valid.sum()),records=out,seed=17)


def main():
    assert not OUT.exists()
    source=ROOT/'results/design2/analysis/lifecycle_01'
    primary=json.loads((source/'summary.json').read_text())
    raw=pd.read_parquet(source/'quality_raw.parquet').rename(columns={'design2':'immediate'})
    keys=['uid','target','request_id','timestamp','label','item_idx','count','writes_since_release']
    reports={};decision_frames={};cost_frames={};state_frames={}
    for name in ('cost_only','deferred'):
        f,c,d,states,seconds=load(name)
        f=f.sort_values(keys).reset_index(drop=True);raw=raw.sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(f[keys],raw[keys],check_dtype=False)
        # New policies cannot change any of the three unchanged reference paths.
        np.testing.assert_allclose(f[['reuse','design1','exact']],raw[['reuse','design1','exact']],rtol=0,atol=0)
        raw[name]=f.design2
        raw[name+'_rebuilds']=f.rebuilds
        raw[name+'_anchored_native']=f.anchored_native
        if name=='deferred':
            # Publication age includes writes BEFORE a delayed rebuild. Only
            # count writes after its first scored request as sustained follow-up.
            anchor=f[f.anchored_native].groupby(['uid','target']).writes_since_release.min()
            origin=pd.Series([anchor.get((int(u),int(t)),np.nan) for u,t in zip(f.uid,f.target)])
            raw[name+'_writes_since_rebuild']=f.writes_since_release.to_numpy()-origin.to_numpy()
        excl={'uid','target','method','rebuild_dependency_closed','validation_extra_geometry'}
        cols=[k for k in c if k not in excl]
        components=dict(c[c.method=='design2'][cols].sum());pop=float(sum(components.values()))
        shared=primary['cost']['shared']['design1' if name=='cost_only' else 'design2']
        # The old three paths also retain exactly the same model arithmetic.
        for m in ('reuse','design1','exact'):
            assert abs(float(c[c.method==m][cols].sum().sum())-primary['cost']['population'][m])<1
        for t,g in d.groupby('target'):
            executed=int((c[(c.method=='design2')&(c.target==t)].rebuild_literal>0).sum())
            reports.setdefault(name,dict(actions=[]))['actions'].append(dict(target=int(t),decided=int(g.action.isin(['rebuild','queue_rebuild']).sum()),executed=executed,
                reasons=g.reason.value_counts().to_dict()))
        reports[name].update(population_FLOPs=pop,shared_FLOPs=shared,components=components,duration_seconds=seconds)
        decision_frames[name]=d;cost_frames[name]=c;state_frames[name]=states
    uids=json.loads((ROOT/'configs/design2/detection_01.json').read_text())['groups']['development']
    per_edge={int(t):{m:binary_metrics(g.label.to_numpy(),g[m].to_numpy()) for m in METHODS} for t,g in raw.groupby('target')}
    means={m:float(np.mean([per_edge[t][m]['ROC_AUC'] for t in TARGETS])) for m in METHODS}
    boot=bootstrap(raw,uids)
    costs={m:dict(population_FLOPs=primary['cost']['population'][old],shared_FLOPs=primary['cost']['shared'][old])
        for m,old in [('reuse','reuse'),('design1','design1'),('exact','exact'),('immediate','design2')]}
    costs.update({m:{k:reports[m][k] for k in ('population_FLOPs','shared_FLOPs')} for m in reports})
    scales=[]
    for n in (1024,30000,100000,1000000):
        exact=costs['exact']['population_FLOPs']*n/1024
        closed=primary['cost']['exact_dependency_closed']*n/1024
        r=dict(users=n,methods={})
        for m,c in costs.items():
            total=c['shared_FLOPs']+c['population_FLOPs']*n/1024
            r['methods'][m]=dict(total_FLOPs=total,ratio_actual=total/exact,ratio_closed=total/closed)
        scales.append(r)
    follow=[]
    for name,col in [('immediate','anchored_native'),('cost_only','cost_only_anchored_native'),('deferred','deferred_anchored_native')]:
        for t in TARGETS:
            base=raw[(raw.target==t)&raw[col]]
            writes=base.deferred_writes_since_rebuild if name=='deferred' else base.writes_since_release
            for label,mask in [('all_real',np.ones(len(base),dtype=bool)),('after_writes',writes.to_numpy()>0)]:
                g=base[mask]
                if len(g):follow.append(dict(method=name,target=t,band=label,users=int(g.uid.nunique()),requests=len(g),
                    metrics={m:binary_metrics(g.label.to_numpy(),g[m].to_numpy()) for m in ('design1','exact',name)}))
    result=dict(status='complete',users=1024,real_requests=len(raw),equal_edge_auc=means,per_edge=per_edge,
        bootstrap=boot,variants=reports,costs=costs,scales=scales,followup=follow,
        common_paths_identical=True,confirmation_read=False,
        selection_scope='Two alternatives fixed after primary development result. No further policy search. Intervals descriptive on reused development cohort; no independent qualification or matched-budget causal claim.')
    OUT.mkdir(parents=True);write_json(OUT/'summary.json',finite_metrics(result))
    raw.to_parquet(OUT/'quality_raw.parquet',index=False)
    for name,d in decision_frames.items():d.to_parquet(OUT/f'{name}_decisions.parquet',index=False)
    render(result)
    print(json.dumps(dict(auc=means,contrasts=[r for r in boot['records'] if r['target']=='equal_edge'],variants=reports,scales=scales)))


def render(s):
    def contrast(l,r):return next(v['interval'] for v in s['bootstrap']['records'] if v['target']=='equal_edge' and v['left']==l and v['right']==r)
    lines=['# Design2真实闭环：主方案与两个固定备选','',
        '原1024开发UID、相同M0—M5真实轨迹和请求。主方案结果后固定两个备选：仅费用重建对照，以及发布决定后延迟到首个实际请求执行重建。没有改变C/H/组/阈值；两项之后结束本轮策略搜索。','',
        '## 1. 核验与评价边界','',
        '两个备选先通过相同8UID canary，再各跑完整1024UID。三个未修改参考分支的所有真实请求/标签/分数均与主实验逐元素一致，费用相同；不同的只是新增策略分支。',
        '延迟方案在发布时不使用未来活动或标签；只有请求实际到达才执行已决定的重建，下一模型取消未执行意图。其重建起点可能晚于发布，并产生新轨迹，因此不宣称保持主方案的质量或后续决策。',
        '仅费用方案没有几何读取或残余校准准备，是较便宜的动作对照，不是同预算比较。共享准备按各方法所需依赖收一次。','',
        '## 2. 真实反馈AUC','',
        '| 发布 | Reuse | Design1 | Exact-All | 立即D2 | 仅费用 | 延迟D2 |','| --- | --- | --- | --- | --- | --- | --- |']
    for t in (1,2,3,4,5):
        q=s['per_edge'].get(t,s['per_edge'].get(str(t)))
        lines.append(f'| M{t} | '+' | '.join(f"{q[m]['ROC_AUC']:.7f}" for m in METHODS)+' |')
    lines+=['| 四边等权 | '+' | '.join(f"{s['equal_edge_auc'][m]:.7f}" for m in METHODS)+' |','',
        '| 对比 | 四边AUC差（百分点） | 配对UID95%区间（百分点） |','| --- | --- | --- |']
    for l,r in [('immediate','design1'),('immediate','cost_only'),('cost_only','design1'),('deferred','design1'),('deferred','immediate'),('deferred','cost_only')]:
        d=100*(s['equal_edge_auc'][l]-s['equal_edge_auc'][r]);interval=[round(100*x,4) for x in contrast(l,r)]
        lines.append(f'| {l} − {r} | {d:+.4f} | {interval} |')
    lines+=['','以上只看真实反馈，排除发布检查面板。M2另列不入四边平均；M5partial不隐去。一次训练seed、反复使用的开发UID与选择后的区间不能当独立确认。','',
        '## 3. 实际重建次数','', '| 方案/发布 | 决定重建 | 实际执行 |','| --- | --- | --- |']
    for name,r in s['variants'].items():
        for a in r['actions']:lines.append(f"| {name}/M{a['target']} | {a['decided']} | {a['executed']} |")
    lines+=['','实际重建后再次发生真实写入的请求如下。延迟方案必须减去重建前的发布年龄，不能将首个请求当场的重建收益算成后续收益；此表已排除那些请求。条件子集不是随机化动作效应。','',
        '| 延迟重建后写入/发布 | UID/请求 | Design1 AUC | 延迟D2 AUC |', '| --- | --- | --- | --- |']
    for r in s['followup']:
        if r['method']=='deferred' and r['band']=='after_writes':
            a=r['metrics']['design1']['ROC_AUC'];b=r['metrics']['deferred']['ROC_AUC']
            av='未定义' if a is None else f'{a:.7f}';bv='未定义' if b is None else f'{b:.7f}'
            lines.append(f"| M{r['target']} | {r['users']}/{r['requests']} | {av} | {bv} |")
    lines+=['','## 4. 完整费用与规模边界','',
        '共同M0、同形状普通读取/追加抵消，真实分支仍独立。沿用主实验完整准备依赖；主表实际prefix图，闭包分母更低而保留全部方法费用，作为保守比较。外推固定本次用户/事件/评分强度，不是人口质量证据。','',
        '| 方法 | 服务增量TF/1024UID | 一次共享准备TF |','| --- | --- | --- |']
    for m,c in s['costs'].items():lines.append(f"| {m} | {c['population_FLOPs']/1e12:.6f} | {c['shared_FLOPs']/1e12:.6f} |")
    lines+=['','| 人口 | 方法 | 总TF | /实际Exact | /闭包Exact |','| --- | --- | --- | --- | --- |']
    for r in s['scales']:
        for m in ('design1','immediate','cost_only','deferred'):
            c=r['methods'][m];lines.append(f"| {r['users']} | {m} | {c['total_FLOPs']/1e12:.3f} | {c['ratio_actual']:.2%} | {c['ratio_closed']:.2%} |")
    lines+=['','## 5. 结论边界','',
        '保留延迟执行作为当前更值得推进的闭环候选。它的服务增量从立即方案5.137TF降至2.915TF（少43.24%），四边平均AUC略升；相对立即方案的质量差区间含零，不能宣称质量确定更好。',
        '延迟方案相对Design1的AUC差+.00127543，配对95%区间[.00000623,.00277282]下界贴近零。结合已在本开发集探索两个备选，只能称正面但偏弱的开发证据，不能按刚好跨过零线作独立验收。M1仍下降，M4/M5有后续收益。',
        '立即/延迟方案均优于仅费用对照的点估计且配对区间为正，说明完整检测触发有额外质量线索；但它们也花费更多，尚未证明同预算下的分配价值。',
        '1024UID下共享准备尚未摊薄，三种方法全部贵于Exact-All；延迟D2在3万/10万/100万的相同负载外推中，完整费用约为更便宜Exact闭包的52.83%/30.00%/21.20%。这不是生产流量或新人口质量验证。',
        '本轮止于主方案和两个备选，不再换检测阈值或搜索第三个策略。下一步优先固定延迟闭环，做同总FLOPs的简单重建对照，并检查M1反例；仍需后续独立证据验证质量。','',
        '![真实闭环质量与费用](../../../../figures/pic/design2/lifecycle_01/lifecycle.png)']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
