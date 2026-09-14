"""CPU-only benchmark coverage, fixed-stratum quality and complete cost audit."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from design.data import ROOT, DAY, histories
from design.report_native_base import ranked_auc, weighted_auc
from design.report_native_flops import rebuild, value
from hstu_kvcache.evaluation.binary_metrics import binary_metrics

OUT = ROOT / 'results/design2/analysis/benchmark_01'
CONFIG = ROOT / 'configs/design2/benchmark_01.json'
SOURCE = ROOT / 'results/design2/analysis/lifecycle_variants_01'
METHODS = ('reuse', 'design1', 'exact', 'immediate', 'cost_only', 'deferred')
STRATA = ('all', 'short', 'old_lineage', 'mixed', 'rare_activity', 'challenge', 'routine')


def clean(v):
    if isinstance(v, dict): return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [clean(x) for x in v]
    if isinstance(v, np.generic): return clean(v.item())
    if isinstance(v, float) and not np.isfinite(v): return None
    return v


def save(path, obj):
    path.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def metadata(cfg, split):
    cal = split['groups']['residual_calibration']
    dev = split['groups']['development']
    assert len(cal) == len(set(cal)) == 512 and len(dev) == len(set(dev)) == 1024
    assert not set(cal) & set(dev)
    history = histories(cal + dev, cfg['stop_day'])
    cuts = np.array(cfg['cutover_days']) * DAY
    rows = []
    for role, users in [('calibration', cal), ('development', dev)]:
        for uid in users:
            ts = history.rows[uid][0]
            for t in cfg['targets']:
                cut = cuts[t-1]
                end = np.searchsorted(ts, cut, side='left')
                kept = ts[max(0, end-1024):end]
                assert len(kept) > 0 and kept[-1] < cut
                producer = np.searchsorted(cuts, kept, side='right')
                rows.append(dict(role=role, uid=uid, target=t, retained_count=len(kept),
                    prior_writes=int(end - np.searchsorted(ts, cut-14*DAY, side='left')),
                    idle_days=float((cut-kept[-1])/DAY),
                    old_fraction=float(np.mean(producer <= t-2)),
                    producers=len(np.unique(producer))))
    f = pd.DataFrame(rows)
    bounds = []
    f['rare_activity'] = False
    for t in cfg['targets']:
        for feature in ('prior_writes', 'idle_days'):
            lo, hi = f[(f.role == 'calibration') & (f.target == t)][feature].quantile(cfg['rarity_quantiles'])
            bounds.append(dict(target=t, feature=feature, lower=lo, upper=hi))
            f.loc[(f.target == t) & ((f[feature] < lo) | (f[feature] > hi)), 'rare_activity'] = True
    f['short'] = f.retained_count <= cfg['short_max']
    f['old_lineage'] = (f.target >= 3) & (f.old_fraction >= cfg['old_fraction_min'])
    f['mixed'] = f.producers >= cfg['mixed_producers_min']
    f['challenge'] = f[['short', 'old_lineage', 'mixed', 'rare_activity']].any(axis=1)
    f['routine'] = ~f.challenge
    f['all'] = True
    assert (f.challenge.astype(int) + f.routine.astype(int) == 1).all()
    return f, bounds


def load_service():
    run = ROOT / 'results/design2/lifecycle_deferred_01'
    queue = json.loads((run/'population/summary.json').read_text())
    costs, states = [], []
    for job in queue['jobs']:
        assert job['exit'] == 0
        p = run / job['name']
        costs.append(pd.read_parquet(p/'costs.parquet'))
        states.extend(json.loads((p/'state_counts.json').read_text()))
    return pd.concat(costs).fillna(0), pd.DataFrame(states)


def quality(f, cfg):
    rows = []
    for name in STRATA:
        for t, g in f[f[name]].groupby('target'):
            support = dict(uids=g.uid.nunique(), requests=len(g), positive=int(g.label.sum()), negative=int((1-g.label).sum()))
            valid = (support['uids'] >= cfg['auc_support']['min_uids'] and
                     min(support['positive'], support['negative']) >= cfg['auc_support']['min_positive'])
            for m in METHODS:
                metrics = binary_metrics(g.label.to_numpy(), g[m].to_numpy())
                # Individual feedback rows are not independent users.
                e = (g[m]-g.exact).abs()
                panel = pd.DataFrame({'uid':g.uid, 'e':e}).groupby('uid').e.max()
                rows.append(dict(stratum=name, target=int(t), method=m, **support,
                    auc_supported=valid, auc=metrics['ROC_AUC'] if valid else None,
                    logloss=metrics['log_loss'], brier=metrics['Brier'],
                    trajectory_uid_max_p95=panel.quantile(.95), trajectory_uid_max_p99=panel.quantile(.99),
                    trajectory_severe_uid_rate=float((panel>.5).mean()),
                    trajectory_excess_mass=float(np.maximum(panel-.5,0).mean())))
    return rows


def contrasts(f, cfg, users):
    lookup = {u:i for i,u in enumerate(users)}
    rng = np.random.default_rng(cfg['bootstrap']['seed'])
    mult = np.array([np.bincount(rng.integers(len(users), size=len(users)), minlength=len(users))
                     for _ in range(cfg['bootstrap']['uid_draws'])])
    records = []
    for name in ('all', 'challenge', 'routine'):
        auc_edges = []
        for t in cfg['targets']:
            g = f[(f.target==t) & f[name]]
            idx = g.uid.map(lookup).to_numpy()
            loss = {m: np.logaddexp(0,g[m].to_numpy())-g.label.to_numpy()*g[m].to_numpy()
                    for m in ('design1','deferred')}
            sums = {m: np.bincount(idx,weights=v,minlength=len(users)) for m,v in loss.items()}
            relative_loss = (mult @ (sums['deferred']-sums['design1'])) / (mult @ sums['design1'])
            loss_ci = np.nanquantile(relative_loss,[.025,.975]).tolist()
            if g.uid.nunique()<30 or min(g.label.sum(), (1-g.label).sum())<20:
                records.append(dict(stratum=name,target=t,status='insufficient_auc_support',relative_logloss_ci=loss_ci))
                continue
            ranks = [ranked_auc(g.label.to_numpy(),g[m].to_numpy(),idx) for m in ('design1','deferred')]
            draws = np.array([weighted_auc(ranks[1], w)-weighted_auc(ranks[0], w) for w in mult])
            auc_edges.append(draws)
            records.append(dict(stratum=name, target=t, delta_auc_ci=np.nanquantile(draws,[.025,.975]).tolist(), relative_logloss_ci=loss_ci, valid=int(np.isfinite(draws).sum())))
        if len(auc_edges)==len(cfg['targets']):
            draws = np.mean(auc_edges,axis=0)
            records.append(dict(stratum=name,target='equal_edge',delta_auc_ci=np.nanquantile(draws,[.025,.975]).tolist(),valid=int(np.isfinite(draws).sum())))
    return records


def main():
    started = time.monotonic()
    cfg = json.loads(CONFIG.read_text())
    split = json.loads((ROOT/cfg['cohort_source']).read_text())
    OUT.mkdir(parents=True, exist_ok=False)
    meta, bounds = metadata(cfg, split)
    meta.to_parquet(OUT/'members.parquet',index=False)
    save(OUT/'calibration_bounds.json',bounds)
    print(json.dumps(dict(stage='metadata',seconds=time.monotonic()-started,
                         counts=meta[meta.role=='development'][list(STRATA)].sum().to_dict())),flush=True)
    members = meta[meta.role=='development'].drop(columns='role')
    raw = pd.read_parquet(SOURCE/'quality_raw.parquet')
    f = raw[raw.target.isin(cfg['targets'])].merge(members,on=['uid','target'],validate='many_to_one')
    assert len(f)==19631 and set(f.uid)<=set(split['groups']['development'])
    c, states = load_service()
    c = c[c.target.isin(cfg['targets'])].merge(members,on=['uid','target'],validate='many_to_one')
    d = pd.read_parquet(SOURCE/'deferred_decisions.parquet')
    windows = members.merge(states[['uid','target','events','requests','request_groups']],on=['uid','target'],validate='one_to_one')
    windows = windows.merge(d[['uid','target','action','reason']],on=['uid','target'],validate='one_to_one')
    components = list(json.loads((SOURCE/'summary.json').read_text())['variants']['deferred']['components'])
    coverage = []
    for name in STRATA:
        for t in cfg['targets']:
            w = windows[(windows.target==t) & windows[name]]
            cc = c[(c.target==t)&c[name]]
            dd = cc[cc.method=='design2']; baseline=cc[cc.method=='design1']
            denom = sum(value(rebuild(int(n))) for n in w.retained_count)
            extra = dd[components].sum().sum()-baseline[components].sum().sum()
            coverage.append(dict(stratum=name,target=t,uids=w.uid.nunique(),states=len(w),
                feedback_uids=int((w.requests>0).sum()),requests=int(w.requests.sum()),events=int(w.events.sum()),
                no_feedback=int((w.requests==0).sum()),planned=int((w.action=='queue_rebuild').sum()),
                rebuilt=int((dd.rebuild_literal>0).sum()),
                intervention_rate=float((dd.rebuild_literal>0).mean()),
                extra_service_flops=extra,own_exact_closed_flops=denom,
                extra_service_ratio=extra/denom if denom else None))
    rows = quality(f,cfg)
    boot = contrasts(f,cfg,split['groups']['development'])
    diag = pd.read_parquet(ROOT/'results/design2/analysis/detection_01/states.parquet',filters=[('role','==','development')])
    continuous = diag[diag.kind=='continuous'].merge(members,on=['uid','target'],validate='one_to_one')
    assert len(continuous)==4096 and (continuous['count']==continuous.retained_count).all()
    diagnostics = []
    for name in STRATA:
        for tau in cfg['tail_thresholds']:
            g=continuous[continuous[name]]
            bad=g.max_abs_error>tau
            diagnostics.append(dict(stratum=name,threshold=tau,states=len(g),
                failure_uids=int(g[bad].uid.nunique()),failure_states=int(bad.sum()),
                uid_equal_rate=float(g.assign(bad=bad).groupby('uid').bad.mean().mean()),
                reference='FreshCurrent at release, original continuous Design1 diagnostic only; no D2 output'))
    s=json.loads((SOURCE/'summary.json').read_text())
    exact=json.loads((ROOT/'results/design2/analysis/lifecycle_01/summary.json').read_text())['cost']['exact_dependency_closed']
    dd=c[c.method=='design2']
    # Components at all five releases are needed, including M2 transition costs.
    allcost, _ = load_service()
    all_d=allcost[allcost.method=='design2']
    assert abs(all_d[components].sum().sum()-s['costs']['deferred']['population_FLOPs'])<1
    pop=s['costs']['deferred']['population_FLOPs']; shared=s['costs']['deferred']['shared_FLOPs']
    closed_delta=float((all_d.rebuild_literal-all_d.rebuild_dependency_closed).sum())
    joined=dd.merge(states[['uid','target','requests']],on=['uid','target'],validate='one_to_one')
    dormant=joined[joined.requests==0][['decision_prepare','decision_C_read','decision_geometry']].sum().to_dict()
    curves=[]
    for point in cfg['cost_envelope']:
        n=point['users']; denom=exact*n/1024
        curves.append(dict(users=n,target=point['total_incremental_ratio_max'],
            current_total_tf=(shared+pop*n/1024)/1e12,
            current_ratio=(shared+pop*n/1024)/denom,
            shared_ratio=shared/denom, service_asymptote=pop/exact,
            maximum_service_GF_per_uid=(point['total_incremental_ratio_max']*denom-shared)/n/1e9,
            current_service_GF_per_uid=pop/1024/1e9,
            closed_rebuild_what_if_ratio=(shared+(pop-closed_delta)*n/1024)/denom))
    intersections=members.groupby(['short','old_lineage','mixed','rare_activity']).size().reset_index(name='states').to_dict('records')
    result=dict(status='complete_existing_evidence_audit',protocol_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        config=cfg,coverage=coverage,quality=rows,auc_intervals=boot,intersections=intersections,
        fresh_current_design1_diagnostic=diagnostics,
        workload_states=diag.groupby('kind').agg(states=('uid','size'),uids=('uid','nunique')).reset_index().to_dict('records'),
        cost_curves=curves,opportunities=dict(
            dependency_closed_rebuild_saving_TF=closed_delta/1e12,
            dormant_decision_components_TF={k:v/1e12 for k,v in dormant.items()},
            dormant_decision_total_TF=sum(dormant.values())/1e12,
            note='what-if bounds only; no executable policy or claimed quality-preserving optimization'),
        missing=['FreshCurrent on actual lifecycle requests for all policies',
                 'prospective fixed-window32/128 independent continuous replay',
                 'matched-total-budget simple rebuild controls',
                 'new held-out validation after repeated development reuse'],
        new_model_forwards=0,confirmation_read=False,elapsed_seconds=time.monotonic()-started)
    save(OUT/'summary.json',result)
    pd.DataFrame(rows).to_csv(OUT/'quality.csv',index=False)
    pd.DataFrame(coverage).to_csv(OUT/'coverage.csv',index=False)
    windows.to_parquet(OUT/'windows.parquet',index=False)
    render(clean(result))
    print(json.dumps(dict(stage='complete',seconds=result['elapsed_seconds'],curves=curves,opportunities=result['opportunities'])),flush=True)


def render(s):
    lines=['# Benchmark v1：已有闭环的覆盖与费用审计','',
        '这次先固定基准，再对旧结果分组重评分；没有新模型前向、改策略或读取确认集。旧总体结果已知，不能称事前注册或独立确认。',
        '规则见[Benchmark定义](../../../../docs/design2/benchmark.md)。静态FreshCurrent与发布Exact轨迹参考严格分开。','',
        '## 自然工作负载覆盖','',
        '| 分组/目标 | UID/状态 | 有反馈UID | 请求 | 实际重建比例 | 额外服务/本组Exact闭包 |',
        '| --- | --- | --- | --- | --- | --- |']
    for r in s['coverage']:
        if r['stratum']=='all': continue
        ratio=r['extra_service_ratio']
        rate='无状态' if r['intervention_rate'] is None else f"{r['intervention_rate']:.2%}"
        cost='无状态' if ratio is None else f"{ratio:.2%}"
        lines.append(f"| {r['stratum']}/M{r['target']} | {r['uids']}/{r['states']} | {r['feedback_uids']} | {r['requests']} | {rate} | {cost} |")
    lines+=['','分组彼此可重叠；challenge并集与routine补集不重叠。背景组不是安全标签，额外服务为D2−D1，完整共享费用见后表。','',
        '## 真实任务质量：延迟D2减Design1','',
        '| 分组/发布 | 有反馈UID | AUC差（百分点） | LogLoss差 | 配对UID95% AUC区间（百分点） |',
        '| --- | --- | --- | --- | --- |']
    for name in ('all','challenge','routine','short','old_lineage','mixed','rare_activity'):
        for t in (1,3,4,5):
            a=next((r for r in s['quality'] if r['stratum']==name and r['target']==t and r['method']=='design1'),None)
            b=next((r for r in s['quality'] if r['stratum']==name and r['target']==t and r['method']=='deferred'),None)
            if a is None: continue
            auc='支持不足' if a['auc'] is None else f"{100*(b['auc']-a['auc']):+.4f}"
            ci=next((r.get('delta_auc_ci') for r in s['auc_intervals'] if r['stratum']==name and r['target']==t),None)
            interval='未计算/支持不足' if ci is None else str([round(100*x,4) for x in ci])
            lines.append(f"| {name}/M{t} | {a['uids']} | {auc} | {b['logloss']-a['logloss']:+.7f} | {interval} |")
    lines+=['','CSV保留全部六方法、Brier及相对ReleaseExact的轨迹残余。后者不是FreshCurrent兼容性误差，不能用于通过主保护门槛。','',
        '## 严重失败的样本是否足够','',
        '| 原连续D1诊断组 | 误差尺度 | 失败UID | 失败状态/总状态 | UID等权严重率 |',
        '| --- | --- | --- | --- | --- |']
    for r in s['fresh_current_design1_diagnostic']:
        lines.append(f"| {r['stratum']} | {r['threshold']} | {r['failure_uids']} | {r['failure_states']}/{r['states']} | {r['uid_equal_rate']:.3%} |")
    lines+=['','该表有当场FreshCurrent，但只有旧连续D1诊断，不能与不同历史的延迟D2服务结果直接相减。主尺度失败UID少于30时，保护有效性结论仍不足；不换尺度求通过。','',
        '## 人口成本曲线与优化空间','',
        '| 人口 | 目标上限 | 当前完整比例 | 共享准备比例 | 达标允许的服务GF/UID | 仅替换重建为闭包的假设比例 |',
        '| --- | --- | --- | --- | --- | --- |']
    for r in s['cost_curves']:
        lines.append(f"| {r['users']} | {r['target']:.0%} | {r['current_ratio']:.2%} | {r['shared_ratio']:.2%} | {r['maximum_service_GF_per_uid']:.4f} | {r['closed_rebuild_what_if_ratio']:.2%} |")
    o=s['opportunities']
    lines+=['',f"当前服务2.8471 GF/UID，渐近比例{s['cost_curves'][0]['service_asymptote']:.2%}；固定K的摊销不能突破该下限。",
        f"已记录的实际重建若可严格改写为KV依赖闭包，算术差额为{o['dependency_closed_rebuild_saving_TF']:.6f} TF/1024UID。上表只替换该项，其余成本不减。尚未实现/核验，不能宣布已省下。",
        f"没有真实反馈的窗口仍支付了{o['dormant_decision_total_TF']:.6f} TF发布检查（准备、C读取、几何）；这是待研究的浪费上限。在线不知道未来是否有反馈，不能用事后活动过滤直接扣账。首次消费才检查会改变输入和决策，需重新回放。",
        '减少校准512人或删场景不是等价优化；当前冻结标尺的准备依赖继续收费。', '',
        '## 当前判定与下一实验','',
        '本CPU审计首先定位到FreshCurrent请求参考、困难主尺度失败样本及同总预算对照的缺口。随后补算的请求参考见[FreshCurrent报告](fresh_current_report.md)：它不改变本表中的原六条服务轨迹或费用。',
        '补齐参考后，主尺度失败仍仅2UID，不能认定普遍保护。下一步先补受控短窗口/困难暴露测试，再用同一Benchmark测依赖闭包重建及首次消费检查。完整费用曲线与困难保护必须一起交付。',
        '尚未运行的窗口32/128负载保留为明确缺口；旧短历史快照不能冒充完成的受控生命周期。',
        f"本CPU分析用时{s['elapsed_seconds']:.2f}s，新增模型前向0。随后FreshCurrent参考的新增前向/评测费用单列在对应报告中。成员、交集、完整六方法指标和成本数值已保存。",'',
        '![成本曲线和服务费用](../../../../figures/pic/design2/benchmark_01/benchmark.png)']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
