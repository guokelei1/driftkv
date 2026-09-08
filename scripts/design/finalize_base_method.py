#!/usr/bin/env python3
"""Retained-ledger release attribution and the single matched representation report."""

import json
from collections import Counter

import numpy as np
import pandas as pd
from design.data import ROOT, histories
from design.report_native_flops import (
    METHODS,
    TARGETS,
    V,
    append_ops,
    calibration_counts,
    population_cost,
    prefix_literal,
    value,
)
from insight_two.common import CUTOVER_DAYS

OUT=ROOT/"results/design/analysis/base_method_final_01"
FLOPS=ROOT/"results/design/analysis/native_flops_01"
RUN=ROOT/"results/design/constant_affine192_01"


def dump(name,x):
    (OUT/name).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False,default=lambda v:v.item())+"\n")


def release_ledger():
    cache=OUT/"calibration_attribution.json"
    if cache.exists():
        counts=json.loads(cache.read_text())
    else:
        cfg=json.loads((ROOT/"results/design/native_base_ablation256_01/configuration.json").read_text())
        c=cfg["cohort"]; fit=cfg["fitting_uids"]; old=c["original_fitting_uids"]
        history=histories(fit,CUTOVER_DAYS[-1]+1)
        counts=dict(C=calibration_counts(history,fit,set(c["lifetime_fitting_uids"])),
                    A=calibration_counts(history,old,set(c["lifetime_fitting_uids"])&set(old)))
        original=json.loads((FLOPS/"calibration_counts.json").read_text())
        for label in counts:
            for key in ("builds","appends","scene_validation"):
                assert counts[label][key]==original[label][key]
        dump("calibration_attribution.json",counts)
    ledger=json.loads((FLOPS/"ledger.json").read_text())
    population=json.loads((FLOPS/"population_counts.json").read_text())
    rows=[]
    for method in METHODS:
        old=ledger[method]; k={t:Counter() for t in TARGETS}
        call_counts=Counter()
        for label,c in counts.items():
            for t in TARGETS:
                b=c["by_target"][str(t)]
                k[t][label+"_prefix"]=sum(value(prefix_literal(int(n)))*v for hist in b["builds"].values() for n,v in hist.items())
                k[t][label+"_replay"]=sum(value(append_ops(*map(int,n.split(','))))*v for hist in b["appends"].values() for n,v in hist.items())
                tokens=sum(int(n)*v for kind,hist in b["builds"].items() if kind!="teacher" for n,v in hist.items())
                tokens+=2*sum(int(n.split(',')[1])*v for kind,hist in b["appends"].items() if "teacher" not in kind for n,v in hist.items())
                k[t][label+"_maintenance"]=V*tokens
                call_counts[t]+=sum(v for field in ("builds","appends") for hist in b[field].values() for v in hist.values())
        for t in TARGETS:
            for name,x in old["shared_FLOPs"].items():
                if name.startswith(f"M{t}_"):
                    k[t][name]=x
            call_counts[t]+=36*counts['C']['scene_validation'][str(t)]['scenes']
        for t in TARGETS:
            k[t]["frequency_coordinate_allowance"]=old['shared_FLOPs']['frequency_and_coordinate_scalar_allowance']*call_counts[t]/sum(call_counts.values())
        np.testing.assert_allclose(sum(sum(v.values()) for v in k.values()),old['K'],rtol=1e-13)
        for t in TARGETS:
            p=population_cost([r for r in population if r['target']==t],method)
            publication=sum(v for key,v in p.items() if key=='init' or key.startswith('publish'))/4091
            a=sum(p.values())/4091; e=old['e']/4
            for u in (30000,100000,1000000):
                rows.append(dict(method=method,target=t,users=u,K=sum(k[t].values()),K_components=dict(k[t]),
                    user_release_FLOPs=publication,user_life_FLOPs=a,exact_FLOPs=u*e,
                    release_ratio=(sum(k[t].values())+u*publication)/(u*e),life_ratio=(sum(k[t].values())+u*a)/(u*e)))
    dump("per_release_flops.json",rows)
    return rows


def representation():
    if not (RUN/"summary.json").exists() or json.loads((RUN/"summary.json").read_text())['status']!='complete':
        return []
    frame=pd.concat([pd.read_parquet(RUN/f"responses_m{t}.parquet") for t in TARGETS],ignore_index=True)
    cfg=json.loads((RUN/'configuration.json').read_text())
    old=json.loads((ROOT/'results/design/mechanism_query_holdout192_01/configuration.json').read_text())
    assert cfg['fitting_uids']==old['fitting_uids'] and cfg['diagnostic_uids']==old['diagnostic_uids']
    assert not frame.duplicated(['target','uid','state_ordinal','layer']).any()
    result=[]
    cols=[f'{m}_{coord}_mse' for m in ('reuse','constant','affine') for coord in ('aggregate','rate')]
    for scope,f in [('all_states',frame),('continuous_cutover',frame[frame.state_group=='continuous_cutover'])]:
        for layer,g in [('all_layers',f),*[(int(l),v) for l,v in f.groupby('layer')]]:
            # First average layers within scene, then states within UID, then equal UID.
            states=g.groupby(['target','group','uid','state_ordinal'])[cols].mean().reset_index()
            users=states.groupby(['target','group','uid'])[cols].mean().reset_index()
            for (t,group),values in users.groupby(['target','group']):
                r=dict(target=int(t),group=group,scope=scope,layer=layer,users=len(values),states=len(states[(states.target==t)&(states.group==group)]))
                for col in cols:
                    r[col]=float(values[col].mean())
                for coord in ('aggregate','rate'):
                    energy=r[f'reuse_{coord}_mse']
                    for m in ('constant','affine'):
                        r[f'{m}_{coord}_ratio']=None if energy==0 else r[f'{m}_{coord}_mse']/energy
                r['zero_signal_uids']=int((values.reuse_aggregate_mse==0).sum())
                result.append(r)
    dump('representation.json',result)
    frame.to_parquet(OUT/'representation_states.parquet',index=False)
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    releases=release_ledger(); mechanism=representation()
    lines=['# 基础方法定稿：观察—设计—证据闭环','',
        '完整方法的唯一叙述源是[论文main.tex](../../../../../paper/main.tex)，本报告索引证据与本轮小型补充，不维护平行Design稿。主方法仍为冻结native C，256×16；两项消融和全部四边均保留。','',
        '## 1. 设计依据表','',
        '| 设计选择 | 直接证据 | 不能扩展的结论 |','| --- | --- | --- |',
        '| 修正读取而非重建KV | [逐场景仿射query留出](../mechanism_query_holdout192_report_01/report.md) | KV漂移必然低秩、任意状态只有32自由度 |',
        '| query条件视图 | 本报告统一公共q的常量/仿射对照；历史时间常量失效 | 任意query都线性、每层常量都无用 |',
        '| 发布共享、用户状态条件化 | [冻结decoder覆盖](../mechanism_decoder_closure192_report_01/report.md)与[实际跨UID质量](../native_base_quality4091_01_report/report.md) | 当前摘要充分、源条件不可替代 |',
        '| native响应输入 | [同容量输入诊断](../mechanism_native_input192_report_01/report.md)和真实反馈fidelity | native在基础AUC上必然胜摘要 |',
        '| native写入、评分修正 | [普通KV/写入canary](../../native_base_quality_canary_02/summary.json) | 连续版本债已解决、评分修正改善后代KV |',
        '| 增量摘要、dirty刷新 | 实際加减维护与[62639刷新计数](../native_flops_01/population_counts.json) | FLOPs少等于低延迟 |','',
        '## 2. 逐发布FLOPs归属','',
        '只分解已审核的同一K，不改变总量。初始M0源和A64坐标准备的初始化收在M1；各目标教师/辅助谱系、坐标与求解收在该目标；发布间源谱系回放收在下一个实际适配目标。因此M1→M2与M2→M3都归M3，M2仍无适配项。跨发布重复准备按原保守账本保留。频率/坐标的低阶余量按对应调用计数分摊，不把整个K均分。','',
        '| 方法 | 目标 | K_e TF | 3万人发布增量/Exact | 3万人生命周期增量/Exact |','| --- | --- | ---: | ---: | ---: |']
    for r in releases:
        if r['users']==30000:
            lines.append(f"| {r['method']} | M{r['target']} | {r['K']/1e12:.4f} | {r['release_ratio']:.2%} | {r['life_ratio']:.2%} |")
    lines+=['','逐边比例以该边同30000用户Exact依赖闭包为分母；四边K和用户费用与原账本求和一致。论文只陈述满足此归属规则的逐边结果，不以四边总比例替代“每次发布<20%”。详细3万/10万/100万见per_release_flops.json。','',
        '## 3. 同公共query的常量与仿射','',
        '旧query-holdout记录缺少原始q和响应，不能直接重拟合常量。本轮只恢复原64拟合/128已使用诊断用户、原真实状态、原fit64/held64面板。每层先由同一条已确定的逐场景仿射下层产生公共q，再在fit面板分别拟合常量均值与原正则仿射，在同一held目标评价。常量不建立自己的下层分支；该比较不是闭环方法排名。','',
        '用规范化FP64坐标评价表示误差，下层沿原FP32仿射路径。首个canary发现FP32反变换抵消误差，保留失败并在FP64代数坐标核对后通过第二个canary；冻结主方法不变。诊断仅每场景拟合表示，不训练新共享方法，不输出新AUC。','',
        '先每场景平均层，再同UID平均场景，最后UID等权。主表为聚合响应坐标（率误差乘N²）；率坐标、逐层和真实cutover子集均保留JSON。比值是同口径均值之比，零分母保留null，未用epsilon。Reuse列指公共q上的目标−源响应能量，不是独立Reuse分支q。','',
        '| 目标/组 | Reuse响应差能量 | 常量held残余 | 仿射held残余 | 常量/能量 | 仿射/能量 |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in mechanism:
        if r['scope']=='all_states' and r['layer']=='all_layers':
            ratios=["未定义" if r[f'{m}_aggregate_ratio'] is None else f"{r[f'{m}_aggregate_ratio']:.4%}" for m in ('constant','affine')]
            lines.append(f"| M{r['target']} {r['group']} | {r['reuse_aggregate_mse']:.6g} | {r['constant_aggregate_mse']:.6g} | {r['affine_aggregate_mse']:.6g} | {' | '.join(ratios)} |")
    lines+=['','统一对照支持的是表示是否需要保留query变化能力，不证明每层query项不可替代，也不把教师拟合的结果视为可部署质量。完整场景/UID和不利层均在representation_states.parquet与representation.json。','',
        '## 4. 主文图表与完成边界','',
        '论文Design章节包含实际函数式、输入/共享/用户维度、损失量纲、A64准备依赖、评分/native写入算法与FLOPs。基础质量表保留Reuse/Exact/native/摘要/无源状态四边及四边等权汇总；主文图为常量/仿射机制图、方法流程图和K+Ua/Ue规模图。源数据与生成器见figures/README.md。','',
        '质量人群4091与FLOPs外推3万/10万/100万在图注分开；相对Exact−0.0473pp不等于质量等价。当前设计冻结为第一项基础方法，不宣称已完成整篇论文、实际时间优势、独立确认、多seed/架构或持续多次迁移。未编译论文，不自动启动第二设计。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(release_rows=len(releases),mechanism_rows=len(mechanism),report=str(OUT/'report.md'))),flush=True)


if __name__=='__main__':
    main()
