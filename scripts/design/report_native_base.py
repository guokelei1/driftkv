#!/usr/bin/env python3
"""Real feedback quality, shared-UID edge bootstrap, and matched domain cost."""

import argparse
import json

import numpy as np
import pandas as pd
from design.data import ROOT
from design.evaluate_native_base import PATHS
from design.native_service import METHODS
from design.run import write_json

from hstu_kvcache.evaluation.binary_metrics import binary_metrics, sigmoid


def ranked_auc(labels,scores,users):
    probabilities=sigmoid(scores)
    order=np.argsort(probabilities,kind="stable")
    starts=np.r_[0,np.flatnonzero(np.diff(probabilities[order])!=0)+1]
    return users[order],labels[order],starts


def weighted_auc(ranked,multiplicity):
    users,labels,starts=ranked
    weights=multiplicity[users]
    positive=np.add.reduceat(weights*labels,starts).astype(np.float64)
    negative=np.add.reduceat(weights*(1-labels),starts).astype(np.float64)
    denominator=positive.sum()*negative.sum()
    return float(np.dot(positive,np.cumsum(negative)-.5*negative)/denominator) if denominator else np.nan


def bootstrap_canary():
    labels=np.array([0,1,0,1,1,0,1,0])
    scores=np.array([0.,0.,-1.,1.,2.,2.,-1.,0.])
    users=np.repeat(np.arange(4),2)
    ranks=ranked_auc(labels,scores,users)
    for counts in (np.ones(4,dtype=int),np.array([2,0,1,1]),np.array([0,3,0,1])):
        idx=np.repeat(np.arange(len(labels)),counts[users])
        reference=binary_metrics(labels[idx],scores[idx])["ROC_AUC"]
        np.testing.assert_allclose(weighted_auc(ranks,counts),reference,atol=1e-15,rtol=1e-15)


def bootstrap(raw,cohort,repetitions):
    """Same UID multiplicity across all edges and methods; no cross-version score pooling."""
    lookup={uid:i for i,uid in enumerate(cohort)}
    targets=sorted(raw.target.unique())
    ranked={}
    for target,f in raw.groupby("target"):
        indices=f.uid.map(lookup).to_numpy()
        for method in PATHS:
            ranked[int(target),method]=ranked_auc(f.label.to_numpy(),f[method].to_numpy(),indices)
            np.testing.assert_allclose(weighted_auc(ranked[int(target),method],np.ones(len(cohort),dtype=int)),
                                       binary_metrics(f.label.to_numpy(),f[method].to_numpy())["ROC_AUC"],atol=1e-14,rtol=1e-14)
    rng=np.random.default_rng(17)
    draws=np.empty((repetitions,len(targets),len(PATHS)))
    for i in range(repetitions):
        counts=np.bincount(rng.integers(0,len(cohort),len(cohort)),minlength=len(cohort))
        for j,target in enumerate(targets):
            for k,method in enumerate(PATHS):
                draws[i,j,k]=weighted_auc(ranked[target,method],counts)
    # Undefined draws remain explicitly counted; never silently fill an AUC.
    valid=np.isfinite(draws).all(axis=(1,2))
    kept=draws[valid]
    differences=[]
    for left,right in (("exact","reuse"),("native","reuse"),("summary_input","reuse"),("no_source","reuse"),
                       ("native","summary_input"),("native","no_source"),
                       ("native","exact"),("summary_input","exact"),("no_source","exact")):
        delta=kept[:,:,PATHS.index(left)]-kept[:,:,PATHS.index(right)]
        for j,target in enumerate(targets):
            differences.append(dict(target=int(target),left=left,right=right,interval=np.quantile(delta[:,j],[.025,.975]).tolist()))
        differences.append(dict(target="equal_edge",left=left,right=right,interval=np.quantile(delta.mean(1),[.025,.975]).tolist()))
    return dict(repetitions=repetitions,valid_draws=int(valid.sum()),seed=17,cohort_users=len(cohort),differences=differences,
        scope="paired UID draws including no-feedback users, same multiplicity across four edges; conditional development evidence at one backbone seed")


def calibration_costs():
    ablation=ROOT / "results/design/native_base_ablation256_01"
    summary=json.loads((ablation / "summary.json").read_text())
    common=sum(summary["state_construction_ledger_seconds"].values())
    result={m:dict(source_teacher_and_lineage_seconds=common,query_and_fit_seconds=0.) for m in METHODS}
    for target in (1,3,4,5):
        records=json.loads((ablation / f"cost_m{target}.json").read_text())
        for r in records:
            result[r["method"]]["query_and_fit_seconds"]+=r["query_acquisition_seconds"]+r["shared_fit_seconds"]
        full=json.loads((ROOT / f"results/design/native_coverage384_01/cost_m{target}.json").read_text())
        r=next(r for r in full if r["method"]=="C")
        result["native"]["query_and_fit_seconds"]+=r["query_acquisition_seconds"]+r["shared_fit_seconds"]
    for value in result.values():
        value["measured_calibration_seconds"]=value["source_teacher_and_lineage_seconds"]+value["query_and_fit_seconds"]
    return result


def matched_cost(summary):
    calibration=calibration_costs()
    rows=[]
    for target,ledger in summary["ledger_seconds"].items():
        # Shared ordinary lineage is charged to each deployable branch, never divided by three methods.
        ordinary=ledger["service_reuse_append"]
        baseline=ordinary+ledger["service_reuse_read"]
        exact=ledger["exact_release_rebuild"]+ledger["service_exact_append"]+ledger["service_exact_read"]
        for method in METHODS:
            shared_condition=sum(ledger.get(f"{p}_{k}",0.) for p in ("publication","refresh") for k in ("source_pack","source_encode"))
            installation=sum(ledger.get(f"{p}_{method}_install",0.) for p in ("publication","refresh"))
            initial=ledger["initial_summary"]+ledger.get("publication_state_metadata",0.)
            maintenance=ledger.get("service_summary_maintenance",0.)
            extra_summary=sum(ledger.get(f"{p}_summary_response_build",0.) for p in ("publication","refresh")) if method=="summary_input" else 0.
            # Report the literal shared-writer implementation for all arms; no-source could remove redundant statistics but has not done so here.
            service=ordinary+ledger["service_"+method+"_read"]+maintenance+initial+shared_condition+installation+extra_summary
            rows.append(dict(target=int(target),method=method,reuse_seconds=baseline,exact_seconds=exact,
                exact_release_seconds=ledger["exact_release_rebuild"],initial_summary_and_release_metadata_seconds=initial,
                shared_condition_seconds=shared_condition,view_install_seconds=installation,summary_input_prepare_seconds=extra_summary,
                summary_maintenance_seconds=maintenance,read_seconds=ledger["service_"+method+"_read"],
                ordinary_append_seconds=ordinary,method_excluding_calibration_seconds=service,increment_over_reuse_seconds=service-baseline))
    totals=[]
    for method in METHODS:
        values=[r for r in rows if r["method"]==method]
        executable=sum(r["method_excluding_calibration_seconds"] for r in values)+calibration[method]["measured_calibration_seconds"]
        exact=sum(r["exact_seconds"] for r in values)
        reuse=sum(r["reuse_seconds"] for r in values)
        rebuild=sum(r["exact_release_seconds"] for r in values)
        incremental=executable-reuse
        totals.append(dict(method=method,calibration=calibration[method],method_seconds=executable,exact_seconds=exact,
            reuse_seconds=reuse,exact_release_rebuild_seconds=rebuild,method_over_exact=executable/exact,
            method_increment_over_reuse_seconds=incremental,increment_over_avoided_exact_rebuild=incremental/rebuild))
    return dict(rows=rows,totals=totals,scope="same4091 selected users, four releases, same eager kernels and replay windows; includes all no-feedback users",
        limitations=["Parent construction is common initial condition and excluded from all incremental comparisons",
                     "C frozen input-coordinate/PCA generation dependencies are retained in historical runs but their marginal cost is not fully separated; current total is a measured subtotal",
                     "Source/teacher construction is measured on identical fitting256 in ablation run; native C query/fit timing reused from its frozen run, same hardware/kernel",
                     "literal no-source path still shares writer/condition preparation; potential simpler implementation savings not claimed",
                     "GPU-synchronized wall timing includes Python orchestration but excludes unisolated data/model I/O, concurrency and long-term waiting; not20% qualification"])


def main(cli):
    bootstrap_canary()
    run=ROOT / "results/design" / cli.run_id
    summary=json.loads((run / "summary.json").read_text())
    assert summary["status"]=="development_complete"
    config=json.loads((run / "configuration.json").read_text())
    out=ROOT / "results/design/analysis" / (cli.run_id+"_report")
    out.mkdir(exist_ok=True)
    raw=pd.read_parquet(run / "quality_raw.parquet")
    assert not raw.duplicated(["target","request_id"]).any()
    assert np.isfinite(raw[list(PATHS)]).all().all()
    assert set(raw.target)=={1,3,4,5}
    cohort=config["development_uids"]
    ci=bootstrap(raw,cohort,cli.bootstrap)
    write_json(out / "bootstrap.json",ci)
    rows,mechanism,uid_rows=[],[],[]
    for target,f in raw.groupby("target"):
        metrics={m:binary_metrics(f.label.to_numpy(),f[m].to_numpy()) for m in PATHS}
        gap=metrics["exact"]["ROC_AUC"]-metrics["reuse"]["ROC_AUC"]
        for method in PATHS:
            gain=metrics[method]["ROC_AUC"]-metrics["reuse"]["ROC_AUC"]
            rows.append(dict(target=int(target),method=method,requests=len(f),feedback_users=int(f.uid.nunique()),
                selected_users=len(cohort),no_feedback_users=len(cohort)-int(f.uid.nunique()),positives=int(f.label.sum()),
                metrics=metrics[method],auc_minus_reuse=gain,exact_minus_reuse=gap,
                auc_gap_recovery=gain/gap if gap>1e-4 else None,ratio_rule="only Exact-Reuse >0.0001; descriptive, not an admission gate"))
            if method!="exact":
                errors=(f[method]-f.exact).pow(2)
                users=pd.DataFrame(dict(uid=f.uid,mse=errors)).groupby("uid").mse.mean()
                reuse=(f.reuse-f.exact).pow(2).groupby(f.uid).mean()
                mechanism.append(dict(target=int(target),method=method,request_mse=float(errors.mean()),uid_mse=float(users.mean()),
                    uid_median=float(users.median()),uid_p95=float(users.quantile(.95)),better_than_reuse_uid_fraction=float((users<reuse).mean())))
                uid_rows.extend(dict(target=int(target),method=method,uid=int(u),mse=float(v)) for u,v in users.items())
    pd.DataFrame(uid_rows).to_parquet(out / "uid_mse.parquet",index=False)
    aggregates={m:float(np.mean([r["auc_minus_reuse"] for r in rows if r["method"]==m])) for m in METHODS}
    costs=matched_cost(summary)
    write_json(out / "cost.json",costs)
    result=dict(status="complete",quality=rows,equal_edge_auc_improvement=aggregates,mechanisms=mechanism,
        selected_users=len(cohort),total_requests=len(raw),confirmation_read=False,seed17_only=True,
        scope="mature initial n_theta0>=1024, independent adjacent migration; post-development protocol; not continuous or blind confirmation")
    write_json(out / "summary.json",result)
    def interval(target,left,right):
        return next(x["interval"] for x in ci["differences"] if x["target"]==target and x["left"]==left and x["right"]==right)
    lines=["# Native响应条件化：成熟用户单次迁移真实质量", "", "完整native设计、冻结C权重；只重新拟合256×16摘要近似输入及无源状态两份必要消融。人群是原development6000中n_theta0≥1024的全部4091人（2026＋2065），不按方法误差筛选、不要求每UID有正负两类反馈。六层/H192/6heads/context1024/seed17，实际legacy ELU+1读取，不是原生SiLU HSTU普遍验证。", "",
        "M1/M3/M4/M5分别以M0/M2/M3/M4精确Parent缓存为同一起点。Reuse和三个修正分支共用普通native写入；Current Exact在发布点重算后独立native回放。同一真实14日服务窗口、因果历史、追加与淘汰。M5保留E14_partial。**这是四次独立相邻迁移，不是连续迁移或免费清除版本债。**", "",
        "## 1. 主结果：真实反馈AUC", "", "每边合并所有合法反馈后计算AUC，不平均分块AUC，也不混合不同版本分数。主汇总量为四边等权绝对AUC改善，以下差值均用百分点（pp）。", "",
        "| M | 反馈数/反馈UID/无反馈UID | Reuse AUC | Exact AUC | native AUC | 摘要近似 AUC | 无源状态 AUC |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for target in (1,3,4,5):
        r={x["method"]:x for x in rows if x["target"]==target}
        lines.append(f"| M{target} | {r['native']['requests']}/{r['native']['feedback_users']}/{r['native']['no_feedback_users']} | "+" | ".join(f"{r[m]['metrics']['ROC_AUC']:.7f}" for m in PATHS)+" |")
    lines += ["", "| M | Exact−Reuse pp | native−Reuse pp [95%描述区间] | native−摘要 pp [区间] | native−无源状态 pp [区间] |", "| --- | ---: | --- | --- | --- |"]
    for target in (1,3,4,5):
        r={x["method"]:x for x in rows if x["target"]==target}
        cells=[]
        for right in ("reuse","summary_input","no_source"):
            delta=100*(r["native"]["metrics"]["ROC_AUC"]-r[right]["metrics"]["ROC_AUC"])
            low,high=interval(target,"native",right)
            cells.append(f"{delta:+.4f} [{100*low:+.4f}, {100*high:+.4f}]")
        lines.append(f"| M{target} | {100*r['native']['exact_minus_reuse']:+.4f} | "+" | ".join(cells)+" |")
    lines += ["", "| 四边等权主汇总 | 相对Reuse绝对AUC改善 pp | 95%描述区间 pp |", "| --- | ---: | --- |"]
    for method in METHODS:
        low,high=interval("equal_edge",method,"reuse")
        lines.append(f"| {method} | {100*aggregates[method]:+.4f} | [{100*low:+.4f}, {100*high:+.4f}] |")
    for right in ("summary_input","no_source"):
        low,high=interval("equal_edge","native",right)
        lines.append(f"| native−{right} | {100*(aggregates['native']-aggregates[right]):+.4f} | [{100*low:+.4f}, {100*high:+.4f}] |")
    lines += ["", f"共{len(raw)}条真实反馈。每个bootstrap重复对同一4091UID抽取一套权重，四条边和五方法共享权重；{ci['valid_draws']}/{ci['repetitions']}次有效。无反馈用户仍在入组与状态账本中；不制造标签。区间仅描述已使用开发人群、单backbone seed17，不是独立确认或多seed证据。", "",
        "恢复比例仅在Exact−Reuse>0.0001时存入JSON；不对小gap/反号生成夸张比值。完整log-loss、Brier、dislike PR-AUC及dislike-only loss见summary.json，各条路径与逐边负结果完整保留。", "",
        "## 2. 设计与两个必要消融", "", "每层在实际query读取后计算 `delta=T_e*a_source(q)+b_e(S,N)+A_e(S,N)*q`，再把修正加回历史聚合，保留原self、projection、gate、residual顺序。发布摘要提供状态条件，native响应提供本次query的历史细节。理想化key不变/value共享线性变换说明这一分工的出发点，不作为实际模型更新假设。", "",
        "完整方法用冻结C；摘要臂只把native响应换成按producer均值生成的响应近似，参数数量、256×16、PCA/输入坐标、ridge=.01、固定mu_A及mask相同。无源状态臂仅保留截距source维，q/native/count/mask仍在，参数更少，属于简化对照；它仍有全局query仿射项，不能把它的效果归为T单独作用。完整/摘要两臂各1475712个W/U系数，无源状态259200个，不含冻结坐标统计。两者均重新联合拟合W/U、使用自己的下层actual q，未直接置零联合权重。", "",
        "不为基础评价重调C；拟合仍使用原256UID场景，评价全部4091合格UID。评价范围是已看到开发边界后的统一协议，不能称盲选成功样本。", "",
        "## 3. 机制解释指标", "", "以下来自同一批真实反馈，先每UID平均请求误差再UID等权；不替代AUC。无反馈UID无此指标，但照常维护。", "",
        "| M | 方法 | UID等权logit MSE | 中位数 | p95 | 优于Reuse UID |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for r in mechanism:
        lines.append(f"| M{r['target']} | {r['method']} | {r['uid_mse']:.6g} | {r['uid_median']:.6g} | {r['uid_p95']:.6g} | {r['better_than_reuse_uid_fraction']:.1%} |")
    lines += ["", "## 4. 同人口、同kernel的匹配计算账本", "", "同4091人、四次更新、相同eager基础kernel与服务时间段。Parent构建是三者共同问题起点，单列并从增量比较中一起排除。普通追加费用完整计入每条可执行分支，不因实验共享轨迹而除以3；无反馈用户也计入。", "",
        "| 路径 | 可计量校准s | 服务/发布/维护＋校准s | 同人口Exact总s | 方法/Exact | 相对Reuse增量/避免的Exact发布重算 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in costs["totals"]:
        lines.append(f"| {r['method']} | {r['calibration']['measured_calibration_seconds']:.2f} | {r['method_seconds']:.2f} | {r['exact_seconds']:.2f} | {r['method_over_exact']:.3f} | {r['increment_over_avoided_exact_rebuild']:.3f} |")
    lines += ["", "校准源/教师/普通谱系按本轮同256UID测量，C的query/fit成本沿用冻结C同hardware/kernel记录。冻结PCA/输入坐标生成依赖有历史记录，但其边际成本尚未全部分离，因此表中仍是可计量小计；不能据此宣称完整20%资格。source-free当前literal实现仍共用writer/条件准备，潜在删减不记成已实现收益。", "",
        "初始摘要、发布安装、刷新、摘要维护、读取、普通追加和Exact发布重算分项见cost.json；无反馈用户账本见原run summary。没有使用旧30000人口或compiled Exact的有利分母。I/O、等待覆盖、并发等未隔离费用另列，不开展kernel融合或生产优化。", "",
        "## 5. 边界与结论", "", "真实连续链、短历史及既知尾部失败保留在前轮报告，不能用本轮独立Parent初始化宣称连续修复。M2不作为适配目标；M3 Parent仍为M2。确认6000未读，未增加校准人数或新原型；不按任何边或UID结果选择赢家。", "",
        "**基础真实质量收益有正面证据。** 完整native四边等权改善+1.1190pp，配对区间[+0.0436,+1.8932]pp。四条边点估计均为正，但逐边仅M3区间全正；不把点估计写成四边均已稳定确认。此结论限定于成熟开发用户、单次相邻迁移、seed17及既定服务窗口。", "",
        "**native特有输入优势尚未由基础AUC证实。** 同容量摘要近似平均改善+1.1815pp，native−摘要为−0.0624pp、区间[−0.2429,+0.0823]pp；M1反而摘要更好。不能把此前proxy优势直接写成真实任务上的native必要性，也不能将区间含0解释为已证明等价。", "",
        "**源状态条件的必要性仍未定。** 完整native比重新拟合的无源状态臂平均高0.2619pp，区间[−0.0319,+0.4725]pp。无源状态仅17.6%的W/U系数，提示简化值得保留，但现有证据既未证明状态条件必需，也未证明可以无损移除。它仍有全局q残差，不是纯T消融。", "",
        "**当前实现尚未实现时间节省。** 同人口/eager-kernel的可计量native小计821.29s，对应Exact577.53s，约1.422倍；相对Reuse新增294.14s，而避免的Exact发布重算约61.93s。这些是时间，不是FLOPs；不得由此判断算法必须削减83%或优先删除源状态路径。理论计算已另按Reuse=0、完整校准依赖与匹配生命周期核算，见[增量FLOPs报告](../native_flops_01/report.md)。不把时间比当成理论预算已失败或已达标。", "",
        "这轮补上了读取修正在基础域中的真实质量效果，尚未补齐‘native信息分工是必要机制且足够便宜’的整条论文论证。保留基础结果与两个消融，不按边选赢家，不再缩人群、追异常UID或追加结构。所有本轮作业已结束；后续讨论应据此调整设计解释。"]
    (out / "report.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(report=str(out / "report.md"),equal_edge_auc_improvement=aggregates),ensure_ascii=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id",default="native_base_quality4091_01")
    parser.add_argument("--bootstrap",type=int,default=1000)
    main(parser.parse_args())
