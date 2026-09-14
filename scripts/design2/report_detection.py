"""Frozen split/threshold analysis of the medium detection experiment."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from design.data import ROOT
from design.report_native_flops import rebuild
from design.run import write_json

SCORES = ("detection","version_age","short_history","source_norm")


def weights(frame):
    per_target = frame.groupby(["uid","target"]).uid.transform("size").to_numpy()
    targets = frame.groupby("uid").target.transform("nunique").to_numpy()
    return 1/(per_target*targets)


def auc(y,score,w):
    order=np.argsort(score,kind="stable")
    y,s,w=y[order],score[order],w[order]
    starts=np.r_[0,np.flatnonzero(np.diff(s))+1]
    positives=np.add.reduceat(w*y,starts)
    negatives=np.add.reduceat(w*(~y),starts)
    total=positives.sum()*negatives.sum()
    return float((positives*(np.cumsum(negatives)-negatives/2)).sum()/total) if total else None


def quantile(x,w,q):
    order=np.argsort(x,kind="stable")
    return float(x[order][min(np.searchsorted(np.cumsum(w[order]),q*w.sum()),len(x)-1)])


def mean(x,w):
    return float(np.sum(x*w)/w.sum()) if w.sum() else None


def budget_curve(frame,score,fractions):
    w=weights(frame)
    serious=frame.max_abs_error.to_numpy()>=.5
    cost=frame.rebuild_flops.to_numpy()
    values=frame[score].to_numpy()/cost
    # UID/target/ordinal order is fixed for ties; no error-dependent selection.
    order=np.argsort(-values,kind="stable")
    cumulative=np.cumsum(cost[order])
    result=[]
    for fraction in fractions:
        chosen=order[cumulative<=fraction*cost.sum()]
        result.append(dict(budget_fraction=fraction,actual_cost_fraction=float(cost[chosen].sum()/cost.sum()),
            selected_states=len(chosen),severe_coverage=float(w[chosen][serious[chosen]].sum()/w[serious].sum()) if serious.any() else None,
            error_energy_coverage=float((w[chosen]*frame.mse.to_numpy()[chosen]).sum()/(w*frame.mse.to_numpy()).sum())))
    return result


def auc_summary(frame,score,rng,draws=500,error_threshold=.5):
    w=weights(frame); y=frame.max_abs_error.to_numpy()>=error_threshold
    s=frame[score].to_numpy()
    value=auc(y,s,w)
    uids,index=np.unique(frame.uid.to_numpy(),return_inverse=True)
    estimates=[]
    for _ in range(draws):
        multiplicity=np.bincount(rng.integers(len(uids),size=len(uids)),minlength=len(uids))
        result=auc(y,s,w*multiplicity[index])
        if result is not None:
            estimates.append(result)
    return dict(auroc=value,uid_bootstrap_95ci=np.quantile(estimates,[.025,.975]).tolist() if estimates else None,
                valid_bootstraps=len(estimates))


def curve(cal,dev,score,quantiles):
    wc,wd=weights(cal),weights(dev)
    sc,sd=cal[score].to_numpy(),dev[score].to_numpy()
    err=dev.max_abs_error.to_numpy()
    result=[]
    sealed=json.loads((ROOT/"configs/design2/thresholds_01.json").read_text())["thresholds"][score]
    for q in quantiles:
        threshold=sealed[str(q)]
        assert np.isclose(threshold,quantile(sc,wc,q),rtol=1e-12,atol=1e-12)
        selected=sd<=threshold
        result.append(dict(calibration_quantile=q,threshold=threshold,
            development_coverage=float(wd[selected].sum()/wd.sum()),states=int(selected.sum()),
            severe_rate=mean((err[selected]>=.5).astype(float),wd[selected]),
            mean_max_abs_error=mean(err[selected],wd[selected]),
            mse=mean(dev.mse.to_numpy()[selected],wd[selected]),
            p95_max_abs_error=quantile(err[selected],wd[selected],.95) if selected.any() else None,
            max_abs_error=float(err[selected].max()) if selected.any() else None))
    return result


def main(cli):
    root=ROOT/"results/design2"
    out=root/cli.output
    # Derived reports can be regenerated; input runs/raw and frozen thresholds cannot.
    out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((ROOT/"configs/design2/detection_01.json").read_text())
    runs=[root/name for name in cli.runs]
    frames=[]; costs={}
    for run in runs:
        summary=json.loads((run/"summary.json").read_text())
        assert summary["status"]=="complete"
        costs[run.name]=summary
        frames.extend(pd.read_parquet(run/f"states_m{t}.parquet") for t in protocol["targets"])
    all_rows=pd.concat(frames,ignore_index=True).sort_values(["role","uid","target","state_ordinal"])
    assert not all_rows.duplicated(["role","uid","target","state_ordinal"]).any()
    for role in all_rows.role.unique():
        assert set(all_rows.loc[all_rows.role==role,"uid"])==set(protocol["groups"][role])
    assert set(all_rows.role)==set(protocol["groups"])
    assert np.isfinite(all_rows[["detection","max_abs_error","mse","source_norm"]]).all().all()
    all_rows["short_history"]=1/all_rows["count"]
    def flops(n):
        values=rebuild(int(n))
        return values["gemm"]+values["scalar"]
    all_rows["rebuild_flops"]=all_rows["count"].map(flops)
    # Special functions are reported by the original formula, not assigned fabricated GEMM-equivalent costs.
    all_rows.to_parquet(out/"states.parquet",index=False)
    cal=all_rows[all_rows.role=="residual_calibration"]
    dev=all_rows[all_rows.role=="development"]
    rng=np.random.default_rng(170908)
    w=weights(dev); severe=dev.max_abs_error.to_numpy()>=.5
    metrics=dict(users=int(dev.uid.nunique()),states=len(dev),queries=len(dev)*16,
                 severe_rate=mean(severe.astype(float),w),
                 severe_uids=int(dev.loc[severe,"uid"].nunique()),
                 mean_max_abs_error=mean(dev.max_abs_error.to_numpy(),w),
                 mse=mean(dev.mse.to_numpy(),w),scores={})
    curves=[]; budgets=[]
    for s in SCORES:
        item=auc_summary(dev,s,rng)
        item["acceptance"]=curve(cal,dev,s,protocol["acceptance_quantiles"])
        item["budget"]=budget_curve(dev,s,protocol["budget_fractions"])
        metrics["scores"][s]=item
        curves.extend(dict(score=s,**v) for v in item["acceptance"])
        budgets.extend(dict(score=s,**v) for v in item["budget"])
    strata=[]
    for field in ("target","kind","length_band"):
        if field=="length_band":
            dev=dev.copy()
            dev[field]=pd.cut(dev["count"],[0,32,255,1023,1024],labels=["1-32","33-255","256-1023","1024"])
        for value,f in dev.groupby(field,observed=True):
            ww=weights(f)
            y=f.max_abs_error.to_numpy()>=.5
            strata.append(dict(field=field,value=str(value),users=int(f.uid.nunique()),states=len(f),
                severe_rate=mean(y.astype(float),ww),
                **{s+"_auroc":auc(y,f[s].to_numpy(),ww) for s in SCORES}))
    pd.DataFrame(strata).to_csv(out/"strata.csv",index=False)
    pd.DataFrame(curves).to_csv(out/"acceptance.csv",index=False)
    pd.DataFrame(budgets).to_csv(out/"budgets.csv",index=False)
    main_score=metrics["scores"]["detection"]
    retain=next(v for v in main_score["acceptance"] if v["calibration_quantile"]==.8)
    budget=next(v for v in main_score["budget"] if v["budget_fraction"]==.2)
    gates=dict(auroc=main_score["auroc"] is not None and main_score["auroc"]>=.7,
        ci=main_score["uid_bootstrap_95ci"] is not None and main_score["uid_bootstrap_95ci"][0]>.5,
        retained_risk=retain["severe_rate"] is not None and retain["severe_rate"]<=metrics["severe_rate"]/2,
        retained_coverage=retain["development_coverage"]>=.7,
        budget_recall=budget["severe_coverage"] is not None and budget["severe_coverage"]>=.4)
    low=all_rows.detection<=retain["threshold"]
    false_negatives=all_rows[low & (all_rows.max_abs_error>=.5)]
    false_negatives.to_parquet(out/"low_score_failures.parquet",index=False)
    false_negatives.sort_values("max_abs_error",ascending=False).groupby("role").head(20).to_csv(out/"false_negative_examples.csv",index=False)
    known=all_rows[all_rows.uid.isin([988060,547600,254830])]
    known.to_csv(out/"known_failures.csv",index=False)
    alternate=[]
    for threshold in protocol["secondary_error_thresholds"]:
        yy=dev.max_abs_error.to_numpy()>=threshold
        ww=weights(dev)
        alternate.append(dict(error_threshold=threshold,severe_rate=mean(yy.astype(float),ww),
            states=int(yy.sum()),uids=int(dev.loc[yy,"uid"].nunique()),
            **{s+"_auroc":auc(yy,dev[s].to_numpy(),ww) for s in SCORES},
            scores={s:auc_summary(dev,s,rng,error_threshold=threshold) for s in SCORES}))
    metrics.update(gates=gates,advance_all_gates=all(gates.values()),strata=strata,secondary_scales=alternate)
    # Count each role separately; fitting and historical outputs are never independent success evidence.
    audits=[]
    for role,f in all_rows.groupby("role"):
        ww=weights(f); yy=f.max_abs_error.to_numpy()>=.5
        audits.append(dict(role=role,users=int(f.uid.nunique()),states=len(f),severe_rate=mean(yy.astype(float),ww),
            low_score_serious_states=int(((f.detection<=retain["threshold"]) & yy).sum()),
            detection_auroc=auc(yy,f.detection.to_numpy(),ww)))
    metrics["roles"]=audits
    # Paired comparisons use the same UID draw for every score.
    uu,ii=np.unique(dev.uid.to_numpy(),return_inverse=True)
    ww=weights(dev); yy=dev.max_abs_error.to_numpy()>=.5
    rng_pair=np.random.default_rng(170909)
    differences=defaultdict(list)
    for _ in range(500):
        multiplicity=np.bincount(rng_pair.integers(len(uu),size=len(uu)),minlength=len(uu))
        wb=ww*multiplicity[ii]
        av={s:auc(yy,dev[s].to_numpy(),wb) for s in SCORES}
        for s in SCORES[1:]:
            if av["detection"] is not None and av[s] is not None:
                differences[s].append(av["detection"]-av[s])
    metrics["paired_auroc_difference"]={s:dict(
        detection_minus_control=metrics["scores"]["detection"]["auroc"]-metrics["scores"][s]["auroc"],
        uid_bootstrap_95ci=np.quantile(v,[.025,.975]).tolist()) for s,v in differences.items()}
    write_json(out/"summary.json",metrics)
    evaluation_ledger=defaultdict(float)
    original_ledger=defaultdict(float)
    reliable_runs=[]
    for name,record in costs.items():
        config=json.loads((root/name/"configuration.json").read_text())
        for key,value in record["ledger_seconds"].items():
            original_ledger[key]+=value
            if config["device"]=="cuda:0":
                evaluation_ledger[key]+=value
        if config["device"]=="cuda:0":
            reliable_runs.append(name)
    geometry=json.loads((root/"geometry_01/summary.json").read_text())
    geometry_checks=[json.loads((root/"geometry_01"/f"checks_m{t}.json").read_text()) for t in protocol["targets"]]
    p,heads,layers,r,aug,d,q=1281,6,6,33,33,192,16
    queries=len(all_rows)*16
    gram_gemm=0
    for check in geometry_checks:
        s=check["scenes"]
        gram_gemm += layers*(2*s*heads*q*aug**2 + 2*r*r*s*heads*aug*aug
                            + 2*s*heads*q*aug*d + 2*heads*r*s*aug*d + 2*s*q*d*d)
    arithmetic=dict(query_panels=len(all_rows),queries=queries,
        triangular_solve_flops_per_query=layers*heads*p*p,
        triangular_solve_total_flops=queries*layers*heads*p*p,
        dense_inverse_quadratic_reference_flops_per_query=2*layers*heads*p*p,
        check_other_arithmetic_per_query=layers*(heads*(2*32+33*33+2*p-1+1)+2*192),
        check_sqrt_calls_per_query=layers*heads,
        gram_main_contraction_flops=gram_gemm,
        cholesky_leading_flops=len(protocol["targets"])*layers*heads*p**3/3,
        factor_storage_bytes=sum(path.stat().st_size for path in (root/"geometry_01").glob("cholesky_m*.pt")),
        scope="FP64 triangular solve actually used; leading preparation contractions, small scalar setup not fully isolated; no time-to-FLOPs conversion")
    timing_probe=json.loads((root/"timing16_02/summary.json").read_text())
    cost_record=dict(runs=costs,valid_cuda0_ledger_seconds=dict(evaluation_ledger),valid_cuda0_runs=reliable_runs,
        original_mixed_device_ledger_seconds_not_valid=dict(original_ledger),
        corrected_timing_probe=timing_probe,
        timing_invalidation="Initial runner omitted set_device; shared timed() synchronized cuda:0. cuda:1/2/3 component timings are invalid; scores/raw/FLOPs/wall remain valid. See medium_01/timing_invalidation.md.",
        geometry_recovery=geometry,
        arithmetic=arithmetic,evaluation_process_seconds=sum(v["elapsed_seconds"] for v in costs.values()),
        medium_queue_wall_seconds=json.loads((root/"medium_01/summary.json").read_text())["elapsed_seconds"],
        peak_gpu_mib=max([v["peak_gpu_mib"] for v in costs.values()]+[geometry["peak_gpu_mib"]]),
        cost_scope="research diagnostics including teachers; frozen Design1 original preparation is an external dependency, not included or claimed free")
    write_json(out/"cost.json",cost_record)
    lines=["# Design 2 第一轮检测：中规模开发结果","",
        "冻结C和原坐标；Current在每个决策cutover重建合法保留前缀。所有16候选均在同一时刻。",
        "这是假设性查询面板上的残余检测，不是推荐AUC、已部署检测或真实重建策略收益。","",
        f"主开发：{metrics['users']} UID，{metrics['states']}状态，{metrics['queries']}查询；校准512 UID。",
        f"严重残余定义：状态面板最大绝对logit误差≥0.5；UID/目标等权严重率 {metrics['severe_rate']:.2%}。",
        "所有阈值在校准组确定，四目标和全部预定状态保留；确认集未读，单backbone seed17。","",
        "| 分数 | 严重失准AUROC [UID 95%描述区间] | 校准80%阈值下开发覆盖 | 续用严重率 | 20%重建FLOPs严重覆盖 |",
        "| --- | --- | --- | --- | --- |"]
    for s,item in metrics["scores"].items():
        cv=next(v for v in item["acceptance"] if v["calibration_quantile"]==.8)
        bc=next(v for v in item["budget"] if v["budget_fraction"]==.2)
        ci=item["uid_bootstrap_95ci"]
        lines.append(f"| {s} | {item['auroc']:.4f} [{ci[0]:.4f}, {ci[1]:.4f}] | {cv['development_coverage']:.2%} | {cv['severe_rate']:.2%} | {bc['severe_coverage']:.2%} |")
    lines += ["",f"预定研究推进条件：{gates}；全部通过：{all(gates.values())}。",
        "这不是统计认证；即使整体条件通过，也须检查各目标/状态类型的低分严重失败。","",
        "完整按目标、状态类型、长度结果见 strata.csv；已知失败见 known_failures.csv，",
        "低分严重失败全量保留于 low_score_failures.parquet，未按结果删除用户。",
        "相同预算用每状态当场重算的矩阵乘加及标量FLOPs，特殊函数另有原公式，未随意换算。",
        "分数/费用排序是离线识别对照，多个替代场景的费用之和不是已实现连续服务账本。",
        "准备、读取、检查、Exact评价与回放时间保留 cost.json。"]
    low10=next(v for v in main_score["acceptance"] if v["calibration_quantile"]==.1)
    lines += ["","## 结论与边界","",
        "**首版全局最大聚合分数没有建立可靠的续用依据；不据此进入压缩部署或真实重建系统。**",
        f"0.5主尺度只有{metrics['severe_uids']}名严重失败UID、{int(severe.sum())}个状态，AUROC区间很宽，"
        "不能将未通过解释成已证明所有内部几何指标无效，也不能宣称简单对照统计显著更好。",
        f"最低约{low10['development_coverage']:.1%}分数区域的严重率反而为{low10['severe_rate']:.3%}，"
        f"高于总体{metrics['severe_rate']:.3%}；低分区域并非单调更安全。",
        "保留80%状态仍有严重残余，说明这不是只要进一步降低同一阈值就能解决的问题。","",
        "**同费用筛查有正面信号，必须与原始质量判据分开。** "
        f"分数/费用在预定5%重建预算下覆盖{main_score['budget'][0]['severe_coverage']:.1%}严重状态；"
        "对照的完整曲线见budgets.csv。它同时改变了历史长度的排序作用，不等同于原始分数准确，"
        "也不是已实现的调度器。预算只含假想重建费用，尚未扣除昂贵的几何检查和准备费用。",
        "原始分数在同长度范围内有更强的点估计信号，而跨长度全局排序弱；这提示尺度可比性值得"
        "后续研究，但不是据本结果立即重调阈值、改归一化或按长度选择不同方法的理由。","",
        "## 预定第二误差尺度","",
        "| 最大绝对残余阈值 | 涉及UID/状态 | 检测AUROC [UID区间] | 年龄AUROC | 短历史AUROC | 源范数AUROC |",
        "| --- | --- | --- | --- | --- | --- |"]
    for item in alternate:
        a=item["scores"]["detection"]; ci=a["uid_bootstrap_95ci"]
        lines.append(f"| {item['error_threshold']} | {item['uids']}/{item['states']} | {a['auroc']:.4f} [{ci[0]:.4f}, {ci[1]:.4f}] | "
            f"{item['version_age_auroc']:.4f} | {item['short_history_auroc']:.4f} | {item['source_norm_auroc']:.4f} |")
    lines += ["","这两个尺度在输出前已固定，不从其中选最有利的一个替代主结果。","",
        "## 低分失败与适用范围","",
        "完整低分严重失败保留于low_score_failures.parquet；拟合组不混入独立成功率。",
        "UID988060原已知拟合内失败在本次同cutover候选面板仍表现为低分、大残余：","",
        "| 目标 | 该UID最大绝对残余 | 该UID最大检测分数 |",
        "| --- | --- | --- |"]
    for t,f in known[known.uid==988060].groupby("target"):
        lines.append(f"| M{t} | {f.max_abs_error.max():.5f} | {f.detection.max():.2f} |")
    lines += ["",f"校准80%检测阈值为{retain['threshold']:.2f}。这些是本次实际cutover的16候选，"
        "与旧fit16/held64的跨时间面板不同，不把旧残余数字当成本次Exact参考。",
        "历史128组和原256拟合组的完整结果保留于summary.json.roles；例如旧短历史难例"
        "可能在本面板没有超过0.5，这同样保留，不能定向挑出新的最坏query。",
        "连续状态的主尺度检测AUROC约0.54；生命周期尾部组没有主尺度严重例，"
        "不能据此估计该组严重失败的识别能力。Exact源控制是原拟合诊断控制；"
        "真实系统若已知精确，应按确定语义绕过修正，不将该控制当一般服务状态。","",
        "## 准备、检查与存储","",
        f"原256用户H恢复{geometry['elapsed_seconds']:.2f}s；四目标FP64 Cholesky文件合计"
        f"{arithmetic['factor_storage_bytes']/2**30:.3f}GiB。各目标原场景与冻结logit匹配，未重拟合C或获取新教师响应。",
        f"1920总UID（512+1024+128+256）的检测评价共{len(all_rows)}状态/{queries}查询。",
        "**计时修订：** 原非0号GPU进程未设置current device，共享timed()同步了cuda:0，"
        "其组件计时失效；原始日志/账本保留，分数、残余、理论FLOPs与队列墙钟不受影响。",
        f"原GPU0有效子集的几何检查{evaluation_ledger['geometry_check']:.2f}s、适配读取"
        f"{evaluation_ledger['adapted_read']:.2f}s；不能把它当全1920UID合计。",
        f"显式set_device后的同16UID四目标探针：几何检查"
        f"{timing_probe['ledger_seconds']['geometry_check']:.2f}s，适配读取"
        f"{timing_probe['ledger_seconds']['adapted_read']:.2f}s，墙钟{timing_probe['elapsed_seconds']:.2f}s。"
        "该重复仅核验计时和输出一致性，不重复纳入校准/开发样本。",
        f"四GPU队列墙钟{cost_record['medium_queue_wall_seconds']:.2f}s，最大单GPU峰值"
        f"{cost_record['peak_gpu_mib']/1024:.2f}GiB。并发任务GPU秒之和不等于墙钟，读取计时含诊断trace。",
        f"实际实现通过三角求解计算准确二次型，主项每查询{arithmetic['triangular_solve_flops_per_query']:,} FLOPs，"
        f"全评价主项约{arithmetic['triangular_solve_total_flops']/1e12:.3f} TFLOPs；"
        "此前1.18亿是稠密逆二次型的参考算法计数，不能误写成本轮三角求解实测。",
        "Gram主要收缩、Cholesky主项、输入变换/范数及sqrt次数另见cost.json；"
        "小标量准备尚未全部隔离，不把这些主项冒称完整发布账本。冻结Design1的原始准备仍是外部依赖。",
        "当前计算不满足廉价在线检查；正面费用排序不包含这一开销，不能宣布净节省。","",
        "后续若继续，应另立小型、预先固定的尺度校准/独立观测研究问题；本轮不继续改阈值、"
        "不加入敏感度、不增加风险网络、不实施调度器或真实重建。"]
    (out/"report.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(output=str(out),primary=metrics["scores"]["detection"],gates=gates)))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs",nargs="+",required=True)
    parser.add_argument("--output",default="analysis/detection_01")
    main(parser.parse_args())
