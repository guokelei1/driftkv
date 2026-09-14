"""Audit the exact source-conditioned quadratic using retained replay evidence."""

import json
from collections import defaultdict

import numpy as np
import pandas as pd

from design.data import ROOT
from design.report_native_flops import query_ops, read_cost
from design2.audit_budget import finite_metrics
from design2.report_detection import weights, mean
from design2.conditional_check import OUT, sha, CONFIG, SOURCE
from hstu_kvcache.design2.conditional import arithmetic

REPORT=ROOT/"results/design2/analysis/conditional225_01"


def cost(frame, preparation):
    n=len(frame); direct=int((frame.stage=="formula").sum()); computed=n-direct
    fallback=int((frame.stage=="accurate_fallback").sum())
    ops=arithmetic(16); old=ops["exact_panel_flops"]
    check=computed*ops["panel_flops"]+fallback*old
    total=check+preparation
    return dict(users=int(frame.uid.nunique()),states=n,
        stages=frame.stage.value_counts().to_dict(),formula_accept=int(((frame.stage=="formula")&frame.accept).sum()),
        formula_reject=int(((frame.stage=="formula")&~frame.accept).sum()),
        numerical_fallback_states=int(((frame.stage=="accurate_fallback")&frame.numerically_invalid).sum()),
        threshold_fallback_states=int(frame.threshold_fallback.sum()),
        mismatches=int((frame.accept!=frame.reference_accept).sum()),
        accurate_original_flops=n*old,formula_only_flops=computed*old,
        state_matrix_flops=computed*(ops["state_gemm"]+ops["state_scalar_allowance"]),
        query_check_flops=computed*(ops["query_gemm"]+ops["query_scalar_allowance"]+2),
        fallback_flops=fallback*old,online_flops=check,
        inverse_preparation_flops=preparation,preparation_and_check_flops=total,
        online_saving_vs_formula=1-check/(computed*old),net_saving_vs_formula=1-total/(computed*old),
        net_saving_vs_original=1-total/(n*old),
        cost_scope="Current prototype state contraction,225D query arithmetic/guards, every fallback and all4-target inverse preparation. Conservative scalar charges, sqrt/comparisons/casts and memory separate. Common Design1/teacher/H dependencies handled separately.")


def main():
    assert not REPORT.exists(),"Preserve completed evidence."
    queue=json.loads((OUT/"population/summary.json").read_text())
    assert queue["status"]=="complete"
    inv=json.loads((OUT/"inverse/summary.json").read_text())
    assert sha(CONFIG)==inv["configuration_sha256"] and sha(SOURCE)==inv["conditional_source_sha256"]
    fitted_path=ROOT/"configs/design2/scale_calibration_01_fitted.json"
    fitted=json.loads(fitted_path.read_text()); tau=fitted["thresholds"]["calibrated"]["0.8"]
    chunks=[OUT/"chunks/residual_calibration_0000"]+[OUT/"chunks"/j["name"] for j in queue["jobs"]]
    frames=[]; runs=[]; ledger=defaultdict(float); qframes=[]
    for path in chunks:
        config=json.loads((path/"configuration.json").read_text())
        assert config["fitted_sha256"]==sha(fitted_path)
        assert config["configuration_sha256"]==sha(CONFIG) and config["conditional_source_sha256"]==sha(SOURCE)
        run=json.loads((path/"summary.json").read_text()); assert run["status"]=="complete"
        runs.append(run)
        for k,v in run["ledger_seconds"].items(): ledger[k]+=v
        for target in (1,3,4,5):
            frames.append(pd.read_parquet(path/f"states_m{target}.parquet").assign(chunk=path.name))
            qframes.append(pd.read_parquet(path/f"queries_m{target}.parquet").assign(role=config["role"]))
    all_rows=pd.concat(frames,ignore_index=True); qrows=pd.concat(qframes,ignore_index=True)
    assert not all_rows.duplicated(["role","uid","target","state_ordinal"]).any()
    protocol=json.loads((ROOT/"configs/design2/detection_01.json").read_text())
    for role in ("residual_calibration","development"):
        assert set(all_rows.loc[all_rows.role==role,"uid"])==set(protocol["groups"][role])
    valid=qrows[qrows.numerically_valid]
    np.testing.assert_allclose(valid.point_u,valid.reference_u,rtol=1e-8,atol=1e-8)
    assert (valid.u_lower<=valid.reference_u).all() and (valid.reference_u<=valid.u_upper).all()
    assert (all_rows.accept==all_rows.reference_accept).all()
    previous=pd.read_parquet(ROOT/"results/design2/analysis/scale_calibration_01/development_predictions.parquet")
    dev=all_rows[all_rows.role=="development"].merge(previous[["uid","target","state_ordinal","calibrated","max_abs_error","rebuild_flops"]],
        on=["uid","target","state_ordinal"],validate="one_to_one")
    assert (dev.accept==(dev.calibrated<=tau)).all()
    old=json.loads((ROOT/"results/design2/analysis/scale_calibration_01/summary.json").read_text())
    w=weights(dev); keep=dev.accept.to_numpy(); e=dev.max_abs_error.to_numpy()
    quality=dict(coverage=float(w[keep].sum()/w.sum()),severe_rates=[mean((e[keep]>=t).astype(float),w[keep]) for t in (.5,.1,1.)],
        severe_states=[int((keep&(e>=t)).sum()) for t in (.5,.1,1.)],
        severe_uids=[int(dev.loc[keep&(e>=t),"uid"].nunique()) for t in (.5,.1,1.)])
    reference=next(v for v in old["methods"]["calibrated"]["acceptance"] if v["quantile"]==.8)
    np.testing.assert_allclose([quality["coverage"],*quality["severe_rates"]],[reference["coverage"],*reference["severe_rates"]],atol=1e-15)
    prep=inv["total_arithmetic_charge"]
    roles={role:cost(f,prep) for role,f in all_rows.groupby("role")}
    both=cost(all_rows,prep); c=roles["development"]
    oldprep=old["costs"]["shared_geometry_preparation_leading_flops"]
    denominator=float(dev.rebuild_flops.sum())
    c.update(shared_old_H_preparation=oldprep,exact_rebuild_all_flops=denominator,
        original_plus_shared_H_flops=c["accurate_original_flops"]+oldprep,
        formula_plus_shared_H_flops=c["formula_only_flops"]+oldprep,
        conditional_plus_shared_H_flops=c["preparation_and_check_flops"]+oldprep,
        conditional_plus_shared_H_over_exact_all=(c["preparation_and_check_flops"]+oldprep)/denominator)
    candidate_counts=[]
    for q in (1,2,4,16):
        a=arithmetic(q); computed=c["states"]-c["stages"].get("formula",0)
        candidate_counts.append(dict(queries=q,**a,
            development_no_fallback_total_flops=computed*a["panel_flops"]+prep,
            formula_exact_comparison_flops=computed*a["exact_panel_flops"],
            scope="For q!=16 this is algebraic cost only, not new quality or fallback measurement; source matrix must be rebuilt for each changed state."))
    # C reads used to restore absent features; retain original batch boundaries.
    read_flops=0; reads=0; specials=defaultdict(int)
    for (chunk,target),f in all_rows.groupby(["chunk","target"],sort=False):
        canary=chunk=="residual_calibration_0000"
        for count,g in f.groupby("count",sort=False):
            for i in range(0,len(g),4):
                batch=g.iloc[i:i+4]
                if not canary and (batch.stage=="formula").all(): continue
                ops=query_ops(int(count),16); reads+=len(batch)
                read_flops+=len(batch)*(ops["gemm"]+ops["scalar"]+read_cost(16,1))
                for k,v in ops.items():
                    if k not in ("gemm","scalar"): specials[k]+=len(batch)*v
    results=[r for run in runs for r in run["results"]]
    assert reads==sum(r["input_read_states"] for r in results)
    summary=dict(status="complete",threshold=tau,roles=roles,combined=both,quality=quality,
        preparation=inv,candidate_counts=candidate_counts,
        max_conditional_query_relative_delta=max(r["max_conditional_query_relative_delta"] for r in results),
        max_original_accurate_relative_delta=max(r["max_original_accurate_relative_delta"] for r in results),
        max_adapt_logit_delta=max(r["max_adapt_logit_delta"] for r in results),
        checked_query_scores=len(valid),max_interval_relative_width=max(r["max_numerical_interval_relative_width"] for r in results),
        input_recovery=dict(read_states=reads,read_flops=read_flops,special_calls=dict(specials),
            note="Frozen C forwards and actual source replay occurred; no new teacher output. C/source view preparation are shared serving dependencies, research rehydration is reported separately."),
        timing=dict(inverse_seconds=inv["elapsed_seconds"],canary_seconds=runs[0]["elapsed_seconds"],
            population_wall_seconds=queue["elapsed_seconds"],valid_summed_gpu_component_seconds=dict(ledger)),
        validation_extra_conditioned_states=sum(r["canary_extra_conditioned_states"] for r in results),
        validation_extra_accurate_states=sum(r["canary_extra_accurate_states"] for r in results),
        shared_storage_bytes=inv["storage_bytes"]+old["costs"]["factor_storage_bytes"],
        retained_cholesky_bytes=old["costs"]["factor_storage_bytes"],
        peak_gpu_mib=max(run["peak_gpu_mib"] for run in runs),new_teacher_outputs=0,C_refitted=False,confirmation_read=False,
        positive_preparation_and_check_saving=c["preparation_and_check_flops"]<c["formula_only_flops"],
        cost_boundary="New inverse preparation and all detector arithmetic are charged; old H preparation shown explicitly. Original C/calibration teachers and full source replay/writer preparation are common remaining dependencies, not zero. Ratio to Exact-All is a detection accounting subtotal, not full Design2 lifecycle/system cost.")
    REPORT.mkdir(parents=True)
    (REPORT/"summary.json").write_text(json.dumps(finite_metrics(summary),indent=2,allow_nan=False)+"\n")
    all_rows.to_parquet(REPORT/"states.parquet",index=False)
    qrows.to_parquet(REPORT/"queries.parquet",index=False)
    dev[keep&(e>=.5)].to_csv(REPORT/"retained_severe_states.csv",index=False)
    counts=all_rows.groupby(["role","target","stage"]).size().unstack(fill_value=0)
    counts.to_csv(REPORT/"stage_counts.csv")
    pd.DataFrame(candidate_counts).to_csv(REPORT/"candidate_costs.csv",index=False)
    write_report(summary)
    print(json.dumps(dict(development=c,quality=quality,max_query_relative_delta=summary["max_conditional_query_relative_delta"])))


def write_report(s):
    c=s["roles"]["development"]; inv=s["preparation"]; q=s["quality"]
    lines=["# Design 2：共享源编码的精确225维状态二次型", "",
        "本轮固定C、H、12组a/b和完整80%阈值，只改变二次型的计算组织。没有低秩近似、尺度重拟合或新教师输出。", "",
        f"准备＋检查相对仅公式短路/准确求解的净节省为{c['net_saving_vs_formula']:.2%}；完整判定保持：{c['mismatches']==0}。",
        "这是检测模块在相同Design1输入/校准依赖下的净节省；全系统生命周期和真实重建收益尚未验证。", "",
        "## 1. 公式与实现", "",
        "实际联合特征为f=[x⊗q;o]，x33维，q33维，o192维。将H⁻¹的源源块按xxᵀ收缩，源/native交叉块按x收缩，native/native块共享，得到225×225矩阵M(x)。",
        "同一状态的16个实际query/native输入仅计算zᵀM(x)z，仍保留全部1281维特征的作用。小测试使用不同的源/query维数，以核查源维在外的排列不是恰巧同为33而掩盖错误。",
        "M按当前批次的一层临时分配，用完释放。源编码/缓存修订或目标模型变化需要重新生成，不为全体UID长期保存。",
        "共享逆矩阵用显式T=solve(L,I)、A=TᵀT生成，核验LT−I并加入标准FP64舍入预算；状态收缩与候选求值的误差预算另加。",
        "非正/非有限/符号不确定的点估计或跨阈值区间均回退原完整求解；只约束区间端点，绝不截断负点估计来制造通过。阈值未调整。", "",
        "## 2. 原512/1024UID完整验证", "",
        "| 集合 | UID/状态 | 公式短路 | 225维检查直接判定 | 原求解回退 | 数值/阈值回退 | 判定差异 |",
        "| --- | --- | --- | --- | --- | --- | --- |"]
    for name,r in s["roles"].items():
        counts=r["stages"]
        lines.append(f"| {name} | {r['users']}/{r['states']} | {counts.get('formula',0)} | {counts.get('conditional',0)} | {counts.get('accurate_fallback',0)} | {r['numerical_fallback_states']}/{r['threshold_fallback_states']} | {r['mismatches']} |")
    lines += ["",f"核对{s['checked_query_scores']}个候选准确分数，最大相对差{s['max_conditional_query_relative_delta']:.3g}；"
        f"恢复的adapt输出最大差{s['max_adapt_logit_delta']:.3g}，所有有效分数落在数值区间内。探针同时逐候选重新精算，人口阶段仅必要时回退并对照既有准确分数。",
        "公式短路状态不需要在线u；探针仍对全部状态核查代数等价，其额外费用单列。不能把未计算的公式状态分数宣称为新测量。",
        f"开发续用覆盖{q['coverage']:.4%}，0.5尺度严重率{q['severe_rates'][0]:.4%}，仍漏检{q['severe_uids'][0]}UID/{q['severe_states'][0]}状态，与冻结准确检测完全一致。",
        f"0.1/1.0尺度续用严重率分别{q['severe_rates'][1]:.4%}/{q['severe_rates'][2]:.4%}，同样未改变。原M1短历史漏检、M5组拒绝代价仍保留，不称质量提升。", "",
        "## 3. 完整新增检测准备和执行费用", "",
        "普通算术按实际矩阵形状和显式求解流程计数，标量/数值预算采用保守收费；sqrt、比较、类型转换与内存访问不伪造为矩阵FLOPs。", "",
        "| 开发面板项目 | TFLOPs |",
        "| --- | --- |"]
    rows=[("原逐候选准确检查",c["accurate_original_flops"]),("仅公式短路＋准确检查",c["formula_only_flops"]),
        ("状态矩阵生成",c["state_matrix_flops"]),("225维查询及数值预算",c["query_check_flops"]),
        ("数值/阈值准确回退",c["fallback_flops"]),("新方法在线合计",c["online_flops"]),
        ("全部四目标逆矩阵准备及核验",c["inverse_preparation_flops"]),
        ("新方法准备＋检查",c["preparation_and_check_flops"])]
    for name,value in rows: lines.append(f"| {name} | {value/1e12:.6f} |")
    lines += ["",f"在线相对仅公式短路节省{c['online_saving_vs_formula']:.2%}；即把四目标全部新增准备都收费给开发面板，仍净省{c['net_saving_vs_formula']:.2%}。",
        f"若一次准备同时供512+1024面板使用，净节省为{s['combined']['net_saving_vs_formula']:.2%}；不会给校准和开发各重复收一次后再相加。",
        f"旧H的Gram/Cholesky共同准备主项另{c['shared_old_H_preparation']/1e12:.6f}TFLOPs。加上它，原准确检测/仅公式短路分别{c['original_plus_shared_H_flops']/1e12:.6f}/{c['formula_plus_shared_H_flops']/1e12:.6f}TFLOPs，"
        f"新方法{c['conditional_plus_shared_H_flops']/1e12:.6f}TFLOPs，约为本开发Exact-All算术的{c['conditional_plus_shared_H_over_exact_all']:.2%}。",
        "最后这个比例是已计的检测准备/检查小计，**不是全部Design2或服务成本比例**。原C拟合、残余校准教师、源回放/写入等依赖未假装免费；同一基线中这些共同工作不因二次型重排改变。", "",
        "## 4. 候选数边界", "",
        "下表每个状态都重新生成M；省略发布准备和回退的每状态列为公式对照。人口合计额外收全部新增逆准备，但q≠16未重新测质量或回退比例。", "",
        "| 候选数 | 新方法每状态GFLOPs | 原求解每状态GFLOPs | 开发新方法合计含新准备（无回退）TFLOPs | 仅公式＋原求解TFLOPs |",
        "| --- | --- | --- | --- | --- |"]
    for v in s["candidate_counts"]:
        lines.append(f"| {v['queries']} | {v['panel_flops']/1e9:.6f} | {v['exact_panel_flops']/1e9:.6f} | {v['development_no_fallback_total_flops']/1e12:.6f} | {v['formula_exact_comparison_flops']/1e12:.6f} |")
    lines += ["", "单候选的新方法更贵；2候选的在线小余量也不能覆盖当前面板的新增准备。不能宣称每种候选负载均节省。",
        "4/16候选列依赖同一状态内复用M；跨缓存或目标修订必须再生成。没有实现跨UID矩阵长期缓存或以未来请求数调度。", "",
        "## 5. 存储、输入恢复与资源", "",
        f"新增共享逆块{inv['storage_bytes']/2**30:.3f}GiB；保留准确回退Cholesky后共{s['shared_storage_bytes']/2**30:.3f}GiB（四目标合计）。",
        f"若将全部36块M保存，每状态需{arithmetic(16)['state_matrix_bytes']/2**20:.2f}MiB；本实现仅在一层4状态批次内临时生成，矩阵本体约{arithmetic(16)['layer_batch4_matrix_bytes']/2**20:.2f}MiB，另有拼接/工作区，不称零内存。",
        f"共享准备{inv['elapsed_seconds']:.2f}s；16UID探针{s['timing']['canary_seconds']:.2f}s；余下四GPU队列{s['timing']['population_wall_seconds']:.2f}s；峰值{s['peak_gpu_mib']/1024:.2f}GiB/GPU。",
        f"本轮恢复输入执行C前向和真实源回放。C读取{s['input_recovery']['read_states']}状态，沿用query/read公式的普通算术主项{s['input_recovery']['read_flops']/1e12:.6f}TFLOPs；源回放、摘要/视图组件时间另记summary。",
        "这是研究恢复未保存特征的实际工作，不能说本轮没有模型前向。源exact_source_control包含Current输入构造，仍没有新教师输出、C/校准重拟合、确认集读取或真实重建策略执行。",
        f"探针额外条件化{s['validation_extra_conditioned_states']}状态、额外准确求解{s['validation_extra_accurate_states']}状态，只用于验证。它们的研究算术另为"
        f"{(s['validation_extra_conditioned_states']*arithmetic(16)['panel_flops']+s['validation_extra_accurate_states']*arithmetic(16)['exact_panel_flops'])/1e9:.3f}GFLOPs，不计为部署必须反复支付的检查。", "",
        "## 6. 结论与下一步", "",
        f"判定等价与新增准备/检查净节省同时通过：{s['positive_preparation_and_check_saving']}。两项小型数值检查及完整512/1024UID核验已完成，未调整阈值或特征方向。",
        "本轮通过，保留此精确重排，后续研究真实状态重建与连续回放；本轮没有实现重建调度器，也未证明端到端AUC/生命周期收益。",
        "当前证据支持结束检测计算方案的本轮搜索；不调整冻结校准/阈值，不继续固定32方向方案。", "",
        "![精确收缩费用与候选数边界](../../../../figures/pic/design2/conditional225_01/conditional225.png)"]
    (REPORT/"report.md").write_text("\n".join(lines)+"\n")


if __name__=="__main__": main()
