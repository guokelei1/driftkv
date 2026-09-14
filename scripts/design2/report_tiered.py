"""Aggregate fixed32 equivalent-decision replay and charge its arithmetic."""

import json
from collections import defaultdict

import numpy as np
import pandas as pd

from design.data import ROOT
from design.report_native_flops import query_ops, read_cost
from design2.audit_budget import check_cost, finite_metrics
from design2.report_detection import mean, weights
from design2.tiered_check import OUT

REPORT = ROOT/"results/design2/analysis/tiered32_01"
STAGES = ("formula", "bound_accept", "bound_reject", "exact_fallback")


def summarize(frame, exact_panel, spectral):
    counts = {s: int((frame.stage == s).sum()) for s in STAGES}
    n = len(frame); coarse = n-counts["formula"]; fallback = counts["exact_fallback"]
    p, k, units = 1281, 32, 16*36
    # Matrix contractions are a verified lower bound independent of scalar/eigen models.
    projection_per_panel = units*2*p*k
    # Includes all feature formation, quadratic arithmetic, guards and affine
    # decision. The128 scalar allowance per unit exceeds the41 counted scalar
    # operations in quadratic_bounds plus broadcast/setup arithmetic.
    coarse_panel = projection_per_panel+16*check_cost()["other_per_query"]+units*(5*k-2+128)+10
    exact_total = n*exact_panel
    fallback_total = fallback*exact_panel
    projection_total = coarse*projection_per_panel
    online_model = fallback_total+coarse*coarse_panel
    prep_verified = spectral["verified_preparation_gemm_flops"]
    prep_model = prep_verified+spectral["preparation_scalar_allowance_flops"]+spectral["eigendecomposition_cost_model_flops"]
    # Recover how many C reads were actually executed: preserve original mixed
    # batches, skip only batches whose every state is handled by formula.
    read_flops, read_states = 0, 0
    special = defaultdict(int)
    for _, chunk in frame.groupby("chunk", sort=False):
        for _, target in chunk.groupby("target", sort=False):
            for length, group in target.groupby("count", sort=False):
                for offset in range(0, len(group), 4):
                    batch = group.iloc[offset:offset+4]
                    if (batch.stage == "formula").all():
                        continue
                    ops = query_ops(int(length), 16)
                    read_states += len(batch)
                    read_flops += len(batch)*(ops["gemm"]+ops["scalar"]+read_cost(16, 1))
                    for key, value in ops.items():
                        if key not in ("gemm", "scalar"):
                            special[key] += len(batch)*value
    return dict(states=n, users=int(frame.uid.nunique()), stages=counts,
        stage_fractions={s: counts[s]/n for s in STAGES}, formula_accept=int(((frame.stage == "formula") & frame.accept).sum()),
        formula_reject=int(((frame.stage == "formula") & ~frame.accept).sum()),
        decision_mismatches=int((frame.accept != frame.reference_accept).sum()),
        exact_check_baseline_flops=exact_total, formula_only_flops=coarse*exact_panel,
        coarse_panel_projection_flops=projection_per_panel, coarse_panel_arithmetic_allowance=coarse_panel,
        coarse_check_flops=coarse*coarse_panel, fallback_flops=fallback_total,
        online_flops_model=online_model, online_saved_fraction=1-online_model/exact_total,
        verified_total_lower_bound_flops=fallback_total+projection_total+prep_verified,
        verified_total_lower_bound_over_exact=(fallback_total+projection_total+prep_verified)/exact_total,
        spectral_preparation_model_flops=prep_model,
        tiered_total_model_flops=online_model+prep_model,
        tiered_total_model_over_exact=(online_model+prep_model)/exact_total,
        break_even_repeated_equal_panels=prep_model/(exact_total-online_model) if online_model<exact_total else None,
        input_recovery_read_states=read_states, input_recovery_read_flops=read_flops,
        input_recovery_special_calls=dict(special),
        bound_special_calls=dict(sqrt=coarse*units*4, clamps=coarse*units*6,
            note="sqrt, clamps, comparisons/casts are not assigned invented GEMM-equivalent FLOPs"))


def main():
    assert not REPORT.exists(), "Preserve completed audit."
    queue = json.loads((OUT/"population/summary.json").read_text())
    assert queue["status"] == "complete"
    chunks = [OUT/"chunks/residual_calibration_0000"]+[OUT/"chunks"/r["name"] for r in queue["jobs"]]
    frames=[]; summaries=[]; ledgers=defaultdict(float)
    for chunk in chunks:
        record=json.loads((chunk/"summary.json").read_text())
        assert record["status"] == "complete"
        summaries.append(record)
        for key, value in record["ledger_seconds"].items():
            ledgers[key] += value
        for target in (1,3,4,5):
            frames.append(pd.read_parquet(chunk/f"states_m{target}.parquet").assign(chunk=chunk.name))
    all_rows=pd.concat(frames, ignore_index=True)
    assert not all_rows.duplicated(["role","uid","target","state_ordinal"]).any()
    protocol=json.loads((ROOT/"configs/design2/detection_01.json").read_text())
    for role in ("residual_calibration","development"):
        assert set(all_rows.loc[all_rows.role==role,"uid"]) == set(protocol["groups"][role])
    previous=pd.read_parquet(ROOT/"results/design2/analysis/scale_calibration_01/development_predictions.parquet")
    dev=all_rows[all_rows.role=="development"].merge(previous[["uid","target","state_ordinal","calibrated","max_abs_error","rebuild_flops"]],
        on=["uid","target","state_ordinal"],validate="one_to_one")
    fitted=json.loads((ROOT/"configs/design2/scale_calibration_01_fitted.json").read_text())
    tau=fitted["thresholds"]["calibrated"]["0.8"]
    assert (dev.accept == (dev.calibrated<=tau)).all()
    assert (all_rows.accept == all_rows.reference_accept).all()
    w=weights(dev); keep=dev.accept.to_numpy(); e=dev.max_abs_error.to_numpy()
    quality=dict(coverage=float(w[keep].sum()/w.sum()),
        severe_rates=[mean((e[keep]>=t).astype(float),w[keep]) for t in (.5,.1,1.)],
        severe_states=[int((keep & (e>=t)).sum()) for t in (.5,.1,1.)],
        severe_uids=[int(dev.loc[keep & (e>=t),"uid"].nunique()) for t in (.5,.1,1.)])
    prior=json.loads((ROOT/"results/design2/analysis/scale_calibration_01/summary.json").read_text())
    old80=next(r for r in prior["methods"]["calibrated"]["acceptance"] if r["quantile"]==.8)
    np.testing.assert_allclose([quality["coverage"],*quality["severe_rates"]], [old80["coverage"],*old80["severe_rates"]],atol=1e-15)
    spectral=json.loads((OUT/"spectrum/summary.json").read_text())
    panel=check_cost()["panel_arithmetic"]+2
    by_role={r:summarize(g,panel,spectral) for r,g in all_rows.groupby("role")}
    both=summarize(all_rows,panel,spectral)
    devcost=by_role["development"]
    old_preparation=prior["costs"]["shared_geometry_preparation_leading_flops"]
    exact_all=float(dev.rebuild_flops.sum())
    devcost.update(exact_rebuild_all_flops=exact_all,
        original_with_common_preparation_over_exact_all=(devcost["exact_check_baseline_flops"]+old_preparation)/exact_all,
        tiered_verified_lower_bound_with_common_preparation_over_exact_all=(devcost["verified_total_lower_bound_flops"]+old_preparation)/exact_all,
        tiered_model_with_common_preparation_over_exact_all=(devcost["tiered_total_model_flops"]+old_preparation)/exact_all)
    for key in ("residual_calibration","development"):
        assert by_role[key]["decision_mismatches"] == 0
    REPORT.mkdir(parents=True)
    all_rows.to_parquet(REPORT/"states.parquet",index=False)
    dev[keep & (e>=.5)].to_csv(REPORT/"retained_severe_states.csv",index=False)
    counts=all_rows.groupby(["role","target","stage"]).size().unstack(fill_value=0)
    counts.to_csv(REPORT/"stage_counts.csv")
    summary=dict(status="complete", threshold=tau, rank=32, roles=by_role, combined=both, quality=quality,
        spectrum=spectral, timing=dict(preparation_seconds=spectral["elapsed_seconds"],population_wall_seconds=queue["elapsed_seconds"],
            canary_seconds=summaries[0]["elapsed_seconds"], valid_summed_gpu_component_seconds=dict(ledgers)),
        max_adapt_logit_delta=max(r["max_adapt_logit_delta"] for s in summaries for r in s["results"]),
        max_accurate_score_relative_delta=max(r["max_accurate_score_relative_delta"] for s in summaries for r in s["results"]),
        canary_extra_full_solves=sum(r["canary_extra_solve_states"] for s in summaries for r in s["results"]),
        peak_gpu_mib=max(s["peak_gpu_mib"] for s in summaries),
        retained_factor_storage_bytes=prior["costs"]["factor_storage_bytes"],
        new_teacher_outputs=0, C_refitted=False, confirmation_read=False,
        frozen_calibration_unchanged=True, stage_counts=counts.reset_index().to_dict("records"),
        positive_total_saving_in_cost_model=devcost["tiered_total_model_over_exact"]<1,
        failure_even_at_verified_lower_bound=devcost["verified_total_lower_bound_over_exact"]>=1,
        cost_scope="Explicit lower bound and stated eigen/scalar charge model, not exact vendor EVD FLOPs. Original calibration teacher, full source replay/writer/PCA preparation and other serving dependencies remain additional. Research input C reads itemized; no timing-to-FLOPs conversion. All same baseline dependencies cancel for incremental comparison.")
    (REPORT/"summary.json").write_text(json.dumps(finite_metrics(summary),indent=2,allow_nan=False)+"\n")
    lines=["# Design 2：固定32方向分级检查", "",
        "保持冻结判定的实现及中规模核验完成；费用结论以本轮已打开的512/1024UID面板为限。", "",
        "**判定保持通过，但固定32方向谱界没有净节省，应停止这个候选的扩张。公式等价短路可单独保留。**" if summary["failure_even_at_verified_lower_bound"] else "费用判别详见下表。", "",
        f"阈值使用完整存储值{tau}；C、H、12组a/b和查询不变。两项局部数值检查通过。",
        "所有比较是判定等价与计算费用，不宣称质量提升；不重拟合、不计算新教师输出、不读确认集或theta3。", "",
        "## 1. 各级处理数量", "",
        "| 集合 | UID/状态 | 公式短路（接受/拒绝） | 谱界接受 | 谱界拒绝 | 完整求解 | 判定差异 |",
        "| --- | --- | --- | --- | --- | --- | --- |"]
    for name,r in by_role.items():
        c=r["stages"]
        lines.append(f"| {name} | {r['users']}/{r['states']} | {c['formula']}（{r['formula_accept']}/{r['formula_reject']}） | {c['bound_accept']} | {c['bound_reject']} | {c['exact_fallback']}（{c['exact_fallback']/r['states']:.2%}） | {r['decision_mismatches']} |")
    lines += ["",f"开发续用覆盖{quality['coverage']:.4%}，严重率{quality['severe_rates'][0]:.4%}，仍漏检{quality['severe_uids'][0]}UID/{quality['severe_states'][0]}状态，与原准确检测器完全一致。",
        "UID192760与897490的M1短历史漏检原样保留；M5组拒绝及拟合内988060的原判定也未改变。已知审查用户未重新执行模型，既有分数代入相同规则的判定不变，不将其冒充新的回放样本。",
        f"重放C的最大logit差{summary['max_adapt_logit_delta']:.3g}，已精算的u最大相对差{summary['max_accurate_score_relative_delta']:.3g}。",
        "探针对每个非公式状态额外精算，直接验证同输入上下界；全量只精算未定状态，并以既有准确u核对每个界和最终判定。", "",
        "## 2. 余项界为何不能省掉", "",
        "对H按升序分解，保留前32个逆特征方向；未展开能量r的贡献保留在[r/dmax,r/d33]。",
        "用完整特征分解残差与正交误差扩张界，再加入投影、范数、最终标量的FP64舍入预算；阈值附近精算。",
        "这是标准浮点模型下的数值界，经实际输入核验，不是误差概率或实际logit安全上界，也不声称形式化区间库认证。",
        f"144个矩阵的d33/dmin中位数{np.median([r['d33']/r['dmin'] for r in spectral['rows']]):.8f}，"
        "正则底部附近大量方向未被32方向展开，余项上界可能宽。维数固定，没有改成64/128择优。", "",
        "## 3. 费用：必须计入粗查、精查与新增准备", "",
        "以下以开发15299状态为例。原准确检查含原几何全部普通算术及一次仿射评分；特殊函数、比较、搬运另列。", "",
        "| 项目 | TFLOPs | 相对原全准确检查 |",
        "| --- | --- | --- |"]
    values=[("原准确检查",devcost["exact_check_baseline_flops"]),
        ("仅公式短路＋其余准确检查",devcost["formula_only_flops"]),
        ("32方向粗查（含标量余量）",devcost["coarse_check_flops"]),
        ("未定状态完整精查",devcost["fallback_flops"]),
        ("分级在线合计",devcost["online_flops_model"]),
        ("新增谱准备：已核实GEMM主项",spectral["verified_preparation_gemm_flops"]),
        ("新增谱准备：另收特征分解模型",spectral["eigendecomposition_cost_model_flops"]),
        ("准备＋检查的已核实下界",devcost["verified_total_lower_bound_flops"]),
        ("准备＋检查的费用模型合计",devcost["tiered_total_model_flops"])]
    for name,value in values:
        lines.append(f"| {name} | {value/1e12:.6f} | {value/devcost['exact_check_baseline_flops']:.2%} |")
    lines += ["", "已核实下界仅用实际投影GEMM、完整回退和谱重构/核验GEMM，尚未收正值的特征分解等剩余费用；若该下界已经超过原法，结论不依赖特征分解模型。",
        f"更直接的对照是仅保留公式短路：第二级额外省下{(devcost['formula_only_flops']-devcost['fallback_flops'])/1e12:.6f}TFLOPs精算，却仅投影GEMM就支付{(devcost['states']-devcost['stages']['formula'])*devcost['coarse_panel_projection_flops']/1e12:.6f}TFLOPs。故第二级在当前状态分布上连在线部分也不优于公式短路；摊薄谱准备不能消除这一点。",
        "特征分解另按每矩阵10p³作明确费用模型，不称为底层LAPACK实际精确计数或普适上界。粗查标量按128次/单元保守收费；完整公式见report_tiered.py。",
        "LAPACK也区分标准特征分解FLOPs与具体三对角算法工作量，不能拿计时填补算法计数：[官方口径](https://www.netlib.org/lapack/lug/node71.html)。",
        f"连同原Gram/Cholesky公共准备主项，原准确检测为Exact-All的{devcost['original_with_common_preparation_over_exact_all']:.2%}；"
        f"分级已核实下界{devcost['tiered_verified_lower_bound_with_common_preparation_over_exact_all']:.2%}，费用模型为{devcost['tiered_model_with_common_preparation_over_exact_all']:.2%}。这些数字均未包含任何真实重建动作。",
        f"若将同一次准备同时摊在512+1024全部面板，已核实下界为原检查的{both['verified_total_lower_bound_over_exact']:.2%}，费用模型为{both['tiered_total_model_over_exact']:.2%}；不只挑较小分母。",
        "更大负载可能摊薄一次谱准备，但那是另一个工作负载条件，不能用假想无限续用替代本轮净节省验收。", "",
        "## 4. 输入恢复、存储和遗留准备依赖", "",
        f"新增32方向投影文件{spectral['projection_storage_bytes']/2**20:.2f}MiB，同时保留回退所需原Cholesky {summary['retained_factor_storage_bytes']/2**30:.3f}GiB；不是将完整因子替换成32方向。",
        f"谱准备墙钟{spectral['elapsed_seconds']:.2f}s，16UID探针{summaries[0]['elapsed_seconds']:.2f}s，余下两GPU队列{queue['elapsed_seconds']:.2f}s；峰值{summary['peak_gpu_mib']/1024:.2f}GiB/GPU。",
        f"恢复原输入涉及C读取{both['input_recovery_read_states']}状态，沿用query/read公式的普通算术主项{both['input_recovery_read_flops']/1e12:.6f}TFLOPs，另有真实源前缀/追加回放、摘要与视图工作，组件时间保存在summary。",
        "这些是本次研究重新获取未存特征的成本，服务比较假定原Design1已产生相同输入；不能同时声称已保存特征或本轮没有模型前向。",
        "原exact_source_control仍是冻结面板的Current输入构造，包含Current计算，但没有新的Exact教师输出或重建动作。原校准教师、原C准备、源回放/写入的未完全隔离算术仍为依赖，未计为0。",
        f"探针另有{summary['canary_extra_full_solves']}个完整状态求解仅为验证，不计作部署必须执行的检查；研究执行费用应额外增加{summary['canary_extra_full_solves']*panel/1e9:.3f}GFLOPs。", "",
        "## 5. 决定", "",
        f"判定保持通过；连已核实准备下界都不节省：{summary['failure_even_at_verified_lower_bound']}。",
        "是否继续实现以实际费用结果为准；不修改阈值、a/b、分组或维数来制造通过。详细分级数量、逐目标结果及低分漏检均留存。"]
    if summary["failure_even_at_verified_lower_bound"]:
        lines += ["", "**停止本轮固定32方向谱界实现的扩张，不进入真实重建/连续调度。** 公式等价短路可保留，但它的局部节省不能替代整个谱界方案的净费用结论。",
            "专家提出的决策保持研究方法合理；本次被否定的是这个固定32方向候选在当前面板的净节省，不是尺度校准指标或整个Design2框架。"]
    lines += ["", "![分级占比与费用](../../../../figures/pic/design2/tiered32_01/tiered32.png)"]
    (REPORT/"report.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(roles=by_role,quality=quality,failure_even_at_verified_lower_bound=summary["failure_even_at_verified_lower_bound"])))


if __name__ == "__main__":
    main()
