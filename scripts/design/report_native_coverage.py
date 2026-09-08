#!/usr/bin/env python3
"""Three predeclared calibration arms, UID-equal outcomes and actual training risk."""

import json

import numpy as np
import pandas as pd
import torch
from design.data import DATASET, ROOT
from design.report_decoder_closure import interval
from design.run import write_json

RUN = ROOT / "results/design/native_coverage384_01"
A = ROOT / "results/design/mechanism_native_input192_01"
AUDIT = ROOT / "results/design/analysis/native_residual_audit_01"
OUT = ROOT / "results/design/analysis/native_coverage384_report_01"


def main():
    run_summary = json.loads((RUN / "summary.json").read_text())
    assert run_summary["status"] == "complete"
    OUT.mkdir(exist_ok=True)
    targets = (1,3,4,5)
    frames = [pd.read_parquet(RUN / f"outputs_m{t}.parquet") for t in targets]
    frame = pd.concat(frames,ignore_index=True)
    assert np.isfinite(frame[["logit_mse","bias_squared","centered_mse"]]).all().all()
    assert not frame.duplicated(["target","uid","state_ordinal","method","panel"]).any()
    np.testing.assert_allclose(frame.logit_mse,frame.bias_squared+frame.centered_mse,atol=1e-12,rtol=1e-12)
    # A is a frozen reference: main tables use its already retained raw on original users.
    old = pd.read_parquet(AUDIT / "states.parquet")
    match = frame.merge(old[["target","uid","state_ordinal","panel","method","logit_mse","bias_squared","centered_mse"]],
                        on=["target","uid","state_ordinal","panel","method"],how="left",suffixes=("","_frozen"),validate="one_to_one")
    for column in ("logit_mse","bias_squared","centered_mse"):
        match[column] = match[column+"_frozen"].fillna(match[column])
    frame = match[frame.columns].copy()
    users = pd.read_parquet(DATASET.parent / "users.parquet",columns=["uid","n_theta0"])
    frame = frame.merge(users,on="uid",validate="many_to_one")
    frame.to_parquet(OUT / "states.parquet",index=False)
    held = frame[frame.panel == "held64"]
    uid = held.groupby(["target","group","method","uid"],as_index=False)[["logit_mse","bias_squared","centered_mse"]].mean()
    uid.to_parquet(OUT / "uids.parquet",index=False)
    rows, contrasts = [],[]
    for (target,group), f in uid.groupby(["target","group"]):
        wide = f.pivot(index="uid",columns="method",values="logit_mse")
        for method in ("A","B","C","reuse","shared15"):
            v = wide[method]
            bias = f[f.method==method].bias_squared.sum()/f[f.method==method].logit_mse.sum()
            rows.append(dict(target=int(target),group=group,method=method,uids=len(v),mse=float(v.mean()),
                median=float(v.median()),p90=float(v.quantile(.9)),p95=float(v.quantile(.95)),max=float(v.max()),
                max_uid_share=float(v.max()/v.sum()),reuse_ratio=float(v.mean()/wide.reuse.mean()),
                better_than_reuse=float((v<wide.reuse).mean()),bias_squared_fraction=float(bias),
                minus_reuse_interval=interval(v-wide.reuse)))
        for left,right in (("C","A"),("B","A"),("C","B"),("C","reuse"),("C","shared15")):
            diff = wide[left]-wide[right]
            improvement = (-diff).clip(lower=0)
            contrasts.append(dict(target=int(target),group=group,left=left,right=right,difference=float(diff.mean()),
                interval=interval(diff),better_uid_fraction=float((diff<0).mean()),
                largest_uid_fraction_of_positive_improvement=float(improvement.max()/improvement.sum()) if improvement.sum()>0 else None))
    # Exact contributions to UID-equal excess MSE; partition sums recover the overall difference.
    contributions = []
    for (target,group), f in held.groupby(["target","group"]):
        baseline = f[f.method=="reuse"][["uid","state_ordinal","logit_mse"]].rename(columns={"logit_mse":"reuse_mse"})
        f = f[f.method.isin(["A","B","C"])].merge(baseline,on=["uid","state_ordinal"],validate="many_to_one")
        sizes = baseline.groupby("uid").size()
        f["uid_scene_weight"] = 1/(baseline.uid.nunique()*f.uid.map(sizes))
        f["excess_contribution"] = (f.logit_mse-f.reuse_mse)*f.uid_scene_weight
        f["cache_length_group"] = pd.cut(f["count"],[0,32,256,1023,1024],labels=["1-32","33-256","257-1023","1024"]).astype(str)
        f["initial_history_group"] = pd.cut(f.n_theta0,[-1,255,1023,4095,np.inf],labels=["<256","256-1023","1024-4095","4096+"]).astype(str)
        f["age_group"] = pd.cut(f.release_age,[-1,0,64,256,1024,np.inf],labels=["0","1-64","65-256","257-1024","1025+"]).astype(str)
        f["dominant_producer"] = f[[f"producer_{p}_fraction" for p in range(int(target)+1)]].idxmax(axis=1)
        f["producer_composition"] = np.where((f[[f"producer_{p}_fraction" for p in range(int(target)+1)]]>0).sum(axis=1)>1,"mixed","single")
        for field in ("cache_length_group","initial_history_group","state_group","age_group","dominant_producer","producer_composition"):
            aggregate = f.groupby(["method",field],as_index=False,observed=True).agg(excess_contribution=("excess_contribution","sum"),
                uid_equivalent_mass=("uid_scene_weight","sum"),uids=("uid","nunique"),scenes=("uid","size"))
            for method in ("A","B","C"):
                np.testing.assert_allclose(aggregate[aggregate.method==method].excess_contribution.sum(),f[f.method==method].excess_contribution.sum(),atol=1e-12)
            contributions.extend(aggregate.rename(columns={field:"stratum"}).assign(field=field,target=int(target),group=group).to_dict("records"))
    write_json(OUT / "strata.json",contributions)
    # Native's actual fitting objective, before deterministic serving masks.
    risks,objective_rows,uid_risks = [],[],[]
    locals_all = pd.concat([pd.read_parquet(RUN / f"responses_m{t}.parquet") for t in targets],ignore_index=True)
    locals_all.to_parquet(OUT / "responses.parquet",index=False)
    for t in targets:
        risk = pd.read_parquet(RUN / f"unmasked_risk_m{t}.parquet")
        records = json.loads((RUN / f"fits_m{t}.json").read_text())
        old_records = json.loads((A / f"fits_m{t}.json").read_text())
        for method in ("A","B","C"):
            fit_panel = "fit64" if method=="A" else "fit16"
            fit_groups = ["original_fitting","additional_fitting"] if method=="C" else ["original_fitting"]
            fit = risk[(risk.method==method)&(risk.panel==fit_panel)&risk.group.isin(fit_groups)].copy()
            fit["actual_loss_contribution"] = fit.unmasked_data/fit.groupby("layer").uid.transform("size")/6
            user_risk = fit.groupby(["target","method","group","uid"],as_index=False).agg(data_contribution=("actual_loss_contribution","sum"),scenes=("scene","nunique"))
            user_risk["loss_share"] = user_risk.data_contribution/user_risk.data_contribution.sum()
            user_risk["base_scenario_share"] = user_risk.scenes/user_risk.scenes.sum()
            user_risk["report_uid_weight"] = 1/len(user_risk)
            uid_risks.append(user_risk)
            params = torch.load((A / f"translator_native_input_m{t}.pt") if method=="A" else (RUN / f"translator_{method}_m{t}.pt"),map_location="cpu",weights_only=True)
            for layer,p in enumerate(params):
                record = next(r for r in (old_records if method=="A" else records) if r["layer"]==layer and r["method"]==("native_input" if method=="A" else method))
                actual = float(fit[fit.layer==layer].unmasked_data.mean())
                np.testing.assert_allclose(actual,record["fitting_aggregate_objective"],atol=1e-4,rtol=3e-4)
                ridge = .01*float(p["weights"].square().sum()+p["read_weights"].square().sum())/192
                objective_rows.append(dict(target=t,method=method,layer=layer,data=actual,ridge=ridge,ridge_to_data=ridge/actual,
                                           recorded_fitting_data=record["fitting_aggregate_objective"]))
            risks.append(fit)
    pd.concat(risks).to_parquet(OUT / "training_state_risks.parquet",index=False)
    user_risk = pd.concat(uid_risks,ignore_index=True)
    user_risk.to_parquet(OUT / "training_uid_risks.parquet",index=False)
    scale_run = ROOT / "results/design/native_response_scale384_01"
    scale_summary = json.loads((scale_run / "summary.json").read_text())
    assert scale_summary["status"] == "complete"
    scales = []
    for t in targets:
        f = pd.read_parquet(scale_run / f"response_scale_m{t}.parquet")
        meta = pd.read_parquet(RUN / f"scenes_m{t}.parquet")
        scales.append(f.merge(meta[["scene","group","state_ordinal","count"]],on="scene",validate="many_to_one"))
    scale = pd.concat(scales,ignore_index=True)
    assert np.isfinite(scale[["response_mse","target_change_energy","source_response_energy"]]).all().all()
    scale.to_parquet(OUT / "response_scale.parquet",index=False)
    scale_uid = scale.groupby(["target","method","panel","group","uid","layer"],as_index=False)[["response_mse","target_change_energy"]].mean()
    scale_uid["relative_response_error"] = scale_uid.response_mse/scale_uid.target_change_energy.replace(0,np.nan)
    scale_uid.to_parquet(OUT / "response_scale_uids.parquet",index=False)
    # Cost is separated into source/teacher construction, query acquisition, fitting and diagnostic overhead.
    cost_rows,prep = [],{}
    for t in targets:
        prep[str(t)] = json.loads((RUN / f"preparation_m{t}.json").read_text())
        measured = json.loads((RUN / f"cost_m{t}.json").read_text())
        meta = pd.read_parquet(RUN / f"scenes_m{t}.parquet")
        old_count = int((meta.group=="original_fitting").sum())
        for method in ("A","B","C"):
            construction = sum(prep[str(t)]["original_fitting"].values())
            if method == "C":
                construction += sum(prep[str(t)]["additional_fitting"].values())
            if method=="A":
                acquisition_fit = json.loads((A / f"cost_m{t}.json").read_text())["native_input_calibration_seconds"]
                row = dict(query_and_fit_seconds=acquisition_fit,fitting_scenes=old_count,fitting_uids=64,query_pairs_per_layer=old_count*64)
            else:
                r = next(r for r in measured if r["method"]==method)
                row = dict(r,query_and_fit_seconds=r["query_acquisition_seconds"]+r["shared_fit_seconds"])
            cost_rows.append(dict(row,target=t,method=method,source_teacher_scene_seconds=construction,
                                  measured_preparation_subtotal_seconds=construction+row["query_and_fit_seconds"]))
    common = {k:v for k,v in run_summary["state_construction_ledger_seconds"].items() if k in ("calibration_source_backfill","calibration_lineage_replay")}
    # Historical denominator retained explicitly; it used compiled Exact and five releases, so not a matched qualification here.
    costs = dict(rows=cost_rows,source_teacher_by_cohort=prep,common_384_uid_ordinary_lineage_seconds=common,
        common_lineage_attribution="Shared build includes128 diagnosis; not silently charged only to fitting. All-common charge provides a conservative preparation upper bound, not marginal measurement.",
        historic_exact_five_release_seconds=204.30332799945026,historic_population=30000,
        historic_initial_summary_proxy_seconds=4.814743253518827,
        historical_budget_scope="Historical compiled Exact five releases vs this eager four-target diagnostic: no matched end-to-end 20% qualification. Deployment, I/O, refresh and waiting coverage still missing.",
        full_run=run_summary,frozen_response_scale_audit=scale_summary,matched_current_population_denominator_available=False)
    write_json(OUT / "cost.json",costs)
    result = dict(rows=rows,contrasts=contrasts,objectives=objective_rows,status="complete",confirmation_read=False,
        decision="Coverage improves M1/M3/M4 and most M5 UIDs, but M5 mean/tail and original-fitting failures do not pass promotion; stop expansion, no6000.",
        scope="fixed six-layer seed17, M1/M3/M4/M5, M2 native only, M5 E14_partial; reused128 diagnostic UIDs; no new6000 evaluation")
    write_json(OUT / "summary.json",result)
    lines = ["# Native校准覆盖：64×64、64×16、256×16", "", "本轮固定native输入结构，仅新增B/C联合校准。A为上一轮冻结64×64；B为原64×固定16，C为256×同16（含原64）。主比较C−A。", "",
        "同一128已使用诊断UID、原held64；每状态先平均query，再每UID平均状态，最后UID等权。以下均为logit MSE，不是AUC恢复。A/Reuse/方案15沿用已核对冻结raw；C新增拟合用户不进入诊断组。", "",
        "## 1. 诊断UID结果", "", "| M | Reuse | 方案15 | A 64×64 | B 64×16 | C 256×16 | C/Reuse | C胜Reuse UID |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for t in targets:
        r = {x["method"]:x for x in rows if x["target"]==t and x["group"]=="diagnostic"}
        lines.append(f"| M{t} | {r['reuse']['mse']:.8g} | {r['shared15']['mse']:.8g} | {r['A']['mse']:.8g} | {r['B']['mse']:.8g} | {r['C']['mse']:.8g} | {r['C']['reuse_ratio']:.4g} | {r['C']['better_than_reuse']:.1%} |")
    lines += ["", "| M | 比较 | 均值差 | 描述性95%配对UID区间 | 改善UID比例 |", "| --- | --- | ---: | --- | ---: |"]
    for r in contrasts:
        if r["group"]=="diagnostic":
            lines.append(f"| M{r['target']} | {r['left']}−{r['right']} | {r['difference']:.7g} | [{r['interval'][0]:.7g}, {r['interval'][1]:.7g}] | {r['better_uid_fraction']:.1%} |")
    lines += ["", "这些区间只描述已反复使用开发UID、单backbone seed17内的变化，不是独立确认。C−A为主比较；B解释查询覆盖，C−B解释16查询下增加用户，不推断未测的256×64交互。", "",
        "## 2. 尾部与拟合用户留出", "", "| M | 方法 | 诊断中位数 | p95 | 最大UID占总误差 | bias²占比 |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for r in rows:
        if r["group"]=="diagnostic" and r["method"] in ("A","B","C"):
            lines.append(f"| M{r['target']} | {r['method']} | {r['median']:.7g} | {r['p95']:.7g} | {r['max_uid_share']:.1%} | {r['bias_squared_fraction']:.1%} |")
    lines += ["", "| M | 拟合人群 | A held MSE | B held MSE | C held MSE | Reuse |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for t in targets:
        for group in ("original_fitting","additional_fitting"):
            r = {x["method"]:x for x in rows if x["target"]==t and x["group"]==group}
            lines.append(f"| M{t} | {group} | {r['A']['mse']:.7g} | {r['B']['mse']:.7g} | {r['C']['mse']:.7g} | {r['reuse']['mse']:.7g} |")
    lines += ["", "新增192仅C参与拟合，对A/B属于未拟合用户；原64对A/B/C均参与拟合。两组不与诊断128混合。全部逐UID与局部响应见uids.parquet、responses.parquet。", "",
        "## 3. 残余与训练风险核对", "", "状态误差严格分解为bias²+centered MSE。M5按实际N、初始历史、场景、年龄、主producer及混合组成的超出Reuse误差贡献见strata.json；每个分区贡献之和精确回到UID等权总差，避免把有更多场景的UID无意放大。分组仅用于解释。", "",
        "拟合目标为 `(1/S) Σ_s (N_s²/mu_A)(1/Q) Σ_j ||rate_error||² + .01||[W,U]||²`（逐head/output；统计同时除以192）。输入标准化冻结，无输出通道标准化；mu_A逐边固定；含截距的全部W/U参与ridge。N²把率误差换回聚合响应误差，不是错误权重。", "",
        "未mask训练风险与服务消退mask后的响应误差分开保存。本次逐场景重建的未mask目标已与原A/B/C拟合记录核对；实际每UID贡献、场景权重和UID报告权重见training_uid_risks.parquet。", "",
        "| M | 方法 | 六层平均数据项 | 六层平均ridge项 | UID988060损失份额 | UID988060 held MSE |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for t in targets:
        for method in ("A","B","C"):
            r = [x for x in objective_rows if x["target"]==t and x["method"]==method]
            share = user_risk[(user_risk.target==t)&(user_risk.method==method)&(user_risk.uid==988060)].loss_share.iloc[0]
            err = uid[(uid.target==t)&(uid.method==method)&(uid.uid==988060)].logit_mse.iloc[0]
            lines.append(f"| M{t} | {method} | {np.mean([x['data'] for x in r]):.7g} | {np.mean([x['ridge'] for x in r]):.7g} | {share:.4%} | {err:.7g} |")
    lines += ["", "名义count²份额不能替代实际损失；ridge项也不可忽略。这里没有证据允许把N²删除称为公式修复；本轮保持全部UID、权重和正则，不安装事后状态偏移。", "",
        "### 同query响应尺度补查", "", "为判断响应是否已经拟合好，另对冻结A/C做一次读取，补齐目标响应变化能量；没有重拟合任何权重或重跑13条干预。每一行用该分支自己的同一q，计算 `E||修正−目标变化||² / E||目标变化||²`。分母是该q下不修正的局部残余，不是独立Reuse前向的logit误差。A/C各自q不同，这些比值不能用于公共q函数优劣归因。", "",
        "| M | UID988060拟合面板 | 层1 | 层2 | 层3 | 层4 | 层5 | 层6 |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for t in targets:
        for method in ("A","C"):
            panel = "fit64" if method=="A" else "fit16"
            f = scale_uid[(scale_uid.target==t)&(scale_uid.method==method)&(scale_uid.panel==panel)&(scale_uid.uid==988060)].sort_values("layer")
            lines.append(f"| M{t} | {method} {panel} | "+" | ".join(f"{x:.4g}" for x in f.relative_response_error)+" |")
    lines += ["", "988060在A的M4层2–6、M5层2–6，局部残余已大于同q目标变化能量；C也存在严重局部过修正。因此不能把这些失败描述成‘响应都拟合好了，仅最终logit不好’。这次没有单层精确下层干预，仍不能把局部失配与错误上下文传播的因果贡献完全分离，更不能直接复用旧方法第一层主导结论。", "",
        "全部拟合/诊断UID、fit/held逐层数据保存在response_scale(_uids).parquet。零目标变化能量的比值保留为未定义，不用epsilon制造有利比值；原绝对残余不删除。", "",
        "### M5新增损失来自哪里", "", "A相对Reuse的诊断总差为+0.0049264，其中实际N=1024分区贡献+0.0065218；不能把原A的诊断失败概括为短历史。C的N≤32分区（1名UID）贡献+0.0085391，其余三个N分区合计−0.0090195。这里是精确可加的UID等权贡献，不是删去该UID后的另一个主指标。", "",
        "| M5诊断UID | 实际缓存N | A MSE | C MSE | Reuse MSE |", "| --- | ---: | ---: | ---: | ---: |"]
    for identifier in (547600,254830):
        values = uid[(uid.target==5)&(uid.uid==identifier)].set_index("method").logit_mse
        n = int(held[(held.target==5)&(held.uid==identifier)]["count"].iloc[0])
        lines.append(f"| {identifier} | {n} | {values['A']:.7g} | {values['C']:.7g} | {values['reuse']:.7g} |")
    lines += ["", "两名UID合计约76.9%的C诊断误差：547600在真实连续/早期混合旧谱系很差，但同UID的Parent初始化辅助状态误差小；254830只有19个token、连续状态是单一旧producer，也明显失败。因此既有混合producer，也有纯旧producer的困难状态；这不足以证明‘混合贡献’是唯一瓶颈，下一步不能自动启动producer原型。", "",
        "## 4. 预算和成本", "", "| M | 方法 | 场景数 | 每层拟合查询对 | 源/教师场景构建s | 响应采集＋拟合s |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for r in cost_rows:
        lines.append(f"| M{r['target']} | {r['method']} | {r['fitting_scenes']} | {r['query_pairs_per_layer']} | {r['source_teacher_scene_seconds']:.3f} | {r['query_and_fit_seconds']:.3f} |")
    lines += ["", "四边源/教师场景构建＋响应/拟合的小计：A 18.49s、B 17.36s、C 62.05s；C约为A的3.36倍，尚未包含普通初始谱系、人口摘要初始化和持续维护。实际每层查询对合计A 61696、C 61344接近，但教师/源构建增加，不能称为等成本实验。", "",
        f"三臂主运行耗时{run_summary['elapsed_seconds']:.2f}s、峰值{run_summary['peak_allocated_mib']:.0f}MiB，包含A审查和全部诊断，不能作为可部署校准成本。共同384UID普通谱系初始化/回放共{sum(common.values()):.3f}s，含128诊断，单列不藏入廉价发布成本。C的新增教师/源构建单列cost.json；共享拟合与响应采集也分账。", "",
        f"冻结A/C响应尺度补查另耗时{scale_summary['elapsed_seconds']:.2f}s、峰值{scale_summary['peak_allocated_mib']:.0f}MiB，属于研究审查费用，保留完整账本；不是第三份校准或可部署成本项。", "",
        "同30000人口的历史五发布Exact分母为204.3033s，20%为40.8607s，初始摘要代理4.8147s。它采用已编译Exact，本轮为eager四目标诊断，不能混用这些数值宣称完整预算达标；本轮尚无匹配的完整人口成本分母。校准/初始化/发布/持续维护/I/O仍须一起计费。", "",
        "新增U仍为每请求221184 MAC，无新增持久摘要或KV扫描；相对两次历史矩阵乘法为96/N：N=16/32/96/256/1024对应6/3/1/0.375/0.09375。该比值不包含Exact的其他算子或服务期，不设N=96阈值，也不为988060定向Exact。", "",
        "## 5. 执行边界", "", "仅B/C两份新校准；固定A mean PCA32、query/native中心尺度、W/U函数族、ridge与mu_A。新增192用户按既定分层/hash一次冻结；原64生命周期覆盖不变。B/C用自己的下层实际query；普通KV谱系共用。固定16子集只有4个item，是查询覆盖预算对照。", "",
        "M1/M3/M4/M5全报，M2仅native过渡，M5保留E14_partial。未做256×64、扩人数、gate、producer原型、自由编码或13路径重跑；没有新6000质量输出，确认6000封存。", "",
        "本轮结论：独立用户覆盖对M1/M3/M4有明确增益，M5多数用户也改善，但没有完成尾部风险收敛，不能归为三臂全面成功。M5 C−A、C−B、C−Reuse区间均含0，且原64拟合用户的M4/M5误差恶化；不进入6000，不补256×64或继续扩人数。", "",
        "下一次讨论应区分状态条件化与聚合响应风险对闭环误差的约束，不能把原型失败再归结为人数不足。当前producer残余有异质性，但本轮没有版本条件化的匹配干预，不能宣布该方向已被证明。保留native信息接口和这次覆盖证据，结束本轮两份校准。"]
    (OUT / "report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines[:15]))


if __name__ == "__main__":
    main()
