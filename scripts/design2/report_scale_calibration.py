"""Render retained scale-calibration evidence; no fitting or model execution."""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/design2/analysis/scale_calibration_01"
LABELS = {"raw": "原几何", "background": "仅组背景", "calibrated": "组背景＋几何"}


def main():
    s = json.loads((OUT / "summary.json").read_text())
    frozen = json.loads((ROOT / "configs/design2/scale_calibration_01_fitted.json").read_text())
    methods, costs = s["methods"], s["costs"]
    at80 = {m: next(r for r in v["acceptance"] if r["quantile"] == .8) for m, v in methods.items()}
    delta = methods["calibrated"]["auroc"][0] - methods["background"]["auroc"][0]
    ci = s["paired"]["calibrated_minus_background_auroc_ci"][0]
    lines = ["# Design 2：一次固定的目标/长度条件残余尺度校准", "",
        "**本轮支持组间尺度校准保留了有用的几何信息，且优于仅长度/目标背景；预定开发推进条件全部通过。**",
        "这不证明预测是误差上界，也不证明同组低分严重失败已经消失。准确检查仍昂贵，尚未进入低成本近似、分级检查或重建调度。", "",
        "原512校准UID/7613状态做5折UID交叉校准，再冻结参数与阈值，评价原1024开发UID/15299状态/244784查询。",
        "冻结Design1、C、H、全部查询和Current-at-decision Exact残余；只读既有结果，新增模型前向和教师计算均为0。",
        "开发数据已用于前轮研究，本轮是开发验证，不能称独立确认；确认集与theta3未读，仍为单模型训练seed17。", "",
        "## 1. 冻结的方法与匹配对照", "",
        "在目标版本×粗长度组内拟合 ê=b+a·u，b,a≥0，以面板最大绝对logit残余为目标，使用UID/目标/状态原权重的加权最小二乘。",
        "同组对照独立拟合常数，即加权平均残余；不是拿几何模型的截距冒充对照。内部RMS仅改善数值条件，系数换回原u，不搜索归一化。",
        "初始长度区间1–32、33–255、256–1023、1024；只按校准UID元数据及固定5折支持合并：任一训练折少于32UID的最左组并向下一组，末组并向前组。",
        "最终各目标均为1–255、256–1023、1024，共12组；最小训练折支持34UID。分组从不读取残余或失败标记。",
        "阈值来自校准UID折外预测的固定分位，随后在512UID上全量拟合一次，冻结后才产生本轮开发预测。同分全部接受，不按残余拆分并列组。",
        "配置见configs/design2/scale_calibration_01.json；分组、UID折、各折系数、最终系数及阈值保存在scale_calibration_01_fitted.json，并记录输入/源码SHA256。", "",
        "## 2. 主尺度：最大绝对logit残余≥0.5", "",
        f"总体严重率{s['overall_severe_rates'][0]:.4%}，涉及{s['serious_uids'][0]}UID/{s['serious_states'][0]}状态；覆盖与风险沿用UID/目标等权，状态/UID数量为实际计数。", "",
        "| 方法 | AUROC [配对UID重采样95%描述区间] | 冻结80%阈值下覆盖 | 续用严重率 | 剩余严重UID/状态 |",
        "| --- | --- | --- | --- | --- |"]
    for m, v in methods.items():
        r, ac = at80[m], v["auroc_uid95"][0]
        lines.append(f"| {LABELS[m]} | {v['auroc'][0]:.4f} [{ac[0]:.4f}, {ac[1]:.4f}] | {r['coverage']:.2%} | "
                     f"{r['severe_rates'][0]:.4%} | {r['severe_uids'][0]}/{r['severe_states'][0]} |")
    lines += ["", f"几何校准减仅背景的AUROC差为{delta:.4f}，配对UID95%描述区间[{ci[0]:.4f}, {ci[1]:.4f}]。",
        f"校准80%阈值{at80['calibrated']['threshold']:.6f}保留{at80['calibrated']['states']}状态，涉及{at80['calibrated']['uids']}UID；同一UID可能同时有被接受与拒绝的不同状态，不能称这些UID永久安全。",
        "三个方法的80%校准分位不保证同一开发覆盖，表中显示实际值；仅背景有整组并列，不能把它的87%覆盖误写成80%。下表给出完整覆盖—风险关系。", "",
        "| 折外分位 | 原几何：开发覆盖 / 严重率 | 仅背景：开发覆盖 / 严重率 | 校准几何：开发覆盖 / 严重率 | 校准后严重UID/状态 |",
        "| --- | --- | --- | --- | --- |"]
    for i, r in enumerate(methods["calibrated"]["acceptance"]):
        cells = [f"{methods[m]['acceptance'][i]['coverage']:.2%} / {methods[m]['acceptance'][i]['severe_rates'][0]:.4%}" for m in LABELS]
        lines.append(f"| {r['quantile']:.0%} | " + " | ".join(cells) + f" | {r['severe_uids'][0]}/{r['severe_states'][0]} |")
    lines += ["", "低10/20/50%阈值区域本次没有0.5严重状态，不能解释成未知人群的零风险上界。",
        "若要保留约70%以上状态，仍会接受低分严重失败；仅降低同一分数阈值必然牺牲覆盖。", "",
        "## 3. 漏检与整组拒绝的代价", "",
        "冻结80%阈值下，以下严重状态仍被校准几何接受（完整31严重状态含所有判定保存在all_severe_development.csv）：", "",
        "| UID | 目标 / N | 状态 | 真实残余 | b分量 | a·u分量 | 预测残余 |",
        "| --- | --- | --- | --- | --- | --- | --- |"]
    severe = pd.read_csv(OUT / "all_severe_development.csv")
    for r in severe[severe["calibrated_keep0.8"]].to_dict("records"):
        lines.append(f"| {r['uid']} | M{r['target']} / {r['count']} | {r['kind']} | {r['max_abs_error']:.6f} | "
                     f"{r['base_component']:.6f} | {r['geometry_component']:.6f} | {r['calibrated']:.6f} |")
    lines += ["", "这些M1短历史失败仍属于同组低u、严重错误：该组原几何与校准后的组内AUROC同为0.3862，非负仿射变换没有修复组内排序。",
        "校准折外也仍漏掉UID866780的M1/N=4 exact_source_control：预测0.032676，残余0.920965。它保留为诊断控制，不冒充真实部署状态。", "",
        "已知拟合内UID988060的M4/M5严重状态现在被拒绝；仅背景对照也拒绝，不能将识别它全部归功于几何。",
        "M4/N=16预测约0.271697（b=0.034855、a·u=0.236842），M5约0.342498（b=0.061148、a·u=0.281350）；真实残余仍为2.9098/3.4651，明显不是预测上界。",
        "历史审查UID547600的M5/N=1024严重状态仅背景仍接受、校准几何拒绝。三名预定审查UID的所有状态（包括不严重的UID254830）保存在known_failures.csv，不混入开发成功率。", "",
        "以下为每组在冻结80%阈值下的接受比例与覆盖损失，明确展示整组拒绝：", "",
        "| 组 | 开发UID/状态 | 主尺度严重状态 | 仅背景接受比例 | 校准几何接受比例 | 校准拒绝占全体覆盖 |",
        "| --- | --- | --- | --- | --- | --- |"]
    for g in s["groups"]:
        lines.append(f"| {g['group']} | {g['development_users']}/{g['development_states']} | {g['serious_states']} | "
                     f"{g['background_retained_fraction']:.2%} | {g['calibrated_retained_fraction']:.2%} | {g['calibrated_lost_global_coverage']:.2%} |")
    lines += ["", "M5的1–255组全部拒绝，损失全体2.32个百分点覆盖；256–1023组拒绝96.41%，再损失2.40个百分点，但后者本次没有0.5严重状态。不能只展示被找出的坏例而忽略这些代价。", "",
        "## 4. 辅助尺度与预算增量", "",
        "| 误差尺度 | 严重UID/状态 | 原几何AUROC | 仅背景AUROC | 校准几何AUROC | 几何减背景配对区间 |",
        "| --- | --- | --- | --- | --- | --- |"]
    for j, t in enumerate(s["errors"]):
        ci = s["paired"]["calibrated_minus_background_auroc_ci"][j]
        lines.append(f"| {t} | {s['serious_uids'][j]}/{s['serious_states'][j]} | {methods['raw']['auroc'][j]:.4f} | "
                     f"{methods['background']['auroc'][j]:.4f} | {methods['calibrated']['auroc'][j]:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] |")
    lines += ["", "0.1/1.0的全部续用风险及状态/UID数量保存在summary.json.methods，各目标与状态类型在strata中完整保留。", "",
        "下表均为重建预算，未扣准确检测费用；分数除以相同状态重建FLOPs，纯费用按升序。", "",
        "| 预算 | 误差尺度 | 原几何/费 | 仅背景/费 | 校准几何/费 | 纯费用 | 校准选中UID/状态 |",
        "| --- | --- | --- | --- | --- | --- | --- |"]
    for i, r in enumerate(s["budgets"]["calibrated"]):
        for j, t in enumerate(s["errors"]):
            cells = [f"{s['budgets'][m][i]['recall'][j]:.2%}" for m in (*LABELS, "cost_only")]
            lines.append(f"| {r['budget']:.0%} | {t} | " + " | ".join(cells) + f" | {r['selected_uids']}/{r['selected_states']} |")
    lines += ["", "主尺度5%预算校准几何覆盖93.30%，高于仅背景64.49%，但低于原几何/费用的100%。尺度修正改善续用辨别，并未改善所有预算排序点。",
        "预算前缀、实际选中UID/状态、年龄/短历史/源范数对照及配对UID覆盖差区间全部保存在summary.json；不按结果删去不利预算。", "",
        "## 5. 交叉校准与费用边界", "",
        "| 预测来源 | 仅背景：80%阈值实际覆盖 | 校准几何：80%阈值实际覆盖 | 校准几何主AUROC |",
        "| --- | --- | --- | --- |"]
    for name, r in (("校准折外", s["calibration"]["oof"]), ("校准全量拟合内", s["calibration"]["final_in_sample"]), ("开发", methods)):
        cov = {m: next(v for v in r[m]["acceptance"] if v["quantile"] == .8)["coverage"] for m in LABELS}
        lines.append(f"| {name} | {cov['background']:.2%} | {cov['calibrated']:.2%} | {r['calibrated']['auroc'][0]:.4f} |")
    lines += ["", "全量拟合内行只用于显示折外到最终参数的尺度漂移，不是独立成功证据；没有据此重选阈值。",
        "500次配对UID区间条件于固定的512校准用户与拟合规则，不覆盖校准样本不确定性或训练seed差异；1.0尺度有498次有效AUROC差重采样，其余两尺度500次。", "",
        f"冻结参数前的校准计时{costs['calibration_seconds']:.2f}s，写最终summary前的开发计时{s['elapsed_seconds']:.2f}s；阶段日志含后续统计/存档分别0.41s和21.37s。模型前向0次，两个数值/分组局部检查通过。",
        f"服务时非负仿射变换每状态仅另加2次算术，但旧准确检查仍需{costs['panel_arithmetic']/1e9:.6f}GFLOPs/16查询面板。",
        f"全开发状态检查{costs['check_all_panel_flops']/1e12:.6f}TFLOPs，占Exact-All的{costs['check_all_over_exact']:.2%}；共享分解{costs['factor_storage_bytes']/2**30:.3f}GiB，已知准备主项另{costs['shared_geometry_preparation_leading_flops']/1e12:.6f}TFLOPs。",
        f"因此校准几何的5%重建点，总普通算术下界仍为{s['total_budget_curves']['calibrated'][0]['total_arithmetic_lower_bound_fraction']:.2%}，不能称5%总费用。",
        "仅背景对照不需要计算u，不能给它强加几何检查成本。所有方法的总量曲线都是下界：包括适用的准确检查及原几何准备主项，但未穷尽校准拟合、原教师获取、源回放与服务操作；这些费用未当作0或声称免费。",
        "已知算术按原FLOPs口径，sqrt、比较另列；未用CPU/GPU计时替代算法预算。短历史费用反转仍存在，N=16检查约为Exact重建27.97倍。", "",
        "## 6. 研究决定", "",
        f"预定条件：{s['gates']}；全部通过：{s['advance']}。",
        "本轮只运行这一套分组、合并与拟合，没有开发结果驱动的再分组或阈值搜索。证据支持尺度校准的研究假设，也支持几何超出长度/目标背景的增量，不能唯一归因所有误差来源。",
        "可以进入下一轮预先固定的低成本近似或分级检查研究；本轮没有实现它们，更没有部署续用许可或重建调度器。",
        "必须保留M1短历史的低分严重失败及M5整组拒绝代价；校准预测是经验误差尺度，不是误差上界、错误概率或任意版本/查询下的安全保证。", "",
        "![续用、预算及分组覆盖](../../../../figures/pic/design2/scale_calibration_01/scale_calibration.png)"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
