"""Render the retained CPU-only budget audit, without rerunning comparisons."""

import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"results/design2/analysis/budget_audit_01"


def main():
    s=json.loads((OUT/"summary.json").read_text())
    c=s["cost"]
    pair=s["paired_original_minus_cost_only"]
    lines=["# Design 2：几何增量价值与完整检查费用复核","",
        "**条件复核支持几何具有长度/费用之外的增量排序信号；原始分数作为续用可信度的路线仍停止。**",
        "这不是尺度问题已被唯一归因，也不是已经获得廉价、可靠的检测器。",
        "仅复用原1024开发UID、15299状态、244784查询的结果，无模型前向、H重算或新拟合。",
        "主尺度仍为最大绝对logit残余≥0.5，辅助尺度0.1/1.0；不改UID、查询、模型、原阈值或预算。","",
        "## 1. 主比较：5%假想重建预算","",
        "| 排序 | 严重状态加权覆盖 | 选中UID | 选中状态 | 捕获严重UID/状态 |",
        "| --- | --- | --- | --- | --- |"]
    for label,key in (("原几何分数/费用","original"),("纯重建费用升序","cost_only")):
        r=s[key][0]
        lines.append(f"| {label} | {r['recall'][0]:.2%} | {r['selected_uids']} | {r['selected_states']} | "
                     f"{r['selected_serious_uids'][0]}/{r['selected_serious_states'][0]} |")
    for label,key in (("同目标/近长度打乱","within_length_shuffle"),("保留UID状态关系打乱","uid_profile_shuffle")):
        r=s["permutations"][key]["curve"][0]
        lines.append(f"| {label} | {r['recall_mean'][0]:.2%} [{r['recall_p025'][0]:.2%}, {r['recall_p975'][0]:.2%}] | "
                     f"{r['selected_uids_mean']:.1f} [{r['selected_uids_range'][0]:.0f}, {r['selected_uids_range'][1]:.0f}] | "
                     f"{r['selected_states_mean']:.1f} [{r['selected_states_range'][0]:.0f}, {r['selected_states_range'][1]:.0f}] | 按次保留 |")
    lines += ["","打乱行给出1000次条件置换的均值与2.5–97.5分位，**不是UID置信区间**；"
        "每次实际选中数量保存在对应npz，不能把置换次数当作独立用户数。",
        f"原法相对纯费用的主点覆盖差为{pair['point'][0][0]*100:.2f}个百分点，"
        f"500次配对UID重采样描述区间[{pair['p025'][0][0]*100:.2f}, {pair['p975'][0][0]*100:.2f}]个百分点。",
        f"事前固定的条件复核判别通过：{s['incremental_signal_rule_pass']}。原法超过两种打乱97.5分位，"
        "且对纯费用的配对区间下界大于0。因此，原先100%覆盖不能仅由偏爱便宜状态解释。",
        "主尺度仍只有13名严重失败UID、31状态，单backbone seed17且复用了开发结果；"
        "这只是继续研究增量价值的证据，不是独立确认或泛化保证。","",
        "## 2. 打乱到底控制了什么","",
        "组固定为目标版本 × floor(4 log2 N)，组内历史长度比小于约1.19，N=1024独立成组。",
        "首个对照在组内置换状态分数；第二个按同UID、同目标/长度组的有序状态类型组成匹配，"
        "整体置换分数向量，保留目标内多快照的分数关系。后者还条件于状态类型与快照数，"
        "不声称保留跨目标的全部UID相关性。两者均不使用残余划组。",
        "小组、相同分数或单UID组可能没有有效跨用户交换，以下公开可打乱程度：","",
        "| 对照 | 全体跨UID供体比例 | 严重状态跨UID供体比例 |",
        "| --- | --- | --- |"]
    for name,v in s["permutations"].items():
        lines.append(f"| {name} | {v['mean_cross_uid_donor_fraction']:.2%} | {v['mean_serious_cross_uid_donor_fraction']:.2%} |")
    lines += ["","这是观察数据的条件置换参照，不依赖“状态相互独立”的显著性宣称。"
        "配对描述区间以UID为单位，重复UID同时携带费用及状态权重，按加权状态整块预算前缀选择。",
        "分数超过长度/费用控制，不等于证明二次型优于其他未比较的便宜状态特征；"
        "当前仍不能区分尺度、模型偏差和信息缺失的因果贡献。","",
        "## 3. 全预算曲线与固定辅助尺度","",
        "| 重建预算 | 误差尺度 | 原分数/费用 | 纯费用 | 状态打乱均值 | UID向量打乱均值 |",
        "| --- | --- | --- | --- | --- | --- |"]
    for i,b in enumerate(s["budgets"]):
        for j,e in enumerate(s["errors"]):
            lines.append(f"| {b:.0%} | {e} | {s['original'][i]['recall'][j]:.2%} | "
                f"{s['cost_only'][i]['recall'][j]:.2%} | "
                f"{s['permutations']['within_length_shuffle']['curve'][i]['recall_mean'][j]:.2%} | "
                f"{s['permutations']['uid_profile_shuffle']['curve'][i]['recall_mean'][j]:.2%} |")
    lines += ["","**0.1尺度在5%预算下，原法38.40%反而低于纯费用40.69%。**"
        "其配对差区间跨0，因此不称几何在每种误差尺度/预算下都更好。"
        "完整各预算选中UID/状态、两种置换范围、配对区间及逐目标结果均在summary.json，未删除不利单元。",
        "所有覆盖率沿用原UID/目标等权口径；实际选中数量是去重UID和状态的直接计数，"
        "不把同UID多个状态算作独立用户。多个替代状态的预算和不是已实现的连续生命周期费用。","",
        "## 4. 完整检查的算术账本","",
        f"三角求解：每查询{c['solve_per_query']:,} FLOPs，16候选为{c['panel_solve_only']/1e9:.6f} GFLOPs。",
        f"归一化、源×query交互、范数和聚合缩放另{c['other_per_query']:,} FLOPs/查询，"
        f"以及每状态6次count×mask。合计 **{c['panel_arithmetic']:,} FLOPs/状态"
        f"（{c['panel_arithmetic']/1e9:.6f} GFLOPs）**。",
        f"另列每状态{c['panel_sqrt_calls']}次sqrt、{c['panel_comparisons']}次比较；"
        "不随意把特殊函数当成矩阵乘加。数据搬运、类型转换、存储带宽不属于这个FLOPs数字。",
        "按当前实际实现，mask关闭的块也执行求解，不能将尚未实现的跳过优化记为已节省。"
        "counts、源latent、query/native响应假定已由Design1提供；若额外造检查查询，还须支付其读取。","",
        "| 保留历史N | Exact重建 GFLOPs | 单面板完整检查 / Exact |",
        "| --- | --- | --- |"]
    for row in c["length_table"]:
        lines.append(f"| {row['length']} | {row['exact_arithmetic']/1e9:.6f} | {row['check_over_exact']:.4f}× |")
    lines += ["","分母沿用第一轮的依赖闭包重建矩阵/标量算术口径；Exact的ELU/SiLU/rsqrt/sincos"
        "调用数在summary.json单列，不为获得更好比率切换口径。1024事件检查约为重建22.66%，"
        "与专家约22.5%的数量级一致；N=16时约为27.97倍。","",
        "## 5. 从5%重建预算到总增量下界","",
        f"本轮15299个候选状态的Exact-All算术总量为{c['exact_all_panel_flops']/1e12:.6f} TFLOPs；"
        f"仅检查全部状态就需{c['check_all_panel_flops']/1e12:.6f} TFLOPs，"
        f"占{c['check_all_over_exact']:.2%}。",
        f"另有恢复H的Gram主要收缩和Cholesky主项{c['shared_geometry_preparation_leading_flops']/1e12:.6f} TFLOPs"
        f"（{c['shared_preparation_over_exact']:.2%}），共享分解存储{c['factor_storage_bytes']/2**30:.3f}GiB。",
        f"因此原5%重建曲线的算术总量下界为 **{s['total_curves'][0]['total_arithmetic_lower_bound_fraction']:.2%} Exact-All**，"
        "不是5%。所有未被选择的状态也已检查，不能只给1487个重建状态计检测费用。",
        "这里完整计入了检测函数的普通算术，但总量仍是下界：准备只计已知主项，"
        "源回放/查询获取/摘要与Design1准备等其余费用未当作0。"
        "未来若在Design1拟合时直接保留Gram，不能重复收取其共同费用；即使省去准备，"
        "当前全量检查本身的26.12%门槛仍然存在。",
        "下表使用同一总算术预算，在先支付全量检查及已知准备后分配剩余重建预算；"
        "它仍是假想选择下界，不是新调度器或实际服务实验。","",
        "| 总预算 | 几何检查是否付得起 | 几何捕获严重状态 | 纯费用捕获严重状态 |",
        "| --- | --- | --- | --- |"]
    for budget,v in s["fixed_total_budget"].items():
        recall="不具备检查预算" if v["geometry"] is None else f"{v['geometry']['recall'][0]:.2%}"
        lines.append(f"| {float(budget):.0%} | {v['geometry_check_affordable']} | {recall} | {v['cost_only']['recall'][0]:.2%} |")
    lines += ["","**当前实现无法落入≤20%的全量检查预算。**"
        "30%下界预算在这份面板上仍能捕获全部主尺度严重状态，但遗漏的准备/服务开销可能使它超额，"
        "不能把该行当作30%完整系统资格。没有用GPU计时替代FLOPs。","",
        "## 6. 决定","",
        "保留几何在预算排序中的增量线索；不结束整个二次型研究，也不恢复“原始低分就是可信”的路线。",
        "这次证据足以支持下一轮预先固定一次长度条件残余尺度校准的研究问题，"
        "但不保证校准能修复拟合偏差或信息缺失。归一化搜索、敏感度、压缩与真实重建本轮均未实施。",
        "Design2的状态续用/重建框架继续保留；被停止的是第一个直接可信度实现。",
        f"本次CPU复核耗时{ s['elapsed_seconds']:.2f}s，模型前向0次；三个局部检查通过。","",
        "![预算与检查费用曲线](../../../../figures/pic/design2/budget_audit_01/budget_audit.png)"]
    (OUT/"report.md").write_text("\n".join(lines)+"\n")
    print(OUT/"report.md")


if __name__=="__main__":
    main()
