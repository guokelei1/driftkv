#!/usr/bin/env python3
"""Write the final research inventory from retained aggregates, without reruns."""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results/partial_refresh_2026_10"


def read(path):
    return json.loads((RESULTS / path).read_text())


def interval(record, digits=5):
    point = record["estimate"]
    bounds = record.get("percentile_95_interval")
    if point is None:
        return "未定义（全量码收益非正或缺少类别）"
    value = f"{point:.{digits}f}"
    if bounds:
        value += f" [{bounds[0]:.{digits}f}, {bounds[1]:.{digits}f}]"
    return value


def main():
    paths = [
        ("原始mask存储", "development4096_selected/summary.json"),
        ("raw存储，1%容忍度", "development4096_raw/summary.json"),
        ("raw存储，0.5%容忍度", "development4096_raw_strict/summary.json"),
        ("冻结strict策略，替代用户群1024", "benchmark1024/quality/summary.json"),
        ("冻结strict策略，全部合格替代用户3360", "benchmark3360/quality/summary.json"),
        ("纯新用户1024，原始策略", "fresh1024_v2/quality/summary.json"),
    ]
    final = read("benchmark3360/quality/summary.json")
    final_paired = final["quality"]["overall"]["paired"]
    lines = ["# 4.4 部分刷新：原型、结果与适用边界", "",
        "实验于2026-10-13启动、2026-10-14完成（Asia/Shanghai）。数据为已有Yambda-500M轨道，冻结Medium HSTU V2→V3→V4→V5：hidden192、6层、6头、head32、历史窗口1024、rank8码。基础模型、物品表示与共享适配权重均沿用保留资产。", "",
        f"冻结策略在全部3360位合格benchmark用户、{final['requests']:,}条真实请求上保留全量compact code收益的{interval(final_paired['full_gain_retained_percent'],2)}%。部分刷新相对Reuse的等发布平均AUC增益为{interval(final_paired['partial_minus_reuse_auc'])}；相对全量码损失为{interval(final_paired['partial_minus_full_auc'])}。发布刷新编码节省9.4053%，完整新增FLOPs节省0.51314%。这是保留大部分收益的正面机制案例，尚未通过整体至少95%保留的区间判据。", "",
        "原型保存实际旧码及独立编码版本，发布后按经过验证的比例刷新历史尾部；缺失码和早于上一版本的码强制刷新。所有条目使用当前Key/Value投影。原生K/V写入不变。", "",
        "后续保留未门控的原始码，在读取时应用已有继承依赖flag，避免永久丢掉下一次发布可能需要的码信息。码宽度仍为rank8，一行全部层/头编码完成后才提交版本；没有增加纠正后的完整K/V副本。", "",
        "共享编码器和Key/Value投影复用原local32/joint16冻结权重，没有重新训练。存储方式、容忍度和候选网格经过既有开发面板探索；每次发布的rho由原256位fit/16位validation用户的前向验证确定。最终strict策略为V3/V4/V5的rho=0/0.875/0.75，评分容忍度为fit纠正能量的0.5%。V3缺失全部码，实际仍全量初始化。比例不假设单调。", "",
        "## 质量", "",
        "保留比例是各发布等权平均的partial−Reuse AUC增益除以full-code−Reuse增益，不平均各发布保留百分比。方括号为1000次配对UID bootstrap的双侧95% percentile区间，描述用户采样不确定性；保留所有发布及闲置用户。比率仅在参考收益为正的draw中定义，有效/无定义draw数保存在JSON；不是模型随机种子的复现区间。", "",
        "| 方案 / 用户群 | 用户 / 请求 | 全量码−Reuse AUC | 部分刷新−Reuse AUC | 收益保留% | 部分刷新−全量码 AUC |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, path in paths:
        if not (RESULTS / path).exists():
            raise FileNotFoundError(f"required final evidence absent: {path}")
        d = read(path)
        paired = d["quality"]["overall"]["paired"]
        lines.append(f"| {name} | {d['users']} / {d['requests']} | {interval(paired['full_minus_reuse_auc'])} | {interval(paired['partial_minus_reuse_auc'])} | {interval(paired['full_gain_retained_percent'],2)} | {interval(paired['partial_minus_full_auc'])} |")
    lines += ["", "4096用户是既有开发面板。替代用户群来自预先存在的benchmark人群，排除当前开发、所有校准预留及正式确认角色。strict策略在其质量评分之前冻结；看到1024结果后，原策略不变，扩展到全部3360位合格用户。扩展不按质量、活跃度或请求可用性挑选；先前1024输入逐项等价后复用。这不是全方法第一次盲确认，也不是新的独立3360复验。", "",
        "95%保留判据检查margin=(partial−Reuse)−0.95×(full−Reuse)，并要求全量码收益为正、margin区间下界大于0。", "",
        "| strict策略人群 | margin及95%区间 | 区间支持至少95%保留 |", "| --- | ---: | --- |"]
    for name, path in paths[2:5]:
        paired = read(path)["quality"]["overall"]["paired"]
        lines.append(f"| {name} | {interval(paired['preservation_margin_auc'],7)} | {'是' if paired['noninferiority_supported_at_threshold'] else '否'} |")
    lines += ["", "最终3360用户的完整逐发布结果如下，V3仍是全量初始化，不能用其100%保留证明跨发布部分复用。", "",
        "| 发布 | 请求 | 全量码−Reuse AUC | 部分刷新−Reuse AUC | 收益保留% |", "| --- | ---: | ---: | ---: | ---: |"]
    for version, values in final["quality"]["by_release"].items():
        paired = values["paired"]
        lines.append(f"| {version} | {values['counts']['requests']} | {interval(paired['full_minus_reuse_auc'])} | {interval(paired['partial_minus_reuse_auc'])} | {interval(paired['full_gain_retained_percent'],2)} |")
    lines += ["", "预定义用户分组全部保留。历史长度与活动由首次发布前事件确定，不使用未来标签；分组间的AUC/比率不可再平均为总体指标。这些是探索性的边界证据，不是经过多重比较校正的确认检验。", "",
        "| 首次发布前用户分组 | 用户 / 请求 | 全量码−Reuse AUC | 部分刷新−Reuse AUC | 收益保留% |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    names = dict(history_le256="历史≤256", history_257_to768="历史257–768", history_gt768="历史>768",
                 past72h_inactive="此前72h无行为", past72h_active="此前72h有行为")
    for key, values in final["cohorts"].items():
        paired = values["overall"]["paired"]
        lines.append(f"| {names[key]} | {values['users']} / {values['requests']} | {interval(paired['full_minus_reuse_auc'])} | {interval(paired['partial_minus_reuse_auc'])} | {interval(paired['full_gain_retained_percent'],2)} |")
    rebuild = final_paired["partial_rebuild_recovery_percent"]
    lines += ["", f"长历史和发布前活跃用户保留得更好；短历史、低活动与V5损失更明显，不能声称统一比例对所有用户同样有效。整体部分刷新对一次Rebuild−Reuse缺口的恢复为{interval(rebuild,2)}%。", "",
        "纯新用户群的全量码和一次Rebuild本身都低于Reuse，因此收益保留率不定义；这是方法适用范围的负面证据，不被替代人群的正面结果覆盖。每个聚合目录均保留逐发布和预定义历史/活动群体结果。", "",
        "## FLOPs", "",
        "完整成本使用另行固定的30000居民用户、393521真实请求；不是对质量样本成本的直接放大。一次Rebuild增量分母为201.640840864172 TF。编码包括实际刷新、追加及强制刷新；共享校准、所有候选前向与验证历史中旧码的准备和维护全部收费。", ""]
    costs = read("population_cost_raw/summary.json")
    policies = costs["policies"]
    lines += ["| 方案 | 发布刷新编码FLOPs节省 | 完整新增FLOPs (TF) | 完整新增成本 / 一次Rebuild |", "| --- | ---: | ---: | ---: |"]
    for key, label in [("original_masked_full_reference","原始全量新码"),("full_refresh_reference","raw全量新码"),("raw_main","raw部分刷新，1%"),("raw_strict","raw部分刷新，0.5%")]:
        c = policies[key]
        lines.append(f"| {label} | {c['refresh_encoder_saving_percent']:.4f}% | {c['complete_added_flops']/1e12:.5f} | {c['complete_percent_of_once_rebuild']:.5f}% |")
    c, base = policies["raw_strict"], policies["original_masked_full_reference"]
    saving = 100 * (base["complete_added_flops"] - c["complete_added_flops"]) / base["complete_added_flops"]
    lines += ["", f"strict相对原始全量新码的完整新增FLOPs节省为{saving:.5f}%；相对raw全量新码另见成本账。不能把发布编码节省百分比当作完整计算节省。", "",
        "读取时门控的85.021622208G FLOPs计入完整成本；1.771283796GB flag传输量单独记账。新增查询工作占原生完整query的20.4694%、原生历史attention的30.6616%；不声明查询开销可忽略或loaded吞吐/p99优势。", "",
        "码、flags、int32版本分别占原生FP32 K/V的12.5%、0.0651%、0.0434%，合计约12.6085%；raw读取使用单层临时门控，不保留第二份码。", "",
        f"主成本沿用已有native输入可用的边界。严格方案单独重构新发布验证输入为{c['complete_with_publication_sources_percent']:.5f}%；单独重构原校准源输入为{c['complete_with_source_reconstruction_percent']:.5f}%；所有阶段无前缀复用的保守上界为{c['complete_with_all_sources_no_reuse_percent']:.5f}%。这些超过20%完整成本上限；最后一项因历史前缀重叠，不表示彼此独立的重构均必不可少。", "",
        "## 检查与复现", "",
        "CPU回归184通过、7项CUDA跳过。新原型覆盖跨版旧码保留、强制刷新、缺失/跳版、FIFO、同版本复用、全层因果退出及重建。GPU全量刷新与保存的原方法分数一致，融合/torch差≤4.77e−7。最终3360轨迹的原生K/V与full-code轨迹数值相等已检查61个快照；理论成本模拟252000项逐项与执行账核对通过，各分片也分别验证。", "",
        "当前为固定模型结构、有效原生K/V的同步原型，选中层全部编码完成后再读。不完整/不兼容K/V的自动检测重建以及第五章流水调度尚未实现。未首次读取期间的追加码延迟准备，以缺失码强制刷新并计账。论文以后需补充保存raw码、读取时应用support gate与因果退休的关系。", "",
        "源码及命令说明在experiments/archive/partial_refresh/README.md；配置、冻结策略、源哈希、用户角色、逐次结果和过程日志均在本结果根目录。原数据、权重、旧结果及论文未改写；没有基础模型训练或论文编译。", "",
        "本次结论限于这些冻结模型、用户群和发布轨迹；验证历史上的评分约束不自动成为所有服务请求的质量保证。", "",
        "关键入口：[最终3360质量](benchmark3360/quality/summary.json)、[完整成本](population_cost_raw/summary.json)、[开发和负结果索引](README.md)、[冻结策略](frozen_raw_candidates.json)、[源码及复现](../../experiments/archive/partial_refresh/README.md)。", ""]
    (RESULTS / "report.md").write_text("\n".join(lines))
    print(RESULTS / "report.md")


if __name__ == "__main__":
    main()
