"""Assemble round2 evidence; completed quality/cost inputs are mandatory."""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[4]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
BASE=ROOT/"results/partial_refresh_2026_10"
R=BASE/"round2"


def read(path):
    return json.loads((R/path).read_text())


def interval(v,digits=5):
    if v["estimate"] is None:return "未定义"
    a,b=v["percentile_95_interval"]
    return f"{v['estimate']:.{digits}f} [{a:.{digits}f}, {b:.{digits}f}]"


def main():
    methods=[("无持久物品表；批内复用","nocache3360/quality","full_static0"),
             ("静态4096物品表；全量码","static4096_3360/quality","full_static4096"),
             ("静态16384物品表＋旧码部分复用","static16384_strict3360/quality","strict_static16384")]
    costs=read("factored_cost/summary.json")
    evidence={}
    for name,path,key in methods:
        d=read(path+"/summary.json");v=read(path+"/cost_verification.json")
        if d["status"]!="complete" or d["users"]!=3360 or d["requests"]!=69361 or v["status"]!="complete":
            raise ValueError(f"incomplete final evidence: {path}")
        evidence[key]=d
    lines=["# 第二轮：更强的紧凑码刷新／维护降本", "",
        "用户明确本轮目标是第4.4节码刷新／维护FLOPs至少节省30%（期望30%–50%），保留原全量compact code收益的95%区间下界>90%。完整新增计算仍单列。长度分组只是一个探索例子。", "",
        "本轮尝试support层/行稀疏、首读长度阈值、仅满窗口、旧码能量/投影变化优先，以及等价编码分解、批内物品复用与发布前静态物品表。最终有效的主要节省来自编码函数内可精确共享的计算；并不是激进少刷历史行本身已达到目标。", "",
        "数据沿用Yambda500m的现有Medium处理版，冻结seed17的Medium HSTU V2→V3→V4→V5、hidden192、6层/6头、head32、context1024、rank8，沿用原local32/joint16适配权重；没有重训基础模型或改写旧证据。质量用户与原fit/validation及已记录校准角色分离，静态表只用原256fit用户的发布前历史。所有新路径位于experiments/archive/partial_refresh/round2/，结果集中在本目录。", "",
        "## 为什么可以节省而几乎不改变评分", "",
        "当前编码器先用固定逐坐标center/scale归一化，再做一次线性变换与SiLU。固定归一化可折叠进权重和偏置，数学上有：", "",
        "```text", "A = W / scale", "b* = b − sum(center × A)",
        "I(item) = CurrentEmbedding(item) × A_item + b*",
        "code(row) = SiLU([native K(row); native V(row)] × A_KV + I(item(row)))", "```", "",
        "每行原始全层编码166464 FLOPs，折叠后每行K/V项37440 FLOPs；每个实际计算的独立物品项110880 FLOPs（包含bias加法），每发布每worker折叠准备221184 FLOPs。同一个编码调用中相同物品只投影一次，再按原行序取回，仍为每个条目生成同一rank8码。NoCache不保留物品表，不跨调用记忆物品。", "",
        "可选静态表在每次发布前，用原256fit用户严格早于该发布的最新1024行为计频率，固定topK物品；未命中在本次调用计算，不写回表。这样不依赖离线按用户遍历的缓存命中顺序，也不使用未来请求/标签选择共享物品。每个静态物品的全部准备、所有未命中与归一化折叠都收费。", "",
        "固定归一化、单线性＋SiLU是本实现的适用条件；数学等价仍允许FP32计算重排误差。本轮以完整真实评分与配对AUC核对，而不是只比较encoder MSE。全量两点rho=1，均重新生成全部所需码；第三点同时保留先前raw strict的0/.875/.75策略。论文后续需据此说明计算复用与少刷行数各自的作用，不能把前者的节省全归因于旧码部分复用。", "",
        "## 冻结后的完整质量回放", "",
        "三点都覆盖同一全部3360位合格benchmark用户、69361请求、所有三发布和闲置用户。共享权重、静态表IDs与配置固定；完整覆盖和原Reuse/full-code/一次Rebuild分数逐项核对。原benchmark此前已用于探索，这不是新的全方法盲确认。收益比是等发布平均的method−Reuse AUC增益除以full-code−Reuse增益，不平均逐发布百分比。区间为1000次配对UID percentile bootstrap，不表示随机种子复现。", "",
        "| 方案 | 全量码−Reuse AUC | 新实现−Reuse AUC | 全量码收益保留% | 新实现−全量码 AUC | 最大完整logit差 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name,path,key in methods:
        d=evidence[key];p=d["quality"]["overall"]["paired"]
        dev=d["quality"]["overall"]["score_deviation"]["pooled_requests"]
        lines.append(f"| {name} | {interval(p['full_minus_reuse_auc'])} | {interval(p['partial_minus_reuse_auc'])} | {interval(p['full_gain_retained_percent'],6)} | {interval(p['partial_minus_full_auc'],9)} | {dev['partial_minus_full_max_absolute_logit_error']:.3g} |")
    lines += ["", "| 方案 | 保留区间下界>90% | 同UID配对90%margin区间 | margin区间下界>0 |", "| --- | --- | ---: | --- |"]
    for name,path,key in methods:
        a=evidence[key]["acceptance"]
        lines.append(f"| {name} | {'是' if a['paired_uid_ratio_lower_95_above_target'] else '否'} | {interval(a['paired_margin_at_current_target'],7)} | {'是' if a['paired_margin_supported_at_current_target'] else '否'} |")
    lines += ["", "全量两点的收益区间几乎为零宽，是编码算子等价及微小浮点差的结果，不能当作新模型能力或跨数据集迁移的证明。完整逐发布与预定义历史/活动群体结果保存在各aggregate中。原方法在纯新1024用户上低于Reuse的负结果仍保留：等价编码优化保留原函数，不能把它解释为修复了整体方法的用户迁移问题。", "",
        "## 固定30k真实事件账", "",
        f"成本另用固定30000居民用户、393521请求，未把质量样本线性放大。刷新／维护分母为原全量初始化＋发布刷新＋实际追加编码＋原support mask，共{costs['baseline_maintenance_flops']/1e12:.12f} TF；纯encoder分母{costs['baseline_encoder_flops']/1e12:.12f} TF另保留。完整新增计算包含共享D2拟合、发布验证、编码与query；一次Rebuild分母201.640840864172 TF。", "",
        "下表采用4worker各自准备的较保守成本：静态表和折叠准备重复4次，共享D2拟合仍一次；所有实际编码调用中的miss与K/V计算均计入。", "",
        "| 方案 | 刷新／维护TF | 刷新／维护节省% | 完整新增TF | 完整新增节省% | 完整新增 / 一次Rebuild% | 原校准source重构敏感性% |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name,path,key in methods:
        c=costs["policies"][key]["independent_preparation_sensitivity"]["4"]
        lines.append(f"| {name} | {c['maintenance_flops']/1e12:.6f} | {c['maintenance_saving_percent']:.5f} | {c['complete_added_flops']/1e12:.6f} | {c['complete_saving_percent_vs_original_masked']:.5f} | {c['complete_percent_of_once_rebuild']:.5f} | {c['source_reconstruction_sensitivity_percent']:.5f} |")
    raw_all=costs["policies"]["strict_static16384"]["complete_with_all_sources_no_reuse_percent"]
    raw_extra=costs["policies"]["strict_static16384"]["independent_preparation_sensitivity"]["4"]["extra_independent_preparation_flops"]
    raw_all+=100*raw_extra/201640840864172
    no_cache=costs["policies"]["full_static0"]["by_release"]
    all_rows=sum(v["counts"]["first_read_encode_rows"]+v["counts"]["append_encode_rows"] for v in no_cache.values())
    unique_terms=sum(v["counts"]["unique_items_in_calls"] for v in no_cache.values())
    lines += ["", f"NoCache的35.10%与静态4096表的49.94%均满足用户指定维护口径；两者完整新增计算分别只省约8.87%和12.61%，不能混用。单次共享准备的静态4096点为49.98713%，4worker分别准备为49.94358%。NoCache含原校准source重构稍超20%；静态4096点含该项仍为19.61258%。raw strict若原发布验证native输入也要另外重构且不复用旧校准前缀，4worker all-source保守敏感性为{raw_all:.5f}%，超过20%。", "",
        f"重复率是成本边界：fixed30k共有{all_rows:,}个实际编码行，逐调用独立物品项合计{unique_terms:,}，U/N={unique_terms/all_rows:.4f}。NoCache至少省30%约需要U/N≤0.7125（另计很小的固定准备）；没有重复时只省约10.88%。因此本数据上的35.10%不自动推广到低重复率数据。静态表收益另外取决于发布前fit热门物品的实际覆盖，不按未来请求重新选表。", "",
        "全量两点新增query工作沿用原方法：占原生整次query17.3636%、history attention26.0094%；raw strict点分别20.4694%、30.6616%。本轮不测或宣称实际延迟、吞吐、p99节省；item-ID去重的整数处理和额外gather/访存不是浮点运算，也不应当据此当作没有开销。", "",
        "## 状态与负结果", "",
        "用户私有码/flag/int32版本仍约占原生FP32 K/V的12.6085%，没有第二份纠正后的完整K/V。NoCache新增全局folded参数0.28235 MiB/发布/worker，无持久物品表；static4096另占4.5 MiB/发布/worker，4worker每个发布共18 MiB表（另有folded参数）。CPU字典/slot metadata、调用内projection/gather scratch在GPU运行inventory中单列；不是loaded serving peak。离线按用户回放实际保留三个发布对象，table合计13.5 MiB/worker、54 MiB/4worker。原型未实现全局共享表退休；单Current表是连续服务所需的工作集说明，不能冒充本次运行实测峰值。", "",
        "| 未选路径 | GPU质量保留%（512用户、100次UIDbootstrap） | 成本／判断 |", "| --- | ---: | --- |"]
    failed=[("长历史>768，尾部50%/25%","length769_tail50_25_512","维护编码约省27%，质量不达标"),
            ("长历史>768，尾部25%/25%","length769_tail25_25_512","维护编码约省29%，质量不达标"),
            ("仅满1024，尾部25%/25%","fullcap_tail25_25_512","质量不达标"),
            ("旧码能量优先","priority_energy25_512","质量不达标"),
            ("新旧投影变化优先","priority_drift25_512","质量不达标")]
    for name,path,note in failed:
        p=read(path+"/summary.json")["quality"]["overall"]["paired"]
        lines.append(f"| {name} | {interval(p['full_gain_retained_percent'],2)} | {note} |")
    lines += ["", "单独support稀疏编码与原masked函数相等，但维护只省约6.75%，不足30%；固定归一化折叠单独也不足30。旧码误差集中在保留码的大窗口用户，MSE更小不保证AUC更好；诊断保留在diagnostics/，不把失败路径删除或改写成成功。", "",
        "## 证据和复现", "",
        "CPU197通过、7CUDA跳过；新增检查覆盖真实维度且非零center/scale的编码等价、静态miss/顺序独立、跨用户收费、三发布生命周期和原生KV一致。GPU真实模型全部请求的评分差与UID区间见上表。每个来源run和最终aggregate分别核对row/miss/static准备/fold/mask/query理论账；来源哈希、用户角色、冻结配置与全部负结果保留。", "",
        "入口：[结果索引](README.md)、[冻结设置](frozen_candidates.json)、[固定30k成本](factored_cost/summary.json)、[源码和命令](../../../experiments/archive/partial_refresh/README.md)、[首轮报告](../report.md)。论文未编译，原模型、数据、旧评分与旧summary未改写。", ""]
    (R/"report.md").write_text("\n".join(lines))
    print(R/"report.md")


if __name__=="__main__":main()
