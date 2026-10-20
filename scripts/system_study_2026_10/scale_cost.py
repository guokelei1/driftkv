#!/usr/bin/env python3
"""Population scaling of the retained 30k next-item arithmetic ledger (CPU only)."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "results/system_study_2026_10/cost/initial/summary.json"
LEDGER = ROOT / "results/query_relay_2026_10/workloads/stride16_candidates16_ledger.json"
POPULATIONS = (1000, 4000, 10000, 30000, 79681, 200000, 1000000)
TABLE_COMPONENTS = {"shared_item_projection_and_composition", "immutable_origin_feature_bank_preparation"}
SMALL_COMPONENTS = {"shared_matrix_compilation", "actual_block_inverse_preparation_estimate"}


def read(path):
    return json.loads(Path(path).read_text())


def decompose(branch, n0):
    price = branch["protocols"]["stride16_candidates16"]
    maintenance = Counter()
    for release in price["by_release"].values():
        maintenance.update(release["maintenance_components_flops"])
    if set(maintenance) - TABLE_COMPONENTS - SMALL_COMPONENTS - {"anchor_current_conversions"}:
        raise ValueError("An unclassified maintenance component cannot be silently dropped")
    fixed = {"correction_calibration_including_origin_once": price["calibration_flops"]}
    fixed.update({name: value for name, value in maintenance.items() if name != "anchor_current_conversions"})
    variable = dict(anchor_current_conversions=maintenance["anchor_current_conversions"],
                    ranking_correction=price["extra_ranking_flops"],
                    behavior_write_correction=price["extra_writer_flops"])
    u = price["common_native_incremental_append_flops"] + price["common_native_whole_query_flops"]
    result = dict(reference_users=n0, catalogue_rows=price["catalogue_table_rows"], replicas=price["preparation_replicas"],
        fixed_components_flops=fixed, variable_components_at_reference_flops=variable,
        fixed_flops=sum(fixed.values()), variable_at_reference_flops=sum(variable.values()),
        common_native_at_reference_flops=u, rebuild_at_reference_flops=price["once_rebuild_flops"],
        source_reconstruction_fixed_flops=price["source_reconstruction_sensitivity_flops"],
        catalogue_linear_preparation_flops=sum(maintenance[key] for key in TABLE_COMPONENTS),
        small_matrix_preparation_flops=sum(maintenance[key] for key in SMALL_COMPONENTS))
    assert result["fixed_flops"] + result["variable_at_reference_flops"] == price["complete_added_flops"]
    assert u + result["rebuild_at_reference_flops"] == price["total_modeled_service_full_flops"]
    assert u + result["fixed_flops"] + result["variable_at_reference_flops"] == price["total_modeled_service_relay_flops"]
    return result


def fixed_cost(parts, *, with_source=False, catalogue_rows=None, replicas=None):
    m = parts["catalogue_rows"] if catalogue_rows is None else catalogue_rows
    p = parts["replicas"] if replicas is None else replicas
    if m <= 0 or p <= 0:
        raise ValueError("Catalogue and replica counts must be positive")
    tables = Fraction(m, parts["catalogue_rows"]) * parts["catalogue_linear_preparation_flops"]
    shared = Fraction(p, parts["replicas"]) * (tables + parts["small_matrix_preparation_flops"])
    result = Fraction(parts["fixed_components_flops"]["correction_calibration_including_origin_once"]) + shared
    if with_source:
        result += parts["source_reconstruction_fixed_flops"]
    return result


def evaluate(parts, users, *, with_source=False, catalogue_rows=None, replicas=None, common_training=0):
    if users <= 0 or common_training < 0:
        raise ValueError("Population must be positive; shared training cannot be negative")
    scale = Fraction(users, parts["reference_users"])
    fixed = fixed_cost(parts, with_source=with_source, catalogue_rows=catalogue_rows, replicas=replicas)
    variable = scale * parts["variable_at_reference_flops"]
    common = scale * parts["common_native_at_reference_flops"]
    rebuild = scale * parts["rebuild_at_reference_flops"]
    full, relay = common + rebuild + common_training, common + variable + fixed + common_training
    saved = full - relay
    return dict(users=users, fixed_flops=float(fixed), variable_flops=float(variable),
        common_native_flops=float(common), once_rebuild_flops=float(rebuild),
        added_flops=float(fixed + variable), full_flops=float(full), relay_flops=float(relay),
        saved_flops=float(saved), net_saved_percent=100 * float(saved / full),
        added_percent_of_rebuild=100 * float((fixed + variable) / rebuild),
        common_foundation_training_flops_for_this_calculation=common_training,
        with_source_reconstruction=with_source,
        catalogue_rows=parts["catalogue_rows"] if catalogue_rows is None else catalogue_rows,
        replicas=parts["replicas"] if replicas is None else replicas)


def threshold(numerator, denominator, *, strict=False):
    """Smallest positive integer N meeting a linear condition; exact rational math."""
    if denominator <= 0:
        return dict(minimum_users=None, boundary_users=None,
                    reason="Unreachable for finite N with positive fixed cost: per-user margin is nonpositive")
    boundary = Fraction(numerator) / Fraction(denominator)
    floor = boundary.numerator // boundary.denominator
    minimum = floor + 1 if strict else -(-boundary.numerator // boundary.denominator)
    return dict(minimum_users=max(1, minimum), boundary_users=float(boundary),
                condition=">" if strict else ">=", reason=None)


def thresholds(parts, with_source=False):
    n0, f = parts["reference_users"], fixed_cost(parts, with_source=with_source)
    r, v, u = (parts[key] for key in ("rebuild_at_reference_flops", "variable_at_reference_flops", "common_native_at_reference_flops"))
    return dict(
        positive_net_saving=threshold(f * n0, r - v, strict=True),
        complete_service_saving={str(pct): threshold(100 * f * n0, 100 * (r - v) - pct * (u + r))
                                 for pct in (10, 20, 30)},
        at_most_20_percent_added_over_rebuild=threshold(5 * f * n0, r - 5 * v),
        asymptotic_complete_service_saving_percent=100 * (r - v) / (u + r),
        asymptotic_added_over_rebuild_percent=100 * v / r)


def memory_at_scale(memory, users, reference_users, replicas=4, catalogue_multiplier=1):
    """Logical final-state storage; excludes models, allocator and working space."""
    scale = Fraction(users, reference_users)
    native = scale * memory["total_population_native_kv_bytes"]
    code = scale * memory["total_population_code_bytes"]
    catalogue = memory["active_compiled_item_table_bytes_per_replica"] + memory["immutable_origin_bank_bytes_per_replica"]
    handoff_catalogue = catalogue + memory["active_compiled_item_table_bytes_per_replica"]
    shared = memory["shared_state_bytes_per_replica"] + (catalogue_multiplier - 1) * catalogue
    handoff = memory["publication_peak_shared_state_bytes_per_producer"] + (catalogue_multiplier - 1) * handoff_catalogue
    return dict(users=users, replicas=replicas, catalogue_multiplier=catalogue_multiplier,
        native_kv_total_bytes=float(native), code_total_bytes=float(code),
        native_and_code_total_bytes=float(native + code),
        per_replica_native_kv_bytes=float(native / replicas), per_replica_code_bytes=float(code / replicas),
        shared_bytes_per_replica=float(shared), shared_bytes_all_replicas=float(replicas * shared),
        handoff_peak_shared_bytes_per_replica=float(handoff), handoff_peak_shared_bytes_all_replicas=float(replicas * handoff),
        per_replica_state_and_shared_lower_bound_bytes=float((native + code) / replicas + shared),
        per_replica_state_and_handoff_shared_lower_bound_bytes=float((native + code) / replicas + handoff))


def describe(report):
    lines = ["# 部署规模与 FLOPs 盈亏平衡", "",
        "主任务固定 next-item（每16次行为触发推荐、每轮16候选）。以下复用真实30k事件形状作条件预测，不是新增质量结果、计时结果或Large/Max人口实证。", "",
        "将全部方法成本拆为固定 F 与随人口增长的 N·v：F包括纠正校准（原起点一次）、完整目录四副本准备、固定特征表与共享小矩阵；v包括逐用户码物化/反锚及逐请求/行为纠正。u为公共原生追加与完整查询，r为每用户每发布首次推荐重建。", "",
        "`Full=N(u+r)`；`Relay=N(u+v)+F`；`省下的FLOPs=N(r−v)−F`。这里N是同一三发布窗口内的部署人口，保留30k账本中的活跃率、历史长度及请求/行为分布；每用户量取原总账/30000。", "",
        "| 分支 | F (TF) | 盈利最小N | 总服务省10% | 省20% | 省30% | 新增/重建≤20% | 总服务节省上限 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    def display(value):
        return "不可达" if value is None else f"{value:,}"
    for name, branch in report["branches"].items():
        t = branch["thresholds"]["source_available"]
        lines.append(f"| {name} | {branch['decomposition']['fixed_flops']/1e12:.3f} | {display(t['positive_net_saving']['minimum_users'])} | "
            + " | ".join(display(t['complete_service_saving'][str(p)]['minimum_users']) for p in (10, 20, 30))
            + f" | {display(t['at_most_20_percent_added_over_rebuild']['minimum_users'])} | {t['asymptotic_complete_service_saving_percent']:.3f}% |")
    lines += ["", "下表每格为“新增/重建% ／ 总服务净节省%”；负净节省表示方法总 FLOPs 更多。", "",
              "| N | current | Q1读6+6 | Q2校准128 |", "|---:|---:|---:|---:|"]
    for i, n in enumerate(POPULATIONS):
        cells = [f"{b['source_available'][i]['added_percent_of_rebuild']:.3f} / {b['source_available'][i]['net_saved_percent']:.3f}"
                 for b in report["branches"].values()]
        lines.append(f"| {n:,} | " + " | ".join(cells) + " |")
    lines += ["", "主账假定校准来源状态可取得，教师重建已计入F。若需重放校准来源，固定校准人口下将该项加入F；初始起点来源只加一次。", "",
              "| 分支 | 含来源F (TF) | 盈利最小N | 总服务省20% | 省30% | 新增/重建≤20% | 30k总服务净节省 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name, branch in report["branches"].items():
        t = branch["thresholds"]["source_rebuilt"]
        row = next(r for r in branch["source_rebuilt"] if r["users"] == 30000)
        lines.append(f"| {name} | {row['fixed_flops']/1e12:.3f} | {display(t['positive_net_saving']['minimum_users'])} | {display(t['complete_service_saving']['20']['minimum_users'])} | {display(t['complete_service_saving']['30']['minimum_users'])} | {display(t['at_most_20_percent_added_over_rebuild']['minimum_users'])} | {row['net_saved_percent']:.3f}% |")
    lines += ["", "这些阈值固定目录大小、四副本、模型维度和64/128名校准用户，不能无条件外推。目录项数M、副本数p改变时，"
              "`F(M,p)=校准 + 来源敏感性 + (p/4)[(M/M0)·目录相关准备 + 小矩阵准备]`；若校准人口也增长，校准和来源项需重新计价。"
              "逐用户转换、每次读取与原生公共计算不会因人口增大自动消失。", "",
              "为显示固定F假设的边界，JSON另列目录翻倍、副本翻倍、二者翻倍及副本随人口增加的条件预测。目录变化仍需保持模型维度，且不提供质量保证；副本增加不把已经分片的用户服务量再乘一次。", "",
              "当前账包含correction校准，未计入两边共同的基础模型训练B(N)。保留训练材料中未找到可直接复用的完整FLOPs账，因此不虚构数值："
              "加入共同训练后，两边总量各加B(N)，绝对节省不变，净节省比例为`[N(r−v)−F]/[N(u+r)+B(N)]`。"
              "上表是已有服务＋correction校准范围，不能写成已经覆盖基础模型训练的整体比例；若B(N)按人口线性增长，连渐近节省上限也会降低。", "",
              "下面列出current的逻辑存储量（十进制GB），假设最终窗口全部保留且平均分给四个副本。每副本共享状态稳态3.182GB、交接峰值4.773GB；后者含旧、新目录表并存。Q2相同，Q1共享状态略增，逐行码容量不变。", "",
              "| N | 原生K/V总GB | 码总GB | 每副本状态＋共享下界GB | 每副本交接下界GB |",
              "|---:|---:|---:|---:|---:|"]
    for row in report["branches"]["current"]["all_state_resident_four_replica_bytes"]:
        lines.append(f"| {row['users']:,} | {row['native_kv_total_bytes']/1e9:.3f} | {row['code_total_bytes']/1e9:.3f} | {row['per_replica_state_and_shared_lower_bound_bytes']/1e9:.3f} | {row['per_replica_state_and_handoff_shared_lower_bound_bytes']/1e9:.3f} |")
    lines += ["", "仅30k用户的原生K/V＋码已约281.254GB，四等分后加共享状态约73.495GB/副本，尚未计模型、工作区与分配器；因此四副本不能解释为4×48GB A40全GPU驻留。"
              "固定目录/副本只用于隔离算术盈亏；扩部署时必须说明主存或缓存分层、驻GPU活跃状态比例与host–GPU传输，并将它们纳入时间模型。副本随人口增长也不能消除此处起点的容量缺口。", "",
              "JSON逐项保存native/code总量、每副本共享状态与交接峰值。目录变化仅放大共享物品表与固定特征表，副本变化复制共享状态、分摊用户行状态；模型、临时转换空间、并发请求缓冲尚未计入，所列值都是容量下界。", "",
              "Q1逐次读取成本更高，其渐近新增/重建比例已超过20%，扩大人口无法把它降到该参考线；仍可保留为潜在质量更高的取舍点。Q2的额外成本主要在固定校准，规模增长可摊薄，但不会改善质量本身。", "",
              "复核：`python -m pytest tests/system_study_2026_10/test_scale_cost.py -q`。重新生成需新目录：`python scripts/system_study_2026_10/scale_cost.py --output <new_directory>`。", ""]
    return "\n".join(lines)


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    screen, ledger = read(INPUT), read(LEDGER)
    n0 = ledger["users"]
    assert n0 == 30000
    branches = {}
    for name, branch in screen["branches"].items():
        parts = decompose(branch, n0)
        rows = {"source_available": [evaluate(parts, n) for n in POPULATIONS],
                "source_rebuilt": [evaluate(parts, n, with_source=True) for n in POPULATIONS]}
        original = branch["protocols"]["stride16_candidates16"]
        check = evaluate(parts, n0)
        for computed, retained in (("added_flops", "complete_added_flops"), ("full_flops", "total_modeled_service_full_flops"),
                                  ("relay_flops", "total_modeled_service_relay_flops"), ("saved_flops", "net_saved_flops")):
            assert check[computed] == original[retained], (name, computed)
        assert abs(check["net_saved_percent"] - original["net_saved_percent_of_total_full_service"]) < 1e-12
        sensitivities = {}
        for label, m, p in (("catalogue_x2", 2 * parts["catalogue_rows"], 4),
                            ("replicas_x2", parts["catalogue_rows"], 8),
                            ("catalogue_and_replicas_x2", 2 * parts["catalogue_rows"], 8)):
            sensitivities[label] = [evaluate(parts, n, catalogue_rows=m, replicas=p) for n in POPULATIONS]
        sensitivities["replicas_grow_to_hold_population_per_replica"] = [
            evaluate(parts, n, replicas=max(4, (4 * n + n0 - 1) // n0)) for n in POPULATIONS]
        memory = branch["memory"]
        for scenario in sensitivities.values():
            for row in scenario:
                row["memory_lower_bound"] = memory_at_scale(memory, row["users"], n0, replicas=row["replicas"],
                    catalogue_multiplier=row["catalogue_rows"] // parts["catalogue_rows"])
        branches[name] = dict(decomposition=parts, **rows,
            thresholds={"source_available": thresholds(parts), "source_rebuilt": thresholds(parts, True)},
            catalogue_replica_sensitivities=sensitivities,
            all_state_resident_four_replica_bytes=[memory_at_scale(memory, n, n0) for n in POPULATIONS],
            quality_status=branch["evidence"])
    report = dict(status="complete_conditional_population_cost_projection", created_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        task="next-item_stride16_candidates16", reference_population=n0, population_grid=list(POPULATIONS),
        formulas=dict(full="N*(u+r)+B(N)", relay="N*(u+v)+F+B(N)", absolute_saved="N*(r-v)-F",
            break_even="N>F/(r-v)", target_total_saving="N*(r-v)-F >= s*[N*(u+r)+B(N)]",
            added_over_rebuild="(F+N*v)/(N*r)", fixed_cost_with_catalogue_replicas="C+S+(p/p0)*[(M/M0)*T+G]"),
        branches=branches,
        assumptions=["User means preserve the complete 30k three-release event distribution, including idle users; no favorable cohort selection.",
            "Catalogue rows, model dimensions, replica count and calibration population fixed in the main grid; sensitivities relax catalogue/replicas explicitly.",
            "79681 and200000 are population counts only, not Large/Max model or dataset evidence.",
            "Calibration source reconstruction is fixed only while calibration roles/counts remain fixed; teachers and initial origin work are not charged twice.",
            "Catalogue preparation is independent on four replicas in the reference. Served users are partitioned, not replicated four times.",
            "Per-replica resident-state estimates assume all final states resident, omit model/scratch, and are not a deployment memory claim.",
            "30k final row state plus shared tables already exceeds four 48GB A40 cards; host/cache-tier placement and transfer must enter a later time model.",
            "No GPU, training, evaluation, arrival timing or old result rewrite performed."],
        common_foundation_training=dict(status="not_priced_in_retained_service_ledger", measured_flops=None,
            inspected_scope="Relevant retained Medium training JSON/Markdown and unified-training docs were searched for training FLOPs; no reusable complete accounting found.",
            evidence_scope="Numerical grid sets B=0 only to preserve the established service-plus-correction-calibration scope; it is not evidence that foundation training is free.",
            effect="Both methods add B(N). Absolute savings are unchanged; positive percentage savings decrease. Fixed B adds s*B to the target-threshold numerator; linear B=N*b subtracts s*b from its per-user margin."),
        checks=dict(reference_30000_exact_costs_match_all_branches=True),
        sources=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in (INPUT, LEDGER, Path(__file__))])
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (output / "summary.md").write_text(describe(report))
    print(json.dumps(dict(status=report["status"], output=str(output.resolve()),
        thresholds={n: b["thresholds"]["source_available"] for n, b in branches.items()}), ensure_ascii=False))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New output directory; never overwrite retained results")
    run(parser.parse_args().output)
