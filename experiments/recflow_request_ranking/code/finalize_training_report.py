"""Produce the handoff from completed audits; never run or change model scores."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .data_common import json_save, sha256
from .temporal_cohort import read_plan

BASE = Path("experiments/recflow_request_ranking")


def read(path):
    return json.loads(Path(path).read_text())


def main():
    temporal = BASE / "runs/temporal_cohort/attempt_1"
    plan = read_plan(temporal / "plan.json")
    original_path = BASE / "runs/version_chain_verification/attempt_1/completion_audit.json"
    temporal_path = temporal / "completion_audit.json"
    old, new = read(original_path), read(temporal_path)
    if (old["status"] != "artifacts_verified" or old["total_models"] != 10
            or old["total_audited_windows"] != 19 or new["status"] != "artifacts_verified"
            or new["total_windows"] != 9 or not new["within_total_budget"]):
        raise ValueError("Both complete, budget-compliant evaluation studies are required")
    if [row["day"] for row in new["windows"]] != list(range(28, 37)):
        raise ValueError("Missing or duplicate temporal window")
    inventory_path = BASE / "reports/model_chain_inventory.json"
    inventory = read(inventory_path)
    if [row["version"] for row in inventory["models"]] != list(range(10)):
        raise ValueError("Model inventory is incomplete")
    for model, verified in zip(inventory["models"], old["models"], strict=True):
        if (model["sha256"] != verified["sha256"]
                or Path(model["checkpoint"]).stat().st_size != verified["bytes"]):
            raise ValueError("Model inventory differs from the independently verified files")
    output = BASE / "reports/final_training_handoff.json"
    result_path = BASE / "RESULTS.md"
    csv_path = BASE / "reports/version_chain_all_populations.csv"
    if any(path.exists() for path in [output, result_path, csv_path]):
        raise ValueError("Keep the existing final handoff; do not silently overwrite it")
    sources = {str(path): sha256(path) for path in [original_path, temporal_path, inventory_path,
               temporal / "training_uid_coverage_audit.json", temporal / "parity/parity.json",
               BASE / "reports/completion_cpu_checks.json", BASE / "reports/temporal_cohort_cpu_checks.json"]}
    rows, total_candidates = [], 0
    for population, windows in [("original", old["windows"]), ("training_role_temporal", new["windows"])]:
        for row in windows:
            path, audit_path = Path(row["summary"]), Path(row["score_audit"])
            summary, audit = read(path), read(audit_path)
            if (audit["status"] != "passed" or audit["history_limit"] != 1024
                    or audit["source_artifact_sha256"][str(path)] != sha256(path)
                    or audit["requests"] != summary["requests"] or summary["world_size"] != 4):
                raise ValueError("Completed full-score audit or summary binding differs")
            sources[str(audit_path)], sources[str(path)] = sha256(audit_path), sha256(path)
            total_candidates += audit["candidate_instances"]
            group = population if population != "original" else ("development" if row["role"] == 2 else "unseen_user_confirmation")
            difference = summary["paired_current_minus"]["reuse"]
            current, reuse = (summary["metrics"][name]["ndcg@10"] for name in ["current", "reuse"])
            rows.append({"population": group, "version": row["version"], "day": summary["days"][0],
                         "requests": summary["requests"], "positive_requests": summary["positive_requests"],
                         "positive_users": summary["positive_users"], "rebuild_ndcg10": current, "reuse_ndcg10": reuse,
                         "rebuild_minus_reuse": difference["difference"], "ci95_low": difference["ci95"][0],
                         "ci95_high": difference["ci95"][1], "reuse_relative_loss_percent": 100*(current-reuse)/current,
                         "summary": str(path), "score_audit": str(audit_path)})
    if len(rows) != 28:
        raise ValueError("The handoff must preserve all nineteen original and nine temporal windows")
    with csv_path.open("x") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    train_seconds = inventory["warmup_wall_seconds"] + sum(row["phase_wall_seconds"] for row in inventory["models"])
    foundation_seconds = inventory["warmup_wall_seconds"] + inventory["models"][0]["phase_wall_seconds"]
    warm = [row for row in rows if row["population"] == "training_role_temporal"]
    positive = [row["version"] for row in warm if row["ci95_low"] > 0]
    negative = [row["version"] for row in warm if row["ci95_high"] < 0]
    ambiguous = [row["version"] for row in warm if row["ci95_low"] <= 0 <= row["ci95_high"]]
    useful = sum(row["clear_cache_loss"] and row["native_beats_baselines"] for row in new["windows"])
    if useful != new["useful_temporal_cache_edges"]:
        raise ValueError("Useful-edge count differs from the frozen criterion")
    table = ["| 更新边 | 原开发差值 | 未见用户确认差值 | 训练分组时序差值 | 时序 UID 95% 区间 | 时序相对损失 |",
             "| --- | ---: | ---: | ---: | --- | ---: |"]
    for version in range(1, 10):
        comparison = {row["population"]: row for row in rows if row["version"] == version and row["day"] <= 36}
        a, b, c = (comparison[name] for name in ["development", "unseen_user_confirmation", "training_role_temporal"])
        table.append(f"| M{version-1}→M{version} | {a['rebuild_minus_reuse']:+.6f} | {b['rebuild_minus_reuse']:+.6f} | "
                     f"{c['rebuild_minus_reuse']:+.6f} | [{c['ci95_low']:+.6f}, {c['ci95_high']:+.6f}] | {c['reuse_relative_loss_percent']:+.2f}% |")
    model_table = ["| 模型 | 本阶段训练日 | 每轮请求 | 轮数 | checkpoint |", "| --- | --- | ---: | ---: | --- |"]
    for model in inventory["models"]:
        days = ", ".join(f"D{day}" for day in model["training_days"])
        model_table.append(f"| M{model['version']} | {days} | {model['requests_per_epoch']} | {model['epochs']} | "
                           f"[{Path(model['checkpoint']).name}]({model['checkpoint']}) |")
    content = f"""# RecFlow Max 训练与版本链交付

十个实际 Max checkpoint、原十九个窗口及追加九个时序窗口均已完成并核对。
历史上限1024，正式日更新和评测的实际历史全部1024；完整已记录排序候选，不额外抽样。
模型16层、宽320、10×32 heads，共1,154,235,993参数；其中1,140,971,840为物品embedding。

## 训练时间与数据量

“25分钟”是262144请求的一轮曝光预热，实测25.05分钟。
完整候选对齐为131072请求×2轮，实测52.82分钟；foundation两阶段合计{foundation_seconds/3600:.2f}小时。
九次日更新合计843144请求，每日完整一轮约19–21分钟。
各训练阶段合计约{train_seconds/3600:.2f}小时；含小模型、验证、两组评测和追加准备的连续预算计时为
{new['global_elapsed_seconds']/3600:.2f}小时，低于10小时。以上为四张A40并行的墙钟时间，不是GPU小时数。
阶段计时的细节与全部SHA见[模型清单](reports/model_chain_inventory.json)。

参数规模主要来自embedding查表，候选共享一次历史编码；不能把这个模型按同参数量的稠密语言模型估算。
这次使用受控训练量，以原生验证选择checkpoint；未证明已经收敛或全量数据只需25分钟。

## 所有更新边

下表差值均为 **Rebuild−Reuse 的 observed-positive NDCG@10**，正数表示复用受损。
三个用户分组独立列出，不合并算显著性。主指标始终NDCG@10，Recall@10及@5/@20保留在逐窗文件中。

{chr(10).join(table)}

原未见用户确认只有M0→M1和M4→M5两条名义区间排零；原开发三条。
额外D37的M8→M9差值−0.00069964、区间跨零，完整保留在[全部28窗CSV](reports/version_chain_all_populations.csv)。
追加组中，复用下降的名义区间排零版本为{positive}；反向版本为{negative}；区间跨零版本为{ambiguous}。
第二条时序边区间下界很接近零，第四条反向边上界也很接近零，不能夸大。

## 追加时序组的含义

在原结果之后，按用户允许扩人口的授权，在D26原5197训练居民中排除384 fit与128 validation缓存校准UID，
再以固定UID哈希选2048人。同一名单用于全部九窗，370105完整请求，无曝光/无正例请求也保留计数。
每个目标checkpoint的实际训练截止日均严格早于对应评测日；人口不按未来活跃交集、标签或分数选择。
名单中2031人已出现于foundation训练目标；后续窗口少数活跃UID尚未被父版监督，均按固定名单保留。
因此这是**训练分组居民的未来请求时序留出，是原研究之后的追加分析**，不能改称独立未见用户确认。
原确认结果不回写。区间是名义配对UID95% bootstrap，不是多重检验后的整体显著性。

## 工程核对与后续使用

原链60项CPU检查、新人口/时间边界10项检查通过；真实Max四卡训练/完整AdamW恢复、最后不完整batch检查通过。
新评测先对原64真实请求逐位对照：Rebuild、parent native、Reuse三路分数最大差全为0，逐请求指标一致。
全部28窗的{sum(row['requests'] for row in rows):,}个请求、{total_candidates:,}个候选实例已与原始候选ID/标签逐项核对并独立重算指标。
九个追加窗口的当前原生模型优于固定基线的数量为{sum(row['native_beats_baselines'] for row in new['windows'])}/9。
正反结果、所有checkpoint、完整训练状态、原始分数和代码绑定均保留。

当前cache比较是**匹配同一因果前缀的版本快照**。持续在线写入/FIFO和论文具体校准方法仍属于后续实验，
本轮结果不证明长历史优于短历史，也不证明论文方法已有效。Rebuild不是质量的理论上界。

{chr(10).join(model_table)}

M0另继承D12–23的预热；各后续模型继承实际父版的完整AdamW状态。模型清单的日期是当前训练阶段日期。
证据入口：[原链审计](runs/version_chain_verification/attempt_1/completion_audit.json)、
[追加组审计](runs/temporal_cohort/attempt_1/completion_audit.json)、[最终交付核对](reports/final_training_handoff.json)。
"""
    result_path.write_text(content)
    sources[str(csv_path)], sources[str(result_path)] = sha256(csv_path), sha256(result_path)
    json_save(output, {"status": "completed", "goal_satisfied": useful >= 3, "models": 10, "windows": 28,
              "history": 1024, "parameters": inventory["total_parameters"], "training_phase_seconds": train_seconds,
              "foundation_phase_seconds": foundation_seconds, "continuous_budget_seconds": new["global_elapsed_seconds"],
              "within_total_budget": new["within_total_budget"], "original_confirmation_clear_edges": 2,
              "temporal_positive_versions": positive, "temporal_negative_versions": negative,
              "temporal_inconclusive_versions": ambiguous, "useful_temporal_edges": useful,
              "requests": sum(row["requests"] for row in rows), "candidate_instances": total_candidates,
              "population_and_inference_scope": plan["independence"], "cache_scope": plan["scope"],
              "statistical_scope": plan["intervals"], "source_sha256": sources,
              "reporter_sha256": sha256(__file__)})
    print(json.dumps({"status": "completed", "goal_satisfied": useful >= 3, "useful_temporal_edges": useful,
                      "positive_versions": positive, "negative_versions": negative, "inconclusive_versions": ambiguous,
                      "continuous_hours": new["global_elapsed_seconds"]/3600, "report": str(result_path)}))


if __name__ == "__main__":
    main()
