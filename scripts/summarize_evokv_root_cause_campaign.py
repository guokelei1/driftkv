from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SOURCES = {
    "qk_sanity": (
        "results/root_cause_campaign/evokv_root_cause_20260807_v0/round_03_0_qk_sanity/result.json",
        "0c7a12f4fab2d02dca388409af508e095652f0c226e435109fc962a609ccf0c3",
    ),
    "qk_attribution": (
        "results/root_cause_campaign/evokv_root_cause_20260807_v0/round_03_a_qk_attribution/result.json",
        "e243ee5b227863b005f0561fd66f002abc7cc2579d7ad4dfcd87773cee424a80",
    ),
    "kuairand_natural_day": (
        "results/root_cause_campaign/evokv_root_cause_20260807_v0/round_03_b_kuairand_natural_day/evaluation.json",
        "3a701cb2c5a263c9e956744263c15ad7ac67630eafe5cdc9f89fcea13b9ca732",
    ),
    "kuairand_path_attribution": (
        "results/root_cause_campaign/evokv_root_cause_20260807_v0/round_03_c1_kuairand_path_attribution/result.json",
        "d4b88441500c1b5bae6b092cdc2b30fe89d862e52ed3da1d8247b4d20bacfc8e",
    ),
    "kuairand_output_only": (
        "results/root_cause_campaign/evokv_root_cause_20260807_v0/round_03_c2_kuairand_cache_compatible/evaluation.json",
        "089be98f5036dcca0f12a823e88bfdb45135d80cb4a3ce8dc6c17f11bc12ab60",
    ),
    "kuairand_kv_invariant": (
        "results/root_cause_campaign/evokv_root_cause_20260807_v0/round_03_c3_kuairand_kv_invariant/evaluation.json",
        "dc09d0f7712d69eb61009aad66d09164ad79e950ba68c7f3c014dd01e3de1f5e",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    temporary.replace(path)


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _load_sources() -> tuple[dict[str, dict[str, object]], dict[str, dict[str, str]]]:
    documents = {}
    bindings = {}
    for name, (raw_path, expected) in SOURCES.items():
        path = Path(raw_path)
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"root-cause source differs: {name}")
        documents[name] = json.loads(path.read_text())
        bindings[name] = {"path": raw_path, "sha256": actual}
    return documents, bindings


def _qk_report(document: dict[str, object], source: dict[str, str]) -> dict[str, object]:
    aggregate = document["aggregate"]
    comparisons = aggregate["fresh_theta2_comparisons"]
    cache = aggregate["cache_relative_error_from_fresh_theta2"]
    methods = [
        "no_prefix",
        "zero_prefix",
        "wrong_user_fresh",
        "recent_4",
        "recent_16",
        "recent_64",
        "stale_theta1",
        "direct_theta0",
        "recursive_theta0_theta1",
    ]
    return {
        "protocol": "evokv_root_cause_qk_report_v0",
        "status": "complete_development_report",
        "scientific_result": False,
        "formal_result": False,
        "source": source,
        "implementation_passed": aggregate["sanity"]["implementation_passed"],
        "records": aggregate["records"],
        "positive_targets": aggregate["positive_targets"],
        "fresh_update_value": aggregate["theta1_to_theta2_fresh_update_value"],
        "fresh_theta2_comparisons": {
            method: {
                metric: comparisons[method][metric]
                for metric in ("cross_entropy", "ndcg_at_10", "mrr", "hit_rate_at_200")
            }
            for method in methods
        },
        "cache_relative_error_mean": {
            method: cache[method].get("mean")
            for method in (
                "stale_theta1",
                "direct_theta0",
                "recursive_theta0_theta1",
                "embedding_table_update_only",
                "projection_update_only",
                "kv_projection_update_only",
                "non_kv_dense_update_only",
            )
        },
        "parameter_path_recovery": aggregate["parameter_path_recovery"],
        "interpretation": [
            "The QK model strongly uses long user history.",
            "The theta1-to-theta2 Fresh update has positive task value.",
            "Adjacent stale K/V has only a small CE tax and no stable ranking tax.",
            "Age-2 direct and recursive caches increase CE tax, but ranking intervals remain weak.",
            "Dense/core parameter changes dominate cache representation drift; embedding/scorer changes are largely cache-compatible in task quality.",
        ],
    }


def _kuairand_report(
    natural: dict[str, object],
    attribution: dict[str, object],
    output_only: dict[str, object],
    invariant: dict[str, object],
    sources: dict[str, dict[str, str]],
) -> dict[str, object]:
    natural_edges = []
    attribution_edges = []
    for edge, path_edge in zip(natural["edges"], attribution["edges"], strict=True):
        natural_edges.append(
            {
                "edge": edge["edge"],
                "positive_targets": edge["positive_targets"],
                "fresh_update_value": edge["previous_to_current_fresh_update_value"],
                "full_horizon_stale_tax": edge["fresh_current_comparisons"]["stale_previous"],
                "full_horizon_prior_prefix_value": edge["fresh_current_comparisons"]["no_prefix"],
            }
        )
        full = path_edge["full_horizon"]
        first = path_edge["stratified"]["positive_ordinal"]["0_1"]
        attribution_edges.append(
            {
                "edge": edge["edge"],
                "first_positive_stale_tax": first["comparisons"]["stale_tax"],
                "first_positive_prior_prefix_value": first["comparisons"]["prior_prefix_value"],
                "cross_entropy_fraction_of_full_update_percent": {
                    method: full["parameter_routes"][method]["cross_entropy"][
                        "fraction_of_full_update_percent"
                    ]
                    for method in (
                        "previous_hidden_current_scorer",
                        "embedding_update_only",
                        "current_hidden_previous_scorer",
                        "dense_update_only",
                    )
                },
            }
        )
    return {
        "protocol": "evokv_root_cause_kuairand_report_v0",
        "status": "complete_development_report",
        "scientific_result": False,
        "formal_result": False,
        "sources": sources,
        "natural_day_edges": natural_edges,
        "temporal_and_parameter_attribution": attribution_edges,
        "output_only_revision": {
            "decision": output_only["decision"],
            "reuse_exact_implementation_passed": output_only["sanity"][
                "implementation_passed"
            ],
            "edge_update_retention": [
                value["update_retention"] for value in output_only["edges"]
            ],
        },
        "kv_invariant_revision": {
            "model_revision": invariant["model_revision"],
            "decision": invariant["decision"],
            "reuse_exact_implementation_passed": invariant["sanity"][
                "implementation_passed"
            ],
            "edge_sanity": [
                {"edge": value["edge"], **value["sanity"]}
                for value in invariant["edges"]
            ],
            "edge_update_retention": [
                value["update_retention"] for value in invariant["edges"]
            ],
        },
        "interpretation": [
            "Both natural-day updates have strong Fresh task value on held-out next days.",
            "Adjacent stale CE tax is positive on edge 1 but not full-horizon stable on edge 2.",
            "The stale CE tax is concentrated in the first post-publication predictions and decays as current-day suffix tokens accumulate.",
            "A separately trained output-only update is real but retains only 22.01% of pooled full-update CE gain.",
            "An untied output head plus the native last-layer Q/gate/output and final norm forms an exactly K/V-invariant subspace and retains 52.23% of pooled full-update CE gain.",
        ],
    }


def _decision(
    bindings: dict[str, dict[str, str]],
    qk: dict[str, object],
    kuairand: dict[str, object],
) -> dict[str, object]:
    invariant_sanity = kuairand["kv_invariant_revision"]["edge_sanity"]
    return {
        "protocol": "evokv_root_cause_cross_dataset_decision_v0",
        "status": "complete_development_decision",
        "scientific_result": False,
        "formal_result": False,
        "campaign": "evokv_root_cause_20260807_v0",
        "sources": bindings,
        "causal_gates": [
            {
                "gate": "L0",
                "question": "Inference, cache lineage, mask, and scoring implementation",
                "qk": "pass",
                "kuairand": "pass",
                "cross_dataset": "pass",
            },
            {
                "gate": "L1",
                "question": "Fresh recommendation model has task value",
                "qk": "pass_development",
                "kuairand": "pass_development",
                "cross_dataset": "pass_development",
            },
            {
                "gate": "L2",
                "question": "Recommendation depends on resident history",
                "qk": "strong_pass",
                "kuairand": "short_horizon_only",
                "cross_dataset": "mixed_by_window_semantics",
            },
            {
                "gate": "L3",
                "question": "New-window model update helps the next held-out window",
                "qk": "pass_one_edge",
                "kuairand": "pass_two_edges",
                "cross_dataset": "pass_development",
            },
            {
                "gate": "L4",
                "question": "Update benefit requires cache-producing parameters",
                "qk": "weak_or_mixed",
                "kuairand": "partial_only",
                "cross_dataset": "cache_safe_subspace_is_material",
            },
            {
                "gate": "L5",
                "question": "Old K/V causes stable broad task-quality loss",
                "qk": "adjacent_fail_age2_ce_only",
                "kuairand": "early_ce_pass_broad_ranking_fail",
                "cross_dataset": "original_d1_opportunity_gate_not_met",
            },
            {
                "gate": "L6",
                "question": "Exact cache maintenance is a measured dominant bottleneck",
                "qk": "not_reopened",
                "kuairand": "not_measured_in_this_revision",
                "cross_dataset": "pending_only_after_model_revision_confirmation",
            },
            {
                "gate": "L7",
                "question": "Select an approximate K/V maintenance mechanism",
                "qk": "blocked",
                "kuairand": "blocked",
                "cross_dataset": "do_not_resume_d1_d2_d3",
            },
        ],
        "primary_direction": {
            "name": "KV-invariant streaming updates with periodic full refresh",
            "development_candidate": "untied all-context output head plus last HSTU layer Q projection, gate projection, output projection, and final norm",
            "measured_properties": {
                "reuse_exact_cache_maximum_absolute_error": max(
                    value["cache_maximum_absolute_error"] for value in invariant_sanity
                ),
                "reuse_exact_hidden_maximum_absolute_error": max(
                    value["hidden_maximum_absolute_error"] for value in invariant_sanity
                ),
                "reuse_exact_nll_maximum_absolute_error": max(
                    value["nll_maximum_absolute_error"] for value in invariant_sanity
                ),
                "reuse_exact_ranks_equal": all(
                    value["ranks_equal"] for value in invariant_sanity
                ),
                "pooled_cross_entropy_full_update_retention_percent": kuairand[
                    "kv_invariant_revision"
                ]["decision"]["pooled_cross_entropy_update_advantage"][
                    "retention_percent"
                ],
                "positive_cross_entropy_ci_by_edge": kuairand[
                    "kv_invariant_revision"
                ]["decision"]["positive_cross_entropy_ci_by_edge"],
                "positive_ranking_ci_by_edge": kuairand[
                    "kv_invariant_revision"
                ]["decision"]["positive_ranking_ci_by_edge"],
            },
            "interpretation": "Most ordinary stream updates should first use parameters that provably do not change resident prefix K/V. Less frequent full-backbone updates are separate cache-breaking events and must perform exact cache renewal or an independently qualified maintenance policy.",
        },
        "stopped_direction": {
            "name": "Broad approximate migration of every adjacent-version K/V cache",
            "reason": "Neither QK nor KuaiRand establishes a stable broad ranking-quality stale tax across two ordinary edges, and the observed risk is small or concentrated in the first post-publication predictions.",
            "actions": [
                "Do not tune training, negatives, or evaluation to manufacture a larger Reuse-Exact gap.",
                "Do not resume affine or low-rank K/V translator search as the headline route.",
                "Do not extend theta3-theta7 solely to seek cumulative failure.",
                "Keep historical D1/D2/D3 artifacts as baselines and infrastructure only.",
            ],
        },
        "d1_decision": {
            "resume": False,
            "reason": "L5, training-seed replication, and measured Exact-cost gates are not jointly satisfied.",
            "future_boundary": "Only a periodic full-refresh boundary may reopen exact-versus-maintained K/V work, under a new model revision and protocol.",
        },
        "next_phase": [
            "Repeat the entire base/full/KV-invariant chain with an independent training seed; update-only stochastic repeats do not count.",
            "Extend the natural-day chain to four to six ordinary updates and predeclare the periodic full-refresh cadence.",
            "Measure the quality-versus-cache-renewal-cost curve for invariant updates between exact full refreshes.",
            "Confirm the native invariant parameter boundary on the large QK model before using QK as the capacity stressor.",
            "Refresh the closest-work screen before making any novelty claim about cache-compatible training.",
        ],
        "limitations": [
            "All campaign outputs are development evidence with one base-model training seed.",
            "The 52.23% gate is pooled CE retention; ranking retention is metric- and edge-dependent and remains weaker.",
            "KuaiRand uses a very large frozen full catalog, so top-k absolute values are low and must not be compared numerically with QK.",
            "No D2/D3 timing, four-rank result, qualification role, or final role was consumed.",
        ],
        "qk_report_source": qk["source"],
    }


def _qk_markdown(report: dict[str, object]) -> str:
    update = report["fresh_update_value"]
    comparisons = report["fresh_theta2_comparisons"]
    rows = []
    for method in (
        "no_prefix",
        "wrong_user_fresh",
        "recent_4",
        "recent_16",
        "recent_64",
        "stale_theta1",
        "direct_theta0",
        "recursive_theta0_theta1",
    ):
        value = comparisons[method]["cross_entropy"]
        rows.append(
            f"| {method} | {value['fresh_theta2_advantage_absolute']:.6f} | "
            f"{value['fresh_theta2_advantage_relative_percent']:.4f}% | "
            f"{value['record_cluster_95_interval']} |"
        )
    return "\n".join(
        [
            "# QK 根因归因报告",
            "",
            "状态：development evidence，非正式论文结果。",
            "",
            f"实现 sanity：`{report['implementation_passed']}`；records={report['records']}，positive targets={report['positive_targets']}。",
            "",
            "## 核心结果",
            "",
            f"theta1→theta2 Fresh CE 改善 `{update['cross_entropy']['theta2_fresh_advantage_absolute']:.6f}`，NDCG@10 相对改善 `{update['ndcg_at_10']['theta2_fresh_advantage_relative_percent']:.3f}%`。",
            "",
            "| 干预 | Fresh CE 优势 | 相对值 | 95% record-cluster CI |",
            "|---|---:|---:|---|",
            *rows,
            "",
            "结论：QK 明显依赖长历史且更新有价值，但相邻 stale 只有很小 CE 差，ranking 不稳定；age-2 CE 有放大但仍不足以恢复 D1。",
            "",
        ]
    )


def _kuairand_markdown(report: dict[str, object]) -> str:
    rows = []
    for edge in report["natural_day_edges"]:
        update = edge["fresh_update_value"]["cross_entropy"]
        stale = edge["full_horizon_stale_tax"]["cross_entropy"]
        rows.append(
            f"| {edge['edge']} | {edge['positive_targets']} | {update['current_fresh_advantage_absolute']:.6f} | "
            f"{stale['fresh_current_advantage_absolute']:.6f} | {stale['user_cluster_95_interval']} |"
        )
    first_rows = []
    for edge in report["temporal_and_parameter_attribution"]:
        stale = edge["first_positive_stale_tax"]["cross_entropy"]
        first_rows.append(
            f"| {edge['edge']} | {stale['current_fresh_advantage_absolute']:.6f} | "
            f"{stale['user_cluster_95_interval']} |"
        )
    output_retention = report["output_only_revision"]["decision"][
        "pooled_cross_entropy_update_advantage"
    ]["retention_percent"]
    invariant_retention = report["kv_invariant_revision"]["decision"][
        "pooled_cross_entropy_update_advantage"
    ]["retention_percent"]
    return "\n".join(
        [
            "# KuaiRand 自然日机会与参数路径报告",
            "",
            "状态：development evidence，非正式论文结果。",
            "",
            "| Edge | Positive targets | Fresh update CE 优势 | 全时域 stale CE tax | 95% user-cluster CI |",
            "|---:|---:|---:|---:|---|",
            *rows,
            "",
            "| Edge | 首个正样本 stale CE tax | 95% user-cluster CI |",
            "|---:|---:|---|",
            *first_rows,
            "",
            f"scorer-only 实际训练保留 pooled full-update CE 收益 `{output_retention:.2f}%`，未通过 25% hybrid gate。",
            f"KV-invariant tail + untied scorer 保留 `{invariant_retention:.2f}%`，通过预声明的 50% primary gate，且 Reuse/Exact cache、hidden、NLL 均逐元素一致。",
            "",
            "结论：自然日更新有强任务价值；stale 风险主要集中在模型发布后的最早请求。更扎实的机会是约束流式更新落在 K/V 不变子空间，而不是为所有相邻版本缓存拟合迁移器。",
            "",
        ]
    )


def _decision_markdown(decision: dict[str, object]) -> str:
    rows = [
        f"| {value['gate']} | {value['qk']} | {value['kuairand']} | {value['cross_dataset']} |"
        for value in decision["causal_gates"]
    ]
    measured = decision["primary_direction"]["measured_properties"]
    return "\n".join(
        [
            "# 03 根因探索：跨数据结论与路线决定",
            "",
            "状态：development decision，非 scientific/formal result。",
            "",
            "| Gate | QK | KuaiRand | 跨数据判断 |",
            "|---|---|---|---|",
            *rows,
            "",
            "## 主要方向",
            "",
            "采用 **KV-invariant streaming updates + periodic full refresh**。当前原生子空间包含 untied output head、最后一层 Q/gate/output projection 和 final norm；所有产生 resident K/V 的参数冻结。",
            "",
            f"两条自然日边 pooled CE 收益保留 `{measured['pooled_cross_entropy_full_update_retention_percent']:.2f}%`；Reuse 与 Exact 的 cache/hidden/NLL 最大误差均为 `0`。",
            f"两条边的逐 target ranks 完全一致：`{measured['reuse_exact_ranks_equal']}`。",
            "",
            "## 明确停止",
            "",
            "停止把“所有相邻版本 K/V 的通用近似迁移”作为当前论文主线；不再通过负样本、窗口、epoch 或指标筛选制造更大的 Reuse–Exact 差值，也不恢复旧 D1/D2/D3 性能扩展。",
            "",
            "## 仍需完成",
            "",
            *[f"- {value}" for value in decision["next_phase"]],
            "",
            "当前结果只有一个 base-model seed，排名收益弱于 CE；因此这是下一阶段候选路线，不是论文结论。",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--campaign-root",
        type=Path,
        default=Path("results/root_cause_campaign/evokv_root_cause_20260807_v0"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    documents, bindings = _load_sources()
    if not all(
        document.get("scientific_result") is False
        and document.get("formal_result") is False
        for document in documents.values()
    ):
        raise ValueError("root-cause source promotion state differs")
    qk = _qk_report(documents["qk_attribution"], bindings["qk_attribution"])
    kuairand = _kuairand_report(
        documents["kuairand_natural_day"],
        documents["kuairand_path_attribution"],
        documents["kuairand_output_only"],
        documents["kuairand_kv_invariant"],
        {
            name: bindings[name]
            for name in (
                "kuairand_natural_day",
                "kuairand_path_attribution",
                "kuairand_output_only",
                "kuairand_kv_invariant",
            )
        },
    )
    decision = _decision(bindings, qk, kuairand)
    qk_root = args.campaign_root / "round_03_a_qk_attribution"
    kuairand_root = args.campaign_root / "round_03_b_kuairand_natural_day"
    decision_root = args.campaign_root / "round_03_c_cross_diagnosis"
    _atomic_json(qk_root / "qk_root_cause_report.json", qk)
    _atomic_text(qk_root / "qk_root_cause_report.md", _qk_markdown(qk))
    _atomic_json(kuairand_root / "kuairand_natural_day_opportunity_report.json", kuairand)
    _atomic_text(
        kuairand_root / "kuairand_natural_day_opportunity_report.md",
        _kuairand_markdown(kuairand),
    )
    _atomic_json(decision_root / "decision.json", decision)
    _atomic_text(decision_root / "root_cause_campaign_report.md", _decision_markdown(decision))
    validation = {
        "status": "valid",
        "scientific_result": False,
        "formal_result": False,
        "program": {"path": str(Path(__file__)), "sha256": _sha256(Path(__file__))},
        "outputs": {
            str(path): _sha256(path)
            for path in (
                qk_root / "qk_root_cause_report.json",
                qk_root / "qk_root_cause_report.md",
                kuairand_root / "kuairand_natural_day_opportunity_report.json",
                kuairand_root / "kuairand_natural_day_opportunity_report.md",
                decision_root / "decision.json",
                decision_root / "root_cause_campaign_report.md",
            )
        },
    }
    _atomic_json(decision_root / "validation.json", validation)
    print(json.dumps(validation, indent=2), flush=True)


if __name__ == "__main__":
    main()
