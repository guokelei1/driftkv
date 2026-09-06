#!/usr/bin/env python3
"""Summarize retained Design runs without executing models or choosing new users."""

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2] / "results/design"


def main(run_ids):
    rows, costs = [], []
    correction_path = ROOT / "analysis/v12_variance_storage_correction.json"
    corrections = {row["run"]:row["corrected_source_view_bytes"]
                   for row in json.loads(correction_path.read_text())["rows"]} if correction_path.exists() else {}
    for run_id in run_ids:
        path = ROOT / run_id
        summary = json.loads((path / "summary.json").read_text())
        config = json.loads((path / "configuration.json").read_text())
        if summary["status"] != "development_complete":
            raise ValueError(f"{run_id}: incomplete; inspect its retained failure or live process")
        diagnostics = not config.get("quality_only", False)
        raw = pd.read_csv(path / "cutover_metrics.csv") if diagnostics else pd.DataFrame(columns=["method", "target"])
        for target, group in raw[raw.method == "learned"].groupby("target", sort=True):
            rows.append(dict(run=run_id, target=target, users=len(group),
                mean_user_probability_recovery=group.probability_gap_recovery.mean(),
                median_user_probability_recovery=group.probability_gap_recovery.median(),
                fraction_users_harmed=(group.probability_gap_recovery < 0).mean(),
                mean_reuse_probability_gap=group.reuse_probability_gap.mean(),
                mean_method_probability_gap=group.observed_probability_gap.mean()))
        ledger = summary["ledger_seconds"]
        reference = config.get("calibration_reference")
        calibration_run = reference["run"] if reference else run_id
        calibration_ledger = (json.loads((ROOT/calibration_run/"summary.json").read_text())["ledger_seconds"]
                              if reference else ledger)
        measured_builds = config["targets"] * len(config["development_uids"])
        counts = summary["state_counts"].values()
        total_appends = sum(value["appends"] for value in counts)
        total_segments = sum(value["translated_segments"] for value in summary["state_counts"].values())
        method_service = sum(ledger.get(key, 0) for key in (
            "service_learned_append", "service_learned_read", "chain_release_translation"))
        exact_service = sum(ledger.get(key, 0) for key in (
            "service_exact_append", "service_exact_read", "chain_exact_release"))
        # Early artifacts explicitly recorded writer bytes but omitted the
        # materialized source read-view. Reconstruct its known 16x2 snapshot size
        # here and flag the correction, preserving the original raw summaries.
        source_view = summary.get("last_user_source_view_bytes", 295680)
        source_view = corrections.get(run_id, source_view)
        snapshot = [summary.get("last_user_source_storage_bytes"), source_view,
                    summary.get("last_user_translated_storage_bytes")]
        costs.append(dict(run=run_id, elapsed_seconds=summary["elapsed_seconds"],
            peak_allocated_mib=summary["peak_allocated_mib"],
            inference_compilation_and_warmup_seconds=ledger.get("inference_compilation_and_warmup",0),
            calibration_reference=calibration_run,
            teacher_and_fit_seconds=sum(calibration_ledger.get(key, 0) for key in (
                "teacher_build", "teacher_replay", "teacher_query", "translator_fit")),
            release_selection_seconds=calibration_ledger.get("release_selection",0),
            calibration_query_selection_seconds=calibration_ledger.get("calibration_query_selection",0),
            calibration_lineage_seconds=calibration_ledger.get("calibration_lineage_replay", 0)+calibration_ledger.get("calibration_lifetime_replay",0),
            source_calibration_build_seconds=sum(calibration_ledger.get(key,0) for key in
                ("calibration_source_backfill","calibration_adjacent_source","calibration_lifetime_source")),
            per_user_initial_backfill_ms=1000*ledger["summary_backfill"]/measured_builds if diagnostics else None,
            per_user_release_translate_ms=1000*ledger["release_translation"]/measured_builds if diagnostics else None,
            per_user_exact_build_ms=1000*ledger["diagnostic_exact_build"]/measured_builds if diagnostics else None,
            real_appends=total_appends, translated_segments=total_segments,
            measured_method_service_seconds=method_service,
            measured_exact_service_seconds=exact_service,
            service_ratio_excluding_global_preparation=method_service/exact_service if exact_service else None,
            last_snapshot_auxiliary_bytes=sum(snapshot) if all(value is not None for value in snapshot) else None,
            source_view_bytes_reconstructed=diagnostics and "last_user_source_view_bytes" not in summary,
            second_moment_view_bytes_corrected=run_id in corrections))
    out = ROOT / "analysis"
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out/"development_comparison.csv", index=False)
    pd.DataFrame(costs).to_csv(out/"development_costs.csv", index=False)
    text = ["# 六层 Design 开发结果", "", "全部为seed17、未拟合的固定开发用户；未读取独立确认。",
            "概率机制恢复采用用户等权、未裁剪比例，不能作为AUC恢复证据。", "",
            "| run | 目标 | 用户 | 平均用户恢复 | 受损用户比例 | Reuse绝对gap | 方法绝对gap |",
            "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        text.append(f"| {row['run']} | M{row['target']} | {row['users']} | "
                    f"{row['mean_user_probability_recovery']:.4f} | {row['fraction_users_harmed']:.3f} | "
                    f"{row['mean_reuse_probability_gap']:.6f} | {row['mean_method_probability_gap']:.6f} |")
    text += ["", "成本CSV分开报告教师/拟合、校准谱系生成、初次摘要backfill和持续服务。",
             "服务比率不包含全局发布准备，因此不是完整方法总成本比率。",
             "早期summary缺少materialized source view的独立字段，本汇总补计已知16段×2slot的295680 bytes；",
             "原summary和raw保留，不静默改写；所有内存量仍不含Python容器开销。",
             "方案12最初两项静态source view漏计的平方矩，按v12_variance_storage_correction.json补计并在CSV标记。",
             "微型开发人口与预先固定的30000部署人口有区别，此处未外推或声称达到20%预算。", ""]
    (out/"report.md").write_text("\n".join(text))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_ids", nargs="+")
    main(parser.parse_args().run_ids)
