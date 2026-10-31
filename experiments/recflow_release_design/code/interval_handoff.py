"""CPU-only final case inventory from completed, already-audited experiments.

No experiments are dispatched and no metrics are recomputed from raw scores.
The registry remains authoritative; this command writes only FINAL_RESULTS.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path

from .interval_protocol import ancestry
from .interval_register import (
    DEFAULT_REGISTRY, TIMEZONE, canonical, csv_rows, load_completed, read,
    same_number, validate_pair,
)


def csv_read(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def require_file(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Required evidence or model file is missing: {path}")
    return path


def primary_window(group):
    windows = group["windows"]
    if "primary_window_id" not in group:
        if len(windows) != 1:
            raise ValueError("Multi-window group needs an explicit primary window")
        return windows[0]
    matches = [window for window in windows if window["window_id"] == group["primary_window_id"]]
    if len(matches) != 1:
        raise ValueError("Missing or ambiguous primary window")
    return matches[0]


def check_run_terminal(run):
    completion = run / "completion.json"
    if not completion.is_file() or read(completion).get("status") != "completed":
        raise ValueError(f"Unfinished experiment remains; no final handoff yet: {run}")
    process_path = run / "active_process.json"
    if process_path.exists():
        process = read(process_path)
        if process.get("status") == "running":
            # A stale activity file alone is not a live process. Inspect its
            # actual command before reporting that final analysis still runs.
            command = Path(f"/proc/{int(process['pid'])}/cmdline")
            try:
                live_command = command.read_bytes().replace(b"\x00", b" ").decode(errors="replace")
            except FileNotFoundError:
                live_command = ""
            if str(run.resolve()) in live_command and "interval_" in live_command:
                raise ValueError(f"A scoring/analysis process is still live: {run}")


def check_group(group, completed):
    window = primary_window(group)
    target, source = group["target_checkpoint"], group["cache_source_checkpoint"]
    lineage = ancestry(target, source)
    if canonical(group["actual_immediate_training_parent_checkpoint"]) != canonical(lineage[-1]["actual_parent"]):
        raise ValueError("Group confuses published cache source with immediate training parent")
    if window["interval"]["days"] != [lineage[-1]["train_end_day"] + 1]:
        raise ValueError("Registered primary evaluation is not the complete next day")
    audit = read(require_file(window["independent_candidate_and_metric_audit_path"]))
    summary = read(require_file(window["summary_path"]))
    if audit.get("status") != "passed" or summary.get("status") not in ("passed", "completed"):
        raise ValueError("Registered group does not have completed audited metrics")
    for name, key in (("request_count", "requests"), ("positive_request_count", "positive_requests")):
        if window[name] != audit[key] or window[name] != summary[key]:
            raise ValueError("Registry and completed audit disagree on request counts")
    for metric, suffix in (("ndcg@10", "ndcg10"), ("recall@10", "recall10")):
        if (not same_number(window[f"rebuild_{suffix}"], summary["metrics"]["current"][metric])
                or not same_number(window[f"reuse_{suffix}"], summary["metrics"]["reuse"][metric])):
            raise ValueError("Registered primary metric differs from its recorded experiment")
    if group.get("development_evidence"):
        replication = completed.get(canonical(window["evaluation_directory"]))
        if replication is None:
            raise ValueError("Registered replication lacks a completed interval run")
        for pilot_record in group["development_evidence"]:
            pilot = completed.get(canonical(pilot_record["evaluation_directory"]))
            if pilot is None:
                raise ValueError("Registered pilot lacks a completed interval run")
            validate_pair(pilot, replication)
        if window.get("evidence_role") != "primary_disjoint_user_replication":
            raise ValueError("New case must report replication-alone primary metrics")
    else:
        # Full-population baseline audit covers complete raw request membership.
        if audit.get("days") != window["interval"]["days"] or summary.get("first_only"):
            raise ValueError("Original baseline is not its recorded full-day window")
        if summary.get("history_limit") != 1024 or summary.get("role") != 2:
            raise ValueError("Original baseline task differs from the current comparison")
    return window


def extra_epoch_evidence(registry):
    metrics_path = require_file(registry["failed_extraepoch_evidence"]["metrics_csv"])
    root = metrics_path.parent
    rows = csv_read(metrics_path)
    expected = {"M3_e2", "M5_e2", "M6_e2", "M7_e2"}
    if {row["branch"] for row in rows} != expected or len(rows) != 4:
        raise ValueError("The four unsuccessful extra-epoch outcomes must all remain recorded")
    completion = read(require_file(root / "completion.json"))
    if completion.get("status") != "completed" or completion.get("completed_branches") != 4:
        raise ValueError("Extra-epoch experiment completion is missing")
    cleanup_path = require_file(root / "extra_epoch_checkpoint_cleanup.json")
    cleanup = read(cleanup_path)
    if {entry["branch"] for entry in cleanup["deleted_checkpoints"]} != expected:
        raise ValueError("Cleanup record does not identify all four extra-epoch checkpoints")
    preserved = []
    for row in rows:
        branch = root / "branches" / row["branch"]
        for relative in ("spec.json", "training/plan.json", "training/completion.json",
                         "evaluation/summary.json", "evaluation/independent_audit.json",
                         "evaluation/original_vs_extra_epoch.json"):
            require_file(branch / relative)
        audit = read(branch / "evaluation/independent_audit.json")
        if audit.get("status") != "passed":
            raise ValueError("An extra-epoch independent metric audit did not pass")
        for rank in range(4):
            require_file(branch / "evaluation" / f"rank_{rank}.csv")
            if next((branch / "evaluation" / f"rank_{rank}_scores").glob("batch_*.npz"), None) is None:
                raise ValueError("An extra-epoch raw-score shard directory is empty")
        logs = list((root / "logs").glob(f"*_{row['branch']}_train.log"))
        if not logs:
            raise ValueError("An extra-epoch training log is missing")
        preserved.append({"branch": row["branch"], "configuration_scores_logs_present": True,
                          "metrics": row, "evidence_directory": str(branch)})
    return {"metrics_csv": str(metrics_path), "completion": completion,
            "cleanup_record": str(cleanup_path), "cleanup": cleanup, "branches": preserved,
            "actual_deleted_checkpoint_presence": [
                {"path": item["path"], "currently_exists": Path(item["path"]).exists()}
                for item in cleanup["deleted_checkpoints"]]}


def build_report(directory, minimum_positive_groups=6):
    directory = Path(directory).resolve()
    registry = read(require_file(directory / "model_groups.json"))
    recorded_csv = csv_read(require_file(directory / "model_groups.csv"))
    expected_csv = [{key: str(value) for key, value in row.items()} for row in csv_rows(registry)]
    if recorded_csv != expected_csv:
        raise ValueError("Model-group CSV does not render the current JSON registry")
    groups = registry["groups"]
    ids = [group["group_id"] for group in groups]
    pairs = [(canonical(group["cache_source_checkpoint"]), canonical(group["target_checkpoint"])) for group in groups]
    if len(set(ids)) != len(ids) or len(set(pairs)) != len(pairs):
        raise ValueError("Group IDs or ordered model pairs are duplicated")
    if registry["distinct_model_group_count"] != len(groups):
        raise ValueError("Registered distinct-model count differs from actual entries")
    for model in registry["model_catalog"].values():
        require_file(model["checkpoint"])
        require_file(model["checkpoint_metadata"])
    run_paths = sorted(path for path in (directory / "runs").iterdir() if path.is_dir())
    for run in run_paths:
        check_run_terminal(run)
    completed, journal = {}, []
    for run in run_paths:
        data = load_completed(run)
        decision = read(require_file(run / "decision.json"))
        completed[canonical(run)] = data
        overall = data["analysis"]["overall"]
        journal.append({"run": str(run), "pair": f"M{data['plan']['source_version']}->M{data['plan']['target_version']}",
                        "question": data["plan"]["question"], "created_at": data["plan"]["created_at"],
                        "completed_at": data["completion"]["completed_at"],
                        "chosen_users": len(data["chosen"]), "overall": overall,
                        "elapsed": data["analysis"]["elapsed"], "decision": decision,
                        "candidate_audit": str(run / "audit.json"),
                        "all_chosen_users_complete_day_requests": True})
    journal.sort(key=lambda item: item["created_at"])
    if not any(Path(item["run"]).name == "m3_m5_pilot_4096" for item in journal):
        raise ValueError("The unsuccessful M3-to-M5 pilot must remain in the complete journal")
    original_path = require_file(registry["all_original_evidence"]["all_nine_adjacent_pairs_metrics_csv"])
    original = [row for row in csv_read(original_path) if int(row["version"]) > 0]
    if {int(row["version"]) for row in original} != set(range(1, 10)) or len(original) != 9:
        raise ValueError("The original nine adjacent-edge outcomes must all remain recorded")
    primary = {group["group_id"]: check_group(group, completed) for group in groups}
    positives = [group for group in groups if group.get("counts_as_usable_positive_group", group["label"] == "retained_original_positive_baseline")]
    if len(positives) < minimum_positive_groups:
        raise ValueError(f"Only {len(positives)} usable positive groups; requested final minimum is {minimum_positive_groups}")
    if any(primary[group["group_id"]]["ndcg10_rebuild_minus_reuse"] <= 0 for group in positives):
        raise ValueError("A counted positive case has nonpositive primary NDCG difference")
    losses = [primary[group["group_id"]]["ndcg10_relative_loss_pct"] for group in positives]
    limits = [
        "These are selected mechanism cases, not an unbiased estimate of how often cache reuse loses quality.",
        "Original 30,000-user results were already observed under immediate-parent reuse. New replication users are disjoint from their pilots, not globally unseen users.",
        "Groups share checkpoints, users or dates. Disjoint-user replication is not independent retraining, another random seed or an independent calendar window.",
        "New-group primary metrics use replication alone; pilot results are reported separately and are not pooled or selected by maximum effect.",
        "Nominal user-cluster confidence intervals are informational; no universal CI threshold or multiplicity-adjusted population claim is made.",
        "Means over the selected positive pairs are descriptive only. Preferred 1–3% effects were not enforced and larger observed values were retained.",
        "All comparisons preserve full logged candidate sets, shared requests and causal histories up to 1024; this is matched-prefix snapshot replay, not persistent-cache/FIFO serving.",
        "Zero-positive requests are counted but omitted from undefined NDCG/Recall means; unexposed items have unknown relevance."]
    return {"generated_at": datetime.now(TIMEZONE).isoformat(), "status": "final_artifacts_verified",
            "goal_status_changed": False, "registry_json": str(directory / "model_groups.json"),
            "registry_csv": str(directory / "model_groups.csv"), "distinct_model_group_count": len(groups),
            "usable_positive_model_group_count": len(positives), "original_models_present": len(registry["model_catalog"]),
            "model_catalog": registry["model_catalog"], "groups": groups,
            "selected_positive_pairs_descriptive_relative_loss_mean_pct": sum(losses)/len(losses),
            "selected_positive_pairs_relative_loss_min_max_pct": [min(losses), max(losses)],
            "completed_interval_experiments": len(journal), "pending_interval_experiments": 0,
            "interval_total_requests_scored_with_repeated_windows_counted": sum(item["overall"]["requests"] for item in journal),
            "interval_elapsed_seconds_sum": sum(item["elapsed"]["scoring_plus_analysis_wall_seconds"] for item in journal),
            "interval_elapsed_note": "Sum of measured scoring-plus-analysis elapsed times; excludes interactive preparation and original training.",
            "experiment_journal": journal, "all_original_nine_edge_metrics": original,
            "all_original_nine_edge_metrics_path": str(original_path),
            "extra_epoch_negative_evidence": extra_epoch_evidence(registry),
            "limitations": limits, "new_training_in_interval_search": False,
            "automatic_successors": False, "verification": "Existing completed audits, semantic model paths, full-window records, chosen-UID disjointness and registry numerical consistency; raw numerical scoring was not repeated."}


def render_text(report):
    lines = ["RecFlow 模型配对最终交付", f"生成时间：{report['generated_at']}",
             f"可用正向模型组：{report['usable_positive_model_group_count']}；唯一模型组：{report['distinct_model_group_count']}。",
             "主指标 NDCG@10，辅助 Recall@10；相对损失=(Rebuild−Reuse)/Rebuild×100%。",
             "新配对只以复核样本报告主指标；pilot 单独保留。相同模型对的多个窗口不重复计数。", ""]
    for group in report["groups"]:
        w = primary_window(group)
        lines += [group["group_id"] + "  [" + group["label"] + "]",
                  f"  缓存来源：{group['cache_source_checkpoint']}", f"  目标模型：{group['target_checkpoint']}",
                  f"  实际直接训练父：{group['actual_immediate_training_parent_checkpoint']}",
                  f"  训练谱系：{' -> '.join(group['training_lineage_model_ids'])}",
                  f"  发布间隔：{group['publication_interval_days']} 天；增量训练日期：{','.join(group['new_training_between_releases']['dates'])}",
                  f"  评测窗口：[{w['interval']['start_inclusive']}, {w['interval']['end_exclusive']})",
                  f"  NDCG@10 Rebuild={w['rebuild_ndcg10']:.9f}, Reuse={w['reuse_ndcg10']:.9f}, 相对损失={w['ndcg10_relative_loss_pct']:.4f}%",
                  f"  Rebuild−Reuse={w['ndcg10_rebuild_minus_reuse']:.9f}; nominal UID95%={w['ndcg10_paired_user_ci95']}",
                  f"  Recall@10 Rebuild={w['rebuild_recall10']:.9f}, Reuse={w['reuse_recall10']:.9f}",
                  f"  请求={w['request_count']}, 活跃UID={w['active_user_count']}, 正反馈请求={w['positive_request_count']}, 正反馈UID={w['positive_user_count']}",
                  f"  证据：{w['evaluation_directory']}", f"  可靠性：{group['reliability']}", ""]
    lines += ["本轮逐步探索记录（包括负向结果）"]
    for item in report["experiment_journal"]:
        gap = item["overall"]["comparisons"]["ndcg@10"]["ancestor_cache_loss"]
        lines += [f"  {Path(item['run']).name}: {item['pair']}, 所选UID={item['chosen_users']}, 请求={item['overall']['requests']}, 损失={gap['relative_loss_pct']:.4f}%, 耗时={item['elapsed']['scoring_plus_analysis_wall_seconds']/60:.2f}分钟",
                  f"    问题：{item['question']}", f"    解释：{item['decision'].get('interpretation', '')}",
                  f"    后续决定：{item['decision'].get('next_action', '')}"]
    lines += ["", f"本轮评测与分析耗时之和：{report['interval_elapsed_seconds_sum']/60:.2f}分钟；没有新增训练。",
              f"选中正向配对相对损失均值：{report['selected_positive_pairs_descriptive_relative_loss_mean_pct']:.4f}%（仅描述所选案例）。",
              f"原始全部9条边：{report['all_original_nine_edge_metrics_path']}", "四个额外epoch反例："]
    extra = report["extra_epoch_negative_evidence"]
    for item in extra["branches"]:
        lines.append(f"  {item['branch']}: 相对损失={float(item['metrics']['relative_loss_pct']):.4f}%；配置、训练日志、分数及核验保留。")
    lines += [f"  checkpoint清理记录：{extra['cleanup_record']}",
              f"  记录释放空间：{extra['cleanup']['released_gib']:.2f} GiB；原始10模型仍在。", "", "解释边界："]
    lines += ["  - " + item for item in report["limitations"]]
    lines += ["", "本文件不修改任务目标状态，不启动任何后继实验。"]
    return "\n".join(lines)+"\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_REGISTRY.parent)
    parser.add_argument("--output", type=Path, help="Optional output directory; default is release_interval")
    parser.add_argument("--minimum-positive-groups", type=int, default=6)
    args = parser.parse_args()
    if args.minimum_positive_groups < 1:
        raise ValueError("Positive final-case minimum is required")
    report = build_report(args.directory, args.minimum_positive_groups)
    output = args.output or args.directory
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (("FINAL_RESULTS.json", json.dumps(report, ensure_ascii=False, indent=2)+"\n"),
                        ("FINAL_RESULTS.txt", render_text(report))):
        path = output / name
        temporary = path.with_name(name+".tmp")
        temporary.write_text(value)
        os.replace(temporary, path)
    print(json.dumps({"output": str(output.resolve()), "usable_positive_groups": report["usable_positive_model_group_count"],
                      "completed_experiments": report["completed_interval_experiments"],
                      "pending_experiments": 0, "goal_status_changed": False}))


if __name__ == "__main__":
    main()
