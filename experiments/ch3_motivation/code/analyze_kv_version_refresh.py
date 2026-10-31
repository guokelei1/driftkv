#!/usr/bin/env python3
"""CPU-only paired analysis of frozen-history KV version-refresh cases.

Each target model supplies its OWN teacher.  Cross-version residual changes
are descriptive: logits from different target models are never one common
ground truth.  A refresh's benefit is compared within the new target only.
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


METRICS = ("score_mse",)
METHODS = {"old_target": ("source_unmapped", "unchanged", "teacher"),
           "new_target": ("source_unmapped", "unchanged", "refresh", "teacher")}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _weights(n, replicates, seed):
    if n < 1 or replicates < 1:
        raise ValueError("paired UID bootstrap requires users and positive replicates")
    rng = np.random.default_rng(seed)
    return np.asarray([np.bincount(rng.integers(n, size=n), minlength=n)
                       for _ in range(replicates)], dtype=np.uint16)


def _interval(values):
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    return dict(ci95=np.percentile(values[valid], [2.5, 97.5]).tolist() if valid.any() else None,
                invalid_bootstrap_draws=int((~valid).sum()))


def _estimate(value, draws, users):
    return dict(mean=float(value) if np.isfinite(value) else None,
                valid_users=int(users), **_interval(draws))


def _mean(array, weights):
    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 1 or len(array) != weights.shape[1] or not np.isfinite(array).all():
        raise ValueError("paired residual array must be finite for every fixed UID")
    return _estimate(array.mean(), weights @ array / weights.sum(1), len(array))


def _ratio(numerator, denominator, weights, *, improvement=False):
    """Ratio of equal-user means, never the mean of unstable per-UID ratios."""
    numerator, denominator = np.asarray(numerator), np.asarray(denominator)
    numerator_draws, denominator_draws = weights @ numerator, weights @ denominator
    with np.errstate(invalid="ignore", divide="ignore"):
        point = float(numerator.mean()) / float(denominator.mean()) if denominator.mean() > 0 else np.nan
        draws = np.where(denominator_draws > 0, numerator_draws / denominator_draws, np.nan)
    if improvement:
        point, draws = 100. * (1. - point), 100. * (1. - draws)
    return _estimate(point, draws, len(numerator))


def _value(user, target, method, metric):
    return user["endpoints"][target][metric][method]


def aggregate_users(users, *, replicates=500, seed=17):
    """Use identical UID multiplicities for all targets, methods and contrasts."""
    uids = [int(row["uid"]) for row in users]
    if len(set(uids)) != len(uids):
        raise ValueError("duplicate UID in fixed-history case")
    weights = _weights(len(uids), replicates, seed)
    endpoints, contrasts = [], []
    for metric in METRICS:
        values = {(target, method): np.asarray([_value(row, target, method, metric)
                  for row in users], dtype=np.float64)
                  for target, methods in METHODS.items() for method in methods}
        if any(not np.isfinite(array).all() or (array < 0).any() for array in values.values()):
            raise ValueError("teacher residuals must be finite and nonnegative")
        for target, methods in METHODS.items():
            if np.any(values[target, "teacher"] != 0):
                raise ValueError("teacher endpoint must have exactly zero self residual")
            baseline = values[target, "source_unmapped"]
            for method in methods:
                array = values[target, method]
                endpoints.append(dict(target=target, method=method, metric=metric,
                    residual=_mean(array, weights),
                    relative_to_source_unmapped=_ratio(array, baseline, weights)))
        old = values["old_target", "unchanged"]
        new = values["new_target", "unchanged"]
        refresh = values["new_target", "refresh"]
        old_baseline = values["old_target", "source_unmapped"]
        new_baseline = values["new_target", "source_unmapped"]
        old_normalized = _ratio(old, old_baseline, weights)
        new_normalized = _ratio(new, new_baseline, weights)
        with np.errstate(invalid="ignore", divide="ignore"):
            old_draws = (weights @ old) / (weights @ old_baseline)
            new_draws = (weights @ new) / (weights @ new_baseline)
        normalized_change = _estimate(
            np.nan if old_normalized["mean"] is None or new_normalized["mean"] is None
            else new_normalized["mean"] - old_normalized["mean"],
            new_draws - old_draws, len(users))
        contrasts.append(dict(metric=metric,
            cross_version_unchanged_residual_change=_mean(new - old, weights),
            cross_version_source_normalized_residual_change=normalized_change,
            new_target_refresh_improvement=_mean(new - refresh, weights),
            new_target_refresh_relative_improvement_percent=_ratio(refresh, new, weights, improvement=True)))
    return dict(unique_users=len(users), paired_uids=uids, endpoints=endpoints, contrasts=contrasts)


def _validate_user(user):
    if len(user["panel_candidates"]) != 16:
        raise ValueError("this diagnostic requires one fixed Q16 candidate panel")
    for target, methods in METHODS.items():
        endpoint = user["endpoints"][target]
        scores = endpoint["scores"]
        teacher = np.asarray(scores["teacher"], dtype=np.float64)
        if teacher.shape != (16,) or not np.isfinite(teacher).all():
            raise ValueError("invalid same-target teacher scores")
        for method in methods:
            actual = np.asarray(scores[method], dtype=np.float64)
            if actual.shape != (16,) or not np.isfinite(actual).all():
                raise ValueError("candidate score panel changed")
            expected = float(np.mean((actual - teacher) ** 2))
            if not np.isclose(endpoint["score_mse"][method], expected, rtol=1e-5, atol=1e-10):
                raise ValueError("stored MSE differs from its OWN target teacher scores")


def load_blocks(input_root):
    """Validate committed block membership and its exact frozen identity."""
    root = Path(input_root).resolve()
    config_path = root / "configuration.json"
    config = json.loads(config_path.read_text())
    if config["format"] != "fixed_history_version_refresh_population_v1":
        raise ValueError("expected the frozen-history version-refresh runner")
    fit = config["refresh_fitting_identity"]
    if (fit["protocol"] != "frozen_mapped_source_version_refresh_v1"
            or fit["source_kind"] != "actual_predecessor_mapped_cache_not_native_old_target"
            or fit["predecessor_checkpoint_sha256"] != config["predecessor_checkpoint_sha256"]
            or fit["history_edge"] != config["history_edge"]
            or fit["target_version"] != config["target_version"]):
        raise ValueError("refresh must be calibrated on the actual predecessor-mapped state")
    identity = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"),
                                        allow_nan=False).encode()).hexdigest()
    records, artifacts = [], []
    for path in sorted((root / "blocks").glob("part_*.json")):
        block = json.loads(path.read_text())
        selected = block["uids"]
        if (block["status"] != "complete" or block["identity_sha256"] != identity
                or block["offset"] != len(records)
                or selected != config["users"][len(records):len(records) + len(selected)]
                or [row["uid"] for row in block["users"]] != selected):
            raise ValueError("committed block identity, UID membership or order changed")
        for user in block["users"]:
            _validate_user(user)
            if (user["endpoints"]["old_target"]["version"] != config["history_edge"]
                    or user["endpoints"]["new_target"]["version"] != config["target_version"]):
                raise ValueError("user target teachers differ from this case's model versions")
        records.extend(block["users"])
        artifacts.append(dict(path=str(path), sha256=_sha(path)))
    if not records:
        raise ValueError("no committed version-refresh users")
    summary_path = root / "summary.json"
    runner = json.loads(summary_path.read_text())
    if (runner["status"] != "complete" or runner["users"] != len(records)
            or runner["diagnostic"] != config["diagnostic"]
            or runner["history_edge"] != config["history_edge"]
            or runner["target_version"] != config["target_version"]
            or runner["feedback_labels_used"] or runner["quality_auc_computed"]
            or runner["artifacts"] != [dict(path=str(Path(x["path"]).relative_to(root)),
                                          sha256=x["sha256"]) for x in artifacts]):
        raise ValueError("runner summary, committed hashes or label-free scope changed")
    return config, records, runner["costs"], dict(
        configuration=dict(path=str(config_path), sha256=_sha(config_path)),
        runner_summary=dict(path=str(summary_path), sha256=_sha(summary_path)),
        identity_sha256=identity, artifacts=artifacts)


def _validate_costs(cost):
    if (cost["current_incremental_refresh_flops"] !=
            cost["current_refresh_calibration_flops"] + cost["current_refresh_population_preparation_flops"]
            or cost["prior_sunk_total_flops"] !=
            cost["prior_sunk_calibration_flops"] + cost["prior_sunk_population_preparation_flops"]):
        raise ValueError("sunk and incremental snapshot costs do not sum")
    denominator = cost["once_rebuild_flops"]
    expected = 100 * cost["current_incremental_refresh_flops"] / denominator if denominator else None
    if expected != cost["current_incremental_percent_of_once_rebuild"]:
        raise ValueError("incremental percent differs from the actual snapshot rebuild reference")


def analyze(input_root, *, replicates=500, seed=17):
    config, users, cost, provenance = load_blocks(input_root)
    _validate_costs(cost)
    aggregate = aggregate_users(users, replicates=replicates, seed=seed)
    complete = len(users) == len(config["users"])
    diagnostic = bool(config["diagnostic"])
    permitted = complete and not diagnostic and len(users) == 2000
    return dict(format="kv_version_refresh_paired_analysis_v1",
        history_edge=config["history_edge"], target_version=config["target_version"],
        requested_users=len(config["users"]), all_requested_users_committed=complete,
        runner_diagnostic=diagnostic, quality_use_permitted=permitted,
        evidence_status="exploratory_fixed_history_mechanism" if permitted else "diagnostic_only",
        **aggregate, costs=cost,
        cost_scope="calibration plus snapshot retained-state preparation on exactly the committed population; excludes common serving reads and research teacher/source replay",
        confidence=dict(method="paired nonparametric UID-cluster percentile bootstrap",
            replicates=replicates, seed=seed,
            pairing="identical fixed UID multiplicities across both targets and every method",
            normalization="ratio of equal-user mean residuals, divided by source-unmapped under that SAME target",
            scope="conditional on this fitted seed and release case, not fitting-seed stability"),
        provenance=dict(provenance, analysis_source_sha256=_sha(Path(__file__))),
        limitations=[
            "Fixed pre-release history, candidate panel and query time; no subsequent events or continuous migration are evaluated.",
            "Each target supplies its own teacher. Cross-version raw residual changes are descriptive, not errors against a common cross-model logit truth.",
            "Source-unmapped normalization is a scale/context reference under each target, not proof that all model-version effects are removed.",
            "Refresh versus unchanged is paired within the new target, but refresh has extra fitting and capacity. Without a matched-budget old-target refresh control, its gain cannot be attributed solely to version adaptation.",
            "No observed feedback labels or AUC; teacher agreement is not downstream recommendation quality.",
            "Two independent frozen-history cases do not demonstrate a continuous multi-release trajectory or any particular compact-code/composable Design.",
            "Single fitting seed; UID bootstrap does not measure training randomness or cross-release stability.",
            "Fewer than512 unique users is diagnostic only; this fixed-population case requires all2000 for its exploratory mechanism report.",
            "Prior F fitting is sunk prior-release cost; incremental refresh preparation/fitting and retained-state mapping are separate. Population teachers are research-only cost."])


def _flatten(prefix, statistic):
    lo, hi = statistic["ci95"] or (None, None)
    return {prefix: statistic["mean"], prefix + "_ci95_low": lo,
            prefix + "_ci95_high": hi, prefix + "_valid_users": statistic["valid_users"],
            prefix + "_invalid_bootstrap_draws": statistic["invalid_bootstrap_draws"]}


def _csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def _number(value):
    return "—" if value is None else f"{value:.6g}"


def _stat(statistic):
    interval = statistic["ci95"]
    return _number(statistic["mean"]) + (" [" + ", ".join(map(_number, interval)) + "]" if interval else " [无有效区间]")


def write_outputs(summary, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    rows = []
    for endpoint in summary["endpoints"]:
        row = {key: endpoint[key] for key in ("target", "method", "metric")}
        row.update(_flatten("residual", endpoint["residual"]))
        row.update(_flatten("relative_to_source_unmapped", endpoint["relative_to_source_unmapped"]))
        rows.append(row)
    _csv(output / "endpoints.csv", rows)
    rows = []
    for contrast in summary["contrasts"]:
        row = dict(metric=contrast["metric"])
        for name, value in contrast.items():
            if name != "metric":
                row.update(_flatten(name, value))
        rows.append(row)
    _csv(output / "paired_contrasts.csv", rows)
    _csv(output / "costs.csv", [{key: value for key, value in summary["costs"].items()
                                 if not isinstance(value, dict)}])
    lines = [f"# 固定历史的版本变化诊断：edge{summary['history_edge']} → targetV{summary['target_version']}", "",
        f"同一组{summary['unique_users']}/{summary['requested_users']}名固定UID；"
        f"{summary['confidence']['replicates']}次配对UID bootstrap，表中为均值[95%区间]。", "",
        "各版本分别使用自己的teacher，不把跨模型logit原值当作共同真值。"
        "跨版本残差只描述各自teacher的一致性变化，同时给出同目标source-unmapped残差归一化。"
        "归一化是同UID均值之比，避免逐用户小分母；它也不能排除全部版本混淆。", ""]
    if not summary["quality_use_permitted"]:
        lines.extend(["本结果仅为程序/覆盖诊断，不能作为正式机制结果；需完整固定2000人且非diagnostic运行。", ""])
    lines.extend(["| 目标 | 方法 | 残差 | 原始均值[95%区间] | 相对同目标source-unmapped |",
                  "|---|---|---|---:|---:|"])
    for point in summary["endpoints"]:
        lines.append(f"| {point['target']} | {point['method']} | {point['metric']} | "
                     f"{_stat(point['residual'])} | {_stat(point['relative_to_source_unmapped'])} |")
    lines.extend(["", "| 残差 | 跨版本unchanged原始变化(new−old) | 同source基准归一化变化(new−old) | 新目标内unchanged−refresh | 新目标内误差下降(%) |",
                  "|---|---:|---:|---:|---:|"])
    for point in summary["contrasts"]:
        fields = ["cross_version_unchanged_residual_change", "cross_version_source_normalized_residual_change",
                  "new_target_refresh_improvement", "new_target_refresh_relative_improvement_percent"]
        lines.append("| " + " | ".join([point["metric"], *[_stat(point[name]) for name in fields]]) + " |")
    cost = summary["costs"]
    lines.extend(["", "正的unchanged−refresh表示在同一新teacher下刷新减少残差；负值和跨零区间同样保留。"
        "刷新多了拟合预算和映射容量，缺少同预算G_old→oldTeacher对照时，不能把收益全部归因于版本适应。", "",
        f"本人口一次新目标重建参考为{cost['once_rebuild_flops']:,} FLOPs。"
        f"当前refresh校准（含准备）{cost['current_refresh_calibration_flops']:,}，"
        f"居民保留状态刷新{cost['current_refresh_population_preparation_flops']:,}，"
        f"本次增量合计{cost['current_incremental_refresh_flops']:,} FLOPs"
        f"（重建参考的{_number(cost['current_incremental_percent_of_once_rebuild'])}%）。", "",
        f"旧F拟合{cost['prior_sunk_calibration_flops']:,}和当时居民物化"
        f"{cost['prior_sunk_population_preparation_flops']:,}，合计{cost['prior_sunk_total_flops']:,} FLOPs，"
        "单独记作prior sunk工作，不在本次增量中重收。人口teacher/source重放属于研究诊断费用。"
        "这里仅算校准与快照状态准备，不含共同serving reads，不是旧230-read端到端费用口径；"
        "只对实际提交人口摊销，不能拿8人诊断费用冒充2000人部署费用。", "",
        "本实验固定发布前历史、候选和查询时间，没有后续事件、真实反馈标签或AUC。"
        "它不证明连续迁移、最终Design有效性，也不证明某种compact code或组合更新形式必要。", "",
        *["- " + item for item in summary["limitations"]]])
    (output / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    result = analyze(args.input, replicates=args.replicates, seed=args.seed)
    write_outputs(result, args.output)
    print(json.dumps(dict(status=result["evidence_status"], users=result["unique_users"], output=str(args.output))))


if __name__ == "__main__":
    main()
