#!/usr/bin/env python3
"""Paired-user analysis of one KV-evolution mechanism case, without GPU work.

Fixed Q16 panels carry teacher logits/read residuals, never observed labels or
AUC.  Real first/last AUC uses only the same users with distinct timestamps.
Every append checkpoint is paired with cutover on its OWN available users.
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
import sys

import numpy as np
import pyarrow.parquet as pq

from hstu_kvcache.evaluation.statistics import weighted_auc

METHODS = ("reuse", "rebuild", "f", "teacher_forced")
SCORE_METHODS = ("reuse", "f", "teacher_forced")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _support(n):
    return dict(unique_users=int(n), quality_use_permitted=n >= 512,
                evidence_status="exploratory_mechanism" if n >= 512 else "diagnostic_only_lt512_users")


def _weights(n, replicates, seed):
    rng = np.random.default_rng(seed)
    return np.asarray([np.bincount(rng.integers(n, size=n), minlength=n)
                       for _ in range(replicates)], dtype=np.uint16)


def _interval(values):
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    return dict(ci95=np.percentile(values[valid], [2.5, 97.5]).tolist() if valid.any() else None,
                invalid_bootstrap_draws=int((~valid).sum()))


def _describe(values, weights):
    array = np.asarray([np.nan if x is None else x for x in values], dtype=np.float64)
    valid = np.isfinite(array)
    if not valid.any():
        return dict(mean=None, valid_users=0, ci95=None,
                    invalid_bootstrap_draws=len(weights), quality_use_permitted=False)
    count = weights[:, valid].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        draws = weights[:, valid] @ array[valid] / count
    return dict(mean=float(array[valid].mean()), valid_users=int(valid.sum()),
                quality_use_permitted=int(valid.sum()) >= 512, **_interval(draws))


def aggregate_snapshots(users, *, replicates=500, seed=17):
    """Paired cutover changes, never compare unmatched checkpoint means."""
    by_user = {}
    for user in users:
        uid = int(user["uid"])
        if uid in by_user:
            raise ValueError("duplicate user in committed evolution blocks")
        snapshots = {row["checkpoint"]: row for row in user["snapshots"]}
        if len(snapshots) != len(user["snapshots"]) or "cutover" not in snapshots:
            raise ValueError("each user needs one cutover and unique checkpoint names")
        by_user[uid] = snapshots
    names = {name: row["target_appends"] for snapshots in by_user.values()
             for name, row in snapshots.items()}
    result = []
    for index, name in enumerate(sorted(names, key=lambda x: (names[x], x))):
        uids = sorted(uid for uid, snapshots in by_user.items() if name in snapshots)
        weights = _weights(len(uids), replicates, seed + index)
        now = [by_user[uid][name] for uid in uids]
        initial = [by_user[uid]["cutover"] for uid in uids]
        score = {}
        for method in SCORE_METHODS:
            current = np.asarray([x["score_mse"][method] for x in now])
            cutover = np.asarray([x["score_mse"][method] for x in initial])
            if not np.isfinite(current).all() or not np.isfinite(cutover).all():
                raise ValueError("nonfinite teacher logit MSE in a committed block")
            score[method] = dict(current=_describe(current, weights),
                                 cutover_same_users=_describe(cutover, weights),
                                 paired_change=_describe(current - cutover, weights))
        forcing_gain = [x["score_mse"]["f"] - x["score_mse"]["teacher_forced"] for x in now]
        mapping_gain = [x["score_mse"]["reuse"] - x["score_mse"]["f"] for x in now]
        reads = {}
        for field in ("raw_read_mse", "length_normalized_read_mse", "centered_energy_ratio"):
            current = [x["candidate_residual"]["summary"][field] for x in now]
            cutover = [x["candidate_residual"]["summary"][field] for x in initial]
            delta = [None if a is None or b is None else a - b for a, b in zip(current, cutover, strict=True)]
            reads[field] = dict(current=_describe(current, weights),
                                cutover_same_users=_describe(cutover, weights),
                                paired_change=_describe(delta, weights))
        result.append(dict(checkpoint=name, target_appends=int(names[name]), **_support(len(uids)),
            available_uids=uids, score_mse=score, f_read_residual=reads,
            mapping_mse_improvement_over_reuse=_describe(mapping_gain, weights),
            teacher_forcing_mse_improvement_over_f=_describe(forcing_gain, weights),
            appended_events_mean=float(np.mean([x["appended_events"] for x in now])),
            inherited_fraction_mean=float(np.mean([x["inherited_fraction"] for x in now])),
            paired_cohort_rule="current checkpoint available UIDs and those same UIDs at cutover"))
    return result


def analyze_feedback(users, rows, *, replicates=500, seed=17):
    """Secondary real-feedback AUC; stages and methods share UID multiplicities."""
    metadata = {int(x["uid"]): x for x in users}
    if any(int(row["uid"]) not in metadata for row in rows):
        raise ValueError("feedback user is absent from committed block membership")
    for row in rows:
        user = metadata[int(row["uid"])]
        expected = {"first": user["first_request_timestamp"], "last": user["last_request_timestamp"]}
        if row["phase"] == "first_last":
            if expected["first"] != expected["last"] or row["query_timestamp"] != expected["first"]:
                raise ValueError("first_last feedback does not represent one shared request time")
        elif row["phase"] not in expected or row["query_timestamp"] != expected[row["phase"]]:
            raise ValueError("feedback phase differs from its user's actual endpoint")
    distinct = {uid for uid, x in metadata.items()
                if x["first_request_timestamp"] is not None and x["last_request_timestamp"] is not None
                and x["first_request_timestamp"] < x["last_request_timestamp"]}
    feedback_users = {int(row["uid"]) for row in rows}
    phase_users = {phase: {int(row["uid"]) for row in rows if row["phase"] == phase}
                   for phase in ("first", "last")}
    common = sorted(distinct & phase_users["first"] & phase_users["last"])
    counts = dict(committed_resident_users=len(metadata), users_with_any_real_feedback=len(feedback_users),
        users_without_real_feedback=len(set(metadata) - feedback_users),
        users_with_distinct_first_last_timestamps=len(distinct),
        single_timestamp_feedback_users=len(feedback_users - distinct),
        primary_matched_distinct_users=len(common),
        distinct_users_missing_a_scored_stage=len(distinct - set(common)))
    if not common:
        return dict(support=counts, **_support(0), points=[], changes=[], primary_uids=[])
    weights = _weights(len(common), replicates, seed)
    population = np.asarray(common, dtype=np.int64)
    allowed = set(common)
    estimates, draws, points = {}, {}, []
    for phase in ("first", "last"):
        groups = {method: sorted([r for r in rows if r["phase"] == phase and
            r["method"] == method and int(r["uid"]) in allowed],
            key=lambda r: (r["uid"], r["query_timestamp"], r["raw_item_id"], r["item_idx"]))
            for method in METHODS}
        keys = lambda group: [(r["uid"], r["query_timestamp"], r["raw_item_id"], r["item_idx"], r["label"])
                              for r in group]
        reference_keys = keys(groups["rebuild"])
        if not reference_keys or any(keys(group) != reference_keys for group in groups.values()):
            raise ValueError("real-feedback events/labels differ across paired methods")
        uid_index = np.searchsorted(population, [r["uid"] for r in groups["rebuild"]])
        labels = np.asarray([r["label"] for r in groups["rebuild"]], dtype=np.float64)
        if not set(labels).issubset({0., 1.}):
            raise ValueError("feedback AUC requires observed binary labels")
        for method, group in groups.items():
            values = np.asarray([r["score"] for r in group], dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError("nonfinite real-feedback score")
            estimates[phase, method] = float(weighted_auc(values, labels, uid_index,
                np.ones((1, len(common)), dtype=np.uint16))[0])
            draws[phase, method] = weighted_auc(values, labels, uid_index, weights)
        for method in METHODS:
            value = estimates[phase, method]
            teacher = estimates[phase, "rebuild"]
            difference = value - teacher
            points.append(dict(phase=phase, method=method, **_support(len(common)),
                feedback_events=len(labels), positive_events=int(labels.sum()),
                auc=value if np.isfinite(value) else None, **_interval(draws[phase, method]),
                auc_minus_teacher=difference if np.isfinite(difference) else None,
                auc_minus_teacher_interval=_interval(draws[phase, method] - draws[phase, "rebuild"])))
    changes = []
    for method in METHODS:
        current = estimates["last", method] - estimates["first", method]
        relative = current - (estimates["last", "rebuild"] - estimates["first", "rebuild"])
        change_draws = draws["last", method] - draws["first", method]
        relative_draws = change_draws - (draws["last", "rebuild"] - draws["first", "rebuild"])
        changes.append(dict(method=method, **_support(len(common)),
            auc_last_minus_first=current if np.isfinite(current) else None,
            auc_change_interval=_interval(change_draws),
            teacher_relative_auc_change=relative if np.isfinite(relative) else None,
            teacher_relative_auc_change_interval=_interval(relative_draws)))
    return dict(support=counts, **_support(len(common)), primary_uids=common, points=points,
        changes=changes, cohort_rule="same UIDs with distinct first<last timestamps and both scored stages; first_last single-request users excluded from temporal AUC",
        scope="secondary observed-feedback discrimination; candidate/label mix can change between stages; not a causal isolation by itself")


def load_blocks(input_root):
    """Read only committed blocks bound to the runner's exact configuration."""
    root = Path(input_root).resolve()
    config_path = root / "configuration.json"
    config = json.loads(config_path.read_text())
    if config["format"] != "kv_evolution_population_v1":
        raise ValueError("expected the KV evolution population runner's configuration")
    # Exact digest convention confirmed with run_kv_evolution.py, not the
    # whitespace-sensitive hash of configuration.json itself.
    identity = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"),
                                        allow_nan=False).encode()).hexdigest()
    records, feedback, artifacts = [], [], []
    for path in sorted((root / "blocks").glob("part_*.json")):
        block = json.loads(path.read_text())
        selected = block["uids"]
        if (block["status"] != "complete" or block["identity_sha256"] != identity
                or block["offset"] != len(records)
                or selected != config["users"][len(records):len(records) + len(selected)]
                or [row["uid"] for row in block["users"]] != selected):
            raise ValueError("committed block identity, UID membership or order changed")
        score_path = root / block["feedback_path"]
        if _sha(score_path) != block["feedback_sha256"]:
            raise ValueError("committed feedback hash changed")
        rows = pq.read_table(score_path).to_pylist()
        selected_set = set(selected)
        if any(row["uid"] not in selected_set for row in rows):
            raise ValueError("feedback shard contains another block's user")
        for user in block["users"]:
            if len(user["panel_candidates"]) != 16:
                raise ValueError("fixed synthetic panel must contain Q16 candidates")
            for snapshot in user["snapshots"]:
                scores = snapshot["fixed_panel_scores"]
                if any(len(scores[method]) != 16 for method in METHODS):
                    raise ValueError("fixed panel score arrays differ")
                teacher = np.asarray(scores["rebuild"], dtype=np.float64)
                for method in SCORE_METHODS:
                    expected = float(np.mean((np.asarray(scores[method]) - teacher) ** 2))
                    if not np.isfinite(expected) or not np.isclose(expected,
                            snapshot["score_mse"][method], rtol=1e-5, atol=1e-10):
                        raise ValueError("stored teacher MSE differs from fixed-panel scores")
        records.extend(block["users"])
        feedback.extend(rows)
        artifacts.extend([dict(path=str(path), sha256=_sha(path)),
                          dict(path=str(score_path), sha256=block["feedback_sha256"])])
    if not records:
        raise ValueError("no committed evolution users to analyze")
    return config, records, feedback, dict(configuration=dict(path=str(config_path), sha256=_sha(config_path)),
                                         identity_sha256=identity, artifacts=artifacts)


def analyze(input_root, *, cost_preflight=None, replicates=500, seed=17):
    if replicates < 1:
        raise ValueError("paired UID bootstrap needs positive replicates")
    config, users, rows, provenance = load_blocks(input_root)
    reference = config["cost_preflight_reference"]
    cost_path = Path(cost_preflight) if cost_preflight is not None else Path(reference["path"])
    if _sha(cost_path) != reference["sha256"]:
        raise ValueError("cost preflight differs from the runner's frozen reference")
    cost = json.loads(cost_path.read_text())
    account = cost["costs"]
    population = account["candidate_reads_230"]["users"]
    if population != 2000:
        raise ValueError("this mechanism study's deployment preflight is the fixed2000 ledger")
    diagnostic = bool(config["diagnostic"])
    checkpoints = aggregate_snapshots(users, replicates=replicates, seed=seed)
    feedback = analyze_feedback(users, rows, replicates=replicates, seed=seed)
    if diagnostic:
        # A runner explicitly launched as a diagnostic cannot become quality
        # evidence merely because it happened to contain >=512 users.
        for point in checkpoints + feedback["points"] + feedback["changes"]:
            point.update(quality_use_permitted=False, evidence_status="runner_diagnostic_only")
        for point in checkpoints:
            for method in point["score_mse"].values():
                for metric in method.values():
                    metric["quality_use_permitted"] = False
            for field in point["f_read_residual"].values():
                for metric in field.values():
                    metric["quality_use_permitted"] = False
            point["mapping_mse_improvement_over_reuse"]["quality_use_permitted"] = False
            point["teacher_forcing_mse_improvement_over_f"]["quality_use_permitted"] = False
        feedback.update(quality_use_permitted=False, evidence_status="runner_diagnostic_only")
    return dict(format="kv_evolution_paired_analysis_v1", edge=config["edge"],
        requested_resident_users=len(config["users"]), committed_resident_users=len(users),
        all_requested_users_committed=len(users) == len(config["users"]),
        runner_diagnostic=diagnostic, exploratory_single_fitting_seed=True,
        checkpoints=checkpoints, real_feedback=feedback,
        confidence=dict(method="paired nonparametric UID-cluster percentile bootstrap",
            replicates=replicates, seed=seed, checkpoint_cohorts="each point's available users paired to their own cutover",
            real_feedback_cohort="same distinct-first-last users jointly resampled across phases and methods",
            scope="conditional on these fitted weights and this edge; not training-seed or five-edge stability"),
        deployment_cost=dict(users=population, complete_230_percent=cost["complete_percent"],
            actual_event_complete_percent=account["percent_of_once_rebuild"],
            within_rebuild_budget=cost["within_rebuild_budget"],
            path=str(cost_path.resolve()), sha256=reference["sha256"],
            scope="ordinary F deployment on fixed2000, not reamortized over feedback/available-checkpoint cohorts; teacher-forced research work is not a serving method"),
        provenance=dict(provenance, analysis_source_sha256=_sha(Path(__file__))),
        limitations=["Synthetic Q16 panels have no observed labels and no AUC.",
            "Each checkpoint uses a potentially different available cohort; use its paired cutover change.",
            "Candidate-centered residual energy is descriptive rich-query variation, not proof of item-only gate necessity.",
            "Teacher-forced new rows are an oracle diagnostic; improvement only diagnoses a possible contribution of write-path error.",
            "Real first/last candidate and label mix can differ even for the same users.",
            "Fewer than512 unique usable users is diagnostic only; all support counts are retained.",
            "This independent edge does not establish a continuous multi-release trajectory or all-five-edge stability."])


def _write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _flatten_metric(prefix, metric):
    low, high = metric["ci95"] or (None, None)
    return {prefix: metric["mean"], prefix + "_ci95_low": low,
            prefix + "_ci95_high": high, prefix + "_valid_users": metric["valid_users"]}


def write_outputs(summary, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    scores, reads, aucs, changes = [], [], [], []
    for point in summary["checkpoints"]:
        base = {k: point[k] for k in ("checkpoint", "target_appends", "unique_users", "quality_use_permitted", "evidence_status")}
        base.update(appended_events_mean=point["appended_events_mean"], inherited_fraction_mean=point["inherited_fraction_mean"])
        for method, metric in point["score_mse"].items():
            row = dict(base, method=method)
            for name in ("current", "cutover_same_users", "paired_change"):
                row.update(_flatten_metric(name + "_teacher_logit_mse", metric[name]))
            row.update(_flatten_metric("teacher_forcing_mse_improvement_over_f", point["teacher_forcing_mse_improvement_over_f"]))
            row.update(_flatten_metric("mapping_mse_improvement_over_reuse", point["mapping_mse_improvement_over_reuse"]))
            scores.append(row)
        for field, metric in point["f_read_residual"].items():
            row = dict(base, metric=field)
            for name in ("current", "cutover_same_users", "paired_change"):
                row.update(_flatten_metric(name, metric[name]))
            row["quality_use_permitted"] = bool(base["quality_use_permitted"] and metric["paired_change"]["quality_use_permitted"])
            if base["quality_use_permitted"] and not row["quality_use_permitted"]:
                row["evidence_status"] = "diagnostic_only_lt512_metric_users"
            reads.append(row)
    for point in summary["real_feedback"]["points"]:
        row = {k: v for k, v in point.items() if not isinstance(v, (dict, list))}
        row["auc_ci95_low"], row["auc_ci95_high"] = point["ci95"] or (None, None)
        row["auc_minus_teacher_ci95_low"], row["auc_minus_teacher_ci95_high"] = point["auc_minus_teacher_interval"]["ci95"] or (None, None)
        aucs.append(row)
    for point in summary["real_feedback"]["changes"]:
        row = {k: v for k, v in point.items() if not isinstance(v, dict)}
        for key in ("auc_change_interval", "teacher_relative_auc_change_interval"):
            row[key + "_ci95_low"], row[key + "_ci95_high"] = point[key]["ci95"] or (None, None)
        changes.append(row)
    _write_csv(output / "checkpoint_scores.csv", scores)
    _write_csv(output / "checkpoint_read_residual.csv", reads)
    _write_csv(output / "real_feedback_auc.csv", aucs)
    _write_csv(output / "real_feedback_changes.csv", changes)
    _write_report(summary, output / "report.md")


def _number(value):
    return "—" if value is None else f"{value:.6g}"


def _write_report(summary, path):
    cost = summary["deployment_cost"]
    lines = [f"# KV映射演化机制：独立版本边{summary['edge']}", "",
        f"已提交{summary['committed_resident_users']}/{summary['requested_resident_users']}名固定居民用户。"
        "本报告是单拟合seed的探索性机制证据，未验证跨五边或训练随机性稳定性。", "",
        "固定Q16面板仅比较教师logit与同查询读取残差，没有真实标签，不计算AUC。"
        "每个追加点都用该点实际可用UID，与这些UID自己的cutover配对；不能把不同点的未配对均值连成同人口退化曲线。", "",
        "| 追加点 | 可用UID | F教师MSE | 同UID cutover MSE | 配对变化 | 强制写入改进(F−forced) | 使用范围 |",
        "|---|---:|---:|---:|---:|---:|---|"]
    if summary["runner_diagnostic"] or summary["committed_resident_users"] < 512:
        lines[2:2] = ["本次为数值/性能诊断；以下数据仅核对程序和输出，不能作为质量或机制结论。", ""]
    for point in summary["checkpoints"]:
        mse = point["score_mse"]["f"]
        lines.append("| " + " | ".join([point["checkpoint"], str(point["unique_users"]),
            _number(mse["current"]["mean"]), _number(mse["cutover_same_users"]["mean"]),
            _number(mse["paired_change"]["mean"]),
            _number(point["teacher_forcing_mse_improvement_over_f"]["mean"]),
            "探索机制" if point["quality_use_permitted"] else "仅诊断"]) + " |")
    support = summary["real_feedback"]["support"]
    lines.extend(["", f"配对MSE变化与强制写入改进的{summary['confidence']['replicates']}次UID区间见JSON/CSV；正的F−forced表示教师新行减少了误差。"
        "教师强制写入是额外研究工作，不能当作可部署方案或其成本优势。", "",
        f"真实反馈支持：{support['users_with_any_real_feedback']}名用户有反馈；"
        f"{support['single_timestamp_feedback_users']}名仅一个请求时间；"
        f"主first/last对照使用{support['primary_matched_distinct_users']}名同时具有不同首尾请求时间且评分齐全的相同用户。"
        "无反馈与单请求用户保留在支持计数中，未混入时序AUC。", "",
        "| 阶段 | 方法 | 配对UID | AUC | 相对教师AUC差 | 使用范围 |",
        "|---|---|---:|---:|---:|---|"])
    for point in summary["real_feedback"]["points"]:
        lines.append(f"| {point['phase']} | {point['method']} | {point['unique_users']} | "
                     f"{_number(point['auc'])} | {_number(point['auc_minus_teacher'])} | "
                     f"{'次要探索' if point['quality_use_permitted'] else '仅诊断'} |")
    lines.extend(["", "真实AUC是次要的观测性判别指标，首尾候选/标签组成可能不同。"
        "同时报告各方法相对教师AUC差的首尾变化，避免把教师自身的阶段差异误称为映射退化。"
        "少于512名可用唯一UID时，仅作诊断。", "",
        "候选去均值残差能量比例只描述当前rich query下的误差变化；q还含时间和历史上下文，"
        "它不证明item-only门控必要，也不证明固定KV无法服务多个候选。"
        "时间演化机制与候选残差机制分别解释。", "",
        f"普通F部署费用引用独立的{cost['users']}人口完整账：230读口径新增FLOPs为Rebuild的"
        f"{cost['complete_230_percent']:.6g}%（实际事件口径{cost['actual_event_complete_percent']:.6g}%）。"
        "不能按本报告较少的反馈用户或追加点可用用户重新摊销。教师诊断开销不属于该部署费用。", "",
        "本报告保留所有方向的结果；不由单边、单seed或局部误差变化推出最终Design有效性。"])
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cost-preflight", type=Path)
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    summary = analyze(args.input, cost_preflight=args.cost_preflight, replicates=args.replicates, seed=args.seed)
    write_outputs(summary, args.output)
    print(json.dumps(dict(status="complete_exploratory_analysis", edge=summary["edge"],
        users=summary["committed_resident_users"], output=str(args.output))))


if __name__ == "__main__":
    main()
