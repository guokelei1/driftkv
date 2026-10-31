#!/usr/bin/env python3
"""CPU paired analysis of a 2x2 refresh calibration/input-source study."""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.ch3_motivation.code.analyze_kv_version_refresh import (
    _csv, _flatten, _mean, _ratio, _sha, _weights,
)

METHODS = ("native_unadapted", "actual_unadapted", "train_native_on_native",
           "train_native_on_actual", "train_actual_on_native", "train_actual_on_actual", "teacher")
BASELINES = {method: ("native_unadapted" if method.endswith("_native") or method == "native_unadapted"
                     else "actual_unadapted") for method in METHODS}


def metrics(user):
    teacher = np.asarray(user["scores"]["teacher"], dtype=np.float64)
    if teacher.shape != (16,) or not np.isfinite(teacher).all() or len(user["panel_candidates"]) != 16:
        raise ValueError("requires the same finite Q16 new-target teacher")
    raw, centered = {}, {}
    for method in METHODS:
        score = np.asarray(user["scores"][method], dtype=np.float64)
        if score.shape != (16,) or not np.isfinite(score).all():
            raise ValueError("invalid crossed-source scores")
        residual = score-teacher
        raw[method] = float(np.mean(residual**2))
        centered[method] = float(np.mean((residual-residual.mean())**2))
        if not np.isclose(raw[method], user["score_mse"][method], rtol=1e-5, atol=1e-10):
            raise ValueError("stored MSE differs from its single new-target teacher")
    return dict(score_mse=raw, centered_score_mse=centered)


def aggregate_users(users, *, replicates=500, seed=17):
    if not users or len({user["uid"] for user in users}) != len(users):
        raise ValueError("requires unique paired users")
    weights = _weights(len(users), replicates, seed)
    prepared = [metrics(user) for user in users]
    endpoints, contrasts = [], []
    for metric in ("score_mse", "centered_score_mse"):
        value = {method: np.asarray([row[metric][method] for row in prepared]) for method in METHODS}
        for method in METHODS:
            base = BASELINES[method]
            endpoints.append(dict(metric=metric, method=method, baseline=base,
                residual=_mean(value[method], weights),
                relative_to_input_unadapted=_ratio(value[method], value[base], weights)))
        advantage_actual = value["train_native_on_actual"]-value["train_actual_on_actual"]
        advantage_native = value["train_actual_on_native"]-value["train_native_on_native"]
        a, n = float(advantage_actual.mean()), float(advantage_native.mean())
        if a > 0 and n > 0:
            pattern = "matched_source_favored_on_each_input_in_point_estimates"
        elif a > 0 and n < 0:
            pattern = "actual_trained_map_wins_both_inputs_in_point_estimates"
        elif a < 0 and n > 0:
            pattern = "native_trained_map_wins_both_inputs_in_point_estimates"
        elif a < 0 and n < 0:
            pattern = "cross_trained_maps_favored_on_each_input_in_point_estimates"
        else:
            pattern = "mixed_or_tied_point_estimates"
        contrasts.append(dict(metric=metric,
            actual_input_matching_advantage=_mean(advantage_actual, weights),
            native_input_matching_advantage=_mean(advantage_native, weights),
            matching_interaction=_mean(advantage_actual+advantage_native, weights),
            point_estimate_pattern=pattern))
    return dict(unique_users=len(users), endpoints=endpoints, contrasts=contrasts)


def load_blocks(root):
    root = Path(root).resolve()
    config = json.loads((root/"configuration.json").read_text())
    if (config["format"] != "crossed_refresh_source_population_v1"
            or config["methods"] != list(METHODS)):
        raise ValueError("expected the frozen crossed-source runner")
    actual, native = config["fitting_identities"]["actual"], config["fitting_identities"]["native"]
    if (actual["source_kind"] != "actual_predecessor_mapped_cache_not_native_old_target"
            or native["source_kind"] != "native_old_target_same_history"
            or native["paired_actual_refresh_checkpoint_sha256"] != config["checkpoints"]["actual"]["sha256"]):
        raise ValueError("native and actual calibration identities differ")
    for key in ("history_edge", "target_version", "history_cutoff", "fit_uids", "validation_users",
                "calibration_candidate_sha256", "width", "depth", "steps", "learning_rate", "seed"):
        if actual[key] != native[key]:
            raise ValueError(f"calibration controls are not matched in {key}")
    if set(config["users"]) & set(actual["fit_uids"]+actual["validation_users"]):
        raise ValueError("evaluation overlaps calibration roles")
    for path, digest in config["sources"].items():
        if _sha(path) != digest:
            raise ValueError("bound numerical source changed after execution")
    for checkpoint in config["checkpoints"].values():
        if _sha(checkpoint["path"]) != checkpoint["sha256"]:
            raise ValueError("bound checkpoint changed after execution")
    identity = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"),
                                        allow_nan=False).encode()).hexdigest()
    users, artifacts = [], []
    for path in sorted((root/"blocks").glob("part_*.json")):
        block = json.loads(path.read_text())
        selected = config["users"][len(users):len(users)+len(block["uids"])]
        if (block["status"] != "complete" or block["identity_sha256"] != identity
                or block["offset"] != len(users) or block["uids"] != selected
                or [u["uid"] for u in block["users"]] != selected):
            raise ValueError("cross-source committed block identity or UID order changed")
        if any(u["target_version"] != config["target_version"]
               or u["history_cutoff"] != config["settings"]["release_day"]*86400 for u in block["users"]):
            raise ValueError("target model or fixed history cutoff changed")
        users.extend(block["users"])
        artifacts.append(dict(path=str(path.relative_to(root)), sha256=_sha(path)))
    runner = json.loads((root/"summary.json").read_text())
    if (runner["status"] != "complete" or runner["users"] != len(users)
            or runner["artifacts"] != artifacts or runner["feedback_labels_used"]
            or runner["quality_auc_computed"] or runner["diagnostic"] != config["diagnostic"]):
        raise ValueError("cross-source summary/hash or label-free scope changed")
    for control in runner["costs"]["controls"].values():
        if control["total_flops"] != control["calibration_flops"]+control["population_mapping_flops"]:
            raise ValueError("control calibration and population cost do not sum")
    return config, users, runner, dict(configuration_sha256=_sha(root/"configuration.json"),
        runner_summary_sha256=_sha(root/"summary.json"), identity_sha256=identity, artifacts=artifacts)


def analyze(root, *, replicates=500, seed=17):
    config, users, runner, provenance = load_blocks(root)
    result = aggregate_users(users, replicates=replicates, seed=seed)
    complete = len(users) == len(config["users"])
    permitted = complete and not config["diagnostic"] and len(users) >= 512
    return dict(format="crossed_refresh_source_analysis_v1", history_edge=config["history_edge"],
        target_version=config["target_version"], requested_users=len(config["users"]),
        all_requested_users_committed=complete, quality_use_permitted=permitted,
        evidence_status="exploratory_fixed_history_mechanism" if permitted else "diagnostic_only",
        **result, costs=runner["costs"], confidence=dict(method="paired UID-cluster percentile bootstrap",
            replicates=replicates, seed=seed, scope="conditional on these fitted weights, not training randomness"),
        provenance=dict(provenance, analysis_source_sha256=_sha(Path(__file__))),
        limitations=["One new-target teacher, fixed history/candidate/time; no real-feedback AUC or continuous trajectory.",
            "Native-source rebuilding is a diagnostic input and is not free retained state.",
            "Fit-only normalization moments change with calibration source; this is part of the calibration treatment.",
            "Same architecture, budget and seed do not prove optimization has converged or remove fitting randomness.",
            "One map winning on both inputs is not evidence that matching the source alone explains its advantage.",
            "Centered score MSE removes a common candidate offset; it remains a teacher-agreement statistic, not ranking accuracy.",
            "No final Design, compact-code or gate necessity claim follows from this control."])


def write_outputs(result, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output/"summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    endpoints = []
    for point in result["endpoints"]:
        endpoints.append({**{key: point[key] for key in ("metric", "method", "baseline")},
            **_flatten("residual", point["residual"]),
            **_flatten("relative_to_input_unadapted", point["relative_to_input_unadapted"])})
    _csv(output/"endpoints.csv", endpoints)
    rows = []
    for point in result["contrasts"]:
        rows.append(dict(metric=point["metric"], point_estimate_pattern=point["point_estimate_pattern"],
            **{key: value for name in ("actual_input_matching_advantage", "native_input_matching_advantage", "matching_interaction")
               for key, value in _flatten(name, point[name]).items()}))
    _csv(output/"paired_contrasts.csv", rows)
    def stat(value):
        lo, hi = value["ci95"] or (None, None)
        return "—" if value["mean"] is None else f"{value['mean']:.6g} [{lo:.6g}, {hi:.6g}]" if lo is not None else f"{value['mean']:.6g} [无区间]"
    lines = [f"# 校准源状态2×2：原history edge{result['history_edge']} → 新目标V{result['target_version']}", "",
        f"{result['unique_users']}/{result['requested_users']}名固定UID，{result['confidence']['replicates']}次配对UID区间；{result['evidence_status']}。", "",
        "N是旧目标的原生重建；A是旧F实际物化的保留状态。RN/RA分别在N/A上校准；同一新目标teacher用于全部格。",
        "架构、拟合人数、候选和预算一致；fit-only归一化在各源池重算，是本calibration treatment的一部分。", "",
        "| 指标 | 路径 | 均值 [UID95%] | 相对对应未校正输入 |", "|---|---|---:|---:|"]
    for point in result["endpoints"]:
        lines.append(f"| {point['metric']} | {point['method']} | {stat(point['residual'])} | {stat(point['relative_to_input_unadapted'])} |")
    lines += ["", "A上的匹配优势=RN(A)−RA(A)；N上的匹配优势=RA(N)−RN(N)；交互是二者之和。正值分别有利于匹配源；不能仅凭交互为正就声称两边都存在匹配优势。", "",
        "| 指标 | A匹配优势 | N匹配优势 | 交互 | 点估计模式 |", "|---|---:|---:|---:|---|"]
    for point in result["contrasts"]:
        fields = [stat(point[key]) for key in ("actual_input_matching_advantage", "native_input_matching_advantage", "matching_interaction")]
        lines.append("| "+" | ".join([point["metric"], *fields, point["point_estimate_pattern"]])+" |")
    lines += ["", "若同一map在N和A上都更好，结果可能反映整体拟合/优化差异，不单独归因为源匹配。centered MSE按每UID的Q16误差减去候选均值后计算。", "",
        "成本按每套校准及一次人口映射分别保留，不把四个诊断格相加作为部署费用。N的旧目标重建另列，旧F列为沉没工作；不扩展到230次服务事件口径。", "",
        "| 校准源 | 校准FLOPs | 全人口映射FLOPs | 合计/一次新目标重建 |", "|---|---:|---:|---:|"]
    for name, cost in result["costs"]["controls"].items():
        lines.append(f"| {name} | {cost['calibration_flops']:,} | {cost['population_mapping_flops']:,} | {cost['percent_once_new_target_rebuild']:.3f}% |")
    lines += ["",
        *["- "+item for item in result["limitations"]]]
    (output/"report.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    result = analyze(args.input, replicates=args.replicates, seed=args.seed)
    write_outputs(result, args.output)
    print(json.dumps(dict(status=result["evidence_status"], users=result["unique_users"], output=str(args.output))))
