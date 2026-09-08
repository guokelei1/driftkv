#!/usr/bin/env python3
"""UID-paired summaries for the pre-specified representation and 2x2 diagnostics."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TARGETS = (1, 3, 4, 5)


def interval(values):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(17)
    draws = values[rng.integers(0, len(values), (1000, len(values)))].mean(1)
    return dict(mean=float(values.mean()), ci95=np.quantile(draws, [.025, .975]).tolist(),
                median=float(np.median(values)), fraction_negative=float((values < 0).mean()))


def load_outputs(path):
    summary = json.loads((path / "summary.json").read_text())
    assert summary["status"] == "diagnostic_complete"
    frame = pd.concat(pd.read_parquet(path / f"outputs_m{t}.parquet").assign(target=t) for t in TARGETS)
    uid = frame.groupby(["target", "group", "panel", "uid", "method"])[["logit_mse", "probability_mse"]].mean().reset_index()
    return frame, uid, summary


def query_report(path):
    frame, uid, summary = load_outputs(path)
    rows = []
    for (target, group), values in uid[uid.panel == "held64"].groupby(["target", "group"]):
        wide = values.pivot(index="uid", columns="method", values="logit_mse")
        rows.append(dict(target=int(target), group=group, users=len(wide),
            means=wide.mean().to_dict(), medians=wide.median().to_dict(),
            oracle64_error_fraction_of_reuse=float(wide.oracle64.mean()/wide.reuse.mean()),
            oracle1024_error_fraction_of_reuse=float(wide.oracle1024.mean()/wide.reuse.mean()),
            oracle64_minus_reuse=interval(wide.oracle64-wide.reuse),
            oracle1024_minus_oracle64=interval(wide.oracle1024-wide.oracle64)))
    geometry_rows, decompositions, aggregate_decompositions, time_rows = [], [], [], []
    for target in TARGETS:
        g = pd.read_parquet(path / f"geometry_m{target}.parquet")
        gu = g.groupby("uid").mean(numeric_only=True)
        geometry_rows.append(dict(target=target,
            effective_rank_median=float(g.effective_rank.median()),
            numeric_rank_median=float(g.numeric_rank.median()),
            condition_number_median=float(g.condition_number.median()),
            coefficient_refit_relative_mse=float(gu.coefficient_refit_mse.mean()/gu.coefficient_energy.mean()),
            held_function_refit_relative_mse=float(gu.held_function_refit_mse.mean()/gu.held_function_energy.mean()),
            scope="per-scene standardized query; refit at same lower view and coordinates, not evidence of sole cause"))
        r = pd.read_parquet(path / f"responses_m{target}.parquet")
        r = r[(r.method == "oracle64") & (r.panel == "held64")]
        counts = frame[frame.target == target][["uid", "scene", "retained_events"]].drop_duplicates()
        physical = r.merge(counts, on=["uid", "scene"], validate="many_to_one")
        energy_columns = ["value_energy", "key_energy", "cross_energy", "total_energy"]
        physical[energy_columns] = physical[energy_columns].mul(physical.retained_events**2, axis=0)
        overall = physical.groupby(["uid", "layer"])[energy_columns].mean().mean()
        aggregate_decompositions.append(dict(target=target,
            ratios=(overall/overall.total_energy).to_dict(),
            scope="actual aggregate response, UID/layer equal, includes cross term; not independent causal effects"))
        for layer, values in r.groupby("layer"):
            per_uid = values.groupby("uid").mean(numeric_only=True)
            energy = per_uid.total_energy.mean()
            decompositions.append(dict(target=target, layer=int(layer),
                value_over_total=float(per_uid.value_energy.mean()/max(energy, 1e-30)),
                key_over_total=float(per_uid.key_energy.mean()/max(energy, 1e-30)),
                cross_over_total=float(per_uid.cross_energy.mean()/max(energy, 1e-30)),
                algebra_residual=float(per_uid.decomposition_residual.max()),
                oracle_response_error_fraction=float(per_uid.error.mean()/max(per_uid.energy.mean(), 1e-30))))
        p = pd.read_parquet(path / f"panels_m{target}.parquet")
        time_rows.append(dict(target=target, scenes=len(p), gap_one_scenes=int((p.gap_seconds == 1).sum()),
            scenes_with_no_time_overlap=int((p.shared_integer_times == 0).sum()),
            mean_fraction_unique_held_times_unseen=float(((p.unique_held_times-p.shared_integer_times)/p.unique_held_times).mean())))
    # The real cutover branch is retained separately from adjacent/lifetime controls.
    config = json.loads((path / "configuration.json").read_text())
    count = len(config["fitting_uids"])+len(config["diagnostic_uids"])
    continuous = []
    for target in TARGETS:
        stride = 2 if target == 1 else 3
        selected = frame[(frame.target == target) & (frame.panel == "held64")
                         & (frame.scene < count*stride) & (frame.scene % stride == 0)]
        for group, values in selected.groupby("group"):
            wide = values.pivot(index="uid", columns="method", values="logit_mse")
            continuous.append(dict(target=target, group=group, means=wide.mean().to_dict(),
                oracle64_error_fraction_of_reuse=float(wide.oracle64.mean()/wide.reuse.mean())))
    return dict(run=path.name, rows=rows, geometry=geometry_rows, update_decomposition=decompositions,
                aggregate_update_decomposition=aggregate_decompositions,
                time_holdout=time_rows, continuous_cutover=continuous,
                elapsed_seconds=summary["elapsed_seconds"], peak_allocated_mib=summary["peak_allocated_mib"])


def factorial_report(path):
    _, uid, summary = load_outputs(path)
    config = json.loads((path / "configuration.json").read_text())
    rows = []
    for (target, group, panel), values in uid.groupby(["target", "group", "panel"]):
        wide = values.pivot(index="uid", columns="method", values="logit_mse")
        mc, mr, fc, fr = (wide[m] for m in ("mean_coefficient", "mean_response", "functional_coefficient", "functional_response"))
        rows.append(dict(target=int(target), group=group, panel=panel, users=len(wide),
            means=wide.mean().to_dict(), medians=wide.median().to_dict(),
            primary_functional_response_minus_mean_coefficient=interval(fr-mc),
            objective_at_mean=interval(mr-mc), objective_at_functional=interval(fr-fc),
            information_at_coefficient=interval(fc-mc), information_at_response=interval(fr-mr),
            interaction=interval((fr-fc)-(mr-mc)),
            maximum_uid_error_fraction={m: float(wide[m].max()/max(wide[m].sum(), 1e-30)) for m in wide}))
    return dict(run=path.name, rows=rows, count_metric=config.get("count_metric", "historical per-event rate"),
                elapsed_seconds=summary["elapsed_seconds"],
                peak_allocated_mib=summary["peak_allocated_mib"], ledger_seconds=summary["ledger_seconds"])


def main(args):
    out = ROOT / "results/design/analysis" / args.report_id
    out.mkdir(parents=True, exist_ok=False)
    report = dict(query=query_report(ROOT / "results/design" / args.query_run),
                  confirmation_read=False, bootstrap="1000 paired UID resamples, seed17; within one backbone seed",
                  multiple_comparisons="descriptive intervals; all planned contrasts shown, no winner selection")
    if args.factorial_run:
        report["factorial"] = factorial_report(ROOT / "results/design" / args.factorial_run)
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# 专家路线机制验证", "", "M1/M3/M4/M5；M2不参与诊断，native谱系继续；M5 E14_partial。",
             "全部为开发诊断，6000确认未读；128 UID-disjoint组不是新盲测。", "",
             "## A：未拟合查询上的逐场景仿射教师", "",
             "| 目标/UID组 | Reuse logit MSE | 64-query oracle | 1024-query oracle | 64残差/Reuse |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for row in report["query"]["rows"]:
        m = row["means"]
        lines.append(f"| M{row['target']} {row['group']} | {m['reuse']:.6g} | {m['oracle64']:.6g} | "
                     f"{m['oracle1024']:.6g} | {100*row['oracle64_error_fraction_of_reuse']:.3f}% |")
    lines += ["", "先逐场景平均，再UID等权；64/1024预算分开。共享方案15参照、连续cutover子集、",
              "query几何、重拟合系数/函数稳定性、时间重合与value/key交叉项见JSON。", ""]
    if "factorial" in report:
        lines += ["## B：匹配二乘二", "", "下表只展示未拟合UID的留出query；全部组/面板/因子区间见JSON。",
                  "损失口径：" + report["factorial"]["count_metric"], "",
                  "共同源PCA32使两种监督容量相同；匹配基线不等同历史方案15。负差值表示误差下降。", "",
                  "| 目标 | 均值/系数 | 均值/响应 | 功能/系数 | 功能/响应 | 主比较差值 [95% UID区间] |",
                  "| --- | ---: | ---: | ---: | ---: | --- |"]
        for row in report["factorial"]["rows"]:
            if row["group"] != "diagnostic_uid" or row["panel"] != "held64":
                continue
            m = row["means"]
            primary = row["primary_functional_response_minus_mean_coefficient"]
            cells = " | ".join(f"{m[k]:.6g}" for k in ("mean_coefficient", "mean_response", "functional_coefficient", "functional_response"))
            lines.append(f"| M{row['target']} | {cells} | {primary['mean']:.4g} "
                         f"[{primary['ci95'][0]:.4g}, {primary['ci95'][1]:.4g}] |")
    lines += ["", "oracle是逐场景教师干预，不能称共享方法质量；proxy误差不等于AUC。",
              "功能输入本轮按实际源KV扫描构建，计入诊断成本；尚非连续在线维护或完整人口成本验证。", ""]
    (out / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-run", required=True)
    parser.add_argument("--factorial-run")
    parser.add_argument("--report-id", required=True)
    main(parser.parse_args())
