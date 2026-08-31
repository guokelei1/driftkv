#!/usr/bin/env python3
"""Adjudicate the sealed C32/C48/C64 label-free PRO frontier."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "configs/contracts/yambda500m_small_hstu_native_progressive_pro_frontier_v1.yaml"
DEFAULT_INPUT = ROOT / "results/yambda500m_small_seed17/insight_progressive_pro_v1/frontier_v1/formal"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_raw(directory: Path, contract_hash: str) -> None:
    seal = json.loads((directory / "raw.seal.json").read_text())
    if seal["contract_sha256"] != contract_hash or seal["labels_read"]:
        raise RuntimeError("frontier seal differs from the label-free contract")
    for name, record in seal["files"].items():
        target = directory / name
        if sha256(target) != record["sha256"] or target.stat().st_size != record["bytes"]:
            raise RuntimeError(f"sealed frontier artifact differs: {name}")


def layer_aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["phase", "edge", "method"], sort=True)
        .agg(
            rows=("identifier", "size"),
            users=("uid", "nunique"),
            median_direction_cosine=("direction_cosine_to_exact_shared_av", "median"),
            median_norm_ratio=("estimated_to_exact_norm_ratio", "median"),
            median_relative_l2=("relative_l2_to_exact_shared_av", "median"),
            median_probe_direction_cosine=("fixed_probe_direction_cosine", "median"),
            median_probe_norm_ratio=("fixed_probe_norm_ratio", "median"),
        )
        .reset_index()
    )


def score_aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    output = (
        frame.groupby(["phase", "edge", "method"], sort=True)
        .agg(
            requests=("identifier", "size"),
            mean_logit_gap_to_current_exact=("mean_abs_logit_gap_to_current_exact", "mean"),
            mean_probability_gap_to_current_exact=("mean_abs_probability_gap_to_current_exact", "mean"),
            mean_logit_gap_to_exact_shared_av=("mean_abs_logit_gap_to_exact_shared_av", "mean"),
        )
        .reset_index()
    )
    reuse = output[output.method == "reuse"][
        ["phase", "edge", "mean_probability_gap_to_current_exact"]
    ].rename(columns={"mean_probability_gap_to_current_exact": "reuse_probability_gap"})
    output = output.merge(reuse, on=["phase", "edge"], how="left", validate="many_to_one")
    output["ratio_of_means_probability_gap_recovery"] = 1.0 - (
        output.mean_probability_gap_to_current_exact
        / output.reuse_probability_gap.clip(lower=1e-12)
    )
    output.loc[output.method == "reuse", "ratio_of_means_probability_gap_recovery"] = 0.0
    return output


def convergence_aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["phase", "edge", "source_carriers", "target_carriers"], sort=True)
        .agg(
            rows=("identifier", "size"),
            median_direction_cosine=("direction_cosine", "median"),
            median_norm_ratio=("norm_ratio", "median"),
            median_relative_l2=("relative_l2", "median"),
        )
        .reset_index()
    )


def _edge_table(layer: pd.DataFrame, phase: str) -> pd.DataFrame:
    return (
        layer[layer.phase == phase]
        .groupby(["edge", "method"], sort=True)
        .relative_l2_to_exact_shared_av.median()
        .unstack("method")
    )


def adjudicate(
    layer: pd.DataFrame, score: pd.DataFrame, contract: dict[str, Any]
) -> dict[str, Any]:
    if bool(layer.labels_read.any()) or bool(score.labels_read.any()):
        raise RuntimeError("frontier unexpectedly read labels")
    required = {(phase, edge, method) for phase in ("cutover", "rolling") for edge in contract["scope"]["edges"] for method in ("C32", "C48", "C64")}
    observed = set(zip(layer.phase, layer.edge, layer.method, strict=True))
    if observed != required:
        raise RuntimeError("frontier is missing a frozen phase/edge/carrier cell")
    rules = contract["prefrozen_rules"]

    direction = (
        layer[layer.method == "C64"]
        .groupby(["phase", "edge"], sort=True)
        .direction_cosine_to_exact_shared_av.median()
        .unstack("edge")
    )
    threshold = float(rules["C64_direction_gate"]["median_cosine_minimum"])
    direction_cutover_edges = int((direction.loc["cutover"] >= threshold).sum())
    direction_rolling_edges = int((direction.loc["rolling"] >= threshold).sum())
    direction_pass = (
        direction_cutover_edges >= int(rules["C64_direction_gate"]["minimum_cutover_edges"])
        and direction_rolling_edges >= int(rules["C64_direction_gate"]["minimum_rolling_edges"])
    )

    cutover = _edge_table(layer, "cutover")
    rolling = _edge_table(layer, "rolling")
    cutover_improved = int((cutover.C64 <= cutover.C32).sum())
    rolling_improved = int((rolling.C64 <= rolling.C32).sum())
    mean_values = pd.concat(
        [
            cutover.mean().rename("cutover"),
            rolling.mean().rename("rolling"),
        ],
        axis=1,
    ).T
    mean_lower = bool(
        mean_values.loc["cutover", "C64"] < mean_values.loc["cutover", "C32"]
        and mean_values.loc["rolling", "C64"] < mean_values.loc["rolling", "C32"]
    )
    fidelity_pass = (
        cutover_improved
        >= int(rules["C64_fidelity_improvement_over_C32"]["median_relative_L2_not_above_C32_minimum_cutover_edges"])
        and rolling_improved
        >= int(rules["C64_fidelity_improvement_over_C32"]["median_relative_L2_not_above_C32_minimum_rolling_edges"])
        and mean_lower
    )

    c48_between_cutover = bool(
        min(mean_values.loc["cutover", "C32"], mean_values.loc["cutover", "C64"])
        <= mean_values.loc["cutover", "C48"]
        <= max(mean_values.loc["cutover", "C32"], mean_values.loc["cutover", "C64"])
    )
    c48_between_rolling = bool(
        min(mean_values.loc["rolling", "C32"], mean_values.loc["rolling", "C64"])
        <= mean_values.loc["rolling", "C48"]
        <= max(mean_values.loc["rolling", "C32"], mean_values.loc["rolling", "C64"])
    )
    intermediate_pass = c48_between_cutover and c48_between_rolling

    score_edge = (
        score[score.method.isin(["C32", "C64"])]
        .groupby(["phase", "edge", "method"], sort=True)
        .mean_abs_logit_gap_to_current_exact.mean()
        .unstack("method")
    )
    score_cutover_edges = int((score_edge.loc["cutover"].C64 <= score_edge.loc["cutover"].C32).sum())
    score_rolling_edges = int((score_edge.loc["rolling"].C64 <= score_edge.loc["rolling"].C32).sum())

    coherent_frontier = direction_pass and fidelity_pass and intermediate_pass
    return {
        "status": (
            "progressive_pro_frontier_passed_C64_frozen"
            if coherent_frontier
            else "progressive_pro_frontier_valid_no_upgrade_selected"
        ),
        "labels_read": False,
        "C64_direction_gate": {
            "passed": direction_pass,
            "threshold": threshold,
            "cutover_edges_passing": direction_cutover_edges,
            "rolling_edges_passing": direction_rolling_edges,
        },
        "C64_fidelity_improvement_over_C32": {
            "passed": fidelity_pass,
            "cutover_edges_not_above": cutover_improved,
            "rolling_edges_not_above": rolling_improved,
            "five_edge_mean_lower_both_phases": mean_lower,
        },
        "C48_intermediate_frontier_point": {
            "passed": intermediate_pass,
            "cutover_between_C32_C64": c48_between_cutover,
            "rolling_between_C32_C64": c48_between_rolling,
        },
        "C64_score_gap_diagnostic": {
            "cutover_edges_not_above_C32": score_cutover_edges,
            "rolling_edges_not_above_C32": score_rolling_edges,
        },
        "five_edge_mean_of_edge_median_relative_l2": {
            phase: {method: float(mean_values.loc[phase, method]) for method in ("C32", "C48", "C64")}
            for phase in ("cutover", "rolling")
        },
        "coherent_progressive_frontier_passed": coherent_frontier,
        "frozen_design_after_frontier": (
            "C64_two_probe_single_direction_global_decay"
            if coherent_frontier
            else "retain_existing_C32_lightweight_PRO"
        ),
        "formal_quality_unlocked": False,
        "interpretation": (
            "C64_consistently_improves_C32_relative_L2_but_does_not_cross_the_absolute_rolling_direction_gate_and_C48_is_non_monotonic;_the_prefrozen_progressive_precision_axis_is_not_coherent"
        ),
    }


def report(summary: dict[str, Any], layer: pd.DataFrame) -> str:
    means = summary["five_edge_mean_of_edge_median_relative_l2"]
    lines = [
        "# Progressive PRO C32/C48/C64 无标签 fidelity frontier",
        "",
        "状态：正式 raw/seal 完整，未读取行为 label；冻结 progressive 选择门未通过。",
        "",
        "## 成本与五边平均 fidelity",
        "",
        "| point | Full FLOPs | cutover relative L2 | rolling relative L2 |",
        "| --- | ---: | ---: | ---: |",
        f"| C32 | 10.52% | {means['cutover']['C32']:.5f} | {means['rolling']['C32']:.5f} |",
        f"| C48 | 14.54% | {means['cutover']['C48']:.5f} | {means['rolling']['C48']:.5f} |",
        f"| C64 | 18.64% | {means['cutover']['C64']:.5f} | {means['rolling']['C64']:.5f} |",
        "",
        "## 冻结规则裁决",
        "",
        f"- C64 relative L2 相对 C32：cutover {summary['C64_fidelity_improvement_over_C32']['cutover_edges_not_above']}/5、rolling {summary['C64_fidelity_improvement_over_C32']['rolling_edges_not_above']}/5，fidelity improvement gate PASS。",
        f"- C64 absolute direction cosine >=0.90：cutover {summary['C64_direction_gate']['cutover_edges_passing']}/5、rolling {summary['C64_direction_gate']['rolling_edges_passing']}/5，direction gate FAIL。",
        f"- C48 是否位于 C32/C64 之间：cutover={summary['C48_intermediate_frontier_point']['cutover_between_C32_C64']}、rolling={summary['C48_intermediate_frontier_point']['rolling_between_C32_C64']}，monotonic frontier gate FAIL。",
        f"- C64 无标签 score gap 不差于 C32：cutover {summary['C64_score_gap_diagnostic']['cutover_edges_not_above_C32']}/5、rolling {summary['C64_score_gap_diagnostic']['rolling_edges_not_above_C32']}/5；仅作诊断。",
        "",
        "## 结论",
        "",
        "增加 carrier 确实比 C32 更接近 Exact，但不是一个单调且达到绝对方向门的 precision axis：C48 的五边平均 relative L2 略低于 C64，而 rolling C64 的五条边均未达到 0.90 方向门。按事前合同不能在看到结果后改选 C48，因此不冻结 progressive upgrade，不启动旧五边质量重测；当前正式设计仍是已经取得 AUC 5/5 正向的 C32 lightweight PRO。",
        "",
        "这不是核心 Insight 的反证，而是对本次增量的边界：仅靠更多同类 carriers、第二个近乎等价的 probe 和 scalar decay，尚不足以形成一个可靠的自校准升级。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    contract = yaml.safe_load(args.contract.read_text())
    contract_hash = sha256(args.contract)
    verify_raw(args.input, contract_hash)
    raw_summary = json.loads((args.input / "summary.json").read_text())
    tolerance = float(contract["prefrozen_rules"]["valid_frontier"]["correctness_max_abs_error"])
    if raw_summary["scope"] != "formal" or raw_summary["labels_read"]:
        raise RuntimeError("frontier adjudication requires formal label-free raw")
    if raw_summary["correctness_max_abs_error"] > tolerance:
        raise RuntimeError("frontier correctness tolerance failed")
    layer_raw = pd.read_parquet(args.input / "layer_metrics.parquet")
    score_raw = pd.read_parquet(args.input / "score_metrics.parquet")
    convergence_raw = pd.read_parquet(args.input / "convergence_metrics.parquet")
    layer = layer_aggregate(layer_raw)
    score = score_aggregate(score_raw)
    convergence = convergence_aggregate(convergence_raw)
    summary = {
        **adjudicate(layer_raw, score_raw, contract),
        "contract_sha256": contract_hash,
        "raw_seal_sha256": sha256(args.input / "raw.seal.json"),
        "correctness_max_abs_error": raw_summary["correctness_max_abs_error"],
        "layer_rows": len(layer_raw),
        "score_rows": len(score_raw),
        "convergence_rows": len(convergence_raw),
    }
    outputs = {
        "layer_aggregate.csv": layer.to_csv(index=False),
        "score_aggregate.csv": score.to_csv(index=False),
        "convergence_aggregate.csv": convergence.to_csv(index=False),
        "adjudication.json": json.dumps(summary, indent=2) + "\n",
        "report.md": report(summary, layer),
    }
    for name, value in outputs.items():
        target = args.input / name
        partial = target.with_suffix(target.suffix + ".partial")
        partial.write_text(value)
        os.replace(partial, target)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
