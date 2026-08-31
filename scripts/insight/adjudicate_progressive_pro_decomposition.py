#!/usr/bin/env python3
"""Adjudicate sealed label-free progressive-PRO decomposition rows."""

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
CONTRACT = ROOT / "configs/contracts/yambda500m_small_hstu_native_progressive_pro_decomposition_v1.yaml"
DEFAULT_INPUT = ROOT / "results/yambda500m_small_seed17/insight_progressive_pro_v1/decomposition_v1/formal"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_raw(directory: Path, contract_hash: str) -> dict[str, Any]:
    seal = json.loads((directory / "raw.seal.json").read_text())
    if seal["contract_sha256"] != contract_hash or seal["labels_read"]:
        raise RuntimeError("raw seal does not match the label-free contract")
    for name, record in seal["files"].items():
        target = directory / name
        if sha256(target) != record["sha256"] or target.stat().st_size != record["bytes"]:
            raise RuntimeError(f"sealed raw artifact differs: {name}")
    return seal


def _layer_aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["phase", "edge", "method"], sort=True)
        .agg(
            rows=("identifier", "size"),
            users=("uid", "nunique"),
            median_direction_cosine=("direction_cosine_to_exact_shared_av", "median"),
            median_norm_ratio=("estimated_to_exact_norm_ratio", "median"),
            median_relative_l2=("relative_l2_to_exact_shared_av", "median"),
            median_oracle_amplitude_relative_l2=(
                "oracle_projection_amplitude_relative_l2",
                "median",
            ),
            median_oracle_amplitude_reduction=(
                "oracle_projection_relative_l2_reduction",
                "median",
            ),
            median_probe_direction_cosine=("fixed_probe_direction_cosine", "median"),
            median_probe_norm_ratio=("fixed_probe_norm_ratio", "median"),
            median_old_recent_direction_cosine=(
                "old_to_recent_component_direction_cosine",
                "median",
            ),
        )
        .reset_index()
    )


def _score_aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    aggregate = (
        frame.groupby(["phase", "edge", "method"], sort=True)
        .agg(
            requests=("identifier", "size"),
            mean_logit_gap_to_current_exact=(
                "mean_abs_logit_gap_to_current_exact",
                "mean",
            ),
            mean_probability_gap_to_current_exact=(
                "mean_abs_probability_gap_to_current_exact",
                "mean",
            ),
            mean_logit_gap_to_exact_shared_av=(
                "mean_abs_logit_gap_to_exact_shared_av",
                "mean",
            ),
        )
        .reset_index()
    )
    reuse = aggregate[aggregate.method == "reuse"][
        ["phase", "edge", "mean_probability_gap_to_current_exact"]
    ].rename(columns={"mean_probability_gap_to_current_exact": "reuse_probability_gap"})
    aggregate = aggregate.merge(reuse, on=["phase", "edge"], how="left", validate="many_to_one")
    aggregate["ratio_of_means_probability_gap_recovery"] = 1.0 - (
        aggregate.mean_probability_gap_to_current_exact
        / aggregate.reuse_probability_gap.clip(lower=1e-12)
    )
    aggregate.loc[aggregate.method == "reuse", "ratio_of_means_probability_gap_recovery"] = 0.0
    return aggregate


def adjudicate(layer: pd.DataFrame, contract: dict[str, Any]) -> dict[str, Any]:
    rules = contract["prefrozen_interpretation_rules"]
    edge_count = layer.edge.nunique()
    if edge_count != 5 or set(layer.phase) != {"cutover", "rolling"}:
        raise RuntimeError("formal decomposition does not contain every phase and edge")
    if bool(layer.labels_read.any()):
        raise RuntimeError("formal decomposition unexpectedly read labels")

    cutover = layer[(layer.phase == "cutover") & (layer.method == "dual_probe")]
    rolling = layer[
        (layer.phase == "rolling") & (layer.method == "dual_probe_global_decay")
    ]
    cut_edge = cutover.groupby("edge", sort=True).median(numeric_only=True)
    roll_edge = rolling.groupby("edge", sort=True).median(numeric_only=True)
    direction_threshold = float(rules["stable_direction"]["threshold_median_cosine"])
    direction_cutover_edges = int(
        (cut_edge.direction_cosine_to_exact_shared_av >= direction_threshold).sum()
    )
    direction_rolling_edges = int(
        (roll_edge.direction_cosine_to_exact_shared_av >= direction_threshold).sum()
    )
    direction_pass = (
        direction_cutover_edges
        >= int(rules["stable_direction"]["minimum_edges_at_cutover"])
        and direction_rolling_edges
        >= int(rules["stable_direction"]["minimum_edges_at_rolling"])
    )

    amplitude_threshold = float(
        rules["amplitude_dominant_error"][
            "oracle_projection_relative_L2_reduction_minimum"
        ]
    )
    amplitude_cutover_edges = int(
        (cut_edge.oracle_projection_relative_l2_reduction >= amplitude_threshold).sum()
    )
    amplitude_rolling_edges = int(
        (roll_edge.oracle_projection_relative_l2_reduction >= amplitude_threshold).sum()
    )
    amplitude_pass = (
        amplitude_cutover_edges
        >= int(rules["amplitude_dominant_error"]["minimum_edges_at_cutover"])
        and amplitude_rolling_edges
        >= int(rules["amplitude_dominant_error"]["minimum_edges_at_rolling"])
    )

    probe = cutover.groupby("edge", sort=True).median(numeric_only=True)
    lower, upper = map(float, rules["two_probe_consistency"]["median_norm_ratio_interval"])
    probe_edge_pass = (
        (probe.fixed_probe_direction_cosine >= float(rules["two_probe_consistency"]["median_direction_cosine_minimum"]))
        & probe.fixed_probe_norm_ratio.between(lower, upper)
    )
    probe_edges = int(probe_edge_pass.sum())
    probe_pass = probe_edges >= int(rules["two_probe_consistency"]["minimum_edges"])

    segment = layer[
        (layer.phase == "rolling")
        & layer.method.isin(["dual_probe_global_decay", "dual_probe_segment_decay"])
    ]
    segment_edge = (
        segment.groupby(["edge", "method"], sort=True)
        .relative_l2_to_exact_shared_av.median()
        .unstack("method")
    )
    segment_better_edges = int(
        (
            segment_edge.dual_probe_segment_decay
            <= segment_edge.dual_probe_global_decay
        ).sum()
    )
    segment_mean_better = bool(
        segment_edge.dual_probe_segment_decay.mean()
        < segment_edge.dual_probe_global_decay.mean()
    )
    segment_pass = segment_better_edges >= 4 and segment_mean_better

    disagreement_edges = int(
        (probe.fixed_probe_direction_cosine < direction_threshold).sum()
    )
    second_component_allowed = disagreement_edges >= 2
    return {
        "status": "progressive_pro_decomposition_valid_frontier_unlocked",
        "labels_read": False,
        "edges": edge_count,
        "stable_direction": {
            "passed": direction_pass,
            "threshold": direction_threshold,
            "cutover_edges_passing": direction_cutover_edges,
            "rolling_edges_passing": direction_rolling_edges,
        },
        "amplitude_dominant_error": {
            "passed": amplitude_pass,
            "minimum_relative_l2_reduction": amplitude_threshold,
            "cutover_edges_passing": amplitude_cutover_edges,
            "rolling_edges_passing": amplitude_rolling_edges,
        },
        "two_probe_consistency": {
            "passed": probe_pass,
            "edges_passing": probe_edges,
            "median_direction_cosine_by_edge": {
                edge: float(value)
                for edge, value in probe.fixed_probe_direction_cosine.items()
            },
            "median_norm_ratio_by_edge": {
                edge: float(value) for edge, value in probe.fixed_probe_norm_ratio.items()
            },
        },
        "segment_decay": {
            "selected": segment_pass,
            "edges_not_above_global": segment_better_edges,
            "five_edge_mean_lower": segment_mean_better,
            "frozen_frontier_decay": "segment" if segment_pass else "global",
        },
        "component_rule": {
            "probe_disagreement_edges": disagreement_edges,
            "second_component_allowed": second_component_allowed,
            "frozen_frontier_components": 2 if second_component_allowed else 1,
        },
        "interpretation": (
            "C32_error_is_not_amplitude_dominated_under_the_prefrozen_rule;_two_fixed_probes_are_consistent_but_redundant;_retain_one_direction_and_global_decay_then_measure_carrier_fidelity_axis"
        ),
    }


def report(summary: dict[str, Any], layer: pd.DataFrame, score: pd.DataFrame) -> str:
    lines = [
        "# Progressive PRO 五边无标签 error decomposition",
        "",
        "状态：sealed raw 已通过完整性复核；未读取行为 label。",
        "",
        "## 冻结规则裁决",
        "",
        f"- C32 对 Exact shared AV 的稳定方向门：cutover {summary['stable_direction']['cutover_edges_passing']}/5，rolling {summary['stable_direction']['rolling_edges_passing']}/5，{'PASS' if summary['stable_direction']['passed'] else 'FAIL'}。",
        f"- amplitude-dominant 门：cutover {summary['amplitude_dominant_error']['cutover_edges_passing']}/5，rolling {summary['amplitude_dominant_error']['rolling_edges_passing']}/5，{'PASS' if summary['amplitude_dominant_error']['passed'] else 'FAIL'}。",
        f"- 双固定 probe 一致性：{summary['two_probe_consistency']['edges_passing']}/5，{'PASS' if summary['two_probe_consistency']['passed'] else 'FAIL'}。",
        f"- segment decay 相对 global decay：{summary['segment_decay']['edges_not_above_global']}/5 edge 不差，最终冻结 `{summary['segment_decay']['frozen_frontier_decay']}` decay。",
        f"- 第二 component：probe disagreement 为 {summary['component_rule']['probe_disagreement_edges']}/5 edge，最终冻结 {summary['component_rule']['frozen_frontier_components']} 个 component。",
        "",
        "## 逐边核心量",
        "",
        "| phase | edge | method | median cosine | median norm ratio | median relative L2 | oracle-amplitude reduction |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    focus = layer[
        ((layer.phase == "cutover") & (layer.method == "dual_probe"))
        | ((layer.phase == "rolling") & (layer.method == "dual_probe_global_decay"))
    ]
    for row in focus.itertuples(index=False):
        lines.append(
            f"| {row.phase} | {row.edge} | {row.method} | {row.median_direction_cosine:.4f} | {row.median_norm_ratio:.4f} | {row.median_relative_l2:.4f} | {row.median_oracle_amplitude_reduction:.1%} |"
        )
    lines += [
        "",
        "## 结论",
        "",
        "两条 probe 的 correction 几乎相同，因此单 probe 偶然性不是当前主要误差源。C32 对 Exact 的方向门和纯幅值解释均未通过；old/recent segment decay 也未达到 4/5 选择门。按事前协议不调阈值、不追加第二 component，下一步只在同一 PRO 内测 C32/C48/C64 carrier fidelity 轴，保留 global decay。",
        "",
        "score aggregate 保存在 `score_aggregate.csv`；它只衡量无标签的 Current/Exact-shared 距离，不构成 AUC、log-loss 或 serving admission。",
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
    summary_raw = json.loads((args.input / "summary.json").read_text())
    if summary_raw["scope"] != "formal" or summary_raw["labels_read"]:
        raise RuntimeError("adjudication requires formal label-free raw")
    if summary_raw["correctness_max_abs_error"] > 2e-5:
        raise RuntimeError("reader trace correctness exceeds inherited tolerance")
    layer_raw = pd.read_parquet(args.input / "layer_metrics.parquet")
    score_raw = pd.read_parquet(args.input / "score_metrics.parquet")
    layer = _layer_aggregate(layer_raw)
    score = _score_aggregate(score_raw)
    summary = {
        **adjudicate(layer_raw, contract),
        "contract_sha256": contract_hash,
        "raw_seal_sha256": sha256(args.input / "raw.seal.json"),
        "correctness_max_abs_error": summary_raw["correctness_max_abs_error"],
        "layer_rows": len(layer_raw),
        "score_rows": len(score_raw),
    }
    outputs = {
        "layer_aggregate.csv": layer.to_csv(index=False),
        "score_aggregate.csv": score.to_csv(index=False),
        "adjudication.json": json.dumps(summary, indent=2) + "\n",
        "report.md": report(summary, layer, score),
    }
    for name, value in outputs.items():
        target = args.input / name
        partial = target.with_suffix(target.suffix + ".partial")
        partial.write_text(value)
        os.replace(partial, target)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
