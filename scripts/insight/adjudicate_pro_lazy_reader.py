#!/usr/bin/env python3
"""Adjudicate the frozen label-free PRO correctness and cost canary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "configs/contracts/yambda500m_small_hstu_native_pro_lazy_reader_v1.yaml"
RESULT = ROOT / "results/yambda500m_small_seed17/insight_pro_lazy_reader_v1/correctness_cost"

import sys

sys.path.insert(0, str(ROOT / "scripts" / "insight"))

from pro_lazy_cost import report as cost_report  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--result", type=Path, default=RESULT)
    args = parser.parse_args()
    contract = yaml.safe_load(args.contract.read_text())
    seal = json.loads((args.result / "raw.seal.json").read_text())
    raw = args.result / "raw.jsonl"
    if seal["contract_sha256"] != sha256(args.contract) or seal["raw_sha256"] != sha256(raw):
        raise RuntimeError("PRO correctness contract or raw seal differs")
    if seal["labels_read"] or seal["rows"] != 5 * 32 * 2:
        raise RuntimeError("PRO correctness scope differs from the prospective contract")
    frame = pd.read_json(raw, lines=True)
    if frame.labels_read.any() or set(frame.carriers) != {16, 32}:
        raise RuntimeError("PRO raw contents differ from the frozen label-free carrier axis")

    maximum_reference_abs = float(frame.fused_reference_max_abs_error.max())
    maximum_reference_relative = float(frame.fused_reference_relative_l2.max())
    maximum_replay_abs = float(frame.fused_replay_max_abs_error.max())
    if "fused_max_abs_error" in contract["correctness_canary"]:
        tolerance = float(contract["correctness_canary"]["fused_max_abs_error"])
        exact_pass = max(maximum_reference_abs, maximum_replay_abs) <= tolerance
        tolerance_description = {"fused_or_replay_max_abs_error": tolerance}
    else:
        relative_tolerance = float(
            contract["correctness_canary"]["fused_reference_relative_l2"]
        )
        replay_tolerance = float(
            contract["correctness_canary"]["replay_max_abs_error"]
        )
        exact_pass = (
            maximum_reference_relative <= relative_tolerance
            and maximum_replay_abs <= replay_tolerance
        )
        tolerance_description = {
            "fused_reference_relative_l2": relative_tolerance,
            "replay_max_abs_error": replay_tolerance,
        }
    compute = cost_report()
    compute_pass = all(
        row["over_full_fraction"] <= contract["cost_contract"]["maximum_fraction_of_full"]
        for row in compute["budgets"]
    )
    structural_pass = bool(
        (frame.materialized_version_translated_prefix_positions_in_action == 0).all()
        and seal["materialized_version_translated_prefix_positions_in_action"] == 0
    )
    progression_pass = exact_pass and compute_pass and structural_pass

    grouped = []
    for (edge, carriers), group in frame.groupby(["edge", "carriers"], sort=True):
        per_layer = list(zip(*group.legacy_layer_relative_l2, strict=True))
        grouped.append(
            {
                "edge": edge,
                "carriers": int(carriers),
                "users": len(group),
                "fused_reference_max_abs_error": float(group.fused_reference_max_abs_error.max()),
                "legacy_sidecar_direction_cosine_mean": float(
                    group.legacy_sidecar_direction_cosine.mean()
                ),
                "legacy_sidecar_direction_cosine_median": float(
                    group.legacy_sidecar_direction_cosine.median()
                ),
                "legacy_sidecar_norm_ratio_median": float(
                    group.legacy_sidecar_norm_ratio.median()
                ),
                "legacy_sidecar_relative_l2_mean": float(
                    group.legacy_sidecar_relative_l2.mean()
                ),
                "legacy_layer_relative_l2_mean": [
                    float(sum(values) / len(values)) for values in per_layer
                ],
            }
        )

    summary = {
        "status": (
            "pro_lazy_reader_correctness_and_cost_passed"
            if progression_pass
            else "pro_lazy_reader_correctness_or_cost_failed"
        ),
        "contract_sha256": sha256(args.contract),
        "raw_sha256": sha256(raw),
        "labels_read": False,
        "score_or_quality_adjudication_performed": False,
        "progression_gate": {
            "passed": progression_pass,
            "fused_equivalence_passed": exact_pass,
            "maximum_fused_reference_abs_error_retained": maximum_reference_abs,
            "maximum_fused_reference_relative_l2": maximum_reference_relative,
            "maximum_replay_abs_error": maximum_replay_abs,
            "tolerance": tolerance_description,
            "compute_passed": compute_pass,
            "structural_no_materialized_prefix_passed": structural_pass,
        },
        "parameter_map_build_seconds_by_edge": seal[
            "parameter_map_build_seconds_by_edge"
        ],
        "edge_carrier_diagnostics": grouped,
        "next_authorization": "return_to_expert_no_score_or_quality_unlocked",
    }
    (args.result / "theoretical_compute.json").write_text(json.dumps(compute, indent=2) + "\n")
    (args.result / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    budget_rows = {row["carriers"]: row for row in compute["budgets"]}
    lines = [
        "# Lightweight PRO fused-reader correctness and cost",
        "",
        f"Progression gate: **{'PASS' if progression_pass else 'FAIL'}**; labels or request scores read: **no**.",
        "",
        "The action materializes zero version-translated prefix positions. A materialized prefix was used only as the sealed numerical reference for the fused AV identity.",
        "",
        "| carriers | GFLOPs/user | of Full | reduction vs Full | logical Parent read | FP32 sidecar write |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for carriers in (16, 32):
        row = budget_rows[carriers]
        lines.append(
            f"| {carriers} | {row['total_flops_per_user'] / 1e9:.3f} | "
            f"{100 * row['over_full_fraction']:.1f}% | "
            f"{100 * row['reduction_vs_full_fraction']:.1f}% | "
            f"{row['conservative_parent_state_stream_bytes_fp32'] / 2**20:.2f} MiB | "
            f"{row['sidecar_write_bytes_fp32'] / 1024:.1f} KiB |"
        )
    lines.extend(
        [
            "",
            f"Maximum fused-reference absolute error retained: `{maximum_reference_abs:.8g}`; maximum relative L2: `{maximum_reference_relative:.8g}`; maximum replay absolute error: `{maximum_replay_abs:.8g}`.",
            "",
            "Outside the release-time headline, bounded-horizon serving adds 512 coverage-scale multiplies per user request and 512 AV additions per candidate (4 layers x 128 width), plus sidecar I/O.",
            "",
            "| edge | carriers | direction cosine mean | direction cosine median | norm ratio median | relative L2 mean |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in grouped:
        lines.append(
            f"| {row['edge'].replace('_to_', ' -> ')} | {row['carriers']} | "
            f"{row['legacy_sidecar_direction_cosine_mean']:.4f} | "
            f"{row['legacy_sidecar_direction_cosine_median']:.4f} | "
            f"{row['legacy_sidecar_norm_ratio_median']:.4f} | "
            f"{row['legacy_sidecar_relative_l2_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "The legacy-sidecar comparison is diagnostic only: it measures the approximation introduced by Parent-conditioned carriers and does not admit score or task quality. The one-time version-map pseudoinverse and logical state bytes are reported separately from the per-user FLOP headline.",
            "",
        ]
    )
    (args.result / "report.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
