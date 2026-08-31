#!/usr/bin/env python3
"""Adjudicate the frozen reader-stage and cross-request persistence gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "configs/contracts/yambda500m_small_hstu_native_reader_compatibility_correction_v1.yaml"
RESULT = ROOT / "results/yambda500m_small_seed17/insight_reader_compatibility_correction_v1"
CONTROLLED = RESULT / "formal_controlled_stage"
REAL = RESULT / "formal_real"
OUTPUT = RESULT / "adjudication"
STAGES = (
    "kv_prefix_contribution",
    "av_aggregation",
    "u_gated_update",
    "layer_hidden",
    "final_readout",
)
POST_AGGREGATION = (
    "av_aggregation",
    "u_gated_update",
    "layer_hidden",
    "final_readout",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _aggregate_stage(score: pd.DataFrame, energy: pd.DataFrame, source: str) -> pd.DataFrame:
    score_agg = (
        score[score.stage != "reuse"]
        .groupby(["edge", "stage"], as_index=False)
        .agg(
            mean_stage_gap=("mean_abs_probability_gap", "mean"),
            mean_reuse_gap=("reuse_probability_gap", "mean"),
        )
    )
    score_agg["mean_gap_recovery"] = 1.0 - (
        score_agg.mean_stage_gap / score_agg.mean_reuse_gap.clip(lower=1e-12)
    )
    energy_agg = (
        energy[energy.total_energy > 1e-20]
        .groupby(["edge", "stage"], as_index=False)
        .agg(mean_shared_energy=("signed_shared_energy_fraction", "mean"))
    )
    output = score_agg.merge(energy_agg, on=["edge", "stage"], validate="one_to_one")
    output.insert(0, "source", source)
    return output


def _bucket_report(pairs: pd.DataFrame, stage: str) -> pd.DataFrame:
    selected = pairs[pairs.stage == stage].copy()
    selected["time_bucket"] = pd.cut(
        selected.seconds_between_requests,
        bins=[-1, 3600, 21600, 86400, 259200, float("inf")],
        labels=["0_1h", "1_6h", "6_24h", "1_3d", "gt_3d"],
    )
    selected["append_bucket"] = pd.cut(
        selected.append_count_difference,
        bins=[-1, 0, 1, 4, 16, float("inf")],
        labels=["0", "1", "2_4", "5_16", "ge_17"],
    )
    selected["remaining_old_fraction"] = selected.current_remaining_old_positions / 512.0
    selected["remaining_bucket"] = pd.cut(
        selected.remaining_old_fraction,
        bins=[-1e-12, 0, 0.25, 0.5, 0.75, 1.0],
        labels=["0", "0_25pct", "25_50pct", "50_75pct", "75_100pct"],
        include_lowest=True,
    )
    frames = []
    for dimension in ("time_bucket", "append_bucket", "remaining_bucket"):
        grouped = (
            selected.groupby(["edge", dimension], observed=False, as_index=False)
            .agg(
                pairs=("uid", "size"),
                median_cosine=("adjacent_request_direction_cosine", "median"),
                median_same_request_recovery=("same_request_gap_recovery", "median"),
                median_prior_recovery=("prior_request_gap_recovery", "median"),
                median_scaled_prior_recovery=(
                    "coverage_scaled_prior_gap_recovery",
                    "median",
                ),
            )
            .rename(columns={dimension: "bucket"})
        )
        grouped.insert(1, "dimension", dimension)
        frames.append(grouped)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    contract = yaml.safe_load(CONTRACT.read_text())
    controlled_summary = json.loads((CONTROLLED / "summary.json").read_text())
    real_summary = json.loads((REAL / "summary.json").read_text())
    contract_hash = sha256(CONTRACT)
    if (
        controlled_summary["status"] != "reader_correction_formal_controlled_passed"
        or controlled_summary["contract_sha256"] != contract_hash
    ):
        raise RuntimeError("formal controlled stage result is not sealed")
    if (
        real_summary["status"] != "reader_correction_real_formal_passed"
        or real_summary["contract_sha256"] != contract_hash
    ):
        raise RuntimeError("formal real-request result is not sealed")

    controlled_score = pd.read_parquet(CONTROLLED / "score_interventions.parquet")
    controlled_energy = pd.read_parquet(CONTROLLED / "stage_energy.parquet")
    real_score = pd.read_parquet(REAL / "stage_score_all_edges.parquet")
    real_energy = pd.read_parquet(REAL / "stage_energy_all_edges.parquet")
    pairs = pd.read_parquet(REAL / "persistence_all_edges.parquet")
    stage = pd.concat(
        [
            _aggregate_stage(controlled_score, controlled_energy, "controlled"),
            _aggregate_stage(real_score, real_energy, "real_exposed"),
        ],
        ignore_index=True,
    )
    energy_threshold = contract["stage_localization"]["earliest_stable_boundary_gate"][
        "shared_energy_fraction_at_least"
    ]
    recovery_threshold = contract["stage_localization"]["earliest_stable_boundary_gate"][
        "same_request_probability_gap_recovery_at_least"
    ]
    stage["passes_edge"] = (
        (stage.mean_shared_energy >= energy_threshold)
        & (stage.mean_gap_recovery >= recovery_threshold)
    )
    counts = (
        stage.groupby(["source", "stage"], as_index=False)
        .passes_edge.sum()
        .rename(columns={"passes_edge": "passing_edges"})
    )
    controlled_min = contract["stage_localization"]["earliest_stable_boundary_gate"][
        "controlled_edges_minimum"
    ]
    real_min = contract["stage_localization"]["earliest_stable_boundary_gate"][
        "real_exposed_edges_minimum"
    ]
    earliest = None
    for candidate in STAGES:
        controlled_count = int(
            counts[(counts.source == "controlled") & (counts.stage == candidate)].passing_edges.iloc[0]
        )
        real_count = int(
            counts[(counts.source == "real_exposed") & (counts.stage == candidate)].passing_edges.iloc[0]
        )
        if controlled_count >= controlled_min and real_count >= real_min:
            earliest = candidate
            break
    stage_passed = earliest is not None
    persistence_stage = None
    if stage_passed:
        persistence_stage = next(
            candidate for candidate in POST_AGGREGATION if STAGES.index(candidate) >= STAGES.index(earliest)
        )

    persistence = (
        pairs.groupby(["edge", "stage"], as_index=False)
        .agg(
            pairs=("uid", "size"),
            users=("uid", "nunique"),
            median_cosine=("adjacent_request_direction_cosine", "median"),
            median_norm_ratio=("current_to_previous_norm_ratio", "median"),
            median_same_request_recovery=("same_request_gap_recovery", "median"),
            median_prior_recovery=("prior_request_gap_recovery", "median"),
            median_scaled_prior_recovery=(
                "coverage_scaled_prior_gap_recovery", "median"
            ),
        )
    )
    persistence_thresholds = contract["cross_request_persistence"][
        "persistence_gate_after_stage_gate"
    ]
    if persistence_stage is not None:
        focus = persistence[persistence.stage == persistence_stage].copy()
        focus["passes_edge"] = (
            (
                focus.median_cosine
                >= persistence_thresholds["median_adjacent_request_cosine_at_least"]
            )
            & (
                focus.median_scaled_prior_recovery
                >= persistence_thresholds[
                    "median_coverage_scaled_prior_gap_recovery_at_least"
                ]
            )
        )
        persistence_edges = int(focus.passes_edge.sum())
        bucket = _bucket_report(pairs, persistence_stage)
    else:
        focus = persistence.iloc[:0].copy()
        persistence_edges = 0
        bucket = pd.DataFrame()
    persistence_passed = stage_passed and persistence_edges >= persistence_thresholds[
        "edges_passing_both_minimum"
    ]
    mechanism_unlocked = stage_passed and persistence_passed
    if not stage_passed:
        status = "reader_stage_gate_failed"
    elif not persistence_passed:
        status = "reader_persistence_gate_failed"
    else:
        status = "reader_stage_and_persistence_gates_passed"

    OUTPUT.mkdir(parents=True)
    stage.to_csv(OUTPUT / "stage_by_edge.csv", index=False)
    counts.to_csv(OUTPUT / "stage_gate_counts.csv", index=False)
    persistence.to_csv(OUTPUT / "persistence_by_edge.csv", index=False)
    bucket.to_csv(OUTPUT / "persistence_buckets.csv", index=False)
    summary = {
        "status": status,
        "contract_sha256": contract_hash,
        "labels_read": False,
        "stage_gate": {
            "passed": stage_passed,
            "earliest_stable_boundary": earliest,
            "shared_energy_threshold": energy_threshold,
            "same_request_recovery_threshold": recovery_threshold,
            "controlled_edges_minimum": controlled_min,
            "real_edges_minimum": real_min,
        },
        "persistence_gate": {
            "passed": persistence_passed,
            "evaluated_stage": persistence_stage,
            "passing_edges": persistence_edges,
            "required_edges": persistence_thresholds["edges_passing_both_minimum"],
            "median_cosine_threshold": persistence_thresholds[
                "median_adjacent_request_cosine_at_least"
            ],
            "median_scaled_recovery_threshold": persistence_thresholds[
                "median_coverage_scaled_prior_gap_recovery_at_least"
            ],
            "full_history_only_reason": (
                "the frozen coverage rule starts from 512 old positions; short-cutover "
                "users remain in same-request stage observation but not persistence pairs"
            ),
        },
        "mechanism_canary_unlocked": mechanism_unlocked,
        "inputs": {
            "controlled_summary": sha256(CONTROLLED / "summary.json"),
            "controlled_score": sha256(CONTROLLED / "score_interventions.parquet"),
            "controlled_energy": sha256(CONTROLLED / "stage_energy.parquet"),
            "real_summary": sha256(REAL / "summary.json"),
            "real_score": sha256(REAL / "stage_score_all_edges.parquet"),
            "real_energy": sha256(REAL / "stage_energy_all_edges.parquet"),
            "persistence": sha256(REAL / "persistence_all_edges.parquet"),
        },
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# Reader compatibility-correction adjudication",
        "",
        f"Status: **{status}**. No label was read.",
        "",
        "This adjudicates a reader compatibility correction, not a materializable history evidence basis. The K/V-prefix stage is already query-dependent (`activated(qK)·V`), so an early boundary there does not imply raw K/V linear substitutability.",
        "",
        "## Stage gate",
        "",
        f"Earliest frozen stable boundary: **{earliest or 'none'}**.",
        "",
        "| source | stage | passing edges |",
        "| --- | --- | ---: |",
    ]
    for row in counts.itertuples(index=False):
        lines.append(f"| {row.source} | {row.stage} | {int(row.passing_edges)} |")
    lines.extend(
        [
            "",
            "## Cross-request persistence gate",
            "",
            f"Evaluated stage: **{persistence_stage or 'not reached'}**; passing edges: **{persistence_edges}/5**.",
            "",
            "| edge | pairs | users | median cosine | same-request recovery | prior recovery | coverage-scaled prior recovery | pass |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in focus.itertuples(index=False):
        lines.append(
            f"| {row.edge} | {int(row.pairs)} | {int(row.users)} | {row.median_cosine:.6f} | "
            f"{row.median_same_request_recovery:.6f} | {row.median_prior_recovery:.6f} | "
            f"{row.median_scaled_prior_recovery:.6f} | {'PASS' if row.passes_edge else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            f"Matched-cost layerwise broadcast-residual canary unlocked: **{'yes' if mechanism_unlocked else 'no'}**.",
            "",
            "All fixed time, append-count and remaining-old-state buckets are retained in `persistence_buckets.csv`.",
        ]
    )
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
