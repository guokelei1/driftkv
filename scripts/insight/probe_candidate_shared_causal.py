#!/usr/bin/env python3
"""Prospective signed/head-wise causal canary for candidate-shared HSTU evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "insight"))

from candidate_shared_causal import (  # noqa: E402
    nested_width_indices,
    signed_head_intervention,
)
from evaluate_yambda500m_foundation_raw import load_histories, load_model  # noqa: E402
from hstu_kvcache.evaluation import (  # noqa: E402
    append_timestamp_group,
    materialize_state,
    timestamp_groups,
)
from probe_recommendation_state_structure import (  # noqa: E402
    CUTOVER_DAYS,
    DAY,
    HISTORY,
    KNOWN_ITEMS,
    candidate_panel,
    checkpoint,
    histories_at_cutover,
)


CONTRACT = ROOT / "configs/contracts/yambda500m_small_hstu_native_candidate_shared_causal_v1.yaml"
POPULATION = (
    ROOT
    / "results/yambda500m_small_seed17/insight_recommendation_state_structure_v1/population.parquet"
)
FIDELITY = (
    ROOT
    / "data/manifests/yambda500m_small_hstu_native_rolling_matrix_fast_v3/requests_fidelity.parquet"
)
DEFAULT_OUTPUT = (
    ROOT / "results/yambda500m_small_seed17/insight_candidate_shared_causal_v1/canary"
)
FORMAL_CONTROLLED_OUTPUT = (
    ROOT
    / "results/yambda500m_small_seed17/insight_candidate_shared_causal_v1/formal_controlled"
)
CONTROLLED_WIDTHS = (64, 32, 16, 8)
EXPOSED_WIDTHS = (16, 8, 4, 2)
PATHS = ("reuse", "shared_only", "residual_only", "full_delta")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_contract() -> dict[str, Any]:
    contract = yaml.safe_load(CONTRACT.read_text())
    frozen = contract["frozen_inputs"]
    for name in (
        "source_observation_contract",
        "population",
        "matrix_manifest",
        "requests_fidelity",
        "requests_quality",
    ):
        record = frozen[name]
        path = ROOT / record["path"]
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"frozen {name} differs from the prospective contract")
    for version in range(6):
        record = frozen["checkpoints"][f"v{version}"]
        path = ROOT / record["path"]
        if path != checkpoint(version) or sha256(path) != record["sha256"]:
            raise RuntimeError(f"frozen v{version} checkpoint differs from the contract")
    return contract


def rank_correlation(reference: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    ref_rank = torch.argsort(torch.argsort(reference, dim=1), dim=1).float()
    other_rank = torch.argsort(torch.argsort(other, dim=1), dim=1).float()
    ref_rank -= ref_rank.mean(dim=1, keepdim=True)
    other_rank -= other_rank.mean(dim=1, keepdim=True)
    return (ref_rank * other_rank).sum(dim=1) / torch.sqrt(
        ref_rank.square().sum(dim=1) * other_rank.square().sum(dim=1)
    ).clamp_min(1e-12)


def score_rows(
    *,
    edge: str,
    bank_source: str,
    identifiers: np.ndarray,
    width: int,
    exact_scores: torch.Tensor,
    reuse_scores: torch.Tensor,
    path_scores: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    exact_probability = torch.sigmoid(exact_scores.float())
    reuse_gap = torch.abs(torch.sigmoid(reuse_scores.float()) - exact_probability).mean(dim=1)
    exact_top1 = exact_scores.argmax(dim=1)
    topk = min(10, width)
    exact_topk = exact_scores.topk(topk, dim=1).indices
    output = []
    for path, scores in path_scores.items():
        gap = torch.abs(torch.sigmoid(scores.float()) - exact_probability).mean(dim=1)
        selected_topk = scores.topk(topk, dim=1).indices
        overlap = (
            (selected_topk[:, :, None] == exact_topk[:, None, :])
            .any(dim=2)
            .sum(dim=1)
            .float()
            / float(topk)
        )
        correlation = rank_correlation(exact_scores, scores)
        for index, identifier in enumerate(identifiers):
            output.append(
                {
                    "edge": edge,
                    "bank_source": bank_source,
                    "identifier": str(identifier),
                    "width": width,
                    "path": path,
                    "mean_abs_logit_gap": float(
                        torch.abs(scores[index] - exact_scores[index]).mean()
                    ),
                    "mean_abs_probability_gap": float(gap[index]),
                    "gap_recovery_over_reuse": float(
                        1.0 - gap[index] / reuse_gap[index].clamp_min(1e-12)
                    ),
                    "top1_agreement": int(scores[index].argmax() == exact_top1[index]),
                    "top10_or_bank_overlap": float(overlap[index]),
                    "rank_correlation": float(correlation[index]),
                    "reuse_probability_gap": float(reuse_gap[index]),
                }
            )
    return output


def head_rows(
    *,
    edge: str,
    bank_source: str,
    identifiers: np.ndarray,
    width: int,
    exact_result,
    reference_shared: tuple[torch.Tensor, ...],
) -> list[dict[str, Any]]:
    output = []
    for layer, (metrics, shared, reference) in enumerate(
        zip(
            exact_result.head_metrics,
            exact_result.shared_components,
            reference_shared,
            strict=True,
        )
    ):
        cosine = torch.nn.functional.cosine_similarity(shared, reference, dim=-1)
        for user_index, identifier in enumerate(identifiers):
            for head in range(shared.shape[1]):
                output.append(
                    {
                        "edge": edge,
                        "bank_source": bank_source,
                        "identifier": str(identifier),
                        "width": width,
                        "layer": layer,
                        "head": head,
                        "signed_shared_energy_fraction": float(
                            metrics["shared_energy_fraction"][user_index, head]
                        ),
                        "signed_residual_energy_fraction": float(
                            metrics["residual_energy"][user_index, head]
                            / metrics["total_energy"][user_index, head].clamp_min(1e-20)
                        ),
                        "orthogonality_error": float(
                            metrics["orthogonality_error"][user_index, head]
                        ),
                        "shared_direction_cosine_to_largest_width": float(
                            cosine[user_index, head]
                        ),
                    }
                )
    return output


@torch.inference_mode()
def evaluate_bank(
    *,
    current,
    exact_cache,
    reuse_cache,
    candidates: torch.Tensor,
    query_deltas: torch.Tensor,
    edge: str,
    bank_source: str,
    identifiers: np.ndarray,
    width: int,
    reference_shared: tuple[torch.Tensor, ...] | None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    tuple[torch.Tensor, ...],
    dict[str, float],
    dict[str, torch.Tensor],
]:
    native_exact, _ = current.observe_cc_reuse(exact_cache, candidates, query_deltas)
    native_reuse, _ = current.observe_cc_reuse(reuse_cache, candidates, query_deltas)
    exact_result = signed_head_intervention(
        current, exact_cache, reuse_cache, candidates, query_deltas, mode="exact"
    )
    path_results = {
        path: signed_head_intervention(
            current, exact_cache, reuse_cache, candidates, query_deltas, mode=path
        )
        for path in ("shared_only", "residual_only", "full_delta")
    }
    path_scores = {"reuse": native_reuse}
    path_scores.update({name: result.scores for name, result in path_results.items()})
    correctness = {
        "native_exact": float(torch.max(torch.abs(native_exact - exact_result.scores))),
        "native_reuse": float(
            torch.max(
                torch.abs(
                    native_reuse
                    - signed_head_intervention(
                        current,
                        exact_cache,
                        reuse_cache,
                        candidates,
                        query_deltas,
                        mode="reuse",
                    ).scores
                )
            )
        ),
        "full_delta": float(
            torch.max(torch.abs(native_exact - path_results["full_delta"].scores))
        ),
    }
    if reference_shared is None:
        reference_shared = exact_result.shared_components
    emitted_scores = {"current_exact": native_exact, **path_scores}
    return (
        score_rows(
            edge=edge,
            bank_source=bank_source,
            identifiers=identifiers,
            width=width,
            exact_scores=native_exact,
            reuse_scores=native_reuse,
            path_scores=path_scores,
        ),
        head_rows(
            edge=edge,
            bank_source=bank_source,
            identifiers=identifiers,
            width=width,
            exact_result=exact_result,
            reference_shared=reference_shared,
        ),
        reference_shared,
        correctness,
        emitted_scores,
    )


def exposed_canary_groups(population_uids: set[int]) -> dict[str, dict[str, Any]]:
    table = pq.read_table(
        FIDELITY,
        columns=[
            "request_id",
            "uid",
            "query_timestamp",
            "item_idx",
            "time_block",
            "target_known",
        ],
    ).to_pandas()
    table = table[
        (table["time_block"] == "matrix_horizon")
        & table["target_known"]
        & table["uid"].astype(np.int64).isin(population_uids)
    ]
    output = {}
    for edge, cutover_day in enumerate(CUTOVER_DAYS):
        start, stop = cutover_day * DAY, (cutover_day + 14) * DAY
        rows = table[
            (table["query_timestamp"] >= start) & (table["query_timestamp"] < stop)
        ]
        sizes = (
            rows.groupby(["uid", "query_timestamp"], sort=True)
            .size()
            .reset_index(name="size")
        )
        eligible = sizes[sizes["size"] >= max(EXPOSED_WIDTHS)].sort_values(
            ["query_timestamp", "uid"]
        )
        if eligible.empty:
            raise RuntimeError(f"no exposed canary group for v{edge}_to_v{edge + 1}")
        selected = eligible.iloc[0]
        group = rows[
            (rows["uid"] == selected.uid)
            & (rows["query_timestamp"] == selected.query_timestamp)
        ].sort_values(["item_idx", "request_id"])
        output[f"v{edge}_to_v{edge + 1}"] = {
            "uid": int(selected.uid),
            "query_timestamp": int(selected.query_timestamp),
            "items": group["item_idx"].to_numpy(dtype=np.int64)[: max(EXPOSED_WIDTHS)],
            "request_ids": group["request_id"].astype(str).to_numpy()[: max(EXPOSED_WIDTHS)],
            "observed_group_size": int(selected["size"]),
        }
    return output


def rolling_caches_for_exposed_group(history, uid: int, query_time: int, cutover: int, parent, current):
    timestamps, items, behaviors = history.rows[uid]
    events = [
        (int(timestamp), int(item), int(behavior))
        for timestamp, item, behavior in zip(timestamps, items, behaviors, strict=True)
    ]
    prefix = [event for event in events if event[0] < cutover]
    if len(prefix) < HISTORY:
        raise RuntimeError("exposed canary user lacks a full cutover history")
    exact = materialize_state(
        current, prefix, producer_version="current", max_length=HISTORY
    )
    reuse = materialize_state(
        parent, prefix, producer_version="parent", max_length=HISTORY
    )
    for timestamp, group in timestamp_groups(
        event for event in events if cutover <= event[0] < query_time
    ):
        del timestamp
        exact = append_timestamp_group(
            current, exact, group, producer_version="current", max_length=HISTORY
        )
        reuse = append_timestamp_group(
            current, reuse, group, producer_version="current", max_length=HISTORY
        )
    if exact.last_timestamp != reuse.last_timestamp or query_time <= exact.last_timestamp:
        raise RuntimeError("exposed rolling states do not share a strict query boundary")
    return exact.cache, reuse.cache, float(query_time - exact.last_timestamp)


def aggregate_gate(score_frame: pd.DataFrame, contract: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    focus = score_frame[
        (score_frame.bank_source == "controlled") & (score_frame.width == 64)
    ]
    aggregate = (
        focus.groupby(["edge", "path"], sort=True)
        .agg(
            mean_probability_gap=("mean_abs_probability_gap", "mean"),
            mean_gap_recovery=("gap_recovery_over_reuse", "mean"),
        )
        .reset_index()
    )
    pivot_gap = aggregate.pivot(
        index="edge", columns="path", values="mean_probability_gap"
    )
    pivot_recovery = aggregate.pivot(
        index="edge", columns="path", values="mean_gap_recovery"
    )
    shared_better = int(
        (pivot_gap["shared_only"] < pivot_gap["residual_only"]).sum()
    )
    shared_positive = int((pivot_recovery["shared_only"] > 0.0).sum())
    frozen = contract["focused_canary"]["progression_gate"]
    passed = (
        shared_better
        >= frozen["controlled_width64_shared_gap_better_than_residual_gap_edges_minimum"]
        and shared_positive
        >= frozen["controlled_width64_shared_gap_recovery_positive_edges_minimum"]
    )
    gate = {
        "passed": passed,
        "shared_gap_better_than_residual_edges": shared_better,
        "shared_gap_recovery_positive_edges": shared_positive,
        "required_shared_better_edges": frozen[
            "controlled_width64_shared_gap_better_than_residual_gap_edges_minimum"
        ],
        "required_shared_positive_edges": frozen[
            "controlled_width64_shared_gap_recovery_positive_edges_minimum"
        ],
    }
    return aggregate, gate


def render_report(
    summary: dict[str, Any],
    aggregate: pd.DataFrame,
    head_frame: pd.DataFrame,
) -> str:
    focus = head_frame[head_frame.width == head_frame.groupby("bank_source").width.transform("max")]
    head_summary = (
        focus.groupby(["bank_source", "edge"], sort=True)
        .agg(
            signed_shared_energy_fraction=("signed_shared_energy_fraction", "mean"),
            orthogonality_error=("orthogonality_error", "max"),
        )
        .reset_index()
    )

    def table(frame: pd.DataFrame) -> list[str]:
        columns = list(frame.columns)
        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
        for row in frame.itertuples(index=False):
            values = []
            for value in row:
                values.append(f"{value:.8g}" if isinstance(value, float) else str(value))
            lines.append("| " + " | ".join(values) + " |")
        return lines

    return "\n".join(
        [
            "# Candidate-shared signed causal canary",
            "",
            "This canary uses signed, per-head HSTU prefix reads without candidate-wise normalization. The shared/residual paths are oracle diagnostic interventions, not executable cache actions.",
            "",
            f"Progression gate: **{'PASS' if summary['progression_gate']['passed'] else 'FAIL'}**.",
            "",
            "## Controlled width-64 score-gap intervention",
            "",
            *table(aggregate),
            "",
            "## Signed head decomposition at the largest bank width",
            "",
            *table(head_summary),
            "",
            f"Maximum native trace error: {summary['correctness']['native_score_max_abs_error']:.8g}.",
            f"Maximum full-delta reconstruction error: {summary['correctness']['full_delta_reconstruction_max_abs_error']:.8g}.",
            "No label was read during the canary.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scope", choices=("canary", "formal-controlled"), default="canary"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.output is None:
        args.output = DEFAULT_OUTPUT if args.scope == "canary" else FORMAL_CONTROLLED_OUTPUT
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    started = time.time()
    contract = verify_contract()
    population = pd.read_parquet(POPULATION)
    if len(population) != contract["controlled_population"]["formal_users"]:
        raise RuntimeError("frozen controlled population count differs from the contract")
    uids = population.uid.to_numpy(dtype=np.int64)
    if args.scope == "formal-controlled":
        canary_summary_path = DEFAULT_OUTPUT / "summary.json"
        if not canary_summary_path.exists():
            raise RuntimeError("formal controlled observation requires the focused canary")
        canary_summary = json.loads(canary_summary_path.read_text())
        if (
            not canary_summary["progression_gate"]["passed"]
            or canary_summary["contract_sha256"] != sha256(CONTRACT)
        ):
            raise RuntimeError("focused canary did not unlock formal controlled observation")
        selected_uids = uids
        exposed = {}
    else:
        selected_uids = uids[: contract["focused_canary"]["controlled_users"]]
        exposed = exposed_canary_groups(set(map(int, uids)))

    device = torch.device(args.device)
    probe, payload = load_model(checkpoint(1), device)
    oov_buckets = int(payload["config"]["num_items"]) - KNOWN_ITEMS
    del probe
    torch.cuda.empty_cache()
    history = load_histories(uids.tolist(), oov_buckets=oov_buckets)

    score_records: list[dict[str, Any]] = []
    head_records: list[dict[str, Any]] = []
    correctness_records: list[dict[str, Any]] = []
    exposed_audit = {}

    for edge, cutover_day in enumerate(CUTOVER_DAYS):
        edge_name = f"v{edge}_to_v{edge + 1}"
        print(json.dumps({"phase": "edge_start", "edge": edge_name}), flush=True)
        cutover = cutover_day * DAY
        _, all_items, _, _, _ = histories_at_cutover(history, uids, cutover)
        full_panel, _, _ = candidate_panel(all_items)
        _, item_np, action_np, delta_np, query_delta_np = histories_at_cutover(
            history, selected_uids, cutover
        )
        panel_np = full_panel[: len(selected_uids)]
        parent, _ = load_model(checkpoint(edge), device)
        current, _ = load_model(checkpoint(edge + 1), device)

        for start in range(0, len(selected_uids), args.batch_size):
            stop = min(start + args.batch_size, len(selected_uids))
            items = torch.as_tensor(item_np[start:stop], dtype=torch.long, device=device)
            actions = torch.as_tensor(action_np[start:stop], dtype=torch.long, device=device)
            deltas = torch.as_tensor(delta_np[start:stop], dtype=torch.float32, device=device)
            query_deltas = torch.as_tensor(
                query_delta_np[start:stop], dtype=torch.float32, device=device
            )
            exact_cache = current.compute_kv(items, actions, deltas)
            reuse_cache = parent.compute_kv(items, actions, deltas)
            identifiers = selected_uids[start:stop]
            reference_shared = None
            for width in CONTROLLED_WIDTHS:
                indices = nested_width_indices(64, width)
                candidates = torch.as_tensor(
                    panel_np[start:stop, indices], dtype=torch.long, device=device
                )
                rows, heads, reference_shared, correctness, _ = evaluate_bank(
                    current=current,
                    exact_cache=exact_cache,
                    reuse_cache=reuse_cache,
                    candidates=candidates,
                    query_deltas=query_deltas,
                    edge=edge_name,
                    bank_source="controlled",
                    identifiers=identifiers,
                    width=width,
                    reference_shared=reference_shared,
                )
                score_records.extend(rows)
                head_records.extend(heads)
                correctness_records.append(
                    {"edge": edge_name, "source": "controlled", "width": width, **correctness}
                )

        if args.scope == "canary":
            group = exposed[edge_name]
            exact_cache, reuse_cache, query_delta = rolling_caches_for_exposed_group(
                history,
                group["uid"],
                group["query_timestamp"],
                cutover,
                parent,
                current,
            )
            full_candidates = group["items"]
            identifiers = np.asarray(
                [f"{group['uid']}:{group['query_timestamp']}"]
            )
            exposed_audit[edge_name] = {
                "uid": group["uid"],
                "query_timestamp": group["query_timestamp"],
                "observed_group_size": group["observed_group_size"],
            }
            reference_shared = None
            for width in EXPOSED_WIDTHS:
                indices = nested_width_indices(16, width)
                candidates = torch.as_tensor(
                    full_candidates[indices][None, :], dtype=torch.long, device=device
                )
                query_deltas = torch.tensor([query_delta], dtype=torch.float32, device=device)
                rows, heads, reference_shared, correctness, _ = evaluate_bank(
                    current=current,
                    exact_cache=exact_cache,
                    reuse_cache=reuse_cache,
                    candidates=candidates,
                    query_deltas=query_deltas,
                    edge=edge_name,
                    bank_source="real_exposed_canary",
                    identifiers=identifiers,
                    width=width,
                    reference_shared=reference_shared,
                )
                score_records.extend(rows)
                head_records.extend(heads)
                correctness_records.append(
                    {"edge": edge_name, "source": "real_exposed_canary", "width": width, **correctness}
                )

        del parent, current
        torch.cuda.empty_cache()
        print(json.dumps({"phase": "edge_complete", "edge": edge_name}), flush=True)

    score_frame = pd.DataFrame(score_records)
    head_frame = pd.DataFrame(head_records)
    correctness_frame = pd.DataFrame(correctness_records)
    aggregate, progression_gate = aggregate_gate(score_frame, contract)
    max_native = float(
        correctness_frame[["native_exact", "native_reuse"]].to_numpy().max()
    )
    max_full = float(correctness_frame["full_delta"].max())
    limits = contract["focused_canary"]["correctness"]
    correctness_passed = (
        max_native <= limits["native_score_max_abs_error"]
        and max_full <= limits["full_delta_reconstruction_max_abs_error"]
    )
    progression_gate["passed"] = bool(progression_gate["passed"] and correctness_passed)
    summary = {
        "status": (
            ("signed_causal_canary_passed" if progression_gate["passed"] else "signed_causal_canary_failed")
            if args.scope == "canary"
            else ("formal_controlled_signed_causal_passed" if progression_gate["passed"] else "formal_controlled_signed_causal_failed")
        ),
        "contract": str(CONTRACT.relative_to(ROOT)),
        "contract_sha256": sha256(CONTRACT),
        "scope": args.scope,
        "controlled_users": len(selected_uids),
        "controlled_edges": 5,
        "controlled_widths": list(CONTROLLED_WIDTHS),
        "real_exposed_groups": 5 if args.scope == "canary" else 0,
        "real_exposed_widths": list(EXPOSED_WIDTHS) if args.scope == "canary" else [],
        "labels_read": False,
        "correctness": {
            "passed": correctness_passed,
            "native_score_max_abs_error": max_native,
            "full_delta_reconstruction_max_abs_error": max_full,
        },
        "progression_gate": progression_gate,
        "exposed_group_audit": exposed_audit,
        "elapsed_seconds": time.time() - started,
    }
    partial = args.output.with_name(args.output.name + ".partial")
    partial.mkdir(parents=True)
    score_frame.to_parquet(partial / "score_interventions.parquet", index=False)
    head_frame.to_parquet(partial / "signed_head_metrics.parquet", index=False)
    correctness_frame.to_csv(partial / "correctness.csv", index=False)
    aggregate.to_csv(partial / "controlled_width64_gate.csv", index=False)
    (partial / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (partial / "report.md").write_text(render_report(summary, aggregate, head_frame))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(partial, args.output)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
