#!/usr/bin/env python3
"""Run the fixed UID-1930/five-edge activation-boundary preflight.

Only the first frozen discovery UID and held-out odd-32 candidate panel are
read.  Labels and the confirmation population remain unread.  Exact endpoint
state is used only by the graph diagnostic and serving oracle; the recursive
boundary replay receives raw history, Parent K/V, and the Current model only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_yambda500m_foundation_raw import load_histories, load_model  # noqa: E402
from insight_one_locality.common import histories_at_cutover  # noqa: E402

from insight_two.activation_boundary_replay import (  # noqa: E402
    build_no_target_boundary_replay_cache,
    intervene_serving_boundary_delta,
    medium_activation_boundary_cost_audit,
    trace_exact_endpoint_graphs,
)
from insight_two.common import (  # noqa: E402
    CONTRACT,
    CUTOVER_DAYS,
    DATASET,
    DAY,
    EDGES,
    HELDOUT_INDICES,
    KNOWN_ITEMS,
    OOV_BUCKETS,
    checkpoint,
    load_frozen_inputs,
    metrics_row,
    score_metrics,
    sha256_file,
    verify_model_payload,
)

UID = 1930
RESULT_DIR = (
    ROOT
    / "results/yambda500m_medium_seed17/insight2_functional_boundary_v1"
    / "activation_boundary_replay_preflight"
)


def verify_preflight_inputs() -> dict[str, str]:
    """Verify immutable inputs while allowing only the living-plan hash drift."""

    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    if contract["scope"]["edges"] != list(EDGES):
        raise RuntimeError("contract edge order differs")
    frozen = contract["frozen_inputs"]
    plan = frozen["research_plan"]
    plan_path = ROOT / plan["path"]
    if not plan_path.is_file():
        raise RuntimeError("living research plan is missing")
    plan_actual = sha256_file(plan_path)
    for name, record in frozen.items():
        if name in {"research_plan", "checkpoints"}:
            continue
        path = ROOT / record["path"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"frozen preflight input differs: {name}")
    for version in range(6):
        record = frozen["checkpoints"][f"v{version}"]
        path = ROOT / record["path"]
        if path != checkpoint(version):
            raise RuntimeError(f"checkpoint path differs for v{version}")
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"checkpoint v{version} differs from contract")
    return {
        "expected_research_plan_sha256": str(plan["sha256"]),
        "actual_research_plan_sha256": plan_actual,
        "research_plan_hash_matches_old_contract": str(plan_actual == plan["sha256"]).lower(),
        "immutable_frozen_inputs_verified": "true",
    }


def _method_row(
    edge: str,
    method: str,
    exact: torch.Tensor,
    reuse: torch.Tensor,
    observed: torch.Tensor,
    *,
    current_exact_used: bool,
) -> dict[str, Any]:
    return {
        "edge": edge,
        "uid": UID,
        "method": method,
        "current_exact_used_by_construction": current_exact_used,
        **metrics_row(score_metrics(exact, reuse, observed)),
    }


def _mean_by_method(rows: list[dict[str, Any]], field: str) -> dict[str, float]:
    methods = sorted({str(row["method"]) for row in rows})
    return {
        method: float(np.mean([float(row[field]) for row in rows if row["method"] == method]))
        for method in methods
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:3")
    args = parser.parse_args()
    if args.device != "cuda:3":
        raise ValueError("this single-UID preflight is fixed to cuda:3")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_num_threads(4)

    verification = verify_preflight_inputs()
    all_uids, all_candidates, _ = load_frozen_inputs()
    if int(all_uids[0]) != UID:
        raise RuntimeError("the first frozen discovery UID is no longer 1930")
    history = load_histories(
        [UID],
        oov_buckets=OOV_BUCKETS,
        dataset_path=DATASET,
        known_vocab_size=KNOWN_ITEMS,
        end_timestamp=(CUTOVER_DAYS[-1] + 1) * DAY,
        threads=4,
    )
    rows: list[dict[str, Any]] = []
    graph_rows: list[dict[str, Any]] = []
    replay_graph_rows: list[dict[str, Any]] = []
    serving_graph_rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for edge_index, edge in enumerate(EDGES):
        parent, parent_payload = load_model(checkpoint(edge_index), device)
        current, current_payload = load_model(checkpoint(edge_index + 1), device)
        verify_model_payload(parent_payload)
        verify_model_payload(current_payload)
        _, items_np, actions_np, deltas_np, query_deltas_np = histories_at_cutover(
            history,
            np.asarray([UID], dtype=np.int64),
            CUTOVER_DAYS[edge_index] * DAY,
        )
        items = torch.as_tensor(items_np, dtype=torch.long, device=device)
        actions = torch.as_tensor(actions_np, dtype=torch.long, device=device)
        deltas = torch.as_tensor(deltas_np, dtype=torch.float32, device=device)
        query_delta = torch.as_tensor(query_deltas_np, dtype=torch.float32, device=device)
        heldout = torch.as_tensor(
            all_candidates[edge_index, 0:1][:, HELDOUT_INDICES],
            dtype=torch.long,
            device=device,
        )

        trace = trace_exact_endpoint_graphs(parent, current, items, actions, deltas)
        exact_scores = current.score_cc_reuse(trace.current_cache, heldout, query_delta)
        reuse_scores = current.score_cc_reuse(trace.parent_cache, heldout, query_delta)
        boundary_oracle = intervene_serving_boundary_delta(
            current,
            trace.current_cache,
            trace.parent_cache,
            heldout,
            query_delta,
            mode="crossing_delta_only",
        )
        unchanged_oracle = intervene_serving_boundary_delta(
            current,
            trace.current_cache,
            trace.parent_cache,
            heldout,
            query_delta,
            mode="unchanged_delta_only",
        )
        no_target_replay = build_no_target_boundary_replay_cache(
            current, trace.parent_cache, items, actions, deltas
        )
        no_target_scores = current.score_cc_reuse(no_target_replay.cache, heldout, query_delta)

        edge_rows = [
            _method_row(
                edge,
                "crossing_delta_serving_oracle",
                exact_scores,
                reuse_scores,
                boundary_oracle.scores,
                current_exact_used=True,
            ),
            _method_row(
                edge,
                "unchanged_delta_serving_oracle",
                exact_scores,
                reuse_scores,
                unchanged_oracle.scores,
                current_exact_used=True,
            ),
            _method_row(
                edge,
                "no_target_recursive_boundary_replay",
                exact_scores,
                reuse_scores,
                no_target_scores,
                current_exact_used=False,
            ),
        ]
        rows.extend(edge_rows)
        graph_rows.extend(
            {"edge": edge, "uid": UID, **asdict(metric)} for metric in trace.layer_metrics
        )
        replay_graph_rows.extend(
            {
                "edge": edge,
                "uid": UID,
                "layer": layer,
                "matched_query_crossing_fraction": fraction,
            }
            for layer, fraction in enumerate(no_target_replay.crossing_fraction_by_active_layer)
        )
        serving_graph_rows.extend(
            {
                "edge": edge,
                "uid": UID,
                "layer": layer,
                "matched_current_query_crossing_fraction": fraction,
            }
            for layer, fraction in enumerate(boundary_oracle.crossing_fraction_by_layer)
        )
        print(
            json.dumps(
                {
                    "edge": edge,
                    "probability_recovery": {
                        str(row["method"]): row["probability_gap_recovery"] for row in edge_rows
                    },
                    "historical_endpoint_crossing_fraction": [
                        metric.activation_region_crossing_fraction for metric in trace.layer_metrics
                    ],
                }
            ),
            flush=True,
        )
        del parent, current, trace, no_target_replay
        torch.cuda.empty_cache()

    cost = medium_activation_boundary_cost_audit()
    probability_means = _mean_by_method(rows, "probability_gap_recovery")
    endpoint_crossing_mean = float(
        np.mean([float(row["activation_region_crossing_fraction"]) for row in graph_rows])
    )
    endpoint_agreement_mean = float(
        np.mean([float(row["activation_region_agreement"]) for row in graph_rows])
    )
    crossing_response_recovery_mean = float(
        np.mean([float(row["crossing_delta_response_gap_recovery"]) for row in graph_rows])
    )
    graph_by_layer: list[dict[str, float | int]] = []
    for layer in range(6):
        selected = [row for row in graph_rows if int(row["layer"]) == layer]
        graph_by_layer.append(
            {
                "layer": layer,
                "activation_region_crossing_fraction": float(
                    np.mean([float(row["activation_region_crossing_fraction"]) for row in selected])
                ),
                "activation_change_l1_on_crossings": float(
                    np.mean([float(row["activation_change_l1_on_crossings"]) for row in selected])
                ),
                "finite_response_delta_crossing_over_joint": float(
                    np.mean(
                        [
                            float(row["finite_response_delta_crossing_over_joint"])
                            for row in selected
                        ]
                    )
                ),
                "crossing_delta_response_gap_recovery": float(
                    np.mean(
                        [float(row["crossing_delta_response_gap_recovery"]) for row in selected]
                    )
                ),
            }
        )

    summary = {
        "status": "nonformal_single_uid_activation_boundary_preflight_complete",
        "uid": UID,
        "device": args.device,
        "candidate_source": "heldout_odd32_only",
        "labels_read": False,
        "confirmation_read": False,
        "rank_or_sampling_used": False,
        "input_verification": verification,
        "elapsed_seconds": time.perf_counter() - started,
        "probability_recovery_by_method_and_edge": {
            method: [
                float(row["probability_gap_recovery"]) for row in rows if row["method"] == method
            ]
            for method in sorted({str(row["method"]) for row in rows})
        },
        "probability_recovery_edge_mean": probability_means,
        "historical_endpoint_graph": {
            "activation_region_agreement_mean": endpoint_agreement_mean,
            "activation_region_crossing_fraction_mean": endpoint_crossing_mean,
            "crossing_delta_response_gap_recovery_mean": crossing_response_recovery_mean,
            "edge_equal_layer_means": graph_by_layer,
        },
        "no_target_replay_crossing_fraction_mean": float(
            np.mean([float(row["matched_query_crossing_fraction"]) for row in replay_graph_rows])
        ),
        "serving_oracle_crossing_fraction_mean": float(
            np.mean(
                [
                    float(row["matched_current_query_crossing_fraction"])
                    for row in serving_graph_rows
                ]
            )
        ),
        "strict_cost_audit": cost.to_dict(),
        "adjudication": {
            "interaction_graph_change_set_is_small": endpoint_crossing_mean <= 0.10,
            "crossing_only_serving_oracle_reaches_80pct": probability_means[
                "crossing_delta_serving_oracle"
            ]
            >= 0.80,
            "no_target_replay_reaches_80pct": probability_means[
                "no_target_recursive_boundary_replay"
            ]
            >= 0.80,
            "strict_constructor_within_20pct": cost.within_twenty_percent_before_response_or_projection,
            "design1_admitted": False,
            "reason": (
                "exact graph discovery QK floor exceeds 20% before response, "
                "projection, gate, or write cost; oracle quality cannot admit "
                "the interaction-boundary object"
            ),
        },
        "rows": rows,
        "historical_graph_rows": graph_rows,
        "no_target_replay_graph_rows": replay_graph_rows,
        "serving_graph_rows": serving_graph_rows,
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULT_DIR / "summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
