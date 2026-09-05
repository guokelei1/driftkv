#!/usr/bin/env python3
"""Run the fixed UID-1930 paired native-response route eliminator.

The legal constructors are completed before Current Exact is materialized.
Only held-out odd-32 candidates are used for evaluation; no label,
confirmation user, rank sweep, or fitted mapping is available to this run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_yambda500m_foundation_raw import load_histories, load_model  # noqa: E402
from insight_one_locality.common import histories_at_cutover  # noqa: E402

from insight_two.common import (  # noqa: E402
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
    verify_model_payload,
)
from insight_two.common_projection_response import (  # noqa: E402
    build_common_projection_response_memory,
    intervene_common_projection_response,
    medium_common_projection_cost,
)
from insight_two.cone_response_memory import (  # noqa: E402
    intervene_cone_response_memory,
)
from insight_two.mode_space_replay import (  # noqa: E402
    approximate_layer0_defect_basis,
    approximate_paired_layer0_defect_basis,
    factorized_reduced_current_replay,
    paired_release_replay,
    splice_shared_modes_from_factorized_replay,
    splice_shared_modes_from_paired_replay,
)
from insight_two.paired_functional_boundary import (  # noqa: E402
    build_paired_factorized_response_memory,
    medium_functional_boundary_cost,
)
from insight_two.paired_native_response import (  # noqa: E402
    build_paired_native_response_memory,
    intervene_paired_native_response,
    medium_paired_native_response_cost,
)
from insight_two.paired_region_delta import (  # noqa: E402
    trace_history_item_region_queries,
)
from insight_two.run_kv_response_coupling_preflight import (  # noqa: E402
    verify_preflight_inputs,
)

UID = 1930
PAIRED_RANK = 4
SINGLE_RANK = 8
OVERSAMPLE = 4
POWER_ITERATIONS = 1
REPLAY_SEED = 17
DEFECT_RANK = 8
DEFECT_OVERSAMPLE = 4
DEFECT_POWER = 0
DEFECT_SEED = 1017
EXACT_ALL_FLOPS = 4_771_282_944
PAIRED_KV_FLOPS = 874_402_376
SINGLE_R8_FLOPS = 853_836_992
SHARED_MODE_READER_FLOPS = 6 * (202_752 + 6_336)


def _row(
    *,
    edge: str,
    method: str,
    exact: torch.Tensor,
    reuse: torch.Tensor,
    observed: torch.Tensor,
    constructor_flops: int,
    persistent_representation_scalars: int,
    persistent_state_role: str,
    reader_flops_per_query: int,
) -> dict[str, Any]:
    fraction = constructor_flops / EXACT_ALL_FLOPS
    return {
        "edge": edge,
        "uid": UID,
        "method": method,
        "constructor_flops_per_user": constructor_flops,
        "constructor_fraction_of_Exact_All": fraction,
        "within_twenty_percent": fraction <= 0.20,
        "persistent_representation_scalars": persistent_representation_scalars,
        "persistent_state_role": persistent_state_role,
        "incremental_reader_flops_per_query": reader_flops_per_query,
        **metrics_row(score_metrics(exact, reuse, observed)),
    }


def _cost_payload() -> dict[str, Any]:
    native = medium_paired_native_response_cost()
    paired_p8 = medium_functional_boundary_cost(
        "paired_r4_functional_moments", matrix_free_initial=True
    )
    common = medium_common_projection_cost()
    return {
        "paired_r4_native_response": {
            "starting_paired_final_KV_ledger_flops": native.paired_kv_ledger_flops,
            "superseded_shared_basis_flops": native.superseded_shared_basis_flops,
            "superseded_signed_core_flops": native.superseded_signed_core_flops,
            "total_constructor_flops": native.total_constructor_flops,
            "constructor_fraction": native.constructor_fraction,
            "within_twenty_percent": native.within_twenty_percent,
            "sidecar_scalars": native.sidecar_scalars,
            "sidecar_fp32_bytes": native.sidecar_fp32_bytes,
            "factor_reads_per_layer_per_query": (
                native.factor_reads_per_layer_per_query
            ),
            "factor_reads_per_query": native.factor_reads_per_query,
            "logical_factor_sidecar_scalar_reads_per_query": (
                native.logical_factor_sidecar_scalar_reads_per_query
            ),
            "incremental_reader_flops_per_query": (
                native.incremental_reader_flops_per_query
            ),
            "native_activation_evaluations_per_query": (
                native.native_activation_evaluations_per_query
            ),
        },
        "paired_r4_KV_splice": {
            "total_constructor_flops": PAIRED_KV_FLOPS,
            "constructor_fraction": PAIRED_KV_FLOPS / EXACT_ALL_FLOPS,
            "incremental_reader_flops_per_query": SHARED_MODE_READER_FLOPS,
        },
        "paired_r4_P8_S4": {
            "total_constructor_flops": paired_p8.total_constructor_flops,
            "constructor_fraction": paired_p8.constructor_fraction,
            "sidecar_scalars": paired_p8.sidecar_scalars,
            "incremental_reader_flops_per_query": (
                paired_p8.incremental_reader_flops_per_query
            ),
        },
        "common_projection_r8_native_response": {
            "total_constructor_flops": common.total_constructor_flops,
            "constructor_fraction": common.constructor_fraction,
            "incremental_reader_flops_per_query": (
                common.incremental_reader_flops_per_query
            ),
        },
        "single_current_r8_controls": {
            "total_constructor_flops": SINGLE_R8_FLOPS,
            "constructor_fraction": SINGLE_R8_FLOPS / EXACT_ALL_FLOPS,
            "shared_mode_incremental_reader_flops_per_query": (
                SHARED_MODE_READER_FLOPS
            ),
        },
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:2")
    args = parser.parse_args()
    if args.device != "cuda:2":
        raise ValueError("this one-UID preflight is fixed to cuda:2")
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
    costs = _cost_payload()
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
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
        query_delta = torch.as_tensor(
            query_deltas_np, dtype=torch.float32, device=device
        )
        heldout = torch.as_tensor(
            all_candidates[edge_index, 0:1][:, HELDOUT_INDICES],
            dtype=torch.long,
            device=device,
        )

        exact_parent = parent.compute_kv(items, actions, deltas)
        parent_embedded = parent.embed_inputs(items, actions, deltas)
        current_embedded = current.embed_inputs(items, actions, deltas)
        paired = paired_release_replay(
            parent,
            current,
            parent_embedded,
            current_embedded,
            rank=PAIRED_RANK,
            compression="fixed_range_finder",
            sketch_oversample=OVERSAMPLE,
            sketch_power_iterations=POWER_ITERATIONS,
            sketch_seed=REPLAY_SEED,
        )
        source_scalars = exact_parent.k.numel() + exact_parent.v.numel()
        native_memory = build_paired_native_response_memory(
            paired, source_kv_scalars=source_scalars
        )
        paired_basis = approximate_paired_layer0_defect_basis(
            paired,
            rank=DEFECT_RANK,
            oversample=DEFECT_OVERSAMPLE,
            power_iterations=DEFECT_POWER,
            seed=DEFECT_SEED,
        )
        paired_kv = splice_shared_modes_from_paired_replay(
            exact_parent, paired, paired_basis
        )

        probe_queries = trace_history_item_region_queries(
            current,
            exact_parent,
            items,
            query_delta,
            probe_count=8,
        )
        paired_p8 = build_paired_factorized_response_memory(
            current,
            paired,
            probe_queries,
            source_kv_scalars=source_scalars,
        )

        single = factorized_reduced_current_replay(
            current,
            current_embedded,
            rank=SINGLE_RANK,
            compression="fixed_range_finder",
            sketch_oversample=OVERSAMPLE,
            sketch_power_iterations=POWER_ITERATIONS,
            sketch_seed=REPLAY_SEED,
        )
        single_basis = approximate_layer0_defect_basis(
            exact_parent,
            single,
            rank=DEFECT_RANK,
            oversample=DEFECT_OVERSAMPLE,
            power_iterations=DEFECT_POWER,
            seed=DEFECT_SEED,
        )
        single_shared = splice_shared_modes_from_factorized_replay(
            exact_parent, single, single_basis
        )
        common_memory = build_common_projection_response_memory(
            single, exact_parent
        )

        # Evaluation-only Current Exact is unavailable to every constructor.
        exact_current = current.compute_kv(items, actions, deltas)
        exact_scores = current.score_cc_reuse(
            exact_current, heldout, query_delta
        )
        reuse_scores = current.score_cc_reuse(
            exact_parent, heldout, query_delta
        )
        native_scores = intervene_paired_native_response(
            current, exact_parent, native_memory, heldout, query_delta
        ).scores
        paired_p8_scores = intervene_cone_response_memory(
            current, exact_parent, paired_p8, heldout, query_delta
        ).scores
        common_scores = intervene_common_projection_response(
            current, exact_parent, common_memory, heldout, query_delta
        ).scores

        methods = (
            (
                "paired_r4_native_response",
                native_scores,
                costs["paired_r4_native_response"]["total_constructor_flops"],
                native_memory.stored_scalars,
                "two_arm_factor_sidecar_over_exact_Parent",
                costs["paired_r4_native_response"][
                    "incremental_reader_flops_per_query"
                ],
            ),
            (
                "paired_r4_KV_splice",
                current.score_cc_reuse(paired_kv.cache, heldout, query_delta),
                PAIRED_KV_FLOPS,
                paired_kv.sidecar_scalars,
                "shared_mode_sidecar_over_exact_Parent",
                SHARED_MODE_READER_FLOPS,
            ),
            (
                "paired_r4_P8_S4",
                paired_p8_scores,
                costs["paired_r4_P8_S4"]["total_constructor_flops"],
                paired_p8.stored_scalars,
                "signed_functional_sidecar_over_exact_Parent",
                costs["paired_r4_P8_S4"][
                    "incremental_reader_flops_per_query"
                ],
            ),
            (
                "common_projection_r8_native_response",
                common_scores,
                costs["common_projection_r8_native_response"][
                    "total_constructor_flops"
                ],
                common_memory.stored_scalars,
                "two_view_factor_sidecar_over_exact_Parent",
                costs["common_projection_r8_native_response"][
                    "incremental_reader_flops_per_query"
                ],
            ),
            (
                "single_current_r8_reduced_cache",
                current.score_cc_reuse(single.cache, heldout, query_delta),
                SINGLE_R8_FLOPS,
                single.cache.k.numel() + single.cache.v.numel(),
                "replacement_dense_approximate_Current_cache",
                0,
            ),
            (
                "single_current_r8_shared_U0",
                current.score_cc_reuse(
                    single_shared.cache, heldout, query_delta
                ),
                SINGLE_R8_FLOPS,
                single_shared.sidecar_scalars,
                "shared_mode_sidecar_over_exact_Parent",
                SHARED_MODE_READER_FLOPS,
            ),
        )
        edge_rows = [
            _row(
                edge=edge,
                method=method,
                exact=exact_scores,
                reuse=reuse_scores,
                observed=observed,
                constructor_flops=int(constructor_flops),
                persistent_representation_scalars=int(scalars),
                persistent_state_role=role,
                reader_flops_per_query=int(reader_flops),
            )
            for method, observed, constructor_flops, scalars, role, reader_flops in methods
        ]
        rows.extend(edge_rows)
        by_method = {str(row["method"]): row for row in edge_rows}
        edge_diagnostic = {
            "edge": edge,
            "native_minus_single_r8_reduced": (
                by_method["paired_r4_native_response"][
                    "probability_gap_recovery"
                ]
                - by_method["single_current_r8_reduced_cache"][
                    "probability_gap_recovery"
                ]
            ),
            "native_minus_single_r8_shared_U0": (
                by_method["paired_r4_native_response"][
                    "probability_gap_recovery"
                ]
                - by_method["single_current_r8_shared_U0"][
                    "probability_gap_recovery"
                ]
            ),
            "native_minus_paired_KV": (
                by_method["paired_r4_native_response"][
                    "probability_gap_recovery"
                ]
                - by_method["paired_r4_KV_splice"][
                    "probability_gap_recovery"
                ]
            ),
        }
        diagnostics.append(edge_diagnostic)
        print(
            json.dumps(
                {
                    "edge": edge,
                    "probability_recovery": {
                        str(row["method"]): row["probability_gap_recovery"]
                        for row in edge_rows
                    },
                    "diagnostic": edge_diagnostic,
                }
            ),
            flush=True,
        )
        del parent, current, exact_parent, exact_current, paired, single
        torch.cuda.empty_cache()

    recoveries: dict[str, list[float]] = {}
    for row in rows:
        recoveries.setdefault(str(row["method"]), []).append(
            float(row["probability_gap_recovery"])
        )
    means = {
        method: float(np.mean(values)) for method, values in recoveries.items()
    }
    native_values = recoveries["paired_r4_native_response"]
    single_values = recoveries["single_current_r8_reduced_cache"]
    shared_values = recoveries["single_current_r8_shared_U0"]
    native_wins_reduced = sum(
        native > single for native, single in zip(native_values, single_values, strict=True)
    )
    native_wins_shared = sum(
        native > shared for native, shared in zip(native_values, shared_values, strict=True)
    )
    passes_scientific_gate = (
        means["paired_r4_native_response"]
        > max(
            means["single_current_r8_reduced_cache"],
            means["single_current_r8_shared_U0"],
        )
        and native_wins_reduced >= 3
        and native_wins_shared >= 3
    )
    summary = {
        "status": "nonformal_single_uid_route_elimination_complete",
        "uid": UID,
        "device": args.device,
        "candidate_source": "heldout_odd32_only",
        "probe_source_for_P8_control_only": "fixed_history_lower_midpoint_P8",
        "primary_uses_probes": False,
        "primary_uses_mapping_or_fit": False,
        "labels_read": False,
        "confirmation_read": False,
        "current_exact_used_only_after_legal_construction": True,
        "input_verification": verification,
        "configuration": {
            "paired_rank_per_arm": PAIRED_RANK,
            "single_control_rank": SINGLE_RANK,
            "oversample": OVERSAMPLE,
            "power_iterations": POWER_ITERATIONS,
            "replay_seed": REPLAY_SEED,
            "no_parameter_sweep": True,
        },
        "costs": costs,
        "probability_recovery_by_method": recoveries,
        "probability_recovery_edge_mean": means,
        "native_response_wins_over_single_reduced_edges": native_wins_reduced,
        "native_response_wins_over_single_shared_edges": native_wins_shared,
        "scientific_gate": {
            "rule": (
                "native edge mean must beat both single-r8 controls and win "
                "at least 3/5 edges against each"
            ),
            "passed": passes_scientific_gate,
            "decision": "retain_for_formal_canary" if passes_scientific_gate else "retire",
        },
        "edge_diagnostics": diagnostics,
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
