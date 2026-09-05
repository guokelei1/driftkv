#!/usr/bin/env python3
"""Run the fixed UID-1930 source-residual-closure route eliminator.

The constructor sees exact Parent K/V but never Current Exact state.  The sole
primary configuration is paired rank4/rank4 with four source-response tests at
each non-terminal layer.  Its deletion control uses exactly the same initial
factors, rank, compression, and native reader without internal source closure.
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
from insight_two.defect_first_replay import (  # noqa: E402
    absolute_replay_from_initial_factors,
    factorized_cache_scalars,
    forward_one_with_factorized_cache,
    forward_one_with_native_response_defect,
    native_response_sidecar_scalars,
)
from insight_two.matrix_free_input_range import (  # noqa: E402
    hstu_input_operator,
    matrix_free_randomized_token_factors,
)
from insight_two.mode_space_replay import (  # noqa: E402
    approximate_layer0_defect_basis,
    splice_shared_modes_from_factorized_replay,
)
from insight_two.run_kv_response_coupling_preflight import (  # noqa: E402
    verify_preflight_inputs,
)
from insight_two.source_residual_closure import (  # noqa: E402
    medium_source_defect_closure_cost,
    medium_source_residual_closure_cost,
    source_defect_closed_replay_from_initial_factors,
    source_residual_closed_replay_from_initial_factors,
)

UID = 1930
PAIRED_RANK = 4
SINGLE_RANK = 8
OVERSAMPLE = 4
POWER_ITERATIONS = 1
SKETCH_SEED = 17
DEFECT_RANK = 8
DEFECT_OVERSAMPLE = 4
DEFECT_POWER = 0
DEFECT_SEED = 1017
EXACT_ALL_FLOPS = 4_771_282_944
PAIRED_NATIVE_FLOPS = 872_238_088
SINGLE_R8_FLOPS = 853_836_992


@torch.inference_mode()
def _score_native_difference(
    model,
    exact_parent,
    approximate_parent,
    approximate_current,
    candidates: torch.Tensor,
    query_delta: torch.Tensor,
) -> torch.Tensor:
    queries = model.embed_query_tokens(candidates, query_delta)
    scores = []
    for candidate in range(candidates.shape[1]):
        readout = forward_one_with_native_response_defect(
            model,
            exact_parent,
            approximate_parent,
            approximate_current,
            queries[:, candidate : candidate + 1],
        )
        scores.append(model.cc_score_head(readout[:, 0]).squeeze(-1))
    return torch.stack(scores, dim=1)


@torch.inference_mode()
def _score_factorized_current(
    model,
    approximate_current,
    candidates: torch.Tensor,
    query_delta: torch.Tensor,
) -> torch.Tensor:
    queries = model.embed_query_tokens(candidates, query_delta)
    scores = []
    for candidate in range(candidates.shape[1]):
        readout = forward_one_with_factorized_cache(
            model,
            approximate_current,
            queries[:, candidate : candidate + 1],
        )
        scores.append(model.cc_score_head(readout[:, 0]).squeeze(-1))
    return torch.stack(scores, dim=1)


def _row(
    *,
    edge: str,
    method: str,
    exact: torch.Tensor,
    reuse: torch.Tensor,
    observed: torch.Tensor,
    constructor_flops: int,
    sidecar_scalars: int,
) -> dict[str, Any]:
    fraction = constructor_flops / EXACT_ALL_FLOPS
    return {
        "edge": edge,
        "uid": UID,
        "method": method,
        "constructor_flops_per_user": constructor_flops,
        "constructor_fraction_of_Exact_All": fraction,
        "within_twenty_percent": fraction <= 0.20,
        "persistent_sidecar_scalars_fp32": sidecar_scalars,
        **metrics_row(score_metrics(exact, reuse, observed)),
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:3")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
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
    primary_cost = medium_source_defect_closure_cost()
    absolute_residual_cost = medium_source_residual_closure_cost()
    if not primary_cost.within_twenty_percent:
        raise RuntimeError("source certificate exceeds the fixed 20% gate")

    rows: list[dict[str, Any]] = []
    certificate_rows: list[dict[str, Any]] = []
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
        parent_operator = hstu_input_operator(parent, items, actions, deltas)
        current_operator = hstu_input_operator(current, items, actions, deltas)
        parent4 = matrix_free_randomized_token_factors(
            parent_operator,
            rank=PAIRED_RANK,
            oversample=OVERSAMPLE,
            power_iterations=POWER_ITERATIONS,
            seed=SKETCH_SEED,
        )
        current4 = matrix_free_randomized_token_factors(
            current_operator,
            rank=PAIRED_RANK,
            oversample=OVERSAMPLE,
            power_iterations=POWER_ITERATIONS,
            seed=SKETCH_SEED,
        )
        source_defect_closed = source_defect_closed_replay_from_initial_factors(
            parent,
            current,
            exact_parent,
            parent4,
            current4,
            rank=PAIRED_RANK,
            sketch_oversample=OVERSAMPLE,
            sketch_power_iterations=POWER_ITERATIONS,
            sketch_seed=SKETCH_SEED,
        )
        source_absolute_closed = source_residual_closed_replay_from_initial_factors(
            parent,
            current,
            exact_parent,
            parent4,
            current4,
            rank=PAIRED_RANK,
            sketch_oversample=OVERSAMPLE,
            sketch_power_iterations=POWER_ITERATIONS,
            sketch_seed=SKETCH_SEED,
        )
        # Hard deletion control: same initial factors, models, numerical
        # resolution and native reader; only exact-source residual closure is
        # absent.
        paired_parent = absolute_replay_from_initial_factors(
            parent,
            parent4,
            rank=PAIRED_RANK,
            sketch_oversample=OVERSAMPLE,
            sketch_power_iterations=POWER_ITERATIONS,
            sketch_seed=SKETCH_SEED,
        )
        paired_current = absolute_replay_from_initial_factors(
            current,
            current4,
            rank=PAIRED_RANK,
            sketch_oversample=OVERSAMPLE,
            sketch_power_iterations=POWER_ITERATIONS,
            sketch_seed=SKETCH_SEED,
        )
        current8 = matrix_free_randomized_token_factors(
            current_operator,
            rank=SINGLE_RANK,
            oversample=OVERSAMPLE,
            power_iterations=POWER_ITERATIONS,
            seed=SKETCH_SEED,
        )
        single = absolute_replay_from_initial_factors(
            current,
            current8,
            rank=SINGLE_RANK,
            sketch_oversample=OVERSAMPLE,
            sketch_power_iterations=POWER_ITERATIONS,
            sketch_seed=SKETCH_SEED,
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

        primary_scores = _score_native_difference(
            current,
            exact_parent,
            source_defect_closed.paired.parent,
            source_defect_closed.paired.current,
            heldout,
            query_delta,
        )
        absolute_residual_scores = _score_native_difference(
            current,
            exact_parent,
            source_absolute_closed.paired.parent,
            source_absolute_closed.paired.current,
            heldout,
            query_delta,
        )
        deletion_scores = _score_native_difference(
            current,
            exact_parent,
            paired_parent,
            paired_current,
            heldout,
            query_delta,
        )
        single_scores = _score_factorized_current(
            current, single, heldout, query_delta
        )
        single_shared_scores = current.score_cc_reuse(
            single_shared.cache, heldout, query_delta
        )

        # Current Exact is evaluation-only and is intentionally materialized
        # after all constructors and scores above have been frozen.
        exact_current = current.compute_kv(items, actions, deltas)
        exact_scores = current.score_cc_reuse(exact_current, heldout, query_delta)
        reuse_scores = current.score_cc_reuse(exact_parent, heldout, query_delta)
        methods = (
            (
                "source_defect_closed_r4_r4_native",
                primary_scores,
                primary_cost.total_constructor_flops,
                native_response_sidecar_scalars(
                    source_defect_closed.paired.parent,
                    source_defect_closed.paired.current,
                ),
            ),
            (
                "source_absolute_residual_closed_r4_r4_negative_control",
                absolute_residual_scores,
                absolute_residual_cost.total_constructor_flops,
                native_response_sidecar_scalars(
                    source_absolute_closed.paired.parent,
                    source_absolute_closed.paired.current,
                ),
            ),
            (
                "paired_r4_r4_native_deletion_control",
                deletion_scores,
                PAIRED_NATIVE_FLOPS,
                native_response_sidecar_scalars(paired_parent, paired_current),
            ),
            (
                "single_current_r8_factorized_control",
                single_scores,
                SINGLE_R8_FLOPS,
                factorized_cache_scalars(single),
            ),
            (
                "single_current_r8_shared_U0_control",
                single_shared_scores,
                SINGLE_R8_FLOPS,
                single_shared.sidecar_scalars,
            ),
        )
        edge_rows = [
            _row(
                edge=edge,
                method=method,
                exact=exact_scores,
                reuse=reuse_scores,
                observed=observed,
                constructor_flops=flops,
                sidecar_scalars=scalars,
            )
            for method, observed, flops, scalars in methods
        ]
        rows.extend(edge_rows)
        for layer, certificate in enumerate(source_defect_closed.certificates):
            certificate_rows.append(
                {
                    "edge": edge,
                    "layer": layer,
                    "positions": [int(value) for value in certificate.positions],
                    "interpolation_condition": certificate.interpolation_condition,
                    "interpolation_max_abs_error": (
                        certificate.interpolation_max_abs_error
                    ),
                    "sampled_source_residual_l2": float(
                        torch.linalg.vector_norm(
                            certificate.sampled_residual.float()
                        )
                    ),
                    "lifted_source_residual_l2": float(
                        torch.linalg.vector_norm(
                            certificate.lifted_residual.float()
                        )
                    ),
                }
            )
        print(
            json.dumps(
                {
                    "edge": edge,
                    "probability_recovery": {
                        str(row["method"]): row["probability_gap_recovery"]
                        for row in edge_rows
                    },
                    "certificate_conditions": [
                        certificate.interpolation_condition
                        for certificate in source_defect_closed.certificates
                    ],
                }
            ),
            flush=True,
        )
        del parent, current, exact_parent, exact_current
        if device.type == "cuda":
            torch.cuda.empty_cache()

    recoveries: dict[str, list[float]] = {}
    for row in rows:
        recoveries.setdefault(str(row["method"]), []).append(
            float(row["probability_gap_recovery"])
        )
    means = {
        method: float(np.mean(values)) for method, values in recoveries.items()
    }
    primary = recoveries["source_defect_closed_r4_r4_native"]
    deletion = recoveries["paired_r4_r4_native_deletion_control"]
    single = recoveries["single_current_r8_factorized_control"]
    single_shared = recoveries["single_current_r8_shared_U0_control"]
    wins_deletion = sum(
        left > right for left, right in zip(primary, deletion, strict=True)
    )
    wins_single = sum(
        left > right for left, right in zip(primary, single, strict=True)
    )
    wins_single_shared = sum(
        left > right for left, right in zip(primary, single_shared, strict=True)
    )
    passes = (
        all(value > 0.0 and value >= 0.80 for value in primary)
        and means["source_defect_closed_r4_r4_native"]
        > max(
            means["paired_r4_r4_native_deletion_control"],
            means["single_current_r8_factorized_control"],
            means["single_current_r8_shared_U0_control"],
        )
        and wins_deletion >= 3
        and wins_single >= 3
        and wins_single_shared >= 3
    )
    summary = {
        "status": "nonformal_single_uid_source_residual_closure_complete",
        "uid": UID,
        "device": args.device,
        "candidate_source": "heldout_odd_32_only",
        "confirmation_read": False,
        "labels_read": False,
        "current_exact_role": "evaluation_target_materialized_after_construction",
        "input_verification": verification,
        "configuration": {
            "paired_rank_per_arm": PAIRED_RANK,
            "source_tests_per_nonterminal_layer": PAIRED_RANK,
            "interpolation": "DEIM rows of carried Parent normalized-state trial space",
            "certified_object": "finite Current-minus-Parent block-update equation",
            "absolute_parent_residual_transport": False,
            "oversample": OVERSAMPLE,
            "power_iterations": POWER_ITERATIONS,
            "seed": SKETCH_SEED,
            "no_parameter_or_layer_sweep": True,
        },
        "cost": primary_cost.to_dict(),
        "absolute_source_residual_negative_control_cost": (
            absolute_residual_cost.to_dict()
        ),
        "probability_recovery_by_method": recoveries,
        "probability_recovery_edge_mean": means,
        "primary_wins_over_deletion_edges": wins_deletion,
        "primary_wins_over_single_r8_edges": wins_single,
        "primary_wins_over_single_r8_shared_U0_edges": wins_single_shared,
        "scientific_gate": {
            "rule": (
                "all five source-defect edges must be positive and >=.80; mean "
                "must beat paired deletion and both single-r8 controls, with at "
                "least 3/5 wins against each"
            ),
            "passed": passes,
            "decision": "retain_for_formal_canary" if passes else "retire",
        },
        "certificate_rows": certificate_rows,
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
