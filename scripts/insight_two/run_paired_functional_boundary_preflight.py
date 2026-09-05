#!/usr/bin/env python3
"""Run the UID-1930/five-edge paired S4 functional-boundary preflight.

This is an informal route-elimination runner.  It reads only the first frozen
discovery UID and the held-out odd-32 candidate panel, creates no contract or
seal, performs no fitting, and exposes no rank/probe sweep.  Candidate
construction finishes before Current Exact is materialized.
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
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_yambda500m_foundation_raw import load_histories, load_model  # noqa: E402
from insight_one_locality.common import histories_at_cutover  # noqa: E402

from insight_two.common import (  # noqa: E402
    ANCHOR_INDICES,
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
from insight_two.cone_response_memory import (  # noqa: E402
    intervene_cone_response_memory,
)
from insight_two.mode_space_replay import (  # noqa: E402
    approximate_paired_layer0_defect_basis,
    factorized_reduced_current_replay,
    paired_release_replay,
    splice_shared_modes_from_paired_replay,
)
from insight_two.paired_functional_boundary import (  # noqa: E402
    PRIMARY_PROBES,
    build_full_exact_response_memory,
    build_paired_factorized_response_memory,
    build_single_arm_factorized_response_memory,
    medium_functional_boundary_cost,
)
from insight_two.paired_region_delta import (  # noqa: E402
    trace_history_item_region_queries,
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
PAIRED_KV_SPLICE_DENSE_INITIAL_FLOPS = 1_041_218_120
PAIRED_KV_SPLICE_MATRIX_FREE_INITIAL_FLOPS = 874_402_376
EXACT_ALL_FLOPS = 4_771_282_944


def verify_preflight_inputs() -> dict[str, str]:
    """Verify frozen evidence while allowing only the living-plan hash drift.

    The repository's exploration plan is intentionally being appended during
    this research turn, so the old formal contract's plan-document hash no
    longer matches.  This informal runner may not use that expected drift to
    relax any data, panel, checkpoint, or prior-evidence seal.
    """

    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    if contract["scope"]["edges"] != list(EDGES):
        raise RuntimeError("contract edge order differs")
    if tuple(contract["candidate_split"]["anchor_indices"]) != ANCHOR_INDICES:
        raise RuntimeError("contract anchor split differs")
    if tuple(contract["candidate_split"]["heldout_indices"]) != HELDOUT_INDICES:
        raise RuntimeError("contract held-out split differs")
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


def _row(
    *,
    edge: str,
    method: str,
    exact: torch.Tensor,
    reuse: torch.Tensor,
    observed: torch.Tensor,
    constructor_flops: int | None,
    sidecar_scalars: int,
    reader_flops_per_query: int,
) -> dict[str, Any]:
    fraction = None if constructor_flops is None else constructor_flops / EXACT_ALL_FLOPS
    return {
        "edge": edge,
        "uid": UID,
        "method": method,
        "constructor_flops_per_user": constructor_flops,
        "constructor_fraction_of_Exact_All": fraction,
        "within_twenty_percent": None if fraction is None else fraction <= 0.20,
        "persistent_sidecar_scalars": sidecar_scalars,
        "incremental_reader_flops_per_query": reader_flops_per_query,
        **metrics_row(score_metrics(exact, reuse, observed)),
    }


def _cost_payload(
    name: str,
    *,
    matrix_free_initial: bool = False,
) -> dict[str, Any]:
    cost = medium_functional_boundary_cost(name, matrix_free_initial=matrix_free_initial)
    return {
        "method": cost.method,
        "initial_factor_executor": (
            "matrix_free_operator" if matrix_free_initial else "dense_then_range_finder"
        ),
        "trajectory_flops": cost.trajectory_flops,
        "anchor_probe_flops": cost.anchor_probe_flops,
        "mask_and_moment_flops": cost.mask_and_moment_flops,
        "signed_moment_subtraction_flops": cost.signed_moment_subtraction_flops,
        "total_constructor_flops": cost.total_constructor_flops,
        "constructor_fraction": cost.constructor_fraction,
        "within_twenty_percent": cost.within_twenty_percent,
        "sidecar_scalars": cost.sidecar_scalars,
        "sidecar_fp32_bytes": cost.sidecar_fp32_bytes,
        "transient_mask_bits": cost.transient_mask_bits,
        "mask_sign_comparisons": cost.mask_sign_comparisons,
        "initial_sin_cos_evaluations": cost.initial_sin_cos_evaluations,
        "initial_gaussian_draws": cost.initial_gaussian_draws,
        "embedding_lookup_scalars": cost.embedding_lookup_scalars,
        "raw_history_scalars": cost.raw_history_scalars,
        "incremental_reader_flops_per_query": (cost.incremental_reader_flops_per_query),
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    if args.device != "cuda:1":
        raise ValueError("this one-UID preflight is fixed to cuda:1")
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
    costs = {
        name: _cost_payload(name)
        for name in (
            "paired_r4_functional_moments",
            "single_current_r8_functional_moments",
            "full_exact_functional_moment_oracle",
        )
    }
    costs["paired_r4_functional_moments_matrix_free_initial"] = _cost_payload(
        "paired_r4_functional_moments", matrix_free_initial=True
    )
    costs["single_current_r8_functional_moments_matrix_free_initial"] = _cost_payload(
        "single_current_r8_functional_moments", matrix_free_initial=True
    )
    rows: list[dict[str, Any]] = []
    edge_diagnostics: list[dict[str, Any]] = []
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

        # Exact Parent is the only persistent upper-layer state available to
        # all legal constructors.  P8 IDs are fixed history midpoints.
        exact_parent = parent.compute_kv(items, actions, deltas)
        probe_queries = trace_history_item_region_queries(
            current,
            exact_parent,
            items,
            query_delta,
            probe_count=PRIMARY_PROBES,
        )
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
        paired_memory = build_paired_factorized_response_memory(
            current,
            paired,
            probe_queries,
            source_kv_scalars=exact_parent.k.numel() + exact_parent.v.numel(),
        )
        paired_basis = approximate_paired_layer0_defect_basis(
            paired,
            rank=DEFECT_RANK,
            oversample=DEFECT_OVERSAMPLE,
            power_iterations=DEFECT_POWER,
            seed=DEFECT_SEED,
        )
        paired_kv_splice = splice_shared_modes_from_paired_replay(
            exact_parent, paired, paired_basis
        )

        single_current = factorized_reduced_current_replay(
            current,
            current_embedded,
            rank=SINGLE_RANK,
            compression="fixed_range_finder",
            sketch_oversample=OVERSAMPLE,
            sketch_power_iterations=POWER_ITERATIONS,
            sketch_seed=REPLAY_SEED,
        )
        single_memory = build_single_arm_factorized_response_memory(
            current, single_current, exact_parent, probe_queries
        )

        # Evaluation-only state is deliberately materialized after all legal
        # paths have been constructed.
        exact_current = current.compute_kv(items, actions, deltas)
        full_exact_memory = build_full_exact_response_memory(
            current, exact_current, exact_parent, probe_queries
        )
        exact_scores = current.score_cc_reuse(exact_current, heldout, query_delta)
        reuse_scores = current.score_cc_reuse(exact_parent, heldout, query_delta)
        paired_kv_scores = current.score_cc_reuse(paired_kv_splice.cache, heldout, query_delta)
        paired_functional_scores = intervene_cone_response_memory(
            current, exact_parent, paired_memory, heldout, query_delta
        ).scores
        single_functional_scores = intervene_cone_response_memory(
            current, exact_parent, single_memory, heldout, query_delta
        ).scores
        oracle_scores = intervene_cone_response_memory(
            current, exact_parent, full_exact_memory, heldout, query_delta
        ).scores

        methods = (
            ("Current_Reuse", reuse_scores, 0, 0, 0),
            (
                "paired_r4_KV_splice",
                paired_kv_scores,
                PAIRED_KV_SPLICE_MATRIX_FREE_INITIAL_FLOPS,
                paired_kv_splice.sidecar_scalars,
                6 * (202_752 + 6_336),
            ),
            (
                "paired_r4_functional_moments",
                paired_functional_scores,
                costs["paired_r4_functional_moments_matrix_free_initial"][
                    "total_constructor_flops"
                ],
                paired_memory.stored_scalars,
                costs["paired_r4_functional_moments_matrix_free_initial"][
                    "incremental_reader_flops_per_query"
                ],
            ),
            (
                "single_current_r8_functional_moments",
                single_functional_scores,
                costs["single_current_r8_functional_moments_matrix_free_initial"][
                    "total_constructor_flops"
                ],
                single_memory.stored_scalars,
                costs["single_current_r8_functional_moments_matrix_free_initial"][
                    "incremental_reader_flops_per_query"
                ],
            ),
            (
                "full_exact_functional_moment_oracle",
                oracle_scores,
                costs["full_exact_functional_moment_oracle"]["total_constructor_flops"],
                full_exact_memory.stored_scalars,
                costs["full_exact_functional_moment_oracle"]["incremental_reader_flops_per_query"],
            ),
        )
        edge_rows = []
        for method, observed, flops, sidecar, reader in methods:
            row = _row(
                edge=edge,
                method=method,
                exact=exact_scores,
                reuse=reuse_scores,
                observed=observed,
                constructor_flops=flops,
                sidecar_scalars=sidecar,
                reader_flops_per_query=reader,
            )
            rows.append(row)
            edge_rows.append(row)
        by_method = {str(row["method"]): row for row in edge_rows}
        diagnostics = {
            "edge": edge,
            "aggregation_gain_over_paired_KV": (
                by_method["paired_r4_functional_moments"]["probability_gap_recovery"]
                - by_method["paired_r4_KV_splice"]["probability_gap_recovery"]
            ),
            "paired_minus_single_functional": (
                by_method["paired_r4_functional_moments"]["probability_gap_recovery"]
                - by_method["single_current_r8_functional_moments"]["probability_gap_recovery"]
            ),
            "paired_memory_scalars": paired_memory.stored_scalars,
            "single_memory_scalars": single_memory.stored_scalars,
            "exact_upper_layer_state_used_by_legal_constructor": False,
        }
        edge_diagnostics.append(diagnostics)
        print(
            json.dumps({"edge": edge, "rows": edge_rows, "diagnostics": diagnostics}),
            flush=True,
        )
        del parent, current
        torch.cuda.empty_cache()

    summary = {
        "status": "nonformal_single_uid_route_elimination_complete",
        "uid": UID,
        "device": "cuda:1",
        "candidate_source": "heldout_odd32_only",
        "probe_source": "fixed_history_lower_midpoint_P8",
        "labels_read": False,
        "confirmation_read": False,
        "input_verification": verification,
        "current_exact_upper_layer_state_used_by_legal_constructors": False,
        "configuration": {
            "paired": {
                "rank_per_arm": PAIRED_RANK,
                "oversample": OVERSAMPLE,
                "power_iterations": POWER_ITERATIONS,
                "seed": REPLAY_SEED,
            },
            "single_current_control": {
                "rank": SINGLE_RANK,
                "oversample": OVERSAMPLE,
                "power_iterations": POWER_ITERATIONS,
                "seed": REPLAY_SEED,
            },
            "probes": PRIMARY_PROBES,
            "reported_legal_cost_executor": "matrix_free_initial_factor",
            "paired_KV_dense_initial_cost_for_sensitivity": (PAIRED_KV_SPLICE_DENSE_INITIAL_FLOPS),
        },
        "costs": costs,
        "elapsed_seconds": time.perf_counter() - started,
        "rows": rows,
        "edge_diagnostics": edge_diagnostics,
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
