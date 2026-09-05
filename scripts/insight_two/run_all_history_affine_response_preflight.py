#!/usr/bin/env python3
"""Run the UID-1930 probe-free all-history affine response falsifier.

This is a single fixed, informal configuration.  It reads only held-out
odd-32 candidates for evaluation, never reads labels or confirmation users,
and creates no contract or result seal.  Every legal candidate is constructed
before Current Exact exists.
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

from insight_two.all_history_affine_response import (  # noqa: E402
    build_full_exact_all_history_response_memory,
    build_single_arm_all_history_response_memory,
    medium_all_history_affine_cost,
    medium_single_r8_kv_splice_cost,
)
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
    approximate_layer0_defect_basis,
    factorized_reduced_current_replay,
    splice_shared_modes_from_factorized_replay,
)
from insight_two.paired_functional_boundary import (  # noqa: E402
    PRIMARY_PROBES,
    build_single_arm_factorized_response_memory,
    medium_functional_boundary_cost,
)
from insight_two.paired_region_delta import (  # noqa: E402
    trace_history_item_region_queries,
)

UID = 1930
RANK = 8
OVERSAMPLE = 4
POWER_ITERATIONS = 1
REPLAY_SEED = 17
DEFECT_RANK = 8
DEFECT_OVERSAMPLE = 4
DEFECT_POWER = 0
DEFECT_SEED = 1017
EXACT_ALL_FLOPS = 4_771_282_944


def verify_preflight_inputs() -> dict[str, str]:
    """Allow only the explicitly recorded living-plan hash drift."""

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
    constructor_flops: int,
    sidecar_scalars: int,
    reader_flops: int,
) -> dict[str, Any]:
    fraction = constructor_flops / EXACT_ALL_FLOPS
    return {
        "edge": edge,
        "uid": UID,
        "method": method,
        "constructor_flops_per_user": constructor_flops,
        "constructor_fraction_of_Exact_All": fraction,
        "within_twenty_percent": fraction <= 0.20,
        "persistent_sidecar_scalars": sidecar_scalars,
        "incremental_reader_flops_per_query": reader_flops,
        **metrics_row(score_metrics(exact, reuse, observed)),
    }


def _all_history_cost_payload(method: str) -> dict[str, Any]:
    cost = medium_all_history_affine_cost(method)
    return {
        "method": cost.method,
        "initial_factor_executor": (
            "full_exact_recompute"
            if method == "full_exact_all_history_affine_oracle"
            else "matrix_free_operator"
        ),
        "trajectory_flops": cost.trajectory_flops,
        "current_moment_flops": cost.current_moment_flops,
        "parent_moment_flops": cost.parent_moment_flops,
        "signed_subtraction_flops": cost.signed_subtraction_flops,
        "total_constructor_flops": cost.total_constructor_flops,
        "constructor_fraction": cost.constructor_fraction,
        "within_twenty_percent": cost.within_twenty_percent,
        "sidecar_scalars": cost.sidecar_scalars,
        "sidecar_fp32_bytes": cost.sidecar_fp32_bytes,
        "incremental_reader_flops_per_query": (cost.incremental_reader_flops_per_query),
        "initial_sin_cos_evaluations": cost.initial_sin_cos_evaluations,
        "initial_gaussian_draws": cost.initial_gaussian_draws,
        "embedding_lookup_scalars": cost.embedding_lookup_scalars,
        "raw_history_scalars": cost.raw_history_scalars,
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
    all_cost = medium_all_history_affine_cost("single_current_r8_all_history_affine")
    full_cost = medium_all_history_affine_cost("full_exact_all_history_affine_oracle")
    p8_cost = medium_functional_boundary_cost(
        "single_current_r8_functional_moments", matrix_free_initial=True
    )
    kv_cost = medium_single_r8_kv_splice_cost()
    costs = {
        "single_current_r8_all_history_affine": _all_history_cost_payload(
            "single_current_r8_all_history_affine"
        ),
        "full_exact_all_history_affine_oracle": _all_history_cost_payload(
            "full_exact_all_history_affine_oracle"
        ),
        "single_current_r8_P8_moments": {
            "total_constructor_flops": p8_cost.total_constructor_flops,
            "constructor_fraction": p8_cost.constructor_fraction,
            "within_twenty_percent": p8_cost.within_twenty_percent,
            "sidecar_scalars": p8_cost.sidecar_scalars,
            "incremental_reader_flops_per_query": (p8_cost.incremental_reader_flops_per_query),
        },
        "single_current_r8_KV_splice": kv_cost,
    }
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
        query_delta = torch.as_tensor(query_deltas_np, dtype=torch.float32, device=device)
        heldout = torch.as_tensor(
            all_candidates[edge_index, 0:1][:, HELDOUT_INDICES],
            dtype=torch.long,
            device=device,
        )

        exact_parent = parent.compute_kv(items, actions, deltas)
        current_replay = factorized_reduced_current_replay(
            current,
            current.embed_inputs(items, actions, deltas),
            rank=RANK,
            compression="fixed_range_finder",
            sketch_oversample=OVERSAMPLE,
            sketch_power_iterations=POWER_ITERATIONS,
            sketch_seed=REPLAY_SEED,
        )
        # Primary path: no probe, candidate or Current-Exact input.
        all_memory = build_single_arm_all_history_response_memory(
            current, current_replay, exact_parent
        )
        kv_basis = approximate_layer0_defect_basis(
            exact_parent,
            current_replay,
            rank=DEFECT_RANK,
            oversample=DEFECT_OVERSAMPLE,
            power_iterations=DEFECT_POWER,
            seed=DEFECT_SEED,
        )
        kv_splice = splice_shared_modes_from_factorized_replay(
            exact_parent, current_replay, kv_basis
        )
        probe_queries = trace_history_item_region_queries(
            current,
            exact_parent,
            items,
            query_delta,
            probe_count=PRIMARY_PROBES,
        )
        p8_memory = build_single_arm_factorized_response_memory(
            current, current_replay, exact_parent, probe_queries
        )

        # Current Exact is evaluation-only for legal rows and construction-only
        # for the explicitly named full-Exact oracle below.
        exact_current = current.compute_kv(items, actions, deltas)
        full_memory = build_full_exact_all_history_response_memory(
            current, exact_current, exact_parent
        )
        exact_scores = current.score_cc_reuse(exact_current, heldout, query_delta)
        reuse_scores = current.score_cc_reuse(exact_parent, heldout, query_delta)
        all_scores = intervene_cone_response_memory(
            current, exact_parent, all_memory, heldout, query_delta
        ).scores
        full_scores = intervene_cone_response_memory(
            current, exact_parent, full_memory, heldout, query_delta
        ).scores
        p8_scores = intervene_cone_response_memory(
            current, exact_parent, p8_memory, heldout, query_delta
        ).scores
        kv_scores = current.score_cc_reuse(kv_splice.cache, heldout, query_delta)

        methods = (
            ("Current_Reuse", reuse_scores, 0, 0, 0),
            (
                "single_current_r8_all_history_affine",
                all_scores,
                all_cost.total_constructor_flops,
                all_memory.stored_scalars,
                all_cost.incremental_reader_flops_per_query,
            ),
            (
                "full_exact_all_history_affine_oracle",
                full_scores,
                full_cost.total_constructor_flops,
                full_memory.stored_scalars,
                full_cost.incremental_reader_flops_per_query,
            ),
            (
                "single_current_r8_P8_moments",
                p8_scores,
                p8_cost.total_constructor_flops,
                p8_memory.stored_scalars,
                p8_cost.incremental_reader_flops_per_query,
            ),
            (
                "single_current_r8_KV_splice",
                kv_scores,
                int(kv_cost["total_constructor_flops"]),
                kv_splice.sidecar_scalars,
                6 * (202_752 + 6_336),
            ),
        )
        edge_rows: list[dict[str, Any]] = []
        for method, observed, flops, sidecar, reader in methods:
            row = _row(
                edge=edge,
                method=method,
                exact=exact_scores,
                reuse=reuse_scores,
                observed=observed,
                constructor_flops=flops,
                sidecar_scalars=sidecar,
                reader_flops=reader,
            )
            edge_rows.append(row)
            rows.append(row)
        by_method = {str(row["method"]): row for row in edge_rows}
        diagnostic = {
            "edge": edge,
            "all_affine_minus_P8": (
                by_method["single_current_r8_all_history_affine"]["probability_gap_recovery"]
                - by_method["single_current_r8_P8_moments"]["probability_gap_recovery"]
            ),
            "all_affine_minus_KV_splice": (
                by_method["single_current_r8_all_history_affine"]["probability_gap_recovery"]
                - by_method["single_current_r8_KV_splice"]["probability_gap_recovery"]
            ),
            "exact_upper_layer_state_used_by_legal_constructors": False,
            "probe_count_for_primary": 0,
        }
        diagnostics.append(diagnostic)
        print(
            json.dumps({"edge": edge, "rows": edge_rows, "diagnostics": diagnostic}),
            flush=True,
        )
        del parent, current
        torch.cuda.empty_cache()

    summary = {
        "status": "nonformal_single_uid_response_operator_falsifier_complete",
        "uid": UID,
        "device": "cuda:1",
        "candidate_source": "heldout_odd32_only",
        "primary_probe_count": 0,
        "labels_read": False,
        "confirmation_read": False,
        "input_verification": verification,
        "current_exact_upper_layer_state_used_by_legal_constructors": False,
        "configuration": {
            "single_current_rank": RANK,
            "oversample": OVERSAMPLE,
            "power_iterations": POWER_ITERATIONS,
            "seed": REPLAY_SEED,
            "reported_cost_executor": "matrix_free_initial_factor",
        },
        "costs": costs,
        "elapsed_seconds": time.perf_counter() - started,
        "rows": rows,
        "edge_diagnostics": diagnostics,
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
