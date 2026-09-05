#!/usr/bin/env python3
"""Run one fixed common-projection response-defect preflight on UID 1930."""

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
    materialize_common_projection_state_splice,
    medium_common_projection_cost,
)
from insight_two.mode_space_replay import (  # noqa: E402
    approximate_layer0_defect_basis,
    factorized_reduced_current_replay,
    splice_shared_modes_from_factorized_replay,
)
from insight_two.run_kv_response_coupling_preflight import (  # noqa: E402
    verify_preflight_inputs,
)

UID = 1930
RANK = 8
OVERSAMPLE = 4
POWER_ITERATIONS = 1
REPLAY_SEED = 17
DEFECT_SEED = 1017


def _row(
    edge: str,
    method: str,
    exact: torch.Tensor,
    reuse: torch.Tensor,
    observed: torch.Tensor,
) -> dict[str, Any]:
    return {
        "edge": edge,
        "uid": UID,
        "method": method,
        **metrics_row(score_metrics(exact, reuse, observed)),
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:3")
    args = parser.parse_args()
    if args.device != "cuda:3":
        raise ValueError("this one-UID preflight is fixed to cuda:3")
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
    sidecar_scalars: list[int] = []
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
        parent_cache = parent.compute_kv(items, actions, deltas)
        replay = factorized_reduced_current_replay(
            current,
            current.embed_inputs(items, actions, deltas),
            rank=RANK,
            compression="fixed_range_finder",
            sketch_oversample=OVERSAMPLE,
            sketch_power_iterations=POWER_ITERATIONS,
            sketch_seed=REPLAY_SEED,
        )
        memory = build_common_projection_response_memory(replay, parent_cache)
        state_splice = materialize_common_projection_state_splice(
            parent_cache, memory
        )
        shared_basis = approximate_layer0_defect_basis(
            parent_cache,
            replay,
            rank=RANK,
            oversample=OVERSAMPLE,
            power_iterations=0,
            seed=DEFECT_SEED,
        )
        shared_splice = splice_shared_modes_from_factorized_replay(
            parent_cache, replay, shared_basis
        )

        # Current Exact is materialized only after all legal states exist.
        current_cache = current.compute_kv(items, actions, deltas)
        exact_scores = current.score_cc_reuse(current_cache, heldout, query_delta)
        reuse_scores = current.score_cc_reuse(parent_cache, heldout, query_delta)
        functional_scores = intervene_common_projection_response(
            current, parent_cache, memory, heldout, query_delta
        ).scores
        methods = (
            ("current_reduced_cache", current.score_cc_reuse(replay.cache, heldout, query_delta)),
            ("common_projection_state_splice", current.score_cc_reuse(state_splice, heldout, query_delta)),
            ("shared_layer0_state_splice", current.score_cc_reuse(shared_splice.cache, heldout, query_delta)),
            ("common_projection_response_defect", functional_scores),
        )
        edge_rows = [
            _row(edge, method, exact_scores, reuse_scores, observed)
            for method, observed in methods
        ]
        rows.extend(edge_rows)
        sidecar_scalars.append(memory.stored_scalars)
        print(
            json.dumps(
                {
                    "edge": edge,
                    "probability_recovery": {
                        str(row["method"]): row["probability_gap_recovery"]
                        for row in edge_rows
                    },
                }
            ),
            flush=True,
        )
        del parent, current, parent_cache, current_cache
        torch.cuda.empty_cache()

    by_method: dict[str, list[float]] = {}
    for row in rows:
        by_method.setdefault(str(row["method"]), []).append(
            float(row["probability_gap_recovery"])
        )
    cost = medium_common_projection_cost()
    summary = {
        "status": "nonformal_single_uid_route_elimination_complete",
        "uid": UID,
        "device": args.device,
        "candidate_source": "heldout_odd32_only",
        "labels_read": False,
        "confirmation_read": False,
        "current_exact_used_only_after_legal_construction": True,
        "input_verification": verification,
        "configuration": {
            "rank": RANK,
            "oversample": OVERSAMPLE,
            "power_iterations": POWER_ITERATIONS,
            "replay_seed": REPLAY_SEED,
            "basis_source": "per_layer_Current_replay_span",
        },
        "cost": {
            "total_constructor_flops": cost.total_constructor_flops,
            "constructor_fraction": cost.constructor_fraction,
            "within_twenty_percent": cost.within_twenty_percent,
            "accounting_is_conservative": True,
            "incremental_reader_flops_per_query": (
                cost.incremental_reader_flops_per_query
            ),
        },
        "sidecar_scalars": sidecar_scalars,
        "elapsed_seconds": time.perf_counter() - started,
        "probability_recovery_by_method": by_method,
        "probability_recovery_edge_mean": {
            method: float(np.mean(values)) for method, values in by_method.items()
        },
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
