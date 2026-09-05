#!/usr/bin/env python3
"""Run the one-user matched Parent-cache differential preflight.

This is intentionally not a formal runner: it reads only the first frozen
discovery UID and the odd 32-candidate held-out panel, writes no result seal,
and exposes no rank or numerical-operator sweep.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

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
    verify_contract,
    verify_model_payload,
)
from insight_two.matched_parent_cache_differential import (  # noqa: E402
    approximate_exact_parent_cache,
    matched_layer0_defect_basis,
    paired_replay_layer0_defect_basis,
    splice_matched_parent_cache_differential,
    splice_paired_replay_differential,
)
from insight_two.mode_space_replay import (  # noqa: E402
    approximate_layer0_defect_basis,
    factorized_reduced_current_replay,
    splice_shared_modes_from_factorized_replay,
)

UID = 1930
ARM_RANK = 4
ARM_OVERSAMPLE = 4
ARM_POWER = 1
ARM_SEED = 17
DEFECT_RANK = 8
DEFECT_OVERSAMPLE = 4
DEFECT_POWER = 0
DEFECT_SEED = 1017


def _method_row(
    *,
    edge: str,
    method: str,
    exact: torch.Tensor,
    reuse: torch.Tensor,
    observed: torch.Tensor,
) -> dict[str, float | int | str]:
    return {
        "edge": edge,
        "uid": UID,
        "method": method,
        **metrics_row(score_metrics(exact, reuse, observed)),
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(4)

    verify_contract()
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
    rows: list[dict[str, float | int | str]] = []
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
        exact_current = current.compute_kv(items, actions, deltas)
        exact_scores = current.score_cc_reuse(
            exact_current, heldout, query_delta
        )
        reuse_scores = current.score_cc_reuse(
            exact_parent, heldout, query_delta
        )

        current_replay = factorized_reduced_current_replay(
            current,
            current.embed_inputs(items, actions, deltas),
            rank=ARM_RANK,
            compression="fixed_range_finder",
            sketch_oversample=ARM_OVERSAMPLE,
            sketch_power_iterations=ARM_POWER,
            sketch_seed=ARM_SEED,
        )

        # Control 1: subtract exact Parent coefficients.  This leaves all
        # Current replay approximation error inside the signed defect.
        exact_parent_basis = approximate_layer0_defect_basis(
            exact_parent,
            current_replay,
            rank=DEFECT_RANK,
            oversample=DEFECT_OVERSAMPLE,
            power_iterations=DEFECT_POWER,
            seed=DEFECT_SEED,
        )
        exact_parent_splice = splice_shared_modes_from_factorized_replay(
            exact_parent, current_replay, exact_parent_basis
        )
        exact_parent_scores = current.score_cc_reuse(
            exact_parent_splice.cache, heldout, query_delta
        )
        rows.append(
            _method_row(
                edge=edge,
                method="current_rank4_minus_exact_parent",
                exact=exact_scores,
                reuse=reuse_scores,
                observed=exact_parent_scores,
            )
        )

        # Primary probe: approximate the persistent Parent cache with the same
        # fixed rank/sketch rule, without executing Parent Transformer blocks.
        matched_parent = approximate_exact_parent_cache(
            exact_parent,
            rank=ARM_RANK,
            oversample=ARM_OVERSAMPLE,
            power_iterations=ARM_POWER,
            seed=ARM_SEED,
        )
        matched_basis = matched_layer0_defect_basis(
            exact_parent,
            current_replay,
            matched_parent,
            rank=DEFECT_RANK,
            oversample=DEFECT_OVERSAMPLE,
            power_iterations=DEFECT_POWER,
            seed=DEFECT_SEED,
        )
        matched_splice = splice_matched_parent_cache_differential(
            exact_parent, current_replay, matched_parent, matched_basis
        )
        matched_scores = current.score_cc_reuse(
            matched_splice.cache, heldout, query_delta
        )
        rows.append(
            _method_row(
                edge=edge,
                method="current_rank4_minus_matched_parent_cache_rank4",
                exact=exact_scores,
                reuse=reuse_scores,
                observed=matched_scores,
            )
        )

        # Control 2: the known equal-resolution two-model replay path.
        parent_replay = factorized_reduced_current_replay(
            parent,
            parent.embed_inputs(items, actions, deltas),
            rank=ARM_RANK,
            compression="fixed_range_finder",
            sketch_oversample=ARM_OVERSAMPLE,
            sketch_power_iterations=ARM_POWER,
            sketch_seed=ARM_SEED,
        )
        paired_basis = paired_replay_layer0_defect_basis(
            exact_parent,
            parent_replay,
            current_replay,
            rank=DEFECT_RANK,
            oversample=DEFECT_OVERSAMPLE,
            power_iterations=DEFECT_POWER,
            seed=DEFECT_SEED,
        )
        paired_splice = splice_paired_replay_differential(
            exact_parent, parent_replay, current_replay, paired_basis
        )
        paired_scores = current.score_cc_reuse(
            paired_splice.cache, heldout, query_delta
        )
        rows.append(
            _method_row(
                edge=edge,
                method="current_rank4_minus_parent_replay_rank4",
                exact=exact_scores,
                reuse=reuse_scores,
                observed=paired_scores,
            )
        )
        print(json.dumps({"edge": edge, "rows": rows[-3:]}), flush=True)
        del parent, current
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary: dict[str, object] = {
        "status": "nonformal_single_uid_preflight_complete",
        "uid": UID,
        "candidate_source": "heldout_odd_32_only",
        "confirmation_read": False,
        "labels_read": False,
        "configuration": {
            "current_arm": {
                "rank": ARM_RANK,
                "oversample": ARM_OVERSAMPLE,
                "power_iterations": ARM_POWER,
                "seed": ARM_SEED,
            },
            "parent_cache_operator": "joint_[K,V]_fixed_range_finder",
            "defect_basis": {
                "rank": DEFECT_RANK,
                "oversample": DEFECT_OVERSAMPLE,
                "power_iterations": DEFECT_POWER,
                "seed": DEFECT_SEED,
            },
        },
        "elapsed_seconds": time.perf_counter() - started,
        "rows": rows,
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
