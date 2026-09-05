#!/usr/bin/env python3
"""Run the fixed one-user defect-coordinate closure diagnostic.

This is not a formal runner.  It reads only discovery UID 1930 and the frozen
odd 32-candidate panel.  There is no rank grid: the sole primary point is
Parent-base rank 2 plus release-defect rank 4.  All controls and numerical
seeds are fixed in source before observing the five-edge result.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

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
from insight_two.defect_first_replay import (  # noqa: E402
    absolute_replay_from_initial_factors,
    approximate_release_defect_basis,
    defect_first_replay_from_initial_factors,
    factorized_cache_scalars,
    forward_one_with_factorized_cache,
    forward_one_with_native_response_defect,
    matrix_free_defect_first_initial_factors,
    medium_defect_first_costs,
    native_response_sidecar_scalars,
    splice_approximate_release_defect,
)
from insight_two.matrix_free_input_range import (  # noqa: E402
    hstu_input_operator,
    matrix_free_randomized_token_factors,
)

UID = 1930
OVERSAMPLE = 4
POWER_ITERATIONS = 1
SKETCH_SEED = 17
STATE_SPLICE_RANK = 8
STATE_SPLICE_OVERSAMPLE = 4
STATE_SPLICE_POWER = 0
STATE_SPLICE_SEED = 1017


def _verify_nonformal_execution_inputs() -> bool:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    if contract["scope"]["edges"] != list(EDGES):
        raise RuntimeError("contract edge order differs")
    if tuple(contract["candidate_split"]["anchor_indices"]) != ANCHOR_INDICES:
        raise RuntimeError("contract anchor split differs")
    if tuple(contract["candidate_split"]["heldout_indices"]) != HELDOUT_INDICES:
        raise RuntimeError("contract held-out split differs")
    plan_matches = True
    for name, record in contract["frozen_inputs"].items():
        if name == "checkpoints":
            continue
        path = ROOT / record["path"]
        matches = path.is_file() and sha256_file(path) == record["sha256"]
        if name == "research_plan":
            plan_matches = matches
        elif not matches:
            raise RuntimeError(f"frozen execution input differs: {name}")
    for version in range(6):
        record = contract["frozen_inputs"]["checkpoints"][f"v{version}"]
        path = ROOT / record["path"]
        if path != checkpoint(version):
            raise RuntimeError(f"checkpoint path differs for v{version}")
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"checkpoint v{version} differs from contract")
    return plan_matches


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
        hidden = forward_one_with_native_response_defect(
            model,
            exact_parent,
            approximate_parent,
            approximate_current,
            queries[:, candidate : candidate + 1],
        )
        scores.append(model.cc_score_head(hidden[:, 0]).squeeze(-1))
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
        hidden = forward_one_with_factorized_cache(
            model,
            approximate_current,
            queries[:, candidate : candidate + 1],
        )
        scores.append(model.cc_score_head(hidden[:, 0]).squeeze(-1))
    return torch.stack(scores, dim=1)


def _row(
    *,
    edge: str,
    method: str,
    exact: torch.Tensor,
    reuse: torch.Tensor,
    observed: torch.Tensor,
    sidecar_scalars: int,
) -> dict[str, float | int | str]:
    return {
        "edge": edge,
        "uid": UID,
        "method": method,
        "persistent_sidecar_scalars_fp32": sidecar_scalars,
        **metrics_row(score_metrics(exact, reuse, observed)),
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:2")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(4)

    plan_matches = _verify_nonformal_execution_inputs()
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
        exact_scores = current.score_cc_reuse(exact_current, heldout, query_delta)
        reuse_scores = current.score_cc_reuse(exact_parent, heldout, query_delta)
        parent_operator = hstu_input_operator(parent, items, actions, deltas)
        current_operator = hstu_input_operator(current, items, actions, deltas)

        # Primary: a small Parent base plus a separately compressed finite
        # release defect.  Current absolute state is never range-compressed.
        primary_parent_initial, primary_defect_initial = (
            matrix_free_defect_first_initial_factors(
                parent_operator,
                current_operator,
                base_rank=2,
                defect_rank=4,
                sketch_oversample=OVERSAMPLE,
                sketch_power_iterations=POWER_ITERATIONS,
                sketch_seed=SKETCH_SEED,
            )
        )
        primary = defect_first_replay_from_initial_factors(
            parent,
            current,
            primary_parent_initial,
            primary_defect_initial,
            base_rank=2,
            defect_rank=4,
            sketch_oversample=OVERSAMPLE,
            sketch_power_iterations=POWER_ITERATIONS,
            sketch_seed=SKETCH_SEED,
        )
        primary_scores = _score_native_difference(
            current,
            exact_parent,
            primary.parent,
            primary.current,
            heldout,
            query_delta,
        )

        # Matched active-rank control: compress the two absolute states
        # independently at Parent rank2 and Current rank6.
        absolute_parent2 = absolute_replay_from_initial_factors(
            parent,
            matrix_free_randomized_token_factors(
                parent_operator,
                rank=2,
                oversample=OVERSAMPLE,
                power_iterations=POWER_ITERATIONS,
                seed=SKETCH_SEED,
            ),
            rank=2,
            sketch_oversample=OVERSAMPLE,
            sketch_power_iterations=POWER_ITERATIONS,
            sketch_seed=SKETCH_SEED,
        )
        absolute_current6 = absolute_replay_from_initial_factors(
            current,
            matrix_free_randomized_token_factors(
                current_operator,
                rank=6,
                oversample=OVERSAMPLE,
                power_iterations=POWER_ITERATIONS,
                seed=SKETCH_SEED,
            ),
            rank=6,
            sketch_oversample=OVERSAMPLE,
            sketch_power_iterations=POWER_ITERATIONS,
            sketch_seed=SKETCH_SEED,
        )
        asymmetric_scores = _score_native_difference(
            current,
            exact_parent,
            absolute_parent2,
            absolute_current6,
            heldout,
            query_delta,
        )

        # Existing paired trajectory at equal rank, now read through the full
        # native response rather than an affine moment approximation.
        absolute_parent4 = absolute_replay_from_initial_factors(
            parent,
            matrix_free_randomized_token_factors(
                parent_operator,
                rank=4,
                oversample=OVERSAMPLE,
                power_iterations=POWER_ITERATIONS,
                seed=SKETCH_SEED,
            ),
            rank=4,
            sketch_oversample=OVERSAMPLE,
            sketch_power_iterations=POWER_ITERATIONS,
            sketch_seed=SKETCH_SEED,
        )
        absolute_current4 = absolute_replay_from_initial_factors(
            current,
            matrix_free_randomized_token_factors(
                current_operator,
                rank=4,
                oversample=OVERSAMPLE,
                power_iterations=POWER_ITERATIONS,
                seed=SKETCH_SEED,
            ),
            rank=4,
            sketch_oversample=OVERSAMPLE,
            sketch_power_iterations=POWER_ITERATIONS,
            sketch_seed=SKETCH_SEED,
        )
        paired_native_scores = _score_native_difference(
            current,
            exact_parent,
            absolute_parent4,
            absolute_current4,
            heldout,
            query_delta,
        )

        # Generic strongest control: one absolute Current rank-8 cache and its
        # native factorized reader.  It has the same factor sidecar scalar count.
        absolute_current8 = absolute_replay_from_initial_factors(
            current,
            matrix_free_randomized_token_factors(
                current_operator,
                rank=8,
                oversample=OVERSAMPLE,
                power_iterations=POWER_ITERATIONS,
                seed=SKETCH_SEED,
            ),
            rank=8,
            sketch_oversample=OVERSAMPLE,
            sketch_power_iterations=POWER_ITERATIONS,
            sketch_seed=SKETCH_SEED,
        )
        single_scores = _score_factorized_current(
            current, absolute_current8, heldout, query_delta
        )

        # Historical representation control only.  This P8 state splice is not
        # used by the defect-first primary or either native-response method.
        paired_basis = approximate_release_defect_basis(
            absolute_parent4,
            absolute_current4,
            rank=STATE_SPLICE_RANK,
            oversample=STATE_SPLICE_OVERSAMPLE,
            power_iterations=STATE_SPLICE_POWER,
            seed=STATE_SPLICE_SEED,
        )
        paired_splice = splice_approximate_release_defect(
            exact_parent, absolute_parent4, absolute_current4, paired_basis
        )
        paired_splice_scores = current.score_cc_reuse(
            paired_splice.cache, heldout, query_delta
        )

        edge_rows = [
            _row(
                edge=edge,
                method="defect_first_b2_d4_native_response",
                exact=exact_scores,
                reuse=reuse_scores,
                observed=primary_scores,
                sidecar_scalars=native_response_sidecar_scalars(
                    primary.parent, primary.current
                ),
            ),
            _row(
                edge=edge,
                method="ordinary_asymmetric_p2_c6_native_response",
                exact=exact_scores,
                reuse=reuse_scores,
                observed=asymmetric_scores,
                sidecar_scalars=native_response_sidecar_scalars(
                    absolute_parent2, absolute_current6
                ),
            ),
            _row(
                edge=edge,
                method="paired_absolute_p4_c4_native_response",
                exact=exact_scores,
                reuse=reuse_scores,
                observed=paired_native_scores,
                sidecar_scalars=native_response_sidecar_scalars(
                    absolute_parent4, absolute_current4
                ),
            ),
            _row(
                edge=edge,
                method="single_absolute_c8_factorized_reader",
                exact=exact_scores,
                reuse=reuse_scores,
                observed=single_scores,
                sidecar_scalars=factorized_cache_scalars(absolute_current8),
            ),
            _row(
                edge=edge,
                method="paired_absolute_p4_c4_state_splice_p8_control",
                exact=exact_scores,
                reuse=reuse_scores,
                observed=paired_splice_scores,
                sidecar_scalars=paired_splice.sidecar_scalars,
            ),
        ]
        rows.extend(edge_rows)
        print(json.dumps({"edge": edge, "rows": edge_rows}), flush=True)
        del parent, current
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary: dict[str, object] = {
        "status": "nonformal_single_uid_defect_first_preflight_complete",
        "uid": UID,
        "candidate_source": "heldout_odd_32_only",
        "confirmation_read": False,
        "labels_read": False,
        "current_exact_role": "evaluation_target_only",
        "research_plan_hash_matches_old_contract": plan_matches,
        "configuration": {
            "primary": "Parent base rank2 + release defect rank4; Current effective rank6",
            "controls": [
                "ordinary absolute Parent2/Current6",
                "paired absolute Parent4/Current4",
                "single absolute Current8",
                "paired4/4 P8 state splice representation control",
            ],
            "range_finder": {
                "oversample": OVERSAMPLE,
                "power_iterations": POWER_ITERATIONS,
                "seed": SKETCH_SEED,
            },
            "primary_reader": (
                "exact Parent native response + approximate Current native response "
                "- approximate Parent native response"
            ),
        },
        "costs": medium_defect_first_costs(),
        "elapsed_seconds": time.perf_counter() - started,
        "rows": rows,
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
