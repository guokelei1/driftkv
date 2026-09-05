#!/usr/bin/env python3
"""Run the fixed one-user coupling-depth mechanism diagnostic.

This is not a formal runner.  It reads only discovery UID 1930 and the frozen
odd 32-candidate held-out panel.  It performs no training, writes no seal, and
does not expose a depth/rank search: the four structural profiles and the one
rank-budget handoff are fixed in source before execution.
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
from insight_two.coupling_depth_replay import (  # noqa: E402
    coupling_depth_layer0_basis,
    factorized_current_rank_handoff_replay,
    factorized_parent_prefix_replay,
    medium_coupling_depth_cost,
    medium_rank_handoff_cost,
    splice_with_coupling_depth,
)
from insight_two.mode_space_replay import (  # noqa: E402
    factorized_reduced_current_replay,
)

UID = 1930
FORMATION_DEPTHS = (1, 3, 5, 6)
ARM_RANK = 4
ARM_OVERSAMPLE = 4
ARM_POWER = 1
ARM_SEED = 17
DEFECT_RANK = 8
DEFECT_OVERSAMPLE = 4
DEFECT_POWER = 0
DEFECT_SEED = 1017


def _verify_nonformal_execution_inputs() -> bool:
    """Verify frozen execution artifacts while allowing the live plan to evolve.

    The functional-boundary contract sealed an earlier hash of the exploration
    plan.  This diagnostic is being developed by appending to that plan, so its
    hash is expected to move.  All data panels and checkpoint hashes remain
    mandatory.  The return value records whether even the plan still matched.
    """

    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    if contract["scope"]["edges"] != list(EDGES):
        raise RuntimeError("contract edge order differs")
    if tuple(contract["candidate_split"]["anchor_indices"]) != ANCHOR_INDICES:
        raise RuntimeError("contract anchor split differs")
    if tuple(contract["candidate_split"]["heldout_indices"]) != HELDOUT_INDICES:
        raise RuntimeError("contract held-out split differs")
    frozen = contract["frozen_inputs"]
    plan_matches = True
    for name, record in frozen.items():
        if name == "checkpoints":
            continue
        path = ROOT / record["path"]
        matches = path.is_file() and sha256_file(path) == record["sha256"]
        if name == "research_plan":
            plan_matches = matches
        elif not matches:
            raise RuntimeError(f"frozen execution input differs: {name}")
    for version in range(6):
        record = frozen["checkpoints"][f"v{version}"]
        path = ROOT / record["path"]
        if path != checkpoint(version):
            raise RuntimeError(f"checkpoint path differs for v{version}")
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"checkpoint v{version} differs from contract")
    return plan_matches


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(4)

    research_plan_matches_contract = _verify_nonformal_execution_inputs()
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
    handoff_rows: list[dict[str, float | int | str]] = []
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
        exact_current = current.compute_kv(items, actions, deltas)
        exact_scores = current.score_cc_reuse(exact_current, heldout, query_delta)
        reuse_scores = current.score_cc_reuse(exact_parent, heldout, query_delta)
        current_replay = factorized_reduced_current_replay(
            current,
            current.embed_inputs(items, actions, deltas),
            rank=ARM_RANK,
            compression="fixed_range_finder",
            sketch_oversample=ARM_OVERSAMPLE,
            sketch_power_iterations=ARM_POWER,
            sketch_seed=ARM_SEED,
        )
        parent_embedded = parent.embed_inputs(items, actions, deltas)
        edge_rows: list[dict[str, float | int | str]] = []
        reference_basis: torch.Tensor | None = None
        depth3_prefix = None
        depth3_basis: torch.Tensor | None = None
        depth3_splice = None
        for depth in FORMATION_DEPTHS:
            parent_prefix = factorized_parent_prefix_replay(
                parent,
                parent_embedded,
                formation_depth=depth,
                rank=ARM_RANK,
                compression="fixed_range_finder",
                sketch_oversample=ARM_OVERSAMPLE,
                sketch_power_iterations=ARM_POWER,
                sketch_seed=ARM_SEED,
            )
            basis = coupling_depth_layer0_basis(
                parent_prefix,
                current_replay,
                rank=DEFECT_RANK,
                oversample=DEFECT_OVERSAMPLE,
                power_iterations=DEFECT_POWER,
                seed=DEFECT_SEED,
            )
            if reference_basis is None:
                reference_basis = basis
            elif not torch.equal(reference_basis, basis):
                raise RuntimeError("U0 changed when only coupling depth changed")
            splice = splice_with_coupling_depth(exact_parent, current_replay, parent_prefix, basis)
            scores = current.score_cc_reuse(splice.cache, heldout, query_delta)
            row: dict[str, float | int | str] = {
                "edge": edge,
                "uid": UID,
                "formation_depth": depth,
                **metrics_row(score_metrics(exact_scores, reuse_scores, scores)),
            }
            rows.append(row)
            edge_rows.append(row)
            if depth == 3:
                depth3_prefix = parent_prefix
                depth3_basis = basis
                depth3_splice = splice

        if depth3_prefix is None or depth3_basis is None or depth3_splice is None:
            raise RuntimeError("the fixed d=3 profile was not constructed")
        handoff_current = factorized_current_rank_handoff_replay(
            current,
            current.embed_inputs(items, actions, deltas),
            handoff_depth=3,
            early_rank=4,
            upper_rank=8,
            sketch_oversample=ARM_OVERSAMPLE,
            sketch_power_iterations=ARM_POWER,
            sketch_seed=ARM_SEED,
        )
        handoff_basis = coupling_depth_layer0_basis(
            depth3_prefix,
            handoff_current,
            rank=DEFECT_RANK,
            oversample=DEFECT_OVERSAMPLE,
            power_iterations=DEFECT_POWER,
            seed=DEFECT_SEED,
        )
        if not torch.equal(handoff_basis, depth3_basis):
            raise RuntimeError("rank handoff changed the frozen layer-0 U0")
        handoff_splice = splice_with_coupling_depth(
            exact_parent, handoff_current, depth3_prefix, handoff_basis
        )
        for layer in range(3):
            if not torch.equal(
                handoff_splice.delta_k_cores[layer],
                depth3_splice.delta_k_cores[layer],
            ) or not torch.equal(
                handoff_splice.delta_v_cores[layer],
                depth3_splice.delta_v_cores[layer],
            ):
                raise RuntimeError("rank handoff changed an early signed core")
        handoff_scores = current.score_cc_reuse(handoff_splice.cache, heldout, query_delta)
        handoff_row: dict[str, float | int | str] = {
            "edge": edge,
            "uid": UID,
            "profile": "d3_parent4_current4_then_current8",
            "formation_depth": 3,
            "early_current_rank": 4,
            "upper_current_rank": 8,
            **metrics_row(score_metrics(exact_scores, reuse_scores, handoff_scores)),
        }
        handoff_rows.append(handoff_row)
        print(
            json.dumps({"edge": edge, "rows": edge_rows, "handoff": handoff_row}),
            flush=True,
        )
        del parent, current
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary: dict[str, object] = {
        "status": "nonformal_single_uid_coupling_depth_preflight_complete",
        "uid": UID,
        "candidate_source": "heldout_odd_32_only",
        "confirmation_read": False,
        "labels_read": False,
        "all_frozen_execution_artifacts_verified": True,
        "research_plan_hash_matches_old_contract": research_plan_matches_contract,
        "formation_depths": list(FORMATION_DEPTHS),
        "configuration": {
            "current_arm": {
                "layers": 6,
                "rank": ARM_RANK,
                "oversample": ARM_OVERSAMPLE,
                "power_iterations": ARM_POWER,
                "seed": ARM_SEED,
            },
            "parent_arm": "same fixed scheme, prefix [0,d) only",
            "defect_basis": {
                "source": "paired approximate layer0 Delta[K,V]",
                "rank": DEFECT_RANK,
                "oversample": DEFECT_OVERSAMPLE,
                "power_iterations": DEFECT_POWER,
                "seed": DEFECT_SEED,
            },
        },
        "costs": [medium_coupling_depth_cost(depth) for depth in FORMATION_DEPTHS],
        "rank_handoff_cost": medium_rank_handoff_cost(),
        "elapsed_seconds": time.perf_counter() - started,
        "rows": rows,
        "rank_handoff_rows": handoff_rows,
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
