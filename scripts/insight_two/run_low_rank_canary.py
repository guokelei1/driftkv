#!/usr/bin/env python3
"""Four-GPU query-conditioned low-rank functional-boundary canary."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_yambda500m_foundation_raw import load_histories, load_model  # noqa: E402
from insight_one_locality.common import histories_at_cutover  # noqa: E402
from insight_two.common import (  # noqa: E402
    ANCHOR_INDICES,
    CANARY_USERS,
    CONTRACT,
    CUTOVER_DAYS,
    DATASET,
    DAY,
    EDGES,
    HELDOUT_INDICES,
    KNOWN_ITEMS,
    OOV_BUCKETS,
    RESULT_ROOT,
    STAGE_PRESENTATION,
    checkpoint,
    load_frozen_inputs,
    metrics_row,
    score_metrics,
    sha256_file,
    verify_contract,
    verify_model_payload,
)
from insight_two.low_rank_correction import (  # noqa: E402
    LAYERED_STAGES,
    low_rank_final_representation,
    low_rank_layered_correction,
)


CANARY_RANKS = (1, 2, 4, 8)
DISCOVERY_RANKS = (0, 1, 2, 4, 8)
STAGES = (*LAYERED_STAGES, "final_readout")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def distributed_context() -> tuple[int, int, int]:
    if "RANK" not in os.environ:
        raise RuntimeError("the frozen canary requires torchrun")
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != 4:
        raise RuntimeError(f"the frozen canary requires four ranks, got {world}")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world


def require_rank0_gate() -> None:
    path = RESULT_ROOT / "canary_rank0/summary.json"
    if not path.is_file():
        raise RuntimeError("low-rank canary requires the rank-0 canary")
    summary = json.loads(path.read_text(encoding="utf-8"))
    if not summary.get("passed") or summary.get("contract_sha256") != sha256_file(CONTRACT):
        raise RuntimeError("rank-0 canary did not pass under the current contract")


def require_discovery_gate() -> None:
    require_rank0_gate()
    low_path = RESULT_ROOT / "canary_low_rank/summary.json"
    estimate_path = RESULT_ROOT / "resource_estimate.json"
    if not low_path.is_file() or not estimate_path.is_file():
        raise RuntimeError("discovery requires low-rank canary and resource estimate")
    low = json.loads(low_path.read_text(encoding="utf-8"))
    estimate = json.loads(estimate_path.read_text(encoding="utf-8"))
    if not low.get("passed") or low.get("contract_sha256") != sha256_file(CONTRACT):
        raise RuntimeError("low-rank canary did not pass under the current contract")
    if estimate.get("contract_sha256") != sha256_file(CONTRACT):
        raise RuntimeError("resource estimate differs from the current contract")
    if estimate.get("discovery_users") != 512 or estimate.get("ranks") != list(DISCOVERY_RANKS):
        raise RuntimeError("resource estimate discovery configuration differs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("canary", "discovery"), default="canary")
    args = parser.parse_args()
    users = CANARY_USERS if args.scope == "canary" else 512
    ranks = CANARY_RANKS if args.scope == "canary" else DISCOVERY_RANKS
    rank, local_rank, world = distributed_context()
    device = torch.device("cuda", local_rank)
    torch.set_num_threads(4)
    started = time.perf_counter()
    verification: list[dict[str, Any] | None] = [None]
    if rank == 0:
        try:
            verify_contract()
            require_rank0_gate() if args.scope == "canary" else require_discovery_gate()
            verification[0] = {"ok": True}
        except BaseException as error:
            verification[0] = {"ok": False, "error": repr(error)}
    dist.broadcast_object_list(verification, src=0)
    assert verification[0] is not None
    if not verification[0]["ok"]:
        raise RuntimeError(f"gate verification failed: {verification[0]['error']}")

    all_uids, all_candidates, _ = load_frozen_inputs()
    selected_indices = np.arange(users, dtype=np.int64)
    local_indices = selected_indices[rank::world]
    local_uids = all_uids[local_indices]
    output = RESULT_ROOT / (
        "canary_low_rank" if args.scope == "canary" else "discovery_functional_boundary"
    )
    partial = output.with_name(output.name + ".partial")
    if rank == 0:
        if output.exists() or partial.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
        partial.mkdir(parents=True)
        atomic_json(
            partial / "configuration.json",
            {
                "contract_sha256": sha256_file(CONTRACT),
                "scope": args.scope,
                "users": users,
                "edges": list(EDGES),
                "stages": list(STAGES),
                "ranks": list(ranks),
                "query_feature": "flattened_Current_Q_at_each_selected_path_layer",
                "ridge_relative": 0.001,
                "labels_read": False,
                "oracle_target_role": "representation_test_only",
            },
        )
    dist.barrier()
    rank_output = partial / f"rank{rank}"
    rank_output.mkdir()

    history_started = time.perf_counter()
    history = load_histories(
        local_uids.tolist(),
        oov_buckets=OOV_BUCKETS,
        dataset_path=DATASET,
        known_vocab_size=KNOWN_ITEMS,
        end_timestamp=(CUTOVER_DAYS[-1] + 1) * DAY,
        threads=8,
    )
    history_seconds = time.perf_counter() - history_started
    score_records: list[dict[str, Any]] = []
    fit_records: list[dict[str, Any]] = []
    edge_records: list[dict[str, Any]] = []
    peak_allocated_mib = 0.0
    peak_reserved_mib = 0.0

    for edge_index, edge in enumerate(EDGES):
        print(json.dumps({"phase": "edge_start", "rank": rank, "edge": edge}), flush=True)
        edge_started = time.perf_counter()
        parent, parent_payload = load_model(checkpoint(edge_index), device)
        current, current_payload = load_model(checkpoint(edge_index + 1), device)
        verify_model_payload(parent_payload)
        verify_model_payload(current_payload)
        _, items_np, actions_np, deltas_np, query_deltas_np = histories_at_cutover(
            history, local_uids, CUTOVER_DAYS[edge_index] * DAY
        )
        candidate_np = all_candidates[edge_index, local_indices]
        torch.cuda.reset_peak_memory_stats(device)

        for local_offset, uid in enumerate(local_uids):
            items = torch.as_tensor(
                items_np[local_offset : local_offset + 1], dtype=torch.long, device=device
            )
            actions = torch.as_tensor(
                actions_np[local_offset : local_offset + 1], dtype=torch.long, device=device
            )
            deltas = torch.as_tensor(
                deltas_np[local_offset : local_offset + 1], dtype=torch.float32, device=device
            )
            query_delta = torch.as_tensor(
                query_deltas_np[local_offset : local_offset + 1],
                dtype=torch.float32,
                device=device,
            )
            anchors = torch.as_tensor(
                candidate_np[local_offset : local_offset + 1, ANCHOR_INDICES],
                dtype=torch.long,
                device=device,
            )
            heldout = torch.as_tensor(
                candidate_np[local_offset : local_offset + 1, HELDOUT_INDICES],
                dtype=torch.long,
                device=device,
            )
            parent_cache = parent.compute_kv(items, actions, deltas)
            current_cache = current.compute_kv(items, actions, deltas)
            exact_scores = current.score_cc_reuse(current_cache, heldout, query_delta)
            reuse_scores = current.score_cc_reuse(parent_cache, heldout, query_delta)

            for stage in STAGES:
                for low_rank in ranks:
                    if stage == "final_readout":
                        result = low_rank_final_representation(
                            current,
                            current_cache,
                            parent_cache,
                            anchors,
                            heldout,
                            query_delta,
                            rank=low_rank,
                        )
                    else:
                        result = low_rank_layered_correction(
                            current,
                            current_cache,
                            parent_cache,
                            anchors,
                            heldout,
                            query_delta,
                            stage=stage,
                            rank=low_rank,
                        )
                    score_records.append(
                        {
                            "edge": edge,
                            "uid": int(uid),
                            "stage": stage,
                            "presentation": STAGE_PRESENTATION[stage],
                            "rank": low_rank,
                            "storage_values_fp32_per_user": result.storage_values_fp32_per_user,
                            **metrics_row(score_metrics(exact_scores, reuse_scores, result.scores)),
                        }
                    )
                    for diagnostic in result.diagnostics:
                        fit_records.append(
                            {
                                "edge": edge,
                                "uid": int(uid),
                                "presentation": STAGE_PRESENTATION[stage],
                                **diagnostic,
                            }
                        )
            peak_allocated_mib = max(
                peak_allocated_mib, torch.cuda.max_memory_allocated(device) / (1 << 20)
            )
            peak_reserved_mib = max(
                peak_reserved_mib, torch.cuda.max_memory_reserved(device) / (1 << 20)
            )
            del (
                items,
                actions,
                deltas,
                query_delta,
                anchors,
                heldout,
                parent_cache,
                current_cache,
                exact_scores,
                reuse_scores,
            )

        edge_seconds = time.perf_counter() - edge_started
        edge_records.append({"edge": edge, "users": len(local_uids), "seconds": edge_seconds})
        print(
            json.dumps(
                {
                    "phase": "edge_complete",
                    "rank": rank,
                    "edge": edge,
                    "seconds": edge_seconds,
                }
            ),
            flush=True,
        )
        del parent, current, parent_payload, current_payload
        torch.cuda.empty_cache()

    score_path = rank_output / "score_records.parquet"
    fit_path = rank_output / "fit_records.parquet"
    pd.DataFrame(score_records).to_parquet(score_path, index=False)
    pd.DataFrame(fit_records).to_parquet(fit_path, index=False)
    rank_summary = {
        "status": "low_rank_canary_rank_complete",
        "rank": rank,
        "uids": local_uids.tolist(),
        "labels_read": False,
        "history_seconds": history_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_mib": peak_allocated_mib,
        "peak_reserved_mib": peak_reserved_mib,
        "edges": edge_records,
        "artifacts": {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in (score_path, fit_path)
        },
    }
    atomic_json(rank_output / "summary.json", rank_summary)
    dist.barrier()

    if rank == 0:
        summaries = [
            json.loads((partial / f"rank{value}/summary.json").read_text(encoding="utf-8"))
            for value in range(world)
        ]
        scores = pd.concat(
            [pd.read_parquet(partial / f"rank{value}/score_records.parquet") for value in range(4)],
            ignore_index=True,
        )
        observed_uids = [uid for summary in summaries for uid in summary["uids"]]
        expected_rows = users * len(EDGES) * len(STAGES) * len(ranks)
        passed = (
            len(observed_uids) == users
            and len(set(observed_uids)) == users
            and len(scores) == expected_rows
            and bool(np.isfinite(scores["probability_gap_recovery"]).all())
            and bool(np.isfinite(scores["logit_gap_recovery"]).all())
        )
        elapsed = max(float(summary["elapsed_seconds"]) for summary in summaries)
        summary = {
            "status": (
                f"{args.scope}_functional_boundary_complete"
                if passed
                else f"{args.scope}_functional_boundary_invalid"
            ),
            "passed": passed,
            "contract_sha256": sha256_file(CONTRACT),
            "scope": args.scope,
            "users": users,
            "edges": list(EDGES),
            "stages": list(STAGES),
            "ranks": list(ranks),
            "rows": len(scores),
            "labels_read": False,
            "elapsed_seconds": elapsed,
            "peak_allocated_mib": max(summary["peak_allocated_mib"] for summary in summaries),
            "peak_reserved_mib": max(summary["peak_reserved_mib"] for summary in summaries),
            "rough_512_user_seconds": elapsed * 512 / users,
            "rank_summaries": [f"rank{value}/summary.json" for value in range(world)],
        }
        atomic_json(partial / "summary.json", summary)
        partial.rename(output)
        print(json.dumps(summary, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
