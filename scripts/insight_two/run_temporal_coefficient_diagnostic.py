#!/usr/bin/env python3
"""Measure oracle temporal coefficients on the frozen cutover S4 direction."""

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
import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "insight"))

from evaluate_yambda500m_foundation_raw import load_histories, load_model  # noqa: E402
from insight_one_locality.common import histories_at_cutover  # noqa: E402
from insight_two.common import (  # noqa: E402
    ANCHOR_INDICES,
    CUTOVER_DAYS,
    DATASET,
    EDGES,
    HELDOUT_INDICES,
    HISTORY,
    KNOWN_ITEMS,
    OOV_BUCKETS,
    RESULT_ROOT,
    checkpoint,
    load_frozen_inputs,
    sha256_file,
    verify_model_payload,
)
from insight_two.run_temporal_persistence_diagnostic import (  # noqa: E402
    PRIMARY_REQUESTS,
    V5_REQUESTS,
    append_metric_rows,
    append_timestamp_events,
    request_groups,
    verify_contract as verify_persistence_contract,
)
from insight_two.temporal_coefficient import (  # noqa: E402
    project_global_coefficient,
    project_layerwise_coefficients,
)
from insight_two.temporal_persistence import (  # noqa: E402
    append_bucket,
    remaining_parent_fraction,
    time_bucket,
)
from reader_compatibility_correction import (  # noqa: E402
    _stage_path,
    intervene_reader_correction,
)


CONTRACT = (
    ROOT
    / "configs/contracts/yambda500m_medium_hstu_native_insight2_temporal_coefficient_v1.yaml"
)
OUTPUT_ROOT = RESULT_ROOT / "diagnostic_temporal_coefficient_v1"
CANARY_USERS = 32
DISCOVERY_USERS = 512
METHODS = (
    "same_request_S4_oracle",
    "frozen_cutover_S4",
    "oracle_global_coefficient_times_cutover_direction",
    "oracle_layerwise_coefficients_times_cutover_directions",
)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def verify_contract() -> tuple[dict[str, Any], str]:
    verify_persistence_contract()
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    if contract["scope"]["edges"] != list(EDGES) or contract["methods"] != list(METHODS):
        raise RuntimeError("temporal-coordinate scope or methods differ")
    for record in contract["frozen_inputs"].values():
        path = ROOT / record["path"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"frozen temporal-coordinate input differs: {path}")
    return contract, sha256_file(CONTRACT)


def require_discovery_gate(contract_hash: str) -> None:
    canary = OUTPUT_ROOT / "canary/summary.json"
    estimate = OUTPUT_ROOT / "resource_estimate.json"
    if not canary.is_file() or not estimate.is_file():
        raise RuntimeError("temporal-coordinate discovery requires canary and resource estimate")
    run = json.loads(canary.read_text(encoding="utf-8"))
    resource = json.loads(estimate.read_text(encoding="utf-8"))
    if not run.get("passed") or run.get("contract_sha256") != contract_hash:
        raise RuntimeError("temporal-coordinate canary did not pass instrumentation")
    if resource.get("contract_sha256") != contract_hash:
        raise RuntimeError("temporal-coordinate resource estimate differs")
    if float(resource.get("estimated_512_user_minutes", 1e9)) > 30:
        raise RuntimeError("temporal-coordinate discovery exceeds 30 minutes")


def distributed_context() -> tuple[int, int, int]:
    if "RANK" not in os.environ:
        raise RuntimeError("temporal-coordinate diagnostic requires torchrun")
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != 4:
        raise RuntimeError(f"temporal-coordinate diagnostic requires four ranks, got {world}")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world


@torch.inference_mode()
def evaluate_user(
    *,
    uid: int,
    user_index: int,
    raw_history: tuple[np.ndarray, np.ndarray, np.ndarray],
    prefix: tuple[np.ndarray, np.ndarray, np.ndarray, float],
    groups: list[dict[str, Any]],
    panel: np.ndarray,
    parent,
    current,
    edge: str,
    cutover: int,
    stop: int,
    metric_records: list[dict[str, Any]],
    coefficient_records: list[dict[str, Any]],
) -> None:
    device = next(current.parameters()).device
    prefix_items, prefix_behaviors, prefix_deltas, cutover_query_delta = prefix
    items = torch.as_tensor(prefix_items[None], dtype=torch.long, device=device)
    behaviors = torch.as_tensor(prefix_behaviors[None], dtype=torch.long, device=device)
    deltas = torch.as_tensor(prefix_deltas[None], dtype=torch.float32, device=device)
    exact_cache = current.compute_kv(items, behaviors, deltas)
    reuse_cache = parent.compute_kv(items, behaviors, deltas)
    anchors = torch.as_tensor(
        panel[None, ANCHOR_INDICES], dtype=torch.long, device=device
    )
    heldout = torch.as_tensor(
        panel[None, HELDOUT_INDICES], dtype=torch.long, device=device
    )
    cutover_delta = torch.tensor(
        [cutover_query_delta], dtype=torch.float32, device=device
    )
    _, frozen, _ = _stage_path(
        current,
        exact_cache,
        reuse_cache,
        anchors,
        cutover_delta,
        stage="av_aggregation",
        mode="shared",
    )
    frozen = tuple(value.detach().clone() for value in frozen)

    timestamps, event_items, event_behaviors = raw_history
    event_position = int(np.searchsorted(timestamps, cutover, side="left"))
    event_end = int(np.searchsorted(timestamps, stop, side="left"))
    last_timestamp = int(cutover - cutover_query_delta)
    append_count = 0
    for group in groups:
        query_timestamp = int(group["query_timestamp"])
        while event_position < event_end and int(timestamps[event_position]) < query_timestamp:
            event_timestamp = int(timestamps[event_position])
            event_stop = event_position + 1
            while event_stop < event_end and int(timestamps[event_stop]) == event_timestamp:
                event_stop += 1
            exact_cache, reuse_cache = append_timestamp_events(
                current,
                exact_cache,
                reuse_cache,
                event_items=event_items[event_position:event_stop],
                event_behaviors=event_behaviors[event_position:event_stop],
                timestamp=event_timestamp,
                last_timestamp=last_timestamp,
                device=device,
            )
            append_count += event_stop - event_position
            last_timestamp = event_timestamp
            event_position = event_stop

        query_delta = torch.tensor(
            [float(query_timestamp - last_timestamp)], dtype=torch.float32, device=device
        )
        _, current_correction, _ = _stage_path(
            current,
            exact_cache,
            reuse_cache,
            anchors,
            query_delta,
            stage="av_aggregation",
            mode="shared",
        )
        global_projection = project_global_coefficient(current_correction, frozen)
        layer_projection = project_layerwise_coefficients(current_correction, frozen)
        seconds = query_timestamp - cutover
        common = {
            "edge": edge,
            "uid": uid,
            "user_index": user_index,
            "phase": "rolling",
            "query_timestamp": query_timestamp,
            "seconds_since_cutover": seconds,
            "time_bucket": time_bucket(seconds),
            "append_count": append_count,
            "append_bucket": append_bucket(append_count),
            "remaining_parent_fraction": remaining_parent_fraction(
                append_count, HISTORY
            ),
        }
        coefficient_records.append(
            {
                **common,
                "global_coefficient": float(global_projection.coefficients[0, 0]),
                "global_projection_relative_l2": float(
                    global_projection.relative_l2[0]
                ),
                "layerwise_projection_relative_l2": float(
                    layer_projection.relative_l2[0]
                ),
                **{
                    f"layer_{layer}_coefficient": float(
                        layer_projection.coefficients[0, layer]
                    )
                    for layer in range(layer_projection.coefficients.shape[1])
                },
            }
        )

        def scores_for(candidates: torch.Tensor) -> tuple[torch.Tensor, ...]:
            exact = current.score_cc_reuse(exact_cache, candidates, query_delta)
            reuse = current.score_cc_reuse(reuse_cache, candidates, query_delta)
            outputs = []
            for correction in (
                current_correction,
                frozen,
                global_projection.correction,
                layer_projection.correction,
            ):
                scores, _ = intervene_reader_correction(
                    current,
                    reuse_cache,
                    candidates,
                    query_delta,
                    stage="av_aggregation",
                    corrections=correction,
                )
                outputs.append(scores)
            return exact, reuse, *outputs

        fixed_values = scores_for(heldout)
        append_metric_rows(
            metric_records,
            common={
                **common,
                "candidate_source": "fixed_heldout_panel",
                "candidate_count": len(HELDOUT_INDICES),
            },
            exact=fixed_values[0],
            reuse=fixed_values[1],
            methods=dict(zip(METHODS, fixed_values[2:], strict=True)),
        )
        real = torch.as_tensor(
            np.asarray(group["items"], dtype=np.int64)[None],
            dtype=torch.long,
            device=device,
        )
        real_values = scores_for(real)
        append_metric_rows(
            metric_records,
            common={
                **common,
                "candidate_source": "real_exposed_items",
                "candidate_count": int(real.shape[1]),
            },
            exact=real_values[0],
            reuse=real_values[1],
            methods=dict(zip(METHODS, real_values[2:], strict=True)),
        )

        while event_position < event_end and int(timestamps[event_position]) == query_timestamp:
            event_stop = event_position + 1
            while event_stop < event_end and int(timestamps[event_stop]) == query_timestamp:
                event_stop += 1
            exact_cache, reuse_cache = append_timestamp_events(
                current,
                exact_cache,
                reuse_cache,
                event_items=event_items[event_position:event_stop],
                event_behaviors=event_behaviors[event_position:event_stop],
                timestamp=query_timestamp,
                last_timestamp=last_timestamp,
                device=device,
            )
            append_count += event_stop - event_position
            last_timestamp = query_timestamp
            event_position = event_stop


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("canary", "discovery"), default="canary")
    args = parser.parse_args()
    users = CANARY_USERS if args.scope == "canary" else DISCOVERY_USERS
    rank, local_rank, world = distributed_context()
    device = torch.device("cuda", local_rank)
    torch.set_num_threads(4)
    started = time.perf_counter()
    verification: list[dict[str, Any] | None] = [None]
    if rank == 0:
        try:
            _, contract_hash = verify_contract()
            if args.scope == "discovery":
                require_discovery_gate(contract_hash)
            verification[0] = {"ok": True, "contract_sha256": contract_hash}
        except BaseException as error:
            verification[0] = {"ok": False, "error": repr(error)}
    dist.broadcast_object_list(verification, src=0)
    assert verification[0] is not None
    if not verification[0]["ok"]:
        raise RuntimeError(f"contract gate failed: {verification[0]['error']}")
    contract_hash = str(verification[0]["contract_sha256"])

    all_uids, all_candidates, _ = load_frozen_inputs()
    selected_indices = np.arange(users, dtype=np.int64)
    local_indices = selected_indices[rank::world]
    local_uids = all_uids[local_indices]
    output = OUTPUT_ROOT / args.scope
    partial = output.with_name(output.name + ".partial")
    if rank == 0:
        if output.exists() or partial.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
        partial.mkdir(parents=True)
        atomic_json(
            partial / "configuration.json",
            {
                "contract_sha256": contract_hash,
                "scope": args.scope,
                "users": users,
                "methods": list(METHODS),
                "coefficient_source": "Current_Exact_oracle",
                "executable_estimator": False,
                "labels_read": False,
            },
        )
    dist.barrier()
    rank_output = partial / f"rank{rank}"
    rank_output.mkdir()

    history = load_histories(
        local_uids.tolist(),
        oov_buckets=OOV_BUCKETS,
        dataset_path=DATASET,
        known_vocab_size=KNOWN_ITEMS,
        end_timestamp=(CUTOVER_DAYS[-1] + 14) * 86_400,
        threads=8,
    )
    metric_records: list[dict[str, Any]] = []
    coefficient_records: list[dict[str, Any]] = []
    coverage_records: list[dict[str, Any]] = []
    edge_records = []

    for edge_index, edge in enumerate(EDGES):
        edge_started = time.perf_counter()
        print(json.dumps({"phase": "edge_start", "rank": rank, "edge": edge}), flush=True)
        cutover = CUTOVER_DAYS[edge_index] * 86_400
        stop = (CUTOVER_DAYS[edge_index] + 14) * 86_400
        groups_by_uid = request_groups(
            V5_REQUESTS if edge_index == 4 else PRIMARY_REQUESTS,
            local_uids,
            cutover,
            stop,
        )
        _, items_np, behaviors_np, deltas_np, query_deltas_np = histories_at_cutover(
            history, local_uids, cutover
        )
        parent, parent_payload = load_model(checkpoint(edge_index), device)
        current, current_payload = load_model(checkpoint(edge_index + 1), device)
        verify_model_payload(parent_payload)
        verify_model_payload(current_payload)
        active = requests = 0
        for offset, (global_index, uid) in enumerate(
            zip(local_indices, local_uids, strict=True)
        ):
            groups = groups_by_uid.get(int(uid), [])
            coverage_records.append(
                {
                    "edge": edge,
                    "uid": int(uid),
                    "user_index": int(global_index),
                    "request_groups": len(groups),
                    "active": bool(groups),
                }
            )
            if not groups:
                continue
            active += 1
            requests += len(groups)
            evaluate_user(
                uid=int(uid),
                user_index=int(global_index),
                raw_history=history.rows[int(uid)],
                prefix=(
                    items_np[offset],
                    behaviors_np[offset],
                    deltas_np[offset],
                    float(query_deltas_np[offset]),
                ),
                groups=groups,
                panel=all_candidates[edge_index, global_index],
                parent=parent,
                current=current,
                edge=edge,
                cutover=cutover,
                stop=stop,
                metric_records=metric_records,
                coefficient_records=coefficient_records,
            )
        edge_records.append(
            {
                "edge": edge,
                "active_users": active,
                "request_groups": requests,
                "seconds": time.perf_counter() - edge_started,
            }
        )
        print(
            json.dumps(
                {
                    "phase": "edge_complete",
                    "rank": rank,
                    **edge_records[-1],
                }
            ),
            flush=True,
        )
        del parent, current
        torch.cuda.empty_cache()

    pd.DataFrame(metric_records).to_parquet(rank_output / "metrics.parquet", index=False)
    pd.DataFrame(coefficient_records).to_parquet(
        rank_output / "coefficients.parquet", index=False
    )
    pd.DataFrame(coverage_records).to_parquet(rank_output / "coverage.parquet", index=False)
    atomic_json(
        rank_output / "summary.json",
        {
            "rank": rank,
            "metric_rows": len(metric_records),
            "coefficient_rows": len(coefficient_records),
            "coverage_rows": len(coverage_records),
            "edges": edge_records,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        },
    )
    dist.barrier()

    if rank == 0:
        artifacts = {}
        combined: dict[str, pd.DataFrame] = {}
        for name in ("metrics", "coefficients", "coverage"):
            frame = pd.concat(
                [
                    pd.read_parquet(partial / f"rank{shard}/{name}.parquet")
                    for shard in range(world)
                ],
                ignore_index=True,
            )
            path = partial / f"{name}.parquet"
            frame.to_parquet(path, index=False)
            combined[name] = frame
            artifacts[path.name] = {
                "rows": len(frame),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        expected_requests = int(combined["coverage"].request_groups.sum())
        finite = all(
            np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all()
            for frame in combined.values()
        )
        passed = bool(
            len(combined["coverage"]) == users * len(EDGES)
            and len(combined["coefficients"]) == expected_requests
            and len(combined["metrics"]) == expected_requests * 2 * len(METHODS)
            and finite
        )
        rank_summaries = [
            json.loads((partial / f"rank{shard}/summary.json").read_text(encoding="utf-8"))
            for shard in range(world)
        ]
        summary = {
            "status": "temporal_coordinate_instrumentation_passed" if passed else "temporal_coordinate_instrumentation_failed",
            "passed": passed,
            "contract_sha256": contract_hash,
            "scope": args.scope,
            "users": users,
            "edges": list(EDGES),
            "request_groups": expected_requests,
            "labels_read": False,
            "oracle_coefficients_only": True,
            "all_numeric_rows_finite": bool(finite),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_allocated_mib": max(row["peak_allocated_mib"] for row in rank_summaries),
            "peak_reserved_mib": max(row["peak_reserved_mib"] for row in rank_summaries),
            "artifacts": artifacts,
        }
        atomic_json(partial / "summary.json", summary)
        partial.replace(output)
        print(json.dumps(summary, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
