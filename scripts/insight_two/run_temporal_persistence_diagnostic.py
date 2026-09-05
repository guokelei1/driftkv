#!/usr/bin/env python3
"""Run the frozen-cutover S4 persistence diagnostic on real Medium timelines."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "insight"))

from evaluate_yambda500m_foundation_raw import load_histories, load_model  # noqa: E402
from hstu_kvcache.models import HSTUKVCache  # noqa: E402
from hstu_kvcache.models.state_transition import append_with_rolling_cap  # noqa: E402
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
    metrics_row,
    score_metrics,
    sha256_file,
    verify_contract as verify_boundary_contract,
    verify_model_payload,
)
from insight_two.temporal_persistence import (  # noqa: E402
    append_bucket,
    correction_drift,
    correction_sha256,
    remaining_parent_fraction,
    scale_correction,
    time_bucket,
)
from reader_compatibility_correction import (  # noqa: E402
    _stage_path,
    intervene_reader_correction,
)


CONTRACT = (
    ROOT
    / "configs/contracts/yambda500m_medium_hstu_native_insight2_temporal_persistence_v1.yaml"
)
OUTPUT_ROOT = RESULT_ROOT / "diagnostic_temporal_persistence_v1"
PRIMARY_REQUESTS = (
    ROOT
    / "data/manifests/yambda500m_medium_hstu_native_d7_d14_v1/requests_fidelity.parquet"
)
V5_REQUESTS = (
    ROOT
    / "data/manifests/yambda500m_medium_hstu_native_d14_v5_extension_v1/requests_fidelity.parquet"
)
CANARY_USERS = 32
DISCOVERY_USERS = 512
REAL_WIDTH_CAP = 64


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def verify_contract() -> tuple[dict[str, Any], str]:
    verify_boundary_contract()
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    if contract["scope"]["edges"] != list(EDGES):
        raise RuntimeError("temporal-persistence edge order differs")
    if int(contract["rolling_semantics"]["cache_cap"]) != HISTORY:
        raise RuntimeError("temporal-persistence cache cap differs")
    for record in contract["frozen_inputs"].values():
        path = ROOT / record["path"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"frozen temporal-persistence input differs: {path}")
    return contract, sha256_file(CONTRACT)


def require_discovery_gate(contract_hash: str) -> None:
    canary = OUTPUT_ROOT / "canary/summary.json"
    estimate = OUTPUT_ROOT / "resource_estimate.json"
    if not canary.is_file() or not estimate.is_file():
        raise RuntimeError("persistence discovery requires canary and resource estimate")
    canary_payload = json.loads(canary.read_text(encoding="utf-8"))
    estimate_payload = json.loads(estimate.read_text(encoding="utf-8"))
    if not canary_payload.get("passed") or canary_payload.get("contract_sha256") != contract_hash:
        raise RuntimeError("persistence canary did not pass instrumentation")
    if estimate_payload.get("contract_sha256") != contract_hash:
        raise RuntimeError("persistence resource estimate differs")
    if float(estimate_payload.get("estimated_512_user_minutes", 1e9)) > 30:
        raise RuntimeError("persistence discovery exceeds the interactive limit")


def distributed_context() -> tuple[int, int, int]:
    if "RANK" not in os.environ:
        raise RuntimeError("temporal-persistence diagnostic requires torchrun")
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != 4:
        raise RuntimeError(f"temporal-persistence diagnostic requires four ranks, got {world}")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world


def request_groups(
    path: Path, uids: np.ndarray, start: int, stop: int
) -> dict[int, list[dict[str, Any]]]:
    table = pq.read_table(
        path,
        filters=[
            ("time_block", "=", "matrix_horizon"),
            ("target_known", "=", True),
            ("query_timestamp", ">=", start),
            ("query_timestamp", "<", stop),
            ("uid", "in", [int(uid) for uid in uids]),
        ],
        columns=["request_id", "uid", "query_timestamp", "item_idx"],
    ).to_pandas()
    output: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for (uid, query_timestamp), rows in table.groupby(
        ["uid", "query_timestamp"], sort=True
    ):
        ordered = rows.sort_values(["item_idx", "request_id"]).iloc[:REAL_WIDTH_CAP]
        output[int(uid)].append(
            {
                "uid": int(uid),
                "query_timestamp": int(query_timestamp),
                "observed_group_size": int(len(rows)),
                "items": ordered.item_idx.to_numpy(dtype=np.int64),
                "request_ids": ordered.request_id.astype(str).tolist(),
            }
        )
    return dict(output)


def stack_pair(exact: HSTUKVCache, reuse: HSTUKVCache) -> HSTUKVCache:
    if exact.seq_len != reuse.seq_len:
        raise ValueError("Exact and Reuse cache lengths differ")
    return HSTUKVCache(
        k=torch.cat((exact.k, reuse.k), dim=1),
        v=torch.cat((exact.v, reuse.v), dim=1),
        seq_len=exact.seq_len,
    )


def split_pair(cache: HSTUKVCache) -> tuple[HSTUKVCache, HSTUKVCache]:
    if cache.k.shape[1] != 2:
        raise ValueError("rolling pair cache must have batch dimension two")
    return (
        HSTUKVCache(cache.k[:, :1], cache.v[:, :1], cache.seq_len),
        HSTUKVCache(cache.k[:, 1:], cache.v[:, 1:], cache.seq_len),
    )


def append_timestamp_events(
    current,
    exact: HSTUKVCache,
    reuse: HSTUKVCache,
    *,
    event_items: np.ndarray,
    event_behaviors: np.ndarray,
    timestamp: int,
    last_timestamp: int,
    device: torch.device,
) -> tuple[HSTUKVCache, HSTUKVCache]:
    count = len(event_items)
    if count < 1 or len(event_behaviors) != count or timestamp < last_timestamp:
        raise ValueError("invalid chronological append group")
    items = torch.as_tensor(event_items[None], dtype=torch.long, device=device)
    behaviors = torch.as_tensor(event_behaviors[None], dtype=torch.long, device=device)
    deltas = torch.zeros((1, count), dtype=torch.float32, device=device)
    deltas[0, 0] = float(max(0, min(7 * 86_400, timestamp - last_timestamp)))
    updated = append_with_rolling_cap(
        current,
        stack_pair(exact, reuse),
        items.repeat(2, 1),
        behaviors.repeat(2, 1),
        deltas.repeat(2, 1),
        HISTORY,
    )
    return split_pair(updated)


def append_metric_rows(
    records: list[dict[str, Any]],
    *,
    common: dict[str, Any],
    exact: torch.Tensor,
    reuse: torch.Tensor,
    methods: dict[str, torch.Tensor],
) -> None:
    for method, scores in methods.items():
        records.append(
            {
                **common,
                "method": method,
                **metrics_row(score_metrics(exact, reuse, scores)),
            }
        )


@torch.inference_mode()
def evaluate_user(
    *,
    uid: int,
    user_index: int,
    raw_history: tuple[np.ndarray, np.ndarray, np.ndarray],
    prefix: tuple[np.ndarray, np.ndarray, np.ndarray, float],
    groups: list[dict[str, Any]],
    panel: np.ndarray,
    current,
    parent,
    edge: str,
    cutover: int,
    stop: int,
    verify_full_delta: bool,
    metric_records: list[dict[str, Any]],
    drift_records: list[dict[str, Any]],
    correctness_records: list[dict[str, Any]],
) -> None:
    device = next(current.parameters()).device
    prefix_items, prefix_behaviors, prefix_deltas, cutover_query_delta = prefix
    item_tensor = torch.as_tensor(prefix_items[None], dtype=torch.long, device=device)
    behavior_tensor = torch.as_tensor(
        prefix_behaviors[None], dtype=torch.long, device=device
    )
    delta_tensor = torch.as_tensor(prefix_deltas[None], dtype=torch.float32, device=device)
    exact_cache = current.compute_kv(item_tensor, behavior_tensor, delta_tensor)
    reuse_cache = parent.compute_kv(item_tensor, behavior_tensor, delta_tensor)
    anchors = torch.as_tensor(
        panel[None, ANCHOR_INDICES], dtype=torch.long, device=device
    )
    heldout = torch.as_tensor(
        panel[None, HELDOUT_INDICES], dtype=torch.long, device=device
    )
    cutover_delta = torch.tensor([cutover_query_delta], dtype=torch.float32, device=device)
    _, cutover_correction, _ = _stage_path(
        current,
        exact_cache,
        reuse_cache,
        anchors,
        cutover_delta,
        stage="av_aggregation",
        mode="shared",
    )
    frozen = tuple(value.detach().clone() for value in cutover_correction)
    frozen_hash = correction_sha256(frozen)

    exact_cutover = current.score_cc_reuse(exact_cache, heldout, cutover_delta)
    reuse_cutover = current.score_cc_reuse(reuse_cache, heldout, cutover_delta)
    frozen_cutover, _ = intervene_reader_correction(
        current,
        reuse_cache,
        heldout,
        cutover_delta,
        stage="av_aggregation",
        corrections=frozen,
    )
    common = {
        "edge": edge,
        "uid": uid,
        "user_index": user_index,
        "phase": "cutover",
        "candidate_source": "fixed_heldout_panel",
        "query_timestamp": cutover,
        "seconds_since_cutover": 0,
        "time_bucket": "cutover",
        "append_count": 0,
        "append_bucket": "0",
        "remaining_parent_fraction": 1.0,
        "candidate_count": len(HELDOUT_INDICES),
    }
    append_metric_rows(
        metric_records,
        common=common,
        exact=exact_cutover,
        reuse=reuse_cutover,
        methods={
            "current_reuse": reuse_cutover,
            "same_request_S4_oracle": frozen_cutover,
            "frozen_cutover_S4": frozen_cutover,
            "coverage_scaled_frozen_cutover_S4": frozen_cutover,
        },
    )
    if verify_full_delta:
        full_readout, _, _ = _stage_path(
            current,
            exact_cache,
            reuse_cache,
            anchors,
            cutover_delta,
            stage="av_aggregation",
            mode="full",
        )
        _, native_exact_readout = current.observe_cc_reuse(
            exact_cache, anchors, cutover_delta
        )
        correctness_records.append(
            {
                "edge": edge,
                "uid": uid,
                "full_S4_delta_max_abs_readout_error": float(
                    torch.max(torch.abs(full_readout - native_exact_readout))
                ),
            }
        )

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

        if query_timestamp < last_timestamp:
            raise RuntimeError("request timeline is not causal")
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
        scale = remaining_parent_fraction(append_count, HISTORY)
        scaled = scale_correction(frozen, scale)
        drift = correction_drift(current_correction, frozen)
        seconds = query_timestamp - cutover
        drift_records.append(
            {
                "edge": edge,
                "uid": uid,
                "user_index": user_index,
                "query_timestamp": query_timestamp,
                "seconds_since_cutover": seconds,
                "time_bucket": time_bucket(seconds),
                "append_count": append_count,
                "append_bucket": append_bucket(append_count),
                "remaining_parent_fraction": scale,
                "direction_cosine": float(drift["direction_cosine"][0]),
                "current_norm": float(drift["current_norm"][0]),
                "frozen_norm": float(drift["frozen_norm"][0]),
                "current_to_frozen_norm_ratio": float(
                    drift["current_to_frozen_norm_ratio"][0]
                ),
                "relative_l2": float(drift["relative_l2"][0]),
                "frozen_correction_sha256": frozen_hash,
                "frozen_hash_unchanged": correction_sha256(frozen) == frozen_hash,
            }
        )

        exact_fixed = current.score_cc_reuse(exact_cache, heldout, query_delta)
        reuse_fixed = current.score_cc_reuse(reuse_cache, heldout, query_delta)
        same_fixed, _ = intervene_reader_correction(
            current,
            reuse_cache,
            heldout,
            query_delta,
            stage="av_aggregation",
            corrections=current_correction,
        )
        frozen_fixed, _ = intervene_reader_correction(
            current,
            reuse_cache,
            heldout,
            query_delta,
            stage="av_aggregation",
            corrections=frozen,
        )
        scaled_fixed, _ = intervene_reader_correction(
            current,
            reuse_cache,
            heldout,
            query_delta,
            stage="av_aggregation",
            corrections=scaled,
        )
        common = {
            "edge": edge,
            "uid": uid,
            "user_index": user_index,
            "phase": "rolling",
            "candidate_source": "fixed_heldout_panel",
            "query_timestamp": query_timestamp,
            "seconds_since_cutover": seconds,
            "time_bucket": time_bucket(seconds),
            "append_count": append_count,
            "append_bucket": append_bucket(append_count),
            "remaining_parent_fraction": scale,
            "candidate_count": len(HELDOUT_INDICES),
        }
        append_metric_rows(
            metric_records,
            common=common,
            exact=exact_fixed,
            reuse=reuse_fixed,
            methods={
                "current_reuse": reuse_fixed,
                "same_request_S4_oracle": same_fixed,
                "frozen_cutover_S4": frozen_fixed,
                "coverage_scaled_frozen_cutover_S4": scaled_fixed,
            },
        )

        real_candidates = torch.as_tensor(
            np.asarray(group["items"], dtype=np.int64)[None],
            dtype=torch.long,
            device=device,
        )
        exact_real = current.score_cc_reuse(exact_cache, real_candidates, query_delta)
        reuse_real = current.score_cc_reuse(reuse_cache, real_candidates, query_delta)
        same_real, _ = intervene_reader_correction(
            current,
            reuse_cache,
            real_candidates,
            query_delta,
            stage="av_aggregation",
            corrections=current_correction,
        )
        frozen_real, _ = intervene_reader_correction(
            current,
            reuse_cache,
            real_candidates,
            query_delta,
            stage="av_aggregation",
            corrections=frozen,
        )
        scaled_real, _ = intervene_reader_correction(
            current,
            reuse_cache,
            real_candidates,
            query_delta,
            stage="av_aggregation",
            corrections=scaled,
        )
        append_metric_rows(
            metric_records,
            common={
                **common,
                "candidate_source": "real_exposed_items",
                "candidate_count": int(real_candidates.shape[1]),
            },
            exact=exact_real,
            reuse=reuse_real,
            methods={
                "current_reuse": reuse_real,
                "same_request_S4_oracle": same_real,
                "frozen_cutover_S4": frozen_real,
                "coverage_scaled_frozen_cutover_S4": scaled_real,
            },
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

    if correction_sha256(frozen) != frozen_hash:
        raise RuntimeError("cutover correction mutated during rolling evaluation")
    if exact_cache.seq_len != reuse_cache.seq_len:
        raise RuntimeError("Exact and Reuse rolling lengths diverged")


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
                "labels_read": False,
                "oracle_boundary_persistence_only": True,
                "primary_method": "coverage_scaled_frozen_cutover_S4",
                "aggregation": "request_mean_within_user_then_user_equal_within_edge_then_edge_equal",
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
    drift_records: list[dict[str, Any]] = []
    coverage_records: list[dict[str, Any]] = []
    correctness_records: list[dict[str, Any]] = []
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
        active = 0
        requests = 0
        for local_offset, (global_index, uid) in enumerate(
            zip(local_indices, local_uids, strict=True)
        ):
            groups = groups_by_uid.get(int(uid), [])
            coverage_records.append(
                {
                    "edge": edge,
                    "uid": int(uid),
                    "user_index": int(global_index),
                    "request_groups": len(groups),
                    "request_candidates": sum(len(group["items"]) for group in groups),
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
                    items_np[local_offset],
                    behaviors_np[local_offset],
                    deltas_np[local_offset],
                    float(query_deltas_np[local_offset]),
                ),
                groups=groups,
                panel=all_candidates[edge_index, global_index],
                current=current,
                parent=parent,
                edge=edge,
                cutover=cutover,
                stop=stop,
                verify_full_delta=True,
                metric_records=metric_records,
                drift_records=drift_records,
                correctness_records=correctness_records,
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
                    "edge": edge,
                    "active_users": active,
                    "request_groups": requests,
                    "seconds": edge_records[-1]["seconds"],
                }
            ),
            flush=True,
        )
        del parent, current
        torch.cuda.empty_cache()

    pd.DataFrame(metric_records).to_parquet(rank_output / "metrics.parquet", index=False)
    pd.DataFrame(drift_records).to_parquet(rank_output / "drift.parquet", index=False)
    pd.DataFrame(coverage_records).to_parquet(rank_output / "coverage.parquet", index=False)
    pd.DataFrame(correctness_records).to_parquet(
        rank_output / "correctness.parquet", index=False
    )
    atomic_json(
        rank_output / "summary.json",
        {
            "rank": rank,
            "uids": len(local_uids),
            "metric_rows": len(metric_records),
            "drift_rows": len(drift_records),
            "coverage_rows": len(coverage_records),
            "correctness_rows": len(correctness_records),
            "edges": edge_records,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        },
    )
    dist.barrier()

    if rank == 0:
        artifacts = {}
        for name in ("metrics", "drift", "coverage", "correctness"):
            frames = [
                pd.read_parquet(partial / f"rank{shard}/{name}.parquet")
                for shard in range(world)
            ]
            combined = pd.concat(frames, ignore_index=True)
            path = partial / f"{name}.parquet"
            combined.to_parquet(path, index=False)
            artifacts[path.name] = {
                "rows": len(combined),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        metrics = pd.read_parquet(partial / "metrics.parquet")
        coverage = pd.read_parquet(partial / "coverage.parquet")
        correctness = pd.read_parquet(partial / "correctness.parquet")
        drift = pd.read_parquet(partial / "drift.parquet")
        maximum_error = (
            float(correctness.full_S4_delta_max_abs_readout_error.max())
            if len(correctness)
            else float("inf")
        )
        finite = bool(
            np.isfinite(metrics.select_dtypes(include=[np.number]).to_numpy()).all()
            and np.isfinite(drift.select_dtypes(include=[np.number]).to_numpy()).all()
        )
        passed = bool(
            len(coverage) == users * len(EDGES)
            and maximum_error <= 2e-5
            and bool(drift.frozen_hash_unchanged.all())
            and finite
        )
        rank_summaries = [
            json.loads((partial / f"rank{shard}/summary.json").read_text(encoding="utf-8"))
            for shard in range(world)
        ]
        summary = {
            "status": "temporal_persistence_instrumentation_passed" if passed else "temporal_persistence_instrumentation_failed",
            "passed": passed,
            "contract_sha256": contract_hash,
            "scope": args.scope,
            "users": users,
            "edges": list(EDGES),
            "labels_read": False,
            "oracle_boundary_persistence_only": True,
            "elapsed_seconds": time.perf_counter() - started,
            "maximum_full_S4_delta_readout_error": maximum_error,
            "all_numeric_rows_finite": finite,
            "active_users_by_edge": coverage.groupby("edge").active.sum().astype(int).to_dict(),
            "request_groups_by_edge": coverage.groupby("edge").request_groups.sum().astype(int).to_dict(),
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
