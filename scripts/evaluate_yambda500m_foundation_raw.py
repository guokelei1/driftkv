#!/usr/bin/env python3
"""Four-GPU raw-first evaluation for contract-fixed Small release edges."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.distributed as dist

from hstu_kvcache.evaluation import (
    OneHopRollingBundle, append_timestamp_group, materialize_state,
    timestamp_groups, validate_raw_table,
)
from hstu_kvcache.data.yambda_history import load_yambda_histories
from hstu_kvcache.models import HSTU, HSTUConfig, HSTUKVCache
from hstu_kvcache.training import FoundationHistoryIndex


DAY = 86_400
ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/processed/yambda500m_unified_v1/scales/small/dataset.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(
    path: Path, device: torch.device, *, allow_canary: bool = False,
) -> tuple[HSTU, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    status = str(payload.get("status", ""))
    status_ok = status.startswith("formal_") and status.endswith("_checkpoint")
    status_ok = status_ok or (
        allow_canary and status in {"four_gpu_canary_checkpoint", "distributed_canary_checkpoint"}
    )
    if payload.get("progress") != 1.0 or not status_ok:
        raise RuntimeError(f"evaluation requires a formal final checkpoint: {path}")
    model = HSTU(HSTUConfig(**payload["config"])
                 ).to(device)
    model.load_state_dict(payload["model"])
    return model.eval(), payload


def balanced_users(rows: list[dict], world: int) -> dict[int, int]:
    counts: dict[int, int] = {}
    for row in rows:
        counts[int(row["uid"])] = counts.get(int(row["uid"]), 0) + 1
    loads = [0] * world; assignment = {}
    for uid, count in sorted(counts.items(), key=lambda value: (-value[1], value[0])):
        rank = min(range(world), key=lambda value: (loads[value], value))
        assignment[uid] = rank; loads[rank] += count
    return assignment


def load_histories(
    uids: list[int], *, oov_buckets: int = 0, dataset_path: Path = DATASET,
    known_vocab_size: int | None = None, start_timestamp: int | None = None,
    end_timestamp: int | None = None, max_history: int | None = None,
    threads: int = 4,
) -> FoundationHistoryIndex:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    known = int(known_vocab_size or dataset["foundation_items"])
    return load_yambda_histories(
        dataset_path, uids, known_vocab_size=known, oov_buckets=oov_buckets,
        start_timestamp=start_timestamp,
        max_pre_events=max_history if start_timestamp is not None else None,
        end_timestamp=int(end_timestamp or (2**63 - 1)),
        threads=threads,
    )


@torch.inference_mode()
def full_observation(model: HSTU, events, candidate: int, query_time: int):
    values = events[-512:]
    device = next(model.parameters()).device
    times = torch.tensor([[event[0] for event in values]], dtype=torch.long, device=device)
    deltas = torch.zeros_like(times, dtype=torch.float32)
    if len(values) > 1:
        deltas[:, 1:] = times[:, 1:] - times[:, :-1]
    items = torch.tensor([[event[1] for event in values]], dtype=torch.long, device=device)
    behaviors = torch.tensor([[event[2] for event in values]], dtype=torch.long, device=device)
    candidate_ids = torch.tensor([[candidate]], dtype=torch.long, device=device)
    query_delta = torch.tensor([float(query_time - values[-1][0])], device=device)
    scores, readout = model.observe_cc_full(items, behaviors, deltas, candidate_ids, query_delta)
    return float(scores[0, 0]), readout[0, 0].float().cpu()


def stacked_cache(states) -> HSTUKVCache:
    values = [getattr(state, "cache", state) for state in states]
    return HSTUKVCache(
        k=torch.cat([state.k for state in values], dim=1),
        v=torch.cat([state.v for state in values], dim=1),
        seq_len=values[0].seq_len,
    )


def select_cache(cache: HSTUKVCache, indices: list[int]) -> HSTUKVCache:
    index = torch.tensor(indices, dtype=torch.long, device=cache.k.device)
    return HSTUKVCache(cache.k.index_select(1, index), cache.v.index_select(1, index), cache.seq_len)


def assign_cache(cache: HSTUKVCache, indices: list[int], values: HSTUKVCache) -> None:
    index = torch.tensor(indices, dtype=torch.long, device=cache.k.device)
    cache.k.index_copy_(1, index, values.k); cache.v.index_copy_(1, index, values.v)


def clone_cache(cache: HSTUKVCache) -> HSTUKVCache:
    return HSTUKVCache(cache.k.clone(), cache.v.clone(), cache.seq_len)


@torch.inference_mode()
def batched_recursive_cache(raw, lineage_models, cutover: int, device: torch.device) -> HSTUKVCache:
    """Build the true persistent lineage for full-cache cohorts."""
    from hstu_kvcache.models.state_transition import append_with_rolling_cap

    base_cutover = 224 * DAY
    prefixes = []
    for timestamps, items, behaviors in raw:
        stop = int(np.searchsorted(timestamps, base_cutover, side="left"))
        if stop < 512:
            raise ValueError("batched recursive lineage requires a full Day-224 cache")
        prefixes.append((timestamps[stop-512:stop], items[stop-512:stop], behaviors[stop-512:stop]))
    times = torch.tensor(np.stack([value[0] for value in prefixes]), dtype=torch.long, device=device)
    items = torch.tensor(np.stack([value[1] for value in prefixes]), dtype=torch.long, device=device)
    behaviors = torch.tensor(np.stack([value[2] for value in prefixes]), dtype=torch.long, device=device)
    deltas = torch.zeros_like(times, dtype=torch.float32); deltas[:, 1:] = times[:, 1:] - times[:, :-1]
    cache = lineage_models[0][1].compute_kv(items, behaviors, deltas)
    last_times = np.asarray([int(value[0][-1]) for value in prefixes], dtype=np.int64)
    positions = np.asarray([
        int(np.searchsorted(value[0], base_cutover, side="left")) for value in raw
    ], dtype=np.int64)
    stops = np.asarray([
        int(np.searchsorted(value[0], cutover, side="left")) for value in raw
    ], dtype=np.int64)
    while True:
        active = np.flatnonzero(positions < stops).tolist()
        if not active:
            break
        by_producer = {}
        for index in active:
            timestamp = int(raw[index][0][positions[index]])
            producer_index = 1 + (timestamp // DAY - 224) // 7
            if not 1 <= producer_index < len(lineage_models):
                raise RuntimeError("recursive event has no registered producer")
            by_producer.setdefault(producer_index, []).append(index)
        for producer_index, indices in sorted(by_producer.items()):
            event_items = torch.tensor(
                [[int(raw[index][1][positions[index]])] for index in indices],
                dtype=torch.long, device=device,
            )
            event_behaviors = torch.tensor(
                [[int(raw[index][2][positions[index]])] for index in indices],
                dtype=torch.long, device=device,
            )
            event_times = np.asarray(
                [int(raw[index][0][positions[index]]) for index in indices], dtype=np.int64
            )
            event_deltas = torch.tensor(
                [[float(max(0, min(7 * DAY, timestamp-last_times[index])))]
                 for index, timestamp in zip(indices, event_times, strict=True)],
                device=device,
            )
            updated = append_with_rolling_cap(
                lineage_models[producer_index][1], select_cache(cache, indices),
                event_items, event_behaviors, event_deltas, 512,
            )
            assign_cache(cache, indices, updated)
            for index, timestamp in zip(indices, event_times, strict=True):
                last_times[index] = timestamp; positions[index] += 1
    return cache


@torch.inference_mode()
def rolling_observations(parent: HSTU, current: HSTU, bundle, candidate: int, query_time: int):
    device = next(current.parameters()).device
    parent_delta = torch.tensor([float(query_time-bundle.parent_exact.last_timestamp)], device=device)
    parent_score, parent_readout = parent.observe_cc_reuse(
        bundle.parent_exact.cache, torch.tensor([[candidate]], device=device), parent_delta
    )
    states = (bundle.current_exact, bundle.one_hop_reuse, bundle.recursive_reuse)
    if len({state.cache.seq_len for state in states}) != 1:
        raise RuntimeError("rolling path cache lengths diverged")
    cache = stacked_cache(states)
    candidates = torch.full((3, 1), candidate, dtype=torch.long, device=device)
    deltas = torch.tensor([float(query_time-state.last_timestamp) for state in states], device=device)
    scores, readouts = current.observe_cc_reuse(cache, candidates, deltas)
    return {
        "parent_exact_rolling": (float(parent_score[0, 0]), parent_readout[0, 0].float().cpu()),
        "current_exact_rolling": (float(scores[0, 0]), readouts[0, 0].float().cpu()),
        "one_hop_reuse_rolling": (float(scores[1, 0]), readouts[1, 0].float().cpu()),
        "recursive_reuse_rolling": (float(scores[2, 0]), readouts[2, 0].float().cpu()),
    }


def recursive_state_for_user(events, lineage_models, cutover: int):
    """Materialize v0 at Day 224, then replay each released producer interval."""
    base_cutover = 224 * DAY
    version, model = lineage_models[0]
    state = materialize_state(
        model, (event for event in events if event[0] < base_cutover),
        producer_version=version, max_length=512,
    )
    for index, (version, model) in enumerate(lineage_models[1:], start=1):
        start = (224 + 7 * (index - 1)) * DAY
        end = min(start + 7 * DAY, cutover)
        for _, group in timestamp_groups(event for event in events if start <= event[0] < end):
            state = append_timestamp_group(
                model, state, group, producer_version=version, max_length=512
            )
    return state


def evaluate_user(*, requests, history, parent, current, parent_name, current_name, edge,
                  checkpoint_hash, parent_hash, manifest_hash, cutover,
                  lineage_models):
    timestamps, items, behaviors = history.rows[int(requests[0]["uid"])]
    events = [(int(t), int(i), int(b)) for t, i, b in zip(timestamps, items, behaviors, strict=True)]
    prefix = [event for event in events if event[0] < cutover]
    recursive_state = recursive_state_for_user(events, lineage_models, cutover)
    bundle = OneHopRollingBundle.at_cutover(
        parent, current, prefix, parent_version=parent_name, current_version=current_name,
        max_length=512, recursive_state=recursive_state,
    )
    post_groups = list(timestamp_groups(event for event in events if event[0] >= cutover))
    group_index = 0; append_count = 0; rolling_evictions = 0; output = []
    request_groups = {}
    for request in requests:
        request_groups.setdefault(int(request["query_timestamp"]), []).append(request)
    for query_time, simultaneous_requests in sorted(request_groups.items()):
        while group_index < len(post_groups) and post_groups[group_index][0] < query_time:
            _, group = post_groups[group_index]
            before = bundle.current_exact.cache.seq_len
            rolling_evictions += max(0, before + len(group) - 512)
            bundle.append_group(parent, current, group, parent_version=parent_name,
                                current_version=current_name, max_length=512)
            append_count += len(group); group_index += 1
        stop = int(np.searchsorted(timestamps, query_time, side="left"))
        full_events = events[max(0, stop-512):stop]
        for request in simultaneous_requests:
            candidate = int(request["item_idx"])
            observations = {
                "parent_full_request_local": full_observation(parent, full_events, candidate, query_time),
                "current_full_request_local": full_observation(current, full_events, candidate, query_time),
                **rolling_observations(parent, current, bundle, candidate, query_time),
            }
            reference = observations["current_exact_rolling"][1]
            for path, (score, readout) in observations.items():
                output.append({
                    "request_id": request["request_id"], "uid": int(request["uid"]),
                    "query_timestamp": query_time, "edge": edge, "path": path,
                    "hstu_logit": score, "architecture": "hstu_native_cc",
                    "append_count_since_cutover": append_count,
                    "seconds_since_cutover": query_time-cutover,
                    "history_length": len(full_events), "cache_length": bundle.current_exact.cache.seq_len,
                    "rolling_evictions": rolling_evictions,
                    "readout_normalized_l2": float((readout-reference).norm()/(reference.norm()+1e-12)),
                    "readout_cosine": float(torch.nn.functional.cosine_similarity(readout[None], reference[None])),
                    "checkpoint_sha256": checkpoint_hash, "parent_checkpoint_sha256": parent_hash,
                    "manifest_sha256": manifest_hash,
                })
        # Atomicity: same-timestamp listens are appended only after every query is scored.
        while group_index < len(post_groups) and post_groups[group_index][0] == query_time:
            _, group = post_groups[group_index]
            before = bundle.current_exact.cache.seq_len
            rolling_evictions += max(0, before + len(group) - 512)
            bundle.append_group(parent, current, group, parent_version=parent_name,
                                current_version=current_name, max_length=512)
            append_count += len(group); group_index += 1
    return output


@torch.inference_mode()
def evaluate_full_cache_cohort(*, uids, by_user, history, parent, current, parent_name,
                               current_name, edge, checkpoint_hash, parent_hash,
                               manifest_hash, cutover, lineage_models,
                               event_end_exclusive: int | None = None,
                               include_request_local: bool = True,
                               include_parent_exact: bool = False,
                               refinement_cast_maps=None,
                               evidence_measure_cast_maps=None,
                               pro_lazy_maps=None,
                               pro_lazy_carriers: int = 32,
                               pro_lazy_repair_width: int = 128,
                               pro_lazy_path: str | None = None,
                               query_chunk_size: int | None = None,
                               max_length: int = 512):
    """Vectorize independent user timelines whose cutover caches are all full."""
    if max_length < 1:
        raise ValueError("max_length must be positive")
    batch = len(uids); device = next(current.parameters()).device
    raw = [history.rows[uid] for uid in uids]
    prefix = []
    for timestamps, items, behaviors in raw:
        stop = int(np.searchsorted(timestamps, cutover, side="left"))
        if stop < max_length:
            raise ValueError("batched cohort requires full cutover caches")
        prefix.append((
            timestamps[stop-max_length:stop], items[stop-max_length:stop],
            behaviors[stop-max_length:stop],
        ))
    times = torch.tensor(np.stack([value[0] for value in prefix]), dtype=torch.long, device=device)
    items = torch.tensor(np.stack([value[1] for value in prefix]), dtype=torch.long, device=device)
    behaviors = torch.tensor(np.stack([value[2] for value in prefix]), dtype=torch.long, device=device)
    deltas = torch.zeros_like(times, dtype=torch.float32); deltas[:, 1:] = times[:, 1:] - times[:, :-1]
    parent_cache = parent.compute_kv(items, behaviors, deltas)
    current_cache = current.compute_kv(items, behaviors, deltas)
    caches = {
        "parent_exact_rolling": parent_cache,
        "current_exact_rolling": current_cache,
        "one_hop_reuse_rolling": clone_cache(parent_cache),
    }
    refinement_path = None
    if refinement_cast_maps is not None:
        from insight.one_release_refinement import OUR_PATH, build_fixed_refinement_cache

        caches[OUR_PATH], layout = build_fixed_refinement_cache(
            parent_cache=parent_cache,
            current=current,
            item_ids=items,
            behaviors=behaviors,
            time_deltas=deltas,
            cast_maps=refinement_cast_maps,
        )
        if (layout.nominal_positions, layout.cast_positions, layout.repair_evidence,
                layout.carriers, layout.padding_positions) != (512, 384, 128, 64, 64):
            raise RuntimeError("full-cache refinement layout differs from frozen r=128,c=64")
        refinement_path = OUR_PATH
    evidence_measure_path = None
    if evidence_measure_cast_maps is not None:
        from insight.one_release_refinement import (
            EVIDENCE_MEASURE_PATH,
            build_evidence_measure_basis_cache,
        )

        caches[EVIDENCE_MEASURE_PATH], layout = build_evidence_measure_basis_cache(
            parent_cache=parent_cache,
            current=current,
            item_ids=items,
            behaviors=behaviors,
            time_deltas=deltas,
            cast_maps=evidence_measure_cast_maps,
        )
        if (layout.nominal_positions, layout.cast_positions, layout.repair_evidence,
                layout.carriers, layout.padding_positions) != (512, 384, 128, 64, 64):
            raise RuntimeError("full-cache evidence-measure layout differs from frozen r=128,c=64")
        evidence_measure_path = EVIDENCE_MEASURE_PATH
    active_pro_path = None
    pro_corrections = None
    if pro_lazy_maps is not None:
        from insight.pro_lazy_reader import (
            build_parent_conditioned_carriers,
            generate_lazy_pro_sidecar,
            pro_path as default_pro_path,
        )

        carrier_cache, layout = build_parent_conditioned_carriers(
            parent_cache=parent_cache,
            current=current,
            item_ids=items,
            behaviors=behaviors,
            time_deltas=deltas,
            repair_width=pro_lazy_repair_width,
            carrier_count=pro_lazy_carriers,
        )
        expected_layout = (
            max_length,
            max_length - pro_lazy_repair_width,
            pro_lazy_repair_width,
            pro_lazy_carriers,
            pro_lazy_repair_width // pro_lazy_carriers,
        )
        if (
            layout.nominal_positions,
            layout.old_positions,
            layout.repair_evidence,
            layout.carriers,
            layout.represented_mass,
        ) != expected_layout:
            raise RuntimeError("full-cache lightweight PRO layout differs from the contract")
        sidecar = generate_lazy_pro_sidecar(
            current,
            parent_cache,
            carrier_cache,
            pro_lazy_maps,
            items[:, -1],
            old_positions=layout.old_positions,
        )
        if sidecar.replay_max_abs_error > 2e-5:
            raise RuntimeError("lightweight PRO cutover replay differs")
        pro_corrections = tuple(value.detach() for value in sidecar.corrections)
        active_pro_path = pro_lazy_path or default_pro_path(pro_lazy_carriers)
    if edge == "v0_to_r0":
        # Producer identity proves the reused and exact rolling states are bitwise
        # identical; retain one state and copy observations after canary validation.
        caches.pop("one_hop_reuse_rolling")
        current_path_names = ("current_exact_rolling",)
    else:
        if len(lineage_models) == 1:
            # On the first natural edge recursive lineage is definitionally one-hop.
            current_path_names = ["current_exact_rolling", "one_hop_reuse_rolling"]
        else:
            caches["recursive_reuse_rolling"] = batched_recursive_cache(
                raw, lineage_models, cutover, device
            )
            current_path_names = [
                "current_exact_rolling", "one_hop_reuse_rolling", "recursive_reuse_rolling"
            ]
        if refinement_path is not None:
            current_path_names.append(refinement_path)
        if evidence_measure_path is not None:
            current_path_names.append(evidence_measure_path)
        current_path_names = tuple(current_path_names)
    last_times = np.asarray([int(value[0][-1]) for value in prefix], dtype=np.int64)
    append_counts = np.zeros(batch, dtype=np.int64); evictions = np.zeros(batch, dtype=np.int64)
    actions = []
    for local, uid in enumerate(uids):
        timestamps, event_items, event_behaviors = raw[local]
        requests_at = {}
        for request in by_user[uid]:
            requests_at.setdefault(int(request["query_timestamp"]), []).append(request)
        post_events = {}
        start = int(np.searchsorted(timestamps, cutover, side="left"))
        for position in range(start, len(timestamps)):
            if event_end_exclusive is not None and int(timestamps[position]) >= event_end_exclusive:
                break
            post_events.setdefault(int(timestamps[position]), []).append(
                (int(timestamps[position]), int(event_items[position]), int(event_behaviors[position]))
            )
        timeline = []
        for timestamp in sorted(set(requests_at) | set(post_events)):
            if timestamp in requests_at:
                timeline.append(("query", timestamp, requests_at[timestamp]))
            for event in sorted(post_events.get(timestamp, ()), key=lambda value: (value[1], value[2])):
                timeline.append(("append", timestamp, event))
        actions.append(timeline)
    positions = np.zeros(batch, dtype=np.int64); output = []
    while True:
        active = [index for index in range(batch) if positions[index] < len(actions[index])]
        if not active:
            break
        append_indices = [index for index in active if actions[index][positions[index]][0] == "append"]
        if append_indices:
            event_values = [actions[index][positions[index]][2] for index in append_indices]
            event_items = torch.tensor([[value[1]] for value in event_values], dtype=torch.long, device=device)
            event_behaviors = torch.tensor([[value[2]] for value in event_values], dtype=torch.long, device=device)
            event_deltas = torch.tensor(
                [[float(max(0, min(7*DAY, value[0]-last_times[index])))]
                 for index, value in zip(append_indices, event_values, strict=True)],
                device=device,
            )
            from hstu_kvcache.models.state_transition import append_with_rolling_cap
            updated_parent = append_with_rolling_cap(
                parent, select_cache(caches["parent_exact_rolling"], append_indices),
                event_items, event_behaviors, event_deltas, max_length,
            )
            assign_cache(caches["parent_exact_rolling"], append_indices, updated_parent)
            current_states = [select_cache(caches[name], append_indices) for name in current_path_names]
            updated_current = append_with_rolling_cap(
                current, stacked_cache(current_states), event_items.repeat(len(current_path_names), 1),
                event_behaviors.repeat(len(current_path_names), 1),
                event_deltas.repeat(len(current_path_names), 1), max_length,
            )
            width = len(append_indices)
            for offset, name in enumerate(current_path_names):
                piece = HSTUKVCache(
                    updated_current.k[:, offset*width:(offset+1)*width],
                    updated_current.v[:, offset*width:(offset+1)*width], max_length,
                )
                assign_cache(caches[name], append_indices, piece)
            for index, value in zip(append_indices, event_values, strict=True):
                last_times[index] = value[0]; append_counts[index] += 1; evictions[index] += 1; positions[index] += 1
        active_after_append = [index for index in range(batch) if positions[index] < len(actions[index])]
        query_indices = [index for index in active_after_append if actions[index][positions[index]][0] == "query"]
        if not query_indices:
            continue
        query_entries = []
        for index in query_indices:
            _, query_time, requests = actions[index][positions[index]]
            query_entries.extend((index, query_time, request) for request in requests)
            positions[index] += 1
        owner = [value[0] for value in query_entries]
        query_times = np.asarray([value[1] for value in query_entries], dtype=np.int64)
        candidates = torch.tensor([[int(value[2]["item_idx"])] for value in query_entries], dtype=torch.long, device=device)
        query_deltas = torch.tensor(
            [float(timestamp-last_times[index]) for index, timestamp in zip(owner, query_times, strict=True)],
            device=device,
        )
        if query_chunk_size is not None and query_chunk_size < 1:
            raise ValueError("query_chunk_size must be positive")
        if query_chunk_size is not None and len(query_entries) > query_chunk_size:
            if include_request_local:
                raise RuntimeError("chunked query reads do not support request-local Full paths")
            parent_score_parts, parent_readout_parts = [], []
            current_score_parts = {name: [] for name in current_path_names}
            current_readout_parts = {name: [] for name in current_path_names}
            for query_start in range(0, len(query_entries), query_chunk_size):
                query_stop = min(query_start + query_chunk_size, len(query_entries))
                chunk_owner = owner[query_start:query_stop]
                chunk_candidates = candidates[query_start:query_stop]
                chunk_deltas = query_deltas[query_start:query_stop]
                if include_parent_exact:
                    scores, readouts = parent.observe_cc_reuse(
                        select_cache(caches["parent_exact_rolling"], chunk_owner),
                        chunk_candidates,
                        chunk_deltas,
                    )
                    parent_score_parts.append(scores.cpu())
                    parent_readout_parts.append(readouts.cpu())
                current_states = [
                    select_cache(caches[name], chunk_owner) for name in current_path_names
                ]
                scores, readouts = current.observe_cc_reuse(
                    stacked_cache(current_states),
                    chunk_candidates.repeat(len(current_path_names), 1),
                    chunk_deltas.repeat(len(current_path_names)),
                )
                chunk_count = query_stop - query_start
                for offset, name in enumerate(current_path_names):
                    current_score_parts[name].append(
                        scores[offset * chunk_count:(offset + 1) * chunk_count].cpu()
                    )
                    current_readout_parts[name].append(
                        readouts[offset * chunk_count:(offset + 1) * chunk_count].cpu()
                    )
            if include_parent_exact:
                parent_scores = torch.cat(parent_score_parts)
                parent_readouts = torch.cat(parent_readout_parts)
            current_scores = torch.cat([
                torch.cat(current_score_parts[name]) for name in current_path_names
            ])
            current_readouts = torch.cat([
                torch.cat(current_readout_parts[name]) for name in current_path_names
            ])
        else:
            if include_request_local or include_parent_exact:
                parent_scores, parent_readouts = parent.observe_cc_reuse(
                    select_cache(caches["parent_exact_rolling"], owner), candidates, query_deltas
                )
            current_states = [select_cache(caches[name], owner) for name in current_path_names]
            current_scores, current_readouts = current.observe_cc_reuse(
                stacked_cache(current_states), candidates.repeat(len(current_path_names), 1),
                query_deltas.repeat(len(current_path_names))
            )
        pro_scores = None
        pro_readouts = None
        if pro_corrections is not None:
            from insight.reader_compatibility_correction import (
                intervene_reader_correction,
                scale_correction,
            )

            pro_score_parts, pro_readout_parts = [], []
            chunk_size = query_chunk_size or len(query_entries)
            for query_start in range(0, len(query_entries), chunk_size):
                query_stop = min(query_start + chunk_size, len(query_entries))
                chunk_owner = owner[query_start:query_stop]
                owner_index = torch.tensor(chunk_owner, dtype=torch.long, device=device)
                factor = torch.as_tensor(
                    np.maximum(0, max_length - evictions[chunk_owner]) / max_length,
                    dtype=torch.float32,
                    device=device,
                )
                selected_corrections = tuple(
                    value.index_select(0, owner_index) for value in pro_corrections
                )
                scaled = scale_correction(selected_corrections, factor)
                scores, readouts = intervene_reader_correction(
                    current,
                    select_cache(caches["one_hop_reuse_rolling"], chunk_owner),
                    candidates[query_start:query_stop],
                    query_deltas[query_start:query_stop],
                    stage="av_aggregation",
                    corrections=scaled,
                )
                pro_score_parts.append(scores.cpu())
                pro_readout_parts.append(readouts.cpu())
            pro_scores = torch.cat(pro_score_parts)
            pro_readouts = torch.cat(pro_readout_parts)
        if include_request_local:
            full_payload = []
            for index, query_time, _ in query_entries:
                timestamps, event_items, event_behaviors = raw[index]
                stop = int(np.searchsorted(timestamps, query_time, side="left")); start = max(0, stop-max_length)
                full_payload.append((timestamps[start:stop], event_items[start:stop], event_behaviors[start:stop]))
            lengths = torch.tensor([len(value[0]) for value in full_payload], dtype=torch.long, device=device)
            width = int(lengths.max())
            full_items = torch.zeros((len(query_entries), width), dtype=torch.long, device=device)
            full_behaviors = torch.zeros_like(full_items); full_deltas = torch.zeros_like(full_items, dtype=torch.float32)
            full_query_deltas = torch.empty(len(query_entries), device=device)
            for row_index, (timestamp_values, item_values, behavior_values) in enumerate(full_payload):
                length = len(timestamp_values)
                full_items[row_index, :length] = torch.as_tensor(item_values, device=device)
                full_behaviors[row_index, :length] = torch.as_tensor(behavior_values, device=device)
                if length > 1:
                    full_deltas[row_index, 1:length] = torch.as_tensor(np.diff(timestamp_values), device=device)
                full_query_deltas[row_index] = float(query_times[row_index]-timestamp_values[-1])
            parent_full_scores, parent_full_readouts = parent.observe_cc_full(
                full_items, full_behaviors, full_deltas, candidates, full_query_deltas, lengths=lengths
            )
            current_full_scores, current_full_readouts = current.observe_cc_full(
                full_items, full_behaviors, full_deltas, candidates, full_query_deltas, lengths=lengths
            )
        count = len(query_entries)
        for row_index, (index, query_time, request) in enumerate(query_entries):
            observations = {
                name: (
                    current_scores[offset * count + row_index, 0],
                    current_readouts[offset * count + row_index, 0],
                )
                for offset, name in enumerate(current_path_names)
            }
            if include_request_local:
                observations.update({
                    "parent_full_request_local": (parent_full_scores[row_index, 0], parent_full_readouts[row_index, 0]),
                    "current_full_request_local": (current_full_scores[row_index, 0], current_full_readouts[row_index, 0]),
                })
            if include_request_local or include_parent_exact:
                observations["parent_exact_rolling"] = (parent_scores[row_index, 0], parent_readouts[row_index, 0])
            if edge == "v0_to_r0":
                observations["one_hop_reuse_rolling"] = observations["current_exact_rolling"]
            if include_request_local and "recursive_reuse_rolling" not in current_path_names:
                observations["recursive_reuse_rolling"] = observations["one_hop_reuse_rolling"]
            if active_pro_path is not None:
                assert pro_scores is not None and pro_readouts is not None
                observations[active_pro_path] = (
                    pro_scores[row_index, 0],
                    pro_readouts[row_index, 0],
                )
            reference = observations["current_exact_rolling"][1].float().cpu()
            for path, (score, readout) in observations.items():
                readout = readout.float().cpu()
                output.append({
                    "request_id": request["request_id"], "uid": int(request["uid"]),
                    "query_timestamp": int(query_time), "edge": edge, "path": path,
                    "hstu_logit": float(score), "architecture": "hstu_native_cc",
                    "append_count_since_cutover": int(append_counts[index]),
                    "seconds_since_cutover": int(query_time-cutover), "history_length": int(lengths[row_index]) if include_request_local else int(min(np.searchsorted(raw[index][0], query_time, side="left"), max_length)),
                    "cache_length": max_length, "rolling_evictions": int(evictions[index]),
                    "readout_normalized_l2": float((readout-reference).norm()/(reference.norm()+1e-12)),
                    "readout_cosine": float(torch.nn.functional.cosine_similarity(readout[None], reference[None])),
                    "checkpoint_sha256": checkpoint_hash, "parent_checkpoint_sha256": parent_hash,
                    "manifest_sha256": manifest_hash,
                })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge", required=True)
    parser.add_argument("--evaluation-block", default="update2")
    parser.add_argument("--cutover-day", type=int, default=224)
    parser.add_argument(
        "--lineage-checkpoints", type=Path, nargs="+",
        help="Ordered v0..parent final checkpoints used to construct recursive reuse",
    )
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-users", type=int, default=0)
    parser.add_argument("--cohort-size", type=int, default=128)
    args = parser.parse_args()
    if args.edge != "v0_to_r0":
        try:
            parent_name, current_name = args.edge.split("_to_")
            parent_index, current_index = int(parent_name[1:]), int(current_name[1:])
        except (ValueError, IndexError) as error:
            raise SystemExit("edge must be vN_to_vN+1 or v0_to_r0") from error
        if current_index != parent_index + 1:
            raise SystemExit("natural edge versions must be consecutive")
        if not args.lineage_checkpoints or len(args.lineage_checkpoints) < current_index:
            raise SystemExit("lineage-checkpoints must contain v0 and end at the parent")
        if args.cutover_day != 224 + 7 * (len(args.lineage_checkpoints) - 1):
            raise SystemExit("cutover day does not match the registered recursive service segments")
    else:
        parent_name, current_name = "v0", "r0"
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(local)
    device = torch.device(f"cuda:{local}")
    try:
        if world != 4:
            raise RuntimeError("formal evaluation requires four ranks")
        if not 1 <= args.cohort_size <= 128:
            raise ValueError("cohort-size must be in [1,128]")
        if rank == 0:
            if args.output.exists():
                raise FileExistsError(f"refusing to overwrite {args.output}")
            args.output.mkdir(parents=True)
        dist.barrier()
        request_path = args.manifest_dir / "requests_fidelity.parquet"
        table = pq.read_table(
            request_path, filters=[("time_block", "=", args.evaluation_block), ("target_known", "=", True)],
            columns=["request_id", "uid", "query_timestamp", "item_idx", "base_features"],
        ).sort_by([("uid", "ascending"), ("query_timestamp", "ascending"), ("request_id", "ascending")])
        all_rows = table.to_pylist(); assignment = balanced_users(all_rows, world)
        selected_uids = sorted(uid for uid, assigned in assignment.items() if assigned == rank)
        if args.max_users:
            selected_uids = selected_uids[:args.max_users]
        selected = set(selected_uids)
        rows = [row for row in all_rows if int(row["uid"]) in selected]
        by_user = {}
        for row in rows:
            by_user.setdefault(int(row["uid"]), []).append(row)
        history = load_histories(selected_uids)
        loaded = {}
        def cached_model(path: Path):
            key = str(path.resolve())
            if key not in loaded:
                loaded[key] = load_model(path, device)
            return loaded[key]
        parent, parent_payload = cached_model(args.parent)
        current, current_payload = cached_model(args.current)
        lineage_models = []
        if args.edge == "v0_to_r0":
            lineage_models = [("v0", parent)]
        else:
            for index, checkpoint in enumerate(args.lineage_checkpoints):
                model, payload = cached_model(checkpoint)
                version = payload.get("version")
                if index == 0 and version != "v0":
                    raise RuntimeError("recursive lineage must begin with v0")
                if index == len(args.lineage_checkpoints) - 1 and version != parent_name:
                    raise RuntimeError("recursive lineage must end with the evaluated parent")
                lineage_models.append((version, model))
            if sha256_file(args.lineage_checkpoints[-1]) != sha256_file(args.parent):
                raise RuntimeError("recursive lineage must end at the evaluated parent")
        if args.edge == "v0_to_r0" and parent_payload["cache_producer_sha256"] != current_payload["cache_producer_sha256"]:
            raise RuntimeError("R0 producer hash is not identical to v0")
        output_rows = []; cutover = args.cutover_day * DAY
        full_uids, fallback_uids = [], []
        post_counts = {}
        for uid in selected_uids:
            timestamps = history.rows[uid][0]
            prefix_length = int(np.searchsorted(timestamps, 224 * DAY, side="left"))
            post_counts[uid] = len(timestamps)-prefix_length
            (full_uids if prefix_length >= 512 else fallback_uids).append(uid)
        full_uids.sort(key=lambda uid: (-post_counts[uid], uid))
        completed_users = 0
        for start in range(0, len(full_uids), args.cohort_size):
            cohort = full_uids[start:start+args.cohort_size]
            output_rows.extend(evaluate_full_cache_cohort(
                uids=cohort, by_user=by_user, history=history, parent=parent, current=current,
                parent_name=parent_name, current_name=current_name,
                edge=args.edge, checkpoint_hash=sha256_file(args.current),
                parent_hash=sha256_file(args.parent), manifest_hash=sha256_file(request_path),
                cutover=cutover, lineage_models=lineage_models,
            ))
            completed_users += len(cohort)
            (args.output/f"progress_rank{rank}.json").write_text(json.dumps({
                "rank": rank, "completed_users": completed_users,
                "assigned_users": len(selected_uids), "raw_rows_buffered": len(output_rows),
                "phase": "batched_full_cache",
            }) + "\n")
        for uid in fallback_uids:
            output_rows.extend(evaluate_user(
                requests=by_user[uid], history=history, parent=parent, current=current,
                parent_name=parent_name, current_name=current_name,
                edge=args.edge, checkpoint_hash=sha256_file(args.current),
                parent_hash=sha256_file(args.parent), manifest_hash=sha256_file(request_path),
                cutover=cutover, lineage_models=lineage_models,
            ))
            completed_users += 1
            (args.output/f"progress_rank{rank}.json").write_text(json.dumps({
                "rank": rank, "completed_users": completed_users,
                "assigned_users": len(selected_uids), "raw_rows_buffered": len(output_rows),
                "phase": "fallback_variable_cache",
            }) + "\n")
        shard = args.output / f"raw_rank{rank}.parquet"
        pq.write_table(pa.Table.from_pylist(output_rows), shard, compression="zstd")
        dist.barrier()
        if rank == 0:
            merged = pa.concat_tables([pq.read_table(args.output/f"raw_rank{value}.parquet") for value in range(world)])
            validate_raw_table(merged)
            raw = args.output / "raw.parquet"; pq.write_table(merged, raw, compression="zstd")
            print(json.dumps({"status": "formal_raw_complete", "edge": args.edge,
                              "rows": merged.num_rows, "requests": merged.num_rows//6,
                              "raw": str(raw)}, indent=2))
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
