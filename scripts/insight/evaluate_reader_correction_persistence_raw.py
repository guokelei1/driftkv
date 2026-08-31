#!/usr/bin/env python3
"""Raw-first real-request stage and persistence observation on four GPUs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.distributed as dist


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "insight"))

from candidate_shared_causal import nested_width_indices  # noqa: E402
from evaluate_candidate_shared_exposed_raw import (  # noqa: E402
    WIDTHS,
    exposed_groups,
    wait_for_paths,
)
from evaluate_yambda500m_foundation_raw import (  # noqa: E402
    DAY,
    assign_cache,
    balanced_users,
    load_histories,
    load_model,
    select_cache,
    stacked_cache,
)
from hstu_kvcache.evaluation import (  # noqa: E402
    append_timestamp_group,
    materialize_state,
    timestamp_groups,
)
from hstu_kvcache.models import HSTUKVCache  # noqa: E402
from hstu_kvcache.models.state_transition import append_with_rolling_cap  # noqa: E402
from probe_reader_compatibility_correction import _energy_rows, _score_rows  # noqa: E402
from reader_compatibility_correction import (  # noqa: E402
    STAGES,
    correction_cosine,
    correction_norm,
    intervene_reader_correction,
    scale_correction,
    trace_reader_correction,
)


PERSISTENCE_STAGES = (
    "av_aggregation",
    "u_gated_update",
    "layer_hidden",
    "final_readout",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _recovery(
    exact: torch.Tensor, reuse: torch.Tensor, candidate: torch.Tensor
) -> torch.Tensor:
    exact_probability = torch.sigmoid(exact.float())
    reuse_gap = torch.abs(torch.sigmoid(reuse.float()) - exact_probability).mean(dim=1)
    candidate_gap = torch.abs(
        torch.sigmoid(candidate.float()) - exact_probability
    ).mean(dim=1)
    return 1.0 - candidate_gap / reuse_gap.clamp_min(1e-12)


def _stack_previous(
    prior: dict[int, dict[str, Any]],
    owners: list[int],
    stage: str,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    layers = len(prior[owners[0]]["corrections"][stage])
    return tuple(
        torch.stack(
            [prior[owner]["corrections"][stage][layer] for owner in owners]
        ).to(device)
        for layer in range(layers)
    )


@torch.inference_mode()
def evaluate_group_batch(
    *,
    current,
    exact_cache: HSTUKVCache,
    reuse_cache: HSTUKVCache,
    groups: list[dict[str, Any]],
    owners: list[int],
    query_deltas: torch.Tensor,
    edge: str,
    append_counts: list[int],
    evictions: list[int],
    cutover: int,
    prior: dict[int, dict[str, Any]] | None,
    verify_full_delta: bool,
    score_records: list[dict[str, Any]],
    energy_records: list[dict[str, Any]],
    persistence_records: list[dict[str, Any]],
    correctness_records: list[dict[str, Any]],
) -> None:
    maximum = groups[0]["max_width"]
    if any(group["max_width"] != maximum for group in groups):
        raise ValueError("request batch mixes maximum widths")
    base_items = np.stack([group["items"] for group in groups])
    identifiers = np.asarray(
        [f"{group['uid']}:{group['query_timestamp']}" for group in groups]
    )
    for width in [value for value in WIDTHS if value <= maximum]:
        selected_indices = nested_width_indices(maximum, width)
        candidates = torch.as_tensor(
            base_items[:, selected_indices], dtype=torch.long, device=exact_cache.k.device
        )
        trace = trace_reader_correction(
            current,
            exact_cache,
            reuse_cache,
            candidates,
            query_deltas,
            verify_full_delta=verify_full_delta and width == maximum,
        )
        score_records.extend(
            _score_rows(
                edge=edge,
                source="real_exposed",
                identifiers=identifiers,
                width=width,
                trace=trace,
            )
        )
        energy_records.extend(
            _energy_rows(
                edge=edge,
                source="real_exposed",
                identifiers=identifiers,
                width=width,
                trace=trace,
            )
        )
        correctness_records.append(
            {"edge": edge, "max_width": maximum, "width": width, **trace.correctness}
        )
        if width != maximum or prior is None:
            continue

        previous_rows = [index for index, owner in enumerate(owners) if owner in prior]
        if previous_rows:
            row_tensor = torch.as_tensor(
                previous_rows, dtype=torch.long, device=exact_cache.k.device
            )
            previous_owners = [owners[index] for index in previous_rows]
            previous_candidates = candidates.index_select(0, row_tensor)
            previous_deltas = query_deltas.index_select(0, row_tensor)
            previous_cache = select_cache(reuse_cache, previous_rows)
            exact_scores = trace.exact_scores.index_select(0, row_tensor)
            reuse_scores = trace.reuse_scores.index_select(0, row_tensor)
            for stage in PERSISTENCE_STAGES:
                previous_correction = _stack_previous(
                    prior, previous_owners, stage, exact_cache.k.device
                )
                current_correction = tuple(
                    value.index_select(0, row_tensor) for value in trace.corrections[stage]
                )
                factors = torch.tensor(
                    [
                        max(0, 512 - evictions[row])
                        / max(1, prior[owners[row]]["remaining_old_positions"])
                        for row in previous_rows
                    ],
                    dtype=torch.float32,
                    device=exact_cache.k.device,
                )
                scaled_previous = scale_correction(previous_correction, factors)
                previous_scores, _ = intervene_reader_correction(
                    current,
                    previous_cache,
                    previous_candidates,
                    previous_deltas,
                    stage=stage,
                    corrections=previous_correction,
                )
                scaled_scores, _ = intervene_reader_correction(
                    current,
                    previous_cache,
                    previous_candidates,
                    previous_deltas,
                    stage=stage,
                    corrections=scaled_previous,
                )
                same_recovery = _recovery(
                    exact_scores,
                    reuse_scores,
                    trace.stage_scores[stage].index_select(0, row_tensor),
                )
                previous_recovery = _recovery(
                    exact_scores, reuse_scores, previous_scores
                )
                scaled_recovery = _recovery(exact_scores, reuse_scores, scaled_scores)
                cosines = correction_cosine(current_correction, previous_correction)
                current_norms = correction_norm(current_correction)
                previous_norms = correction_norm(previous_correction)
                for local, row in enumerate(previous_rows):
                    owner = owners[row]
                    group = groups[row]
                    old = prior[owner]
                    persistence_records.append(
                        {
                            "edge": edge,
                            "uid": group["uid"],
                            "previous_query_timestamp": old["query_timestamp"],
                            "current_query_timestamp": group["query_timestamp"],
                            "seconds_between_requests": (
                                group["query_timestamp"] - old["query_timestamp"]
                            ),
                            "previous_append_count": old["append_count"],
                            "current_append_count": append_counts[row],
                            "append_count_difference": (
                                append_counts[row] - old["append_count"]
                            ),
                            "previous_remaining_old_positions": old[
                                "remaining_old_positions"
                            ],
                            "current_remaining_old_positions": max(
                                0, 512 - evictions[row]
                            ),
                            "coverage_scale": float(factors[local]),
                            "previous_max_width": old["max_width"],
                            "current_max_width": maximum,
                            "stage": stage,
                            "adjacent_request_direction_cosine": float(cosines[local]),
                            "current_correction_norm": float(current_norms[local]),
                            "previous_correction_norm": float(previous_norms[local]),
                            "current_to_previous_norm_ratio": float(
                                current_norms[local] / previous_norms[local].clamp_min(1e-12)
                            ),
                            "same_request_gap_recovery": float(same_recovery[local]),
                            "prior_request_gap_recovery": float(previous_recovery[local]),
                            "coverage_scaled_prior_gap_recovery": float(
                                scaled_recovery[local]
                            ),
                        }
                    )

        for row, owner in enumerate(owners):
            prior[owner] = {
                "query_timestamp": groups[row]["query_timestamp"],
                "append_count": append_counts[row],
                "remaining_old_positions": max(0, 512 - evictions[row]),
                "max_width": maximum,
                "corrections": {
                    stage: tuple(
                        value[row].detach().cpu() for value in trace.corrections[stage]
                    )
                    for stage in PERSISTENCE_STAGES
                },
            }


@torch.inference_mode()
def evaluate_full_cohort(
    *,
    uids: list[int],
    groups_by_user: dict[int, list[dict[str, Any]]],
    history,
    parent,
    current,
    edge: str,
    cutover: int,
    stop: int,
    verify_full_delta: bool,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    device = next(current.parameters()).device
    raw_history = [history.rows[uid] for uid in uids]
    prefix = []
    for timestamps, items, behaviors in raw_history:
        boundary = int(np.searchsorted(timestamps, cutover, side="left"))
        if boundary < 512:
            raise ValueError("full cohort contains a short cutover history")
        prefix.append(
            (
                timestamps[boundary - 512 : boundary],
                items[boundary - 512 : boundary],
                behaviors[boundary - 512 : boundary],
            )
        )
    times = torch.as_tensor(np.stack([row[0] for row in prefix]), device=device)
    items = torch.as_tensor(np.stack([row[1] for row in prefix]), dtype=torch.long, device=device)
    behaviors = torch.as_tensor(
        np.stack([row[2] for row in prefix]), dtype=torch.long, device=device
    )
    deltas = torch.zeros_like(times, dtype=torch.float32)
    deltas[:, 1:] = times[:, 1:] - times[:, :-1]
    exact_cache = current.compute_kv(items, behaviors, deltas)
    reuse_cache = parent.compute_kv(items, behaviors, deltas)
    last_times = np.asarray([int(row[0][-1]) for row in prefix], dtype=np.int64)
    append_counts = np.zeros(len(uids), dtype=np.int64)
    evictions = np.zeros(len(uids), dtype=np.int64)

    actions = []
    for index, uid in enumerate(uids):
        timestamps, event_items, event_behaviors = raw_history[index]
        requests_at = {
            group["query_timestamp"]: group for group in groups_by_user[uid]
        }
        post_events: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
        start = int(np.searchsorted(timestamps, cutover, side="left"))
        end = int(np.searchsorted(timestamps, stop, side="left"))
        for position in range(start, end):
            post_events[int(timestamps[position])].append(
                (
                    int(timestamps[position]),
                    int(event_items[position]),
                    int(event_behaviors[position]),
                )
            )
        timeline = []
        for timestamp in sorted(set(requests_at) | set(post_events)):
            if timestamp in requests_at:
                timeline.append(("query", timestamp, requests_at[timestamp]))
            for event in sorted(
                post_events.get(timestamp, ()), key=lambda value: (value[1], value[2])
            ):
                timeline.append(("append", timestamp, event))
        actions.append(timeline)

    positions = np.zeros(len(uids), dtype=np.int64)
    prior: dict[int, dict[str, Any]] = {}
    score_records: list[dict[str, Any]] = []
    energy_records: list[dict[str, Any]] = []
    persistence_records: list[dict[str, Any]] = []
    correctness_records: list[dict[str, Any]] = []
    while True:
        active = [i for i in range(len(uids)) if positions[i] < len(actions[i])]
        if not active:
            break
        append_indices = [i for i in active if actions[i][positions[i]][0] == "append"]
        if append_indices:
            events = [actions[i][positions[i]][2] for i in append_indices]
            event_items = torch.tensor(
                [[event[1]] for event in events], dtype=torch.long, device=device
            )
            event_behaviors = torch.tensor(
                [[event[2]] for event in events], dtype=torch.long, device=device
            )
            event_deltas = torch.tensor(
                [
                    [float(max(0, min(7 * DAY, event[0] - last_times[index])))]
                    for index, event in zip(append_indices, events, strict=True)
                ],
                device=device,
            )
            selected = stacked_cache(
                [
                    select_cache(exact_cache, append_indices),
                    select_cache(reuse_cache, append_indices),
                ]
            )
            updated = append_with_rolling_cap(
                current,
                selected,
                event_items.repeat(2, 1),
                event_behaviors.repeat(2, 1),
                event_deltas.repeat(2, 1),
                512,
            )
            count = len(append_indices)
            assign_cache(
                exact_cache,
                append_indices,
                HSTUKVCache(updated.k[:, :count], updated.v[:, :count], 512),
            )
            assign_cache(
                reuse_cache,
                append_indices,
                HSTUKVCache(updated.k[:, count:], updated.v[:, count:], 512),
            )
            for index, event in zip(append_indices, events, strict=True):
                last_times[index] = event[0]
                append_counts[index] += 1
                evictions[index] += 1
                positions[index] += 1

        remaining = [i for i in range(len(uids)) if positions[i] < len(actions[i])]
        query_indices = [i for i in remaining if actions[i][positions[i]][0] == "query"]
        if not query_indices:
            continue
        entries = []
        for index in query_indices:
            _, query_time, group = actions[index][positions[index]]
            entries.append((index, query_time, group))
            positions[index] += 1
        for maximum in WIDTHS:
            selected_entries = [entry for entry in entries if entry[2]["max_width"] == maximum]
            if not selected_entries:
                continue
            owners = [entry[0] for entry in selected_entries]
            groups = [entry[2] for entry in selected_entries]
            query_deltas = torch.tensor(
                [
                    float(query_time - last_times[index])
                    for index, query_time, _ in selected_entries
                ],
                dtype=torch.float32,
                device=device,
            )
            evaluate_group_batch(
                current=current,
                exact_cache=select_cache(exact_cache, owners),
                reuse_cache=select_cache(reuse_cache, owners),
                groups=groups,
                owners=owners,
                query_deltas=query_deltas,
                edge=edge,
                append_counts=[int(append_counts[index]) for index in owners],
                evictions=[int(evictions[index]) for index in owners],
                cutover=cutover,
                prior=prior,
                verify_full_delta=verify_full_delta,
                score_records=score_records,
                energy_records=energy_records,
                persistence_records=persistence_records,
                correctness_records=correctness_records,
            )
    return score_records, energy_records, persistence_records, correctness_records


@torch.inference_mode()
def evaluate_short_user(
    *,
    uid: int,
    groups: list[dict[str, Any]],
    history,
    parent,
    current,
    edge: str,
    cutover: int,
    verify_full_delta: bool,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    timestamps, items, behaviors = history.rows[uid]
    events = [
        (int(timestamp), int(item), int(behavior))
        for timestamp, item, behavior in zip(timestamps, items, behaviors, strict=True)
    ]
    prefix = [event for event in events if event[0] < cutover]
    exact = materialize_state(current, prefix, producer_version="current", max_length=512)
    reuse = materialize_state(parent, prefix, producer_version="parent", max_length=512)
    post = list(timestamp_groups(event for event in events if event[0] >= cutover))
    post_index = 0
    append_count = 0
    evictions = 0
    score_records: list[dict[str, Any]] = []
    energy_records: list[dict[str, Any]] = []
    persistence_records: list[dict[str, Any]] = []
    correctness_records: list[dict[str, Any]] = []
    for group in groups:
        query_time = group["query_timestamp"]
        while post_index < len(post) and post[post_index][0] < query_time:
            _, event_group = post[post_index]
            evictions += max(0, exact.cache.seq_len + len(event_group) - 512)
            exact = append_timestamp_group(
                current, exact, event_group, producer_version="current", max_length=512
            )
            reuse = append_timestamp_group(
                current, reuse, event_group, producer_version="current", max_length=512
            )
            append_count += len(event_group)
            post_index += 1
        query_delta = torch.tensor(
            [float(query_time - exact.last_timestamp)],
            dtype=torch.float32,
            device=exact.cache.k.device,
        )
        evaluate_group_batch(
            current=current,
            exact_cache=exact.cache,
            reuse_cache=reuse.cache,
            groups=[group],
            owners=[uid],
            query_deltas=query_delta,
            edge=edge,
            append_counts=[append_count],
            evictions=[evictions],
            cutover=cutover,
            prior=None,
            verify_full_delta=verify_full_delta,
            score_records=score_records,
            energy_records=energy_records,
            persistence_records=persistence_records,
            correctness_records=correctness_records,
        )
    return score_records, energy_records, persistence_records, correctness_records


def _extend(targets: tuple[list[dict], ...], values: tuple[list[dict], ...]) -> None:
    for target, value in zip(targets, values, strict=True):
        target.extend(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge", required=True)
    parser.add_argument("--cutover-day", type=int, required=True)
    parser.add_argument("--end-day", type=int, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cohort-size", type=int, default=16)
    parser.add_argument("--max-users", type=int, default=0)
    args = parser.parse_args()
    if args.end_day != args.cutover_day + 14:
        raise ValueError("real persistence evaluator requires the frozen E14 window")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    started = time.time()
    try:
        if world != 4:
            raise RuntimeError("real persistence observation requires four ranks")
        if rank == 0:
            if args.output.exists():
                raise FileExistsError(f"refusing to overwrite {args.output}")
            args.output.mkdir(parents=True)
            (args.output / ".directory_ready").write_text("ready\n")
        wait_for_paths([args.output / ".directory_ready"], "rank-zero output directory")

        parent, parent_payload = load_model(args.parent, device)
        current, current_payload = load_model(args.current, device)
        if parent_payload["config"] != current_payload["config"]:
            raise RuntimeError("Parent and Current configurations differ")
        request_path = args.manifest_dir / "requests_fidelity.parquet"
        all_groups = exposed_groups(
            request_path, args.cutover_day * DAY, args.end_day * DAY
        )
        groups_by_user = {
            uid: groups for uid, groups in all_groups.items() if len(groups) >= 2
        }
        assignment_rows = [
            {"uid": uid}
            for uid, groups in groups_by_user.items()
            for _ in range(sum(group["max_width"] for group in groups))
        ]
        assignment = balanced_users(assignment_rows, world)
        selected_uids = sorted(uid for uid, owner in assignment.items() if owner == rank)
        if args.max_users:
            selected_uids = selected_uids[: args.max_users]
        oov_buckets = int(current_payload["config"]["num_items"]) - 781_678
        history = load_histories(selected_uids, oov_buckets=oov_buckets)
        cutover = args.cutover_day * DAY
        full_uids, short_uids = [], []
        for uid in selected_uids:
            boundary = int(np.searchsorted(history.rows[uid][0], cutover, side="left"))
            (full_uids if boundary >= 512 else short_uids).append(uid)

        outputs: tuple[list[dict], ...] = ([], [], [], [])
        completed = 0
        for start in range(0, len(full_uids), args.cohort_size):
            cohort = full_uids[start : start + args.cohort_size]
            values = evaluate_full_cohort(
                uids=cohort,
                groups_by_user=groups_by_user,
                history=history,
                parent=parent,
                current=current,
                edge=args.edge,
                cutover=cutover,
                stop=args.end_day * DAY,
                verify_full_delta=bool(args.max_users),
            )
            _extend(outputs, values)
            completed += len(cohort)
            (args.output / f"progress_rank{rank}.json").write_text(
                json.dumps(
                    {
                        "rank": rank,
                        "completed_users": completed,
                        "assigned_users": len(selected_uids),
                        "phase": "full_history",
                    }
                )
                + "\n"
            )
        for uid in short_uids:
            values = evaluate_short_user(
                uid=uid,
                groups=groups_by_user[uid],
                history=history,
                parent=parent,
                current=current,
                edge=args.edge,
                cutover=cutover,
                verify_full_delta=bool(args.max_users),
            )
            _extend(outputs, values)
            completed += 1

        frames = {
            "stage_score": pd.DataFrame(outputs[0]),
            "stage_energy": pd.DataFrame(outputs[1]),
            "persistence": pd.DataFrame(outputs[2]),
            "correctness": pd.DataFrame(outputs[3]),
        }
        for name, frame in frames.items():
            target = args.output / f"{name}_rank{rank}.parquet"
            partial = target.with_suffix(".parquet.partial")
            frame.to_parquet(partial, index=False)
            os.replace(partial, target)

        if rank == 0:
            shards = {
                name: [args.output / f"{name}_rank{value}.parquet" for value in range(world)]
                for name in frames
            }
            wait_for_paths(
                [path for paths in shards.values() for path in paths],
                "all reader-correction shards",
            )
            merged = {
                name: pa.concat_tables([pq.read_table(path) for path in paths])
                for name, paths in shards.items()
            }
            targets = {}
            for name, table in merged.items():
                if "label" in table.column_names:
                    raise RuntimeError("reader-correction raw artifact contains a label")
                target = args.output / f"{name}.parquet"
                partial = target.with_suffix(".parquet.partial")
                pq.write_table(table, partial, compression="zstd")
                os.replace(partial, target)
                targets[name] = target
            scores = merged["stage_score"].to_pandas()
            persistence = merged["persistence"].to_pandas()
            correctness = merged["correctness"].to_pandas()
            error_columns = [
                "native_exact",
                "native_reuse",
                "final_full_delta",
                "layer_stage_full_delta",
            ]
            maximum_error = float(correctness[error_columns].max(skipna=True).max())
            seal = {
                "status": "reader_correction_real_raw_sealed",
                "edge": args.edge,
                "users": int(scores.identifier.str.split(":").str[0].nunique()),
                "request_groups": int(scores.identifier.nunique()),
                "full_history_persistence_pairs": int(
                    persistence[
                        ["uid", "previous_query_timestamp", "current_query_timestamp"]
                    ].drop_duplicates().shape[0]
                ),
                "widths": sorted(scores.width.unique().tolist()),
                "stages": list(STAGES),
                "labels_read": False,
                "correctness_max_abs_error": maximum_error,
                "elapsed_seconds": time.time() - started,
                "artifacts": {
                    name: {"path": str(path), "sha256": sha256(path)}
                    for name, path in targets.items()
                },
                "parent_checkpoint_sha256": sha256(args.parent),
                "checkpoint_sha256": sha256(args.current),
                "requests_fidelity_sha256": sha256(request_path),
            }
            (args.output / "raw.seal.json").write_text(json.dumps(seal, indent=2) + "\n")
            for paths in shards.values():
                for path in paths:
                    path.unlink()
            (args.output / ".raw_complete").write_text("complete\n")
            print(json.dumps(seal, indent=2), flush=True)
        else:
            wait_for_paths([args.output / ".raw_complete"], "rank-zero merge")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
