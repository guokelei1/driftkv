#!/usr/bin/env python3
"""Four-GPU raw-first signed causal evaluation on real exposed request groups."""

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
from probe_candidate_shared_causal import evaluate_bank  # noqa: E402


WIDTHS = (16, 8, 4, 2)
RAW_PATHS = ("current_exact", "reuse", "shared_only", "residual_only", "full_delta")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def wait_for_paths(paths: list[Path], description: str, timeout_seconds: float = 900.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while not all(path.exists() for path in paths):
        if time.monotonic() >= deadline:
            missing = [str(path) for path in paths if not path.exists()]
            raise TimeoutError(f"timed out waiting for {description}: {missing}")
        time.sleep(0.1)


def largest_width(size: int) -> int | None:
    return next((width for width in WIDTHS if size >= width), None)


def exposed_groups(request_path: Path, start: int, stop: int) -> dict[int, list[dict[str, Any]]]:
    table = pq.read_table(
        request_path,
        filters=[
            ("time_block", "=", "matrix_horizon"),
            ("target_known", "=", True),
            ("query_timestamp", ">=", start),
            ("query_timestamp", "<", stop),
        ],
        columns=["request_id", "uid", "query_timestamp", "item_idx"],
    ).to_pandas()
    output: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for (uid, query_timestamp), rows in table.groupby(
        ["uid", "query_timestamp"], sort=True
    ):
        width = largest_width(len(rows))
        if width is None:
            continue
        ordered = rows.sort_values(["item_idx", "request_id"]).iloc[:width]
        output[int(uid)].append(
            {
                "uid": int(uid),
                "query_timestamp": int(query_timestamp),
                "observed_group_size": len(rows),
                "max_width": width,
                "items": ordered.item_idx.to_numpy(dtype=np.int64),
                "request_ids": ordered.request_id.astype(str).to_numpy(),
            }
        )
    for groups in output.values():
        groups.sort(key=lambda group: group["query_timestamp"])
    return dict(output)


def append_raw_rows(
    output: list[dict[str, Any]],
    *,
    edge: str,
    width: int,
    groups: list[dict[str, Any]],
    selected_indices: np.ndarray,
    scores: dict[str, torch.Tensor],
    append_counts: list[int],
    evictions: list[int],
    cutover: int,
) -> None:
    for bank_index, group in enumerate(groups):
        request_ids = group["request_ids"][selected_indices]
        items = group["items"][selected_indices]
        bank_id = f"{group['uid']}:{group['query_timestamp']}"
        for candidate_position, (request_id, item_idx) in enumerate(
            zip(request_ids, items, strict=True)
        ):
            for path in RAW_PATHS:
                output.append(
                    {
                        "request_id": str(request_id),
                        "uid": group["uid"],
                        "query_timestamp": group["query_timestamp"],
                        "edge": edge,
                        "bank_id": bank_id,
                        "observed_group_size": group["observed_group_size"],
                        "width": width,
                        "candidate_position": candidate_position,
                        "item_idx": int(item_idx),
                        "path": path,
                        "hstu_logit": float(scores[path][bank_index, candidate_position]),
                        "append_count_since_cutover": append_counts[bank_index],
                        "rolling_evictions": evictions[bank_index],
                        "seconds_since_cutover": group["query_timestamp"] - cutover,
                    }
                )


@torch.inference_mode()
def evaluate_group_batch(
    *,
    current,
    exact_cache: HSTUKVCache,
    reuse_cache: HSTUKVCache,
    groups: list[dict[str, Any]],
    query_deltas: torch.Tensor,
    edge: str,
    append_counts: list[int],
    evictions: list[int],
    cutover: int,
    score_records: list[dict[str, Any]],
    head_records: list[dict[str, Any]],
    raw_records: list[dict[str, Any]],
    correctness_records: list[dict[str, Any]],
) -> None:
    if not groups:
        return
    maximum = groups[0]["max_width"]
    if any(group["max_width"] != maximum for group in groups):
        raise ValueError("group batch mixes maximum candidate widths")
    base_items = np.stack([group["items"] for group in groups])
    identifiers = np.asarray(
        [f"{group['uid']}:{group['query_timestamp']}" for group in groups]
    )
    reference_shared = None
    for width in [value for value in WIDTHS if value <= maximum]:
        indices = nested_width_indices(maximum, width)
        candidates = torch.as_tensor(
            base_items[:, indices], dtype=torch.long, device=exact_cache.k.device
        )
        rows, heads, reference_shared, correctness, scores = evaluate_bank(
            current=current,
            exact_cache=exact_cache,
            reuse_cache=reuse_cache,
            candidates=candidates,
            query_deltas=query_deltas,
            edge=edge,
            bank_source="real_exposed_formal",
            identifiers=identifiers,
            width=width,
            reference_shared=reference_shared,
        )
        score_records.extend(rows)
        head_records.extend(heads)
        correctness_records.append(
            {"edge": edge, "max_width": maximum, "width": width, **correctness}
        )
        append_raw_rows(
            raw_records,
            edge=edge,
            width=width,
            groups=groups,
            selected_indices=indices,
            scores=scores,
            append_counts=append_counts,
            evictions=evictions,
            cutover=cutover,
        )


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
    score_records: list[dict[str, Any]] = []
    head_records: list[dict[str, Any]] = []
    raw_records: list[dict[str, Any]] = []
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
                query_deltas=query_deltas,
                edge=edge,
                append_counts=[int(append_counts[index]) for index in owners],
                evictions=[int(evictions[index]) for index in owners],
                cutover=cutover,
                score_records=score_records,
                head_records=head_records,
                raw_records=raw_records,
                correctness_records=correctness_records,
            )
    return score_records, head_records, raw_records, correctness_records


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
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    timestamps, items, behaviors = history.rows[uid]
    events = [
        (int(timestamp), int(item), int(behavior))
        for timestamp, item, behavior in zip(timestamps, items, behaviors, strict=True)
    ]
    prefix = [event for event in events if event[0] < cutover]
    if not prefix:
        raise RuntimeError("real exposed user has no cutover prefix")
    exact = materialize_state(current, prefix, producer_version="current", max_length=512)
    reuse = materialize_state(parent, prefix, producer_version="parent", max_length=512)
    post = list(timestamp_groups(event for event in events if event[0] >= cutover))
    post_index = 0
    append_count = 0
    evictions = 0
    score_records: list[dict[str, Any]] = []
    head_records: list[dict[str, Any]] = []
    raw_records: list[dict[str, Any]] = []
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
            query_deltas=query_delta,
            edge=edge,
            append_counts=[append_count],
            evictions=[evictions],
            cutover=cutover,
            score_records=score_records,
            head_records=head_records,
            raw_records=raw_records,
            correctness_records=correctness_records,
        )
    return score_records, head_records, raw_records, correctness_records


def merge_records(target: list[dict], values: tuple[list[dict], ...]) -> None:
    target.extend(values)


def validate_raw(table: pa.Table) -> tuple[int, int]:
    if "label" in table.column_names:
        raise RuntimeError("raw exposed artifact must not contain labels")
    frame = table.select(["bank_id", "width", "request_id", "path"]).to_pandas()
    paths = frame.groupby(["bank_id", "width", "request_id"]).path.agg(set)
    expected = set(RAW_PATHS)
    if not paths.map(lambda value: value == expected).all():
        raise RuntimeError("every selected exposed request must have all diagnostic paths")
    if frame.duplicated(["bank_id", "width", "request_id", "path"]).any():
        raise RuntimeError("duplicate exposed request/path rows")
    return int(frame[["bank_id", "width"]].drop_duplicates().shape[0]), int(
        frame[["request_id", "width"]].drop_duplicates().shape[0]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--edge", required=True)
    parser.add_argument("--cutover-day", type=int, required=True)
    parser.add_argument("--start-day", type=int, required=True)
    parser.add_argument("--end-day", type=int, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cohort-size", type=int, default=32)
    parser.add_argument("--max-users", type=int, default=0)
    args = parser.parse_args()
    if args.end_day <= args.start_day or args.start_day != args.cutover_day:
        raise ValueError("formal exposed evaluation must start at cutover")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    try:
        if world != 4:
            raise RuntimeError("formal real-exposed evaluation requires four ranks")
        if rank == 0:
            if args.output.exists():
                raise FileExistsError(f"refusing to overwrite {args.output}")
            args.output.mkdir(parents=True)
            (args.output / ".directory_ready").write_text("ready\n")
        wait_for_paths([args.output / ".directory_ready"], "rank-zero output directory")

        parent, parent_payload = load_model(args.parent, device)
        current, current_payload = load_model(args.current, device)
        if parent_payload["config"] != current_payload["config"]:
            raise RuntimeError("Parent and Current model configurations differ")
        request_path = args.manifest_dir / "requests_fidelity.parquet"
        groups_by_user = exposed_groups(
            request_path, args.start_day * DAY, args.end_day * DAY
        )
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
            boundary = int(
                np.searchsorted(history.rows[uid][0], cutover, side="left")
            )
            (full_uids if boundary >= 512 else short_uids).append(uid)

        score_records: list[dict[str, Any]] = []
        head_records: list[dict[str, Any]] = []
        raw_records: list[dict[str, Any]] = []
        correctness_records: list[dict[str, Any]] = []
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
            )
            merge_records(score_records, values[0])
            merge_records(head_records, values[1])
            merge_records(raw_records, values[2])
            merge_records(correctness_records, values[3])
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
            )
            merge_records(score_records, values[0])
            merge_records(head_records, values[1])
            merge_records(raw_records, values[2])
            merge_records(correctness_records, values[3])
            completed += 1
            (args.output / f"progress_rank{rank}.json").write_text(
                json.dumps(
                    {
                        "rank": rank,
                        "completed_users": completed,
                        "assigned_users": len(selected_uids),
                        "phase": "short_history",
                    }
                )
                + "\n"
            )

        frames = {
            "raw": pd.DataFrame(raw_records),
            "bank": pd.DataFrame(score_records),
            "head": pd.DataFrame(head_records),
            "correctness": pd.DataFrame(correctness_records),
        }
        for name, frame in frames.items():
            target = args.output / f"{name}_rank{rank}.parquet"
            partial = target.with_suffix(".parquet.partial")
            frame.to_parquet(partial, index=False)
            os.replace(partial, target)

        if rank == 0:
            shard_paths = {
                name: [args.output / f"{name}_rank{value}.parquet" for value in range(world)]
                for name in frames
            }
            wait_for_paths(
                [path for paths in shard_paths.values() for path in paths],
                "all exposed raw and metric shards",
            )
            merged = {
                name: pa.concat_tables([pq.read_table(path) for path in paths])
                for name, paths in shard_paths.items()
            }
            banks, selected_requests = validate_raw(merged["raw"])
            targets = {}
            for name, table in merged.items():
                target = args.output / f"{name}.parquet"
                partial = target.with_suffix(".parquet.partial")
                pq.write_table(table, partial, compression="zstd")
                os.replace(partial, target)
                targets[name] = target
            correctness = merged["correctness"].to_pandas()
            seal = {
                "status": "candidate_shared_real_exposed_raw_sealed_before_label_join",
                "stage": args.stage,
                "edge": args.edge,
                "users": int(merged["raw"].to_pandas()["uid"].nunique()),
                "banks_across_widths": banks,
                "selected_requests_across_widths": selected_requests,
                "widths": list(reversed(WIDTHS)),
                "labels_read": False,
                "native_score_max_abs_error": float(
                    correctness[["native_exact", "native_reuse"]].to_numpy().max()
                ),
                "full_delta_reconstruction_max_abs_error": float(
                    correctness["full_delta"].max()
                ),
                "artifacts": {
                    name: {"path": str(path), "sha256": sha256(path)}
                    for name, path in targets.items()
                },
                "checkpoint_sha256": sha256(args.current),
                "parent_checkpoint_sha256": sha256(args.parent),
                "requests_fidelity_sha256": sha256(request_path),
            }
            (args.output / "raw.seal.json").write_text(json.dumps(seal, indent=2) + "\n")
            for paths in shard_paths.values():
                for path in paths:
                    path.unlink()
            (args.output / ".raw_complete").write_text("complete\n")
            print(json.dumps(seal, indent=2), flush=True)
        else:
            wait_for_paths([args.output / ".raw_complete"], "rank-zero exposed merge")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
