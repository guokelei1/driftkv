#!/usr/bin/env python3
"""Four-GPU raw-first D14 one-hop Reuse versus exact rolling evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.distributed as dist

from evaluate_yambda500m_foundation_raw import (
    DAY, balanced_users, evaluate_full_cache_cohort, load_histories, load_model,
)
from hstu_kvcache.evaluation import (
    VersionedCacheState, append_timestamp_group, materialize_state, observe_rolling,
    timestamp_groups,
)
from insight.one_release_refinement import (
    OUR_PATH, build_fixed_refinement_cache, parameter_cast_maps,
)


PAIR_PATHS = ("current_exact_rolling", "one_hop_reuse_rolling")
RELEASE_DEBT_PATHS = ("parent_exact_rolling", *PAIR_PATHS)
REFINEMENT_PATHS = (*RELEASE_DEBT_PATHS, OUR_PATH)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pair_rows(
    rows: list[dict], *, include_parent_exact: bool, include_fixed_refinement: bool
) -> list[dict]:
    paths = RELEASE_DEBT_PATHS if include_parent_exact else PAIR_PATHS
    if include_fixed_refinement:
        paths = (*paths, OUR_PATH)
    return [row for row in rows if row["path"] in paths]


def wait_for_paths(paths: list[Path], *, description: str, timeout_seconds: float = 600.0) -> None:
    """Filesystem rendezvous avoids an NCCL barrier after GPU computation."""
    deadline = time.monotonic() + timeout_seconds
    while not all(path.exists() for path in paths):
        if time.monotonic() >= deadline:
            absent = [str(path) for path in paths if not path.exists()]
            raise TimeoutError(f"timed out waiting for {description}: {absent}")
        time.sleep(0.1)


@torch.inference_mode()
def evaluate_fallback_user(*, requests: list[dict], history, parent, current, edge: str,
                           parent_name: str, current_name: str, cutover: int,
                           current_hash: str, parent_hash: str, manifest_hash: str,
                           include_parent_exact: bool,
                           refinement_cast_maps=None) -> list[dict]:
    timestamps, items, behaviors = history.rows[int(requests[0]["uid"])]
    events = [(int(t), int(i), int(b)) for t, i, b in zip(timestamps, items, behaviors, strict=True)]
    prefix = [event for event in events if event[0] < cutover]
    if not prefix:
        raise RuntimeError("qualified request has no strictly-prior prefix")
    parent_state = None
    if include_parent_exact:
        parent_state = materialize_state(parent, prefix, producer_version=parent_name, max_length=512)
    current_state = materialize_state(current, prefix, producer_version=current_name, max_length=512)
    reuse_state = materialize_state(parent, prefix, producer_version=parent_name, max_length=512)
    refinement_state = None
    if refinement_cast_maps is not None:
        ordered = [event for _, group in timestamp_groups(prefix) for event in group][-512:]
        prefix_times = torch.tensor(
            [[event[0] for event in ordered]], dtype=torch.long, device=reuse_state.cache.k.device
        )
        prefix_deltas = torch.zeros_like(prefix_times, dtype=torch.float32)
        if len(ordered) > 1:
            prefix_deltas[:, 1:] = prefix_times[:, 1:] - prefix_times[:, :-1]
        prefix_items = torch.tensor(
            [[event[1] for event in ordered]], dtype=torch.long, device=reuse_state.cache.k.device
        )
        prefix_behaviors = torch.tensor(
            [[event[2] for event in ordered]], dtype=torch.long, device=reuse_state.cache.k.device
        )
        refinement_cache, layout = build_fixed_refinement_cache(
            parent_cache=reuse_state.cache,
            current=current,
            item_ids=prefix_items,
            behaviors=prefix_behaviors,
            time_deltas=prefix_deltas,
            cast_maps=refinement_cast_maps,
        )
        producers = (
            ("evokv_zero_padding",) * layout.padding_positions
            + (f"{parent_name}_cast_to_{current_name}",) * layout.cast_positions
            + (f"{current_name}_group_patch_scale",) * layout.carriers
        )
        refinement_state = VersionedCacheState(
            refinement_cache, reuse_state.last_timestamp, producers
        )
    post_groups = list(timestamp_groups(event for event in events if event[0] >= cutover))
    request_groups: dict[int, list[dict]] = {}
    for request in requests:
        request_groups.setdefault(int(request["query_timestamp"]), []).append(request)
    group_index = 0
    append_count = 0
    evictions = 0
    output: list[dict] = []
    for query_time, simultaneous in sorted(request_groups.items()):
        while group_index < len(post_groups) and post_groups[group_index][0] < query_time:
            _, group = post_groups[group_index]
            evictions += max(0, current_state.cache.seq_len + len(group) - 512)
            if parent_state is not None:
                parent_state = append_timestamp_group(parent, parent_state, group, producer_version=parent_name, max_length=512)
            current_state = append_timestamp_group(current, current_state, group, producer_version=current_name, max_length=512)
            reuse_state = append_timestamp_group(current, reuse_state, group, producer_version=current_name, max_length=512)
            if refinement_state is not None:
                refinement_state = append_timestamp_group(
                    current, refinement_state, group,
                    producer_version=current_name, max_length=512,
                )
            append_count += len(group)
            group_index += 1
        stop = int(np.searchsorted(timestamps, query_time, side="left"))
        history_length = min(stop, 512)
        for request in simultaneous:
            candidate = int(request["item_idx"])
            if parent_state is not None:
                parent_score, parent_readout = observe_rolling(parent, parent_state, candidate_id=candidate, query_timestamp=query_time)
            current_score, current_readout = observe_rolling(current, current_state, candidate_id=candidate, query_timestamp=query_time)
            reuse_score, reuse_readout = observe_rolling(current, reuse_state, candidate_id=candidate, query_timestamp=query_time)
            if refinement_state is not None:
                refinement_score, refinement_readout = observe_rolling(
                    current, refinement_state,
                    candidate_id=candidate, query_timestamp=query_time,
                )
            common = {
                "request_id": request["request_id"], "uid": int(request["uid"]),
                "query_timestamp": query_time, "edge": edge, "architecture": "hstu_native_cc",
                "append_count_since_cutover": append_count, "seconds_since_cutover": query_time - cutover,
                "history_length": history_length, "cache_length": current_state.cache.seq_len,
                "rolling_evictions": evictions, "checkpoint_sha256": current_hash,
                "parent_checkpoint_sha256": parent_hash, "manifest_sha256": manifest_hash,
            }
            if parent_state is not None:
                output.append({**common, "path": "parent_exact_rolling", "hstu_logit": parent_score, "readout_normalized_l2": float((parent_readout-current_readout).norm()/(current_readout.norm()+1e-12)), "readout_cosine": float(torch.nn.functional.cosine_similarity(parent_readout[None], current_readout[None]))})
            output.append({**common, "path": "current_exact_rolling", "hstu_logit": current_score, "readout_normalized_l2": 0.0, "readout_cosine": 1.0})
            output.append({**common, "path": "one_hop_reuse_rolling", "hstu_logit": reuse_score, "readout_normalized_l2": float((reuse_readout-current_readout).norm()/(current_readout.norm()+1e-12)), "readout_cosine": float(torch.nn.functional.cosine_similarity(reuse_readout[None], current_readout[None]))})
            if refinement_state is not None:
                output.append({**common, "path": OUR_PATH, "hstu_logit": refinement_score, "readout_normalized_l2": float((refinement_readout-current_readout).norm()/(current_readout.norm()+1e-12)), "readout_cosine": float(torch.nn.functional.cosine_similarity(refinement_readout[None], current_readout[None]))})
        # Same-timestamp events append only after every query has been scored.
        while group_index < len(post_groups) and post_groups[group_index][0] == query_time:
            _, group = post_groups[group_index]
            evictions += max(0, current_state.cache.seq_len + len(group) - 512)
            if parent_state is not None:
                parent_state = append_timestamp_group(parent, parent_state, group, producer_version=parent_name, max_length=512)
            current_state = append_timestamp_group(current, current_state, group, producer_version=current_name, max_length=512)
            reuse_state = append_timestamp_group(current, reuse_state, group, producer_version=current_name, max_length=512)
            if refinement_state is not None:
                refinement_state = append_timestamp_group(
                    current, refinement_state, group,
                    producer_version=current_name, max_length=512,
                )
            append_count += len(group)
            group_index += 1
    return output


def validate_pair_raw(table: pa.Table, *, expected_paths: tuple[str, ...] | None = None) -> int:
    required = {"request_id", "uid", "query_timestamp", "edge", "path", "hstu_logit", "append_count_since_cutover", "checkpoint_sha256", "parent_checkpoint_sha256", "manifest_sha256"}
    missing = sorted(required - set(table.column_names))
    if missing:
        raise RuntimeError(f"raw table missing columns: {missing}")
    if "label" in table.column_names:
        raise RuntimeError("raw artifact must not contain labels")
    values = table.to_pylist()
    by_request: dict[str, set[str]] = {}
    for row in values:
        by_request.setdefault(row["request_id"], set()).add(row["path"])
    observed_paths = set(next(iter(by_request.values()))) if by_request else set()
    valid_paths = (set(PAIR_PATHS), set(RELEASE_DEBT_PATHS))
    if expected_paths is not None:
        valid_paths = (set(expected_paths),)
    if any(not any(paths == permitted for permitted in valid_paths) for paths in by_request.values()):
        raise RuntimeError("every request must have exactly the contracted rolling observations")
    if len(values) != len(observed_paths) * len(by_request):
        raise RuntimeError("duplicate request/path observations")
    return len(by_request)


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
    parser.add_argument("--cohort-size", type=int, default=128)
    parser.add_argument("--max-users", type=int, default=0)
    parser.add_argument("--include-parent-exact", action="store_true")
    parser.add_argument(
        "--include-fixed-refinement", action="store_true",
        help="add the frozen one-release CAST+GROUP/PATCH+SCALE r=128,c=64 path",
    )
    parser.add_argument("--force-fallback", action="store_true", help="score every user through the existing scalar rolling fallback")
    args = parser.parse_args()
    if args.end_day <= args.start_day or args.start_day < args.cutover_day:
        raise ValueError("evaluation must be a nonempty post-cutover interval")
    parent_name, current_name = args.edge.split("_to_")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    try:
        if world != 4:
            raise RuntimeError("formal one-hop evaluation requires four ranks")
        if not 1 <= args.cohort_size <= 128:
            raise ValueError("cohort-size must be in [1,128]")
        if rank == 0:
            if args.output.exists():
                raise FileExistsError(f"refusing to overwrite {args.output}")
            args.output.mkdir(parents=True)
            (args.output / ".directory_ready").write_text("ready\n")
        wait_for_paths([args.output / ".directory_ready"], description="rank-zero output directory")

        parent, parent_payload = load_model(args.parent, device)
        current, current_payload = load_model(args.current, device)
        if parent_payload["config"] != current_payload["config"]:
            raise RuntimeError("Parent and Current must share an identical HSTU configuration")
        cast_maps = parameter_cast_maps(parent, current) if args.include_fixed_refinement else None
        oov_buckets = int(current_payload["config"]["num_items"]) - 781678
        if oov_buckets < 0:
            raise RuntimeError("checkpoint vocabulary is smaller than the frozen known vocabulary")
        request_path = args.manifest_dir / "requests_fidelity.parquet"
        request_table = pq.read_table(
            request_path,
            filters=[("time_block", "=", "matrix_horizon"), ("target_known", "=", True),
                     ("query_timestamp", ">=", args.start_day * DAY), ("query_timestamp", "<", args.end_day * DAY)],
            columns=["request_id", "uid", "query_timestamp", "item_idx"],
        ).sort_by([("uid", "ascending"), ("query_timestamp", "ascending"), ("request_id", "ascending")])
        rows = request_table.to_pylist()
        assignment = balanced_users(rows, world)
        selected_uids = sorted(uid for uid, owner in assignment.items() if owner == rank)
        if args.max_users:
            selected_uids = selected_uids[:args.max_users]
        selected = set(selected_uids)
        by_user: dict[int, list[dict]] = {}
        for row in rows:
            if int(row["uid"]) in selected:
                by_user.setdefault(int(row["uid"]), []).append(row)
        history = load_histories(selected_uids, oov_buckets=oov_buckets)
        cutover = args.cutover_day * DAY
        full_uids, fallback_uids, post_counts = [], [], {}
        for uid in selected_uids:
            times = history.rows[uid][0]
            prefix = int(np.searchsorted(times, cutover, side="left"))
            post_counts[uid] = len(times) - prefix
            (fallback_uids if args.force_fallback or prefix < 512 else full_uids).append(uid)
        # Keep every batch representative of the rank's activity mix.  Sorting
        # by post-cutover volume groups all extreme users into the first batch,
        # creating a long GPU tail without changing the evaluated population.
        full_uids.sort()
        current_hash, parent_hash, manifest_hash = sha256_file(args.current), sha256_file(args.parent), sha256_file(request_path)
        output: list[dict] = []
        complete = 0
        for start in range(0, len(full_uids), args.cohort_size):
            cohort = full_uids[start:start + args.cohort_size]
            values = evaluate_full_cache_cohort(
                uids=cohort, by_user=by_user, history=history, parent=parent, current=current,
                parent_name=parent_name, current_name=current_name, edge=args.edge,
                checkpoint_hash=current_hash, parent_hash=parent_hash, manifest_hash=manifest_hash,
                cutover=cutover, lineage_models=[(parent_name, parent)],
                event_end_exclusive=args.end_day * DAY, include_request_local=False,
                include_parent_exact=args.include_parent_exact,
                refinement_cast_maps=cast_maps,
                query_chunk_size=256,
            )
            output.extend(pair_rows(
                values,
                include_parent_exact=args.include_parent_exact,
                include_fixed_refinement=args.include_fixed_refinement,
            ))
            complete += len(cohort)
            (args.output / f"progress_rank{rank}.json").write_text(json.dumps({"rank": rank, "completed_users": complete, "assigned_users": len(selected_uids), "phase": "batched_full_cache"}) + "\n")
        for uid in fallback_uids:
            output.extend(evaluate_fallback_user(
                requests=by_user[uid], history=history, parent=parent, current=current, edge=args.edge,
                parent_name=parent_name, current_name=current_name, cutover=cutover,
                current_hash=current_hash, parent_hash=parent_hash, manifest_hash=manifest_hash,
                include_parent_exact=args.include_parent_exact,
                refinement_cast_maps=cast_maps,
            ))
            complete += 1
            (args.output / f"progress_rank{rank}.json").write_text(json.dumps({"rank": rank, "completed_users": complete, "assigned_users": len(selected_uids), "phase": "fallback_variable_cache"}) + "\n")
        shard = args.output / f"raw_rank{rank}.parquet"
        partial_shard = args.output / f"raw_rank{rank}.parquet.partial"
        pq.write_table(pa.Table.from_pylist(output), partial_shard, compression="zstd")
        os.replace(partial_shard, shard)
        if rank == 0:
            shards = [args.output / f"raw_rank{value}.parquet" for value in range(world)]
            wait_for_paths(shards, description="all raw shards")
            merged = pa.concat_tables([pq.read_table(args.output / f"raw_rank{value}.parquet") for value in range(world)])
            expected_paths = RELEASE_DEBT_PATHS if args.include_parent_exact else PAIR_PATHS
            if args.include_fixed_refinement:
                expected_paths = (*expected_paths, OUR_PATH)
            requests = validate_pair_raw(merged, expected_paths=expected_paths)
            raw = args.output / "raw.parquet"
            partial_raw = args.output / "raw.parquet.partial"
            pq.write_table(merged, partial_raw, compression="zstd")
            os.replace(partial_raw, raw)
            seal = {"status": "native_onehop_reuse_raw_sealed_before_label_join", "raw_sha256": sha256_file(raw), "rows": merged.num_rows, "requests": requests, "stage": args.stage, "edge": args.edge, "cutover_day": args.cutover_day, "evaluation_day_range": [args.start_day, args.end_day], "contains_reuse": True, "contains_parent_exact_rolling": args.include_parent_exact, "contains_fixed_one_release_refinement": args.include_fixed_refinement, "fixed_refinement_path": OUR_PATH if args.include_fixed_refinement else None, "fixed_refinement_plan": {"repair_width": 128, "group_size": 2, "full_history_carriers": 64, "recursive_reuse": False} if args.include_fixed_refinement else None, "recursive_reuse": False, "architecture": "hstu_native_cc"}
            (args.output / "raw.seal.json").write_text(json.dumps(seal, indent=2) + "\n")
            (args.output / ".raw_complete").write_text("complete\n")
            print(json.dumps(seal, indent=2))
        else:
            wait_for_paths([args.output / ".raw_complete"], description="rank-zero raw merge")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
