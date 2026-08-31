#!/usr/bin/env python3
"""Four-GPU raw-first score canary for the compact-probe AV sidecar."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.distributed as dist


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "insight"))

from broadcast_residual import generate_av_broadcast_residual  # noqa: E402
from evaluate_yambda500m_foundation_raw import (  # noqa: E402
    DAY,
    balanced_users,
    load_histories,
    load_model,
)
from hstu_kvcache.evaluation import (  # noqa: E402
    VersionedCacheState,
    append_timestamp_group,
    materialize_state,
    observe_rolling,
    timestamp_groups,
)
from one_release_refinement import (  # noqa: E402
    BROADCAST_RESIDUAL_PATH,
    OUR_PATH,
    build_broadcast_probe_source_cache,
    build_fixed_refinement_cache,
    parameter_cast_maps,
)
from reader_compatibility_correction import (  # noqa: E402
    intervene_reader_correction,
    scale_correction,
)


PATHS = (
    "current_exact_rolling",
    "one_hop_reuse_rolling",
    OUR_PATH,
    BROADCAST_RESIDUAL_PATH,
)


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
            raise TimeoutError(f"timed out waiting for {description}")
        time.sleep(0.1)


@torch.inference_mode()
def evaluate_user(
    *,
    requests: list[dict],
    history,
    parent,
    current,
    cast_maps,
    edge: str,
    cutover: int,
    end: int,
    current_hash: str,
    parent_hash: str,
    manifest_hash: str,
) -> tuple[list[dict], float]:
    uid = int(requests[0]["uid"])
    timestamps, items, behaviors = history.rows[uid]
    events = [
        (int(timestamp), int(item), int(behavior))
        for timestamp, item, behavior in zip(timestamps, items, behaviors, strict=True)
    ]
    prefix = [event for event in events if event[0] < cutover]
    ordered = [event for _, group in timestamp_groups(prefix) for event in group][-512:]
    if len(ordered) != 512:
        raise RuntimeError("broadcast-residual canary requires a full cutover history")
    current_state = materialize_state(
        current, ordered, producer_version=edge.split("_to_")[1], max_length=512
    )
    reuse_state = materialize_state(
        parent, ordered, producer_version=edge.split("_to_")[0], max_length=512
    )
    device = reuse_state.cache.k.device
    prefix_times = torch.tensor([[event[0] for event in ordered]], device=device)
    prefix_deltas = torch.zeros_like(prefix_times, dtype=torch.float32)
    prefix_deltas[:, 1:] = prefix_times[:, 1:] - prefix_times[:, :-1]
    prefix_items = torch.tensor(
        [[event[1] for event in ordered]], dtype=torch.long, device=device
    )
    prefix_behaviors = torch.tensor(
        [[event[2] for event in ordered]], dtype=torch.long, device=device
    )
    design0_cache, design0_layout = build_fixed_refinement_cache(
        parent_cache=reuse_state.cache,
        current=current,
        item_ids=prefix_items,
        behaviors=prefix_behaviors,
        time_deltas=prefix_deltas,
        cast_maps=cast_maps,
    )
    if design0_layout.carriers != 64:
        raise RuntimeError("Design 0 canary layout differs")
    design0_state = VersionedCacheState(
        design0_cache,
        reuse_state.last_timestamp,
        ("design0",) * design0_cache.seq_len,
    )
    probe_source, source_layout = build_broadcast_probe_source_cache(
        parent_cache=reuse_state.cache,
        current=current,
        item_ids=prefix_items,
        behaviors=prefix_behaviors,
        time_deltas=prefix_deltas,
        cast_maps=cast_maps,
    )
    if source_layout.carriers != 32:
        raise RuntimeError("compact probe source must have 32 carriers")
    sidecar = generate_av_broadcast_residual(
        current, probe_source, reuse_state.cache, prefix_items[:, -1]
    )
    corrections = tuple(value.detach() for value in sidecar.corrections)

    post_groups = list(
        timestamp_groups(event for event in events if cutover <= event[0] < end)
    )
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
            current_state = append_timestamp_group(
                current, current_state, group, producer_version="current", max_length=512
            )
            reuse_state = append_timestamp_group(
                current, reuse_state, group, producer_version="current", max_length=512
            )
            design0_state = append_timestamp_group(
                current, design0_state, group, producer_version="current", max_length=512
            )
            append_count += len(group)
            group_index += 1
        factor = torch.tensor(
            [max(0, 512 - evictions) / 512.0], dtype=torch.float32, device=device
        )
        scaled = scale_correction(corrections, factor)
        for request in simultaneous:
            candidate_id = int(request["item_idx"])
            current_score, current_readout = observe_rolling(
                current,
                current_state,
                candidate_id=candidate_id,
                query_timestamp=query_time,
            )
            reuse_score, reuse_readout = observe_rolling(
                current,
                reuse_state,
                candidate_id=candidate_id,
                query_timestamp=query_time,
            )
            design0_score, design0_readout = observe_rolling(
                current,
                design0_state,
                candidate_id=candidate_id,
                query_timestamp=query_time,
            )
            candidate = torch.tensor([[candidate_id]], dtype=torch.long, device=device)
            query_delta = torch.tensor(
                [float(query_time - reuse_state.last_timestamp)], device=device
            )
            broadcast_scores, broadcast_readout = intervene_reader_correction(
                current,
                reuse_state.cache,
                candidate,
                query_delta,
                stage="av_aggregation",
                corrections=scaled,
            )
            broadcast_score = float(broadcast_scores[0, 0])
            broadcast_readout_cpu = broadcast_readout[0, 0].detach().cpu()
            common = {
                "request_id": str(request["request_id"]),
                "uid": uid,
                "query_timestamp": query_time,
                "edge": edge,
                "append_count_since_cutover": append_count,
                "rolling_evictions": evictions,
                "coverage_scale": float(factor[0]),
                "checkpoint_sha256": current_hash,
                "parent_checkpoint_sha256": parent_hash,
                "manifest_sha256": manifest_hash,
            }
            paths = (
                ("current_exact_rolling", current_score, current_readout),
                ("one_hop_reuse_rolling", reuse_score, reuse_readout),
                (OUR_PATH, design0_score, design0_readout),
                (BROADCAST_RESIDUAL_PATH, broadcast_score, broadcast_readout_cpu),
            )
            for path, score, readout in paths:
                output.append(
                    {
                        **common,
                        "path": path,
                        "hstu_logit": score,
                        "readout_normalized_l2": float(
                            (readout - current_readout).norm()
                            / (current_readout.norm() + 1e-12)
                        ),
                        "readout_cosine": float(
                            torch.nn.functional.cosine_similarity(
                                readout[None], current_readout[None]
                            )
                        ),
                    }
                )
        while group_index < len(post_groups) and post_groups[group_index][0] == query_time:
            _, group = post_groups[group_index]
            evictions += max(0, current_state.cache.seq_len + len(group) - 512)
            current_state = append_timestamp_group(
                current, current_state, group, producer_version="current", max_length=512
            )
            reuse_state = append_timestamp_group(
                current, reuse_state, group, producer_version="current", max_length=512
            )
            design0_state = append_timestamp_group(
                current, design0_state, group, producer_version="current", max_length=512
            )
            append_count += len(group)
            group_index += 1
    return output, sidecar.replay_max_abs_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge", required=True)
    parser.add_argument("--cutover-day", type=int, required=True)
    parser.add_argument("--end-day", type=int, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-users", type=int, default=8)
    args = parser.parse_args()
    if args.end_day != args.cutover_day + 14 or args.max_users != 8:
        raise ValueError("only the frozen E14 eight-user-per-rank canary is allowed")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    try:
        if world != 4:
            raise RuntimeError("broadcast-residual canary requires four ranks")
        if rank == 0:
            if args.output.exists():
                raise FileExistsError(f"refusing to overwrite {args.output}")
            args.output.mkdir(parents=True)
            (args.output / ".directory_ready").write_text("ready\n")
        wait_for_paths([args.output / ".directory_ready"], "output directory")

        parent, parent_payload = load_model(args.parent, device)
        current, current_payload = load_model(args.current, device)
        if parent_payload["config"] != current_payload["config"]:
            raise RuntimeError("Parent and Current configurations differ")
        cast_maps = parameter_cast_maps(parent, current)
        population = set(pd.read_parquet(args.population).uid.astype(int))
        request_path = args.manifest_dir / "requests_fidelity.parquet"
        request_table = pq.read_table(
            request_path,
            filters=[
                ("time_block", "=", "matrix_horizon"),
                ("target_known", "=", True),
                ("query_timestamp", ">=", args.cutover_day * DAY),
                ("query_timestamp", "<", args.end_day * DAY),
            ],
            columns=["request_id", "uid", "query_timestamp", "item_idx"],
        ).to_pandas()
        request_table = request_table[request_table.uid.isin(population)].sort_values(
            ["uid", "query_timestamp", "request_id"]
        )
        rows = request_table.to_dict("records")
        assignment = balanced_users(rows, world)
        selected_uids = sorted(uid for uid, owner in assignment.items() if owner == rank)[
            : args.max_users
        ]
        selected = set(selected_uids)
        by_user: dict[int, list[dict]] = {}
        for row in rows:
            if int(row["uid"]) in selected:
                by_user.setdefault(int(row["uid"]), []).append(row)
        oov_buckets = int(current_payload["config"]["num_items"]) - 781_678
        history = load_histories(selected_uids, oov_buckets=oov_buckets)
        current_hash, parent_hash, manifest_hash = (
            sha256(args.current),
            sha256(args.parent),
            sha256(request_path),
        )
        output: list[dict] = []
        replay_errors = []
        for uid in selected_uids:
            rows_out, replay_error = evaluate_user(
                requests=by_user[uid],
                history=history,
                parent=parent,
                current=current,
                cast_maps=cast_maps,
                edge=args.edge,
                cutover=args.cutover_day * DAY,
                end=args.end_day * DAY,
                current_hash=current_hash,
                parent_hash=parent_hash,
                manifest_hash=manifest_hash,
            )
            output.extend(rows_out)
            replay_errors.append(replay_error)
        frame = pd.DataFrame(output)
        if set(frame.path.unique()) != set(PATHS):
            raise RuntimeError("canary path set differs")
        target = args.output / f"raw_rank{rank}.parquet"
        frame.to_parquet(target.with_suffix(".parquet.partial"), index=False)
        os.replace(target.with_suffix(".parquet.partial"), target)
        correctness = args.output / f"correctness_rank{rank}.json"
        correctness.write_text(
            json.dumps(
                {
                    "rank": rank,
                    "users": len(selected_uids),
                    "probe_replay_max_abs_error": max(replay_errors, default=0.0),
                }
            )
            + "\n"
        )
        if rank == 0:
            shards = [args.output / f"raw_rank{value}.parquet" for value in range(world)]
            checks = [args.output / f"correctness_rank{value}.json" for value in range(world)]
            wait_for_paths([*shards, *checks], "all canary shards")
            merged = pd.concat([pd.read_parquet(path) for path in shards], ignore_index=True)
            path_sets = merged.groupby("request_id").path.agg(set)
            if not path_sets.map(lambda value: value == set(PATHS)).all():
                raise RuntimeError("every request must have every canary path")
            raw = args.output / "raw.parquet"
            merged.to_parquet(raw.with_suffix(".parquet.partial"), index=False)
            os.replace(raw.with_suffix(".parquet.partial"), raw)
            maximum_error = max(
                json.loads(path.read_text())["probe_replay_max_abs_error"] for path in checks
            )
            seal = {
                "status": "av_broadcast_residual_canary_raw_sealed",
                "edge": args.edge,
                "users": int(merged.uid.nunique()),
                "requests": int(merged.request_id.nunique()),
                "paths": list(PATHS),
                "labels_read": False,
                "probe_replay_max_abs_error": maximum_error,
                "raw_sha256": sha256(raw),
                "checkpoint_sha256": current_hash,
                "parent_checkpoint_sha256": parent_hash,
                "requests_fidelity_sha256": manifest_hash,
            }
            (args.output / "raw.seal.json").write_text(json.dumps(seal, indent=2) + "\n")
            for path in [*shards, *checks]:
                path.unlink()
            (args.output / ".raw_complete").write_text("complete\n")
            print(json.dumps(seal, indent=2), flush=True)
        else:
            wait_for_paths([args.output / ".raw_complete"], "rank-zero merge")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
