#!/usr/bin/env python3
"""Evaluate one P9.9 cell with one migrated state per uid and true rolling append."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
import yaml

import eval_p7_h_raw as p7eval
import eval_p8_release_raw as p8raw
import eval_p9_materialized_lineage_canary as rolling
import train_p7_theta0 as p7
from hstu_kvcache.data.p7_training import load_p7_requests
from hstu_kvcache.models import HSTUKVCache, append_with_rolling_cap


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/p9_9_heldout_rolling_quality_contract_v1.yaml"
MANIFEST = ROOT / "data/manifests/p8_release_v1"
POPULATION = ROOT / "data/manifests/p9_full_population_v1"
LISTENS = ROOT / "data/raw/yambda/flat/50m/listens.parquet"
OUTPUT_ROOT = ROOT / "results/p9/heldout_rolling_quality_raw"
ACTIONS = (
    "noop", "layer0_recent128", "layer0_middle", "layer0_full",
    "hybrid_tail128", "exact_all",
)


class RangeReader:
    """Read monotonic raw ranges while retaining only a small row-group cache."""

    def __init__(self, path: Path) -> None:
        self.parquet = pq.ParquetFile(path)
        sizes = [self.parquet.metadata.row_group(i).num_rows for i in range(self.parquet.num_row_groups)]
        self.ends = np.cumsum(sizes).tolist()
        self.cache: dict[int, pa.Table] = {}

    def rows(self, start: int, end: int) -> pa.Table:
        pieces = []
        cursor = start
        while cursor < end:
            group = bisect.bisect_right(self.ends, cursor)
            group_start = 0 if group == 0 else self.ends[group - 1]
            group_end = self.ends[group]
            if group not in self.cache:
                self.cache[group] = self.parquet.read_row_group(
                    group, columns=["uid", "timestamp", "item_id", "is_organic"]
                )
                if len(self.cache) > 3:
                    self.cache.pop(next(iter(self.cache)))
            table = self.cache[group]
            count = min(end, group_end) - cursor
            pieces.append(table.slice(cursor - group_start, count))
            cursor += count
        return pa.concat_tables(pieces)


def validate_contract() -> dict:
    contract = yaml.safe_load(CONTRACT.read_text())
    paths = {
        "p9_8_contract_sha256": ROOT / "configs/contracts/p9_8_cutover_profiler_contract_v1.yaml",
        "p9_8_raw_seal_sha256": ROOT / "results/p9/p9_8_cutover_profiler_raw_seal_v1.json",
        "p9_8_adjudication_sha256": ROOT / "results/p9/p9_8_cutover_profiler_v1.json",
        "p9_7_uid_canary_sha256": ROOT / "results/p9/p9_7_uid_executor_canary_v1.json",
        "state_transition_source_sha256": ROOT / "src/hstu_kvcache/models/state_transition.py",
        "p8_materialization_summary_sha256": ROOT / "data/manifests/p8_release_v1/materialization_summary.json",
    }
    for key, path in paths.items():
        if p7.sha256_file(path) != contract["inputs"][key]:
            raise RuntimeError(f"P9.9 input hash mismatch: {key}")
    if tuple(contract["scope"]["actions"]) != ACTIONS:
        raise RuntimeError("P9.9 action set changed")
    return contract


def request_pointers(split: str) -> dict[str, int]:
    index_path = MANIFEST / split / "manifest.index.json"
    index = json.loads(index_path.read_text())
    tables = [pq.read_table(index_path.parent / shard["path"]) for shard in index["request_shards"]]
    table = pa.concat_tables(tables)
    table = table.filter(pc.and_(pc.equal(table["workload"], "F"), pc.equal(table["manifest_kind"], "quality")))
    return {
        str(request_id): int(end)
        for request_id, end in zip(table["request_id"].to_pylist(), table["raw_prefix_end_exclusive"].to_pylist(), strict=True)
    }


def make_event_arrays(table: pa.Table) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        table["timestamp"].to_numpy(zero_copy_only=False).astype(np.int64),
        table["item_id"].to_numpy(zero_copy_only=False).astype(np.int64),
        table["is_organic"].to_numpy(zero_copy_only=False).astype(np.int64),
    )


def batch_snapshot_tensors(records: list[dict], device: torch.device):
    items = np.stack([record["items"][: record["snapshot_length"] ] for record in records])
    organic = np.stack([record["organic"][: record["snapshot_length"] ] for record in records])
    timestamps = np.stack([record["timestamps"][: record["snapshot_length"] ] for record in records])
    deltas = np.zeros(timestamps.shape, dtype=np.float32)
    if timestamps.shape[1] > 1:
        deltas[:, 1:] = np.diff(timestamps, axis=1).clip(0, 7 * 86_400)
    return (
        torch.tensor(items, dtype=torch.long, device=device),
        torch.tensor(1 + (1 - organic), dtype=torch.long, device=device),
        torch.tensor(deltas, dtype=torch.float32, device=device),
    )


def select_cache(cache: HSTUKVCache, index: int) -> HSTUKVCache:
    return HSTUKVCache(
        k=cache.k[:, index : index + 1].clone(),
        v=cache.v[:, index : index + 1].clone(),
        seq_len=cache.seq_len,
    )


def stack_action_caches(caches: list[HSTUKVCache], index: int) -> HSTUKVCache:
    return HSTUKVCache(
        k=torch.cat([cache.k[:, index : index + 1] for cache in caches], dim=1),
        v=torch.cat([cache.v[:, index : index + 1] for cache in caches], dim=1),
        seq_len=caches[0].seq_len,
    )


def append_arrays(current, state, timestamps, items, organic, previous_timestamp, device):
    if not len(timestamps):
        return state
    deltas = np.zeros(len(timestamps), dtype=np.float32)
    deltas[0] = np.clip(int(timestamps[0]) - int(previous_timestamp), 0, 7 * 86_400)
    if len(timestamps) > 1:
        deltas[1:] = np.diff(timestamps).clip(0, 7 * 86_400)
    batch = len(ACTIONS)
    tensors = (
        torch.tensor(items[None, :], dtype=torch.long, device=device).repeat(batch, 1),
        torch.tensor((1 + (1 - organic))[None, :], dtype=torch.long, device=device).repeat(batch, 1),
        torch.tensor(deltas[None, :], dtype=torch.float32, device=device).repeat(batch, 1),
    )
    return append_with_rolling_cap(current, state, *tensors, max_length=512)


def append_user_groups(current, states, previous_timestamps, payloads, device) -> int:
    """Append variable per-user suffixes while batching equal-length states."""
    maximum = max((len(payload["timestamps"]) for payload in payloads), default=0)
    appended = 0
    for position in range(maximum):
        active = [payload for payload in payloads if position < len(payload["timestamps"])]
        by_length: dict[int, list[dict]] = {}
        for payload in active:
            by_length.setdefault(states[payload["index"]].seq_len, []).append(payload)
        for length in sorted(by_length):
            selected = by_length[length]
            selected_states = [states[payload["index"]] for payload in selected]
            merged = HSTUKVCache(
                k=torch.cat([value.k for value in selected_states], dim=1),
                v=torch.cat([value.v for value in selected_states], dim=1),
                seq_len=length,
            )
            event_items = []
            event_behaviors = []
            event_deltas = []
            for payload in selected:
                index = payload["index"]
                timestamp = int(payload["timestamps"][position])
                event_items.extend([int(payload["items"][position])] * len(ACTIONS))
                event_behaviors.extend([1 + (1 - int(payload["organic"][position]))] * len(ACTIONS))
                event_deltas.extend([
                    float(np.clip(timestamp - previous_timestamps[index], 0, 7 * 86_400))
                ] * len(ACTIONS))
                previous_timestamps[index] = timestamp
            merged = append_with_rolling_cap(
                current, merged,
                torch.tensor(event_items, dtype=torch.long, device=device)[:, None],
                torch.tensor(event_behaviors, dtype=torch.long, device=device)[:, None],
                torch.tensor(event_deltas, dtype=torch.float32, device=device)[:, None],
                max_length=512,
            )
            for local, payload in enumerate(selected):
                start = local * len(ACTIONS)
                states[payload["index"]] = HSTUKVCache(
                    k=merged.k[:, start : start + len(ACTIONS)].clone(),
                    v=merged.v[:, start : start + len(ACTIONS)].clone(),
                    seq_len=merged.seq_len,
                )
            appended += len(selected)
    return appended


def score_action_groups(current, base, states, requests, last_timestamps, device) -> np.ndarray:
    if any(len(request.candidate_ids) != 1 for request in requests):
        raise RuntimeError("P9.9 F quality request must contain one observed candidate")
    action_count = len(ACTIONS)
    state = HSTUKVCache(
        k=torch.cat([value.k for value in states], dim=1),
        v=torch.cat([value.v for value in states], dim=1),
        seq_len=states[0].seq_len,
    )
    candidate = torch.tensor(
        np.stack([request.candidate_ids for request in requests]), dtype=torch.long, device=device
    ).repeat_interleave(action_count, dim=0)
    query_delta = torch.tensor([
        float(np.clip(request.query_timestamp - last_timestamp, 0, 7 * 86_400))
        for request, last_timestamp in zip(requests, last_timestamps, strict=True)
    ], dtype=torch.float32, device=device).repeat_interleave(action_count)
    residual = current.score_cc_reuse(
        state, candidate, query_delta,
        prefix_lengths=torch.full((len(requests) * action_count,), state.seq_len, dtype=torch.long, device=device),
        query_type_ids=torch.full((len(requests) * action_count,), 2, dtype=torch.long, device=device),
    )[:, 0].reshape(len(requests), action_count)
    base_features = torch.tensor(
        np.stack([request.base_features for request in requests]), dtype=torch.float32, device=device
    )
    base_logits = base(base_features)[:, 0]
    return (residual + base_logits[:, None]).detach().cpu().numpy().astype(np.float64)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", choices=("r0", "r1_edge1", "r1_edge2", "r2"), required=True)
    parser.add_argument("--model", choices=("m0_f", "m1"), required=True)
    parser.add_argument("--seed", type=int, choices=(17, 37, 71), required=True)
    parser.add_argument("--device", choices=("cuda:0", "cuda:1"), required=True)
    parser.add_argument("--max-users", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    contract = validate_contract()
    device = torch.device(args.device)
    split, cutover = p8raw.RELEASE_EDGE[args.release]
    edge = "edge2" if split.startswith("edge2") else "edge1"
    output = (args.output or OUTPUT_ROOT / args.release / f"{args.model}_seed{args.seed}").resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    started = time.perf_counter()
    manifest_requests = load_p7_requests(MANIFEST, LISTENS, split, "F", manifest_kind="quality")
    sealed_request_source = (
        ROOT / "results/p8/staleness_raw" / args.release
        / f"{args.model}_seed{args.seed}" / "F_quality.parquet"
    )
    sealed_request_ids = set(
        str(value) for value in pq.read_table(sealed_request_source, columns=["request_id"])["request_id"].to_pylist()
    )
    requests = [request for request in manifest_requests if request.request_id in sealed_request_ids]
    if len(requests) != len(sealed_request_ids):
        raise RuntimeError("P9.9 could not reconstruct every sealed P8 quality request")
    grouped: dict[int, list] = {}
    for request in requests:
        grouped.setdefault(request.uid, []).append(request)
    for values in grouped.values():
        values.sort(key=lambda row: (row.query_timestamp, row.request_id))
    pointers = request_pointers(split)
    population_rows = pq.read_table(POPULATION / edge / "states.parquet").to_pylist()
    population = {int(row["uid"]): row for row in population_rows}
    cold_start_users = set(grouped) - set(population)
    cold_start_requests = sum(len(grouped[uid]) for uid in cold_start_users)
    if cold_start_users:
        grouped = {uid: values for uid, values in grouped.items() if uid in population}
    if args.max_users is not None:
        selected = sorted(grouped, key=lambda uid: hashlib.sha256(str(uid).encode()).digest())[: args.max_users]
        grouped = {uid: grouped[uid] for uid in selected}

    reader = RangeReader(LISTENS)
    records = []
    for uid in sorted(grouped, key=lambda value: int(population[value]["raw_prefix_end_exclusive"])):
        state_row = population[uid]
        snapshot_end = int(state_row["raw_prefix_end_exclusive"])
        snapshot_length = int(state_row["effective_prefix_length"])
        max_end = max(pointers[request.request_id] for request in grouped[uid])
        table = reader.rows(snapshot_end - snapshot_length, max_end)
        if not np.all(table["uid"].to_numpy(zero_copy_only=False) == uid):
            raise RuntimeError("P9.9 raw range crossed uid boundary")
        timestamps, items, organic = make_event_arrays(table)
        records.append({
            "uid": uid, "snapshot_end": snapshot_end, "snapshot_length": snapshot_length,
            "timestamps": timestamps, "items": items, "organic": organic,
            "requests": grouped[uid],
        })

    checkpoint = p8raw.TRAIN_ROOT / args.release / f"{args.model}_seed{args.seed}" / "selected.pt"
    current, child = p8raw.load_model(checkpoint, device)
    if not child["admitted"]:
        raise RuntimeError("P9.9 refuses a non-admitted release checkpoint")
    parent, _ = p8raw.load_model(ROOT / child["parent_checkpoint"], device)
    base = p7eval.load_base("F", device)
    rows = []
    migration_invocations = 0
    appended_action_events = 0
    by_length: dict[int, list[dict]] = {}
    for record in records:
        by_length.setdefault(record["snapshot_length"], []).append(record)
    for length in sorted(by_length):
        values = by_length[length]
        for begin in range(0, len(values), int(contract["execution"]["cutover_batch_size"])):
            batch_records = values[begin : begin + int(contract["execution"]["cutover_batch_size"])]
            tensors = batch_snapshot_tensors(batch_records, device)
            parent_snapshot = parent.compute_kv(*tensors)
            migrated = [rolling.migrate(action, current, parent_snapshot, tensors) for action in ACTIONS]
            migration_invocations += len(batch_records) * len(ACTIONS)
            states = [stack_action_caches(migrated, index) for index in range(len(batch_records))]
            cursors = [0] * len(batch_records)
            positions = [0] * len(batch_records)
            previous_timestamps = [
                int(record["timestamps"][record["snapshot_length"] - 1]) for record in batch_records
            ]
            while any(position < len(record["requests"]) for position, record in zip(positions, batch_records, strict=True)):
                pending = []
                append_payloads = []
                for batch_index, record in enumerate(batch_records):
                    if positions[batch_index] >= len(record["requests"]):
                        continue
                    request = record["requests"][positions[batch_index]]
                    target_end = pointers[request.request_id] - record["snapshot_end"]
                    if target_end < cursors[batch_index] or target_end > len(record["timestamps"]) - record["snapshot_length"]:
                        raise RuntimeError("P9.9 request raw pointer is inconsistent")
                    absolute_start = record["snapshot_length"] + cursors[batch_index]
                    absolute_end = record["snapshot_length"] + target_end
                    new_timestamps = record["timestamps"][absolute_start:absolute_end]
                    if len(new_timestamps) and np.any(new_timestamps >= request.query_timestamp):
                        raise RuntimeError("P9.9 appended a coincident/future event")
                    append_payloads.append({
                        "index": batch_index,
                        "timestamps": new_timestamps,
                        "items": record["items"][absolute_start:absolute_end],
                        "organic": record["organic"][absolute_start:absolute_end],
                    })
                    cursors[batch_index] = target_end
                    pending.append((batch_index, request))
                appended_action_events += (
                    append_user_groups(current, states, previous_timestamps, append_payloads, device)
                    * len(ACTIONS)
                )
                by_state_length: dict[int, list[tuple[int, object]]] = {}
                for item in pending:
                    by_state_length.setdefault(states[item[0]].seq_len, []).append(item)
                for state_length in sorted(by_state_length):
                    selected = by_state_length[state_length]
                    selected_indices = [item[0] for item in selected]
                    selected_requests = [item[1] for item in selected]
                    logits = score_action_groups(
                        current, base, [states[index] for index in selected_indices], selected_requests,
                        [previous_timestamps[index] for index in selected_indices], device,
                    )
                    for request, request_logits in zip(selected_requests, logits, strict=True):
                        current_logit = float(request_logits[ACTIONS.index("exact_all")])
                        for action, action_logit in zip(ACTIONS, request_logits, strict=True):
                            rows.append({
                                "request_id": request.request_id, "uid": request.uid,
                                "query_timestamp": request.query_timestamp, "action": action,
                                "action_logit": float(action_logit), "current_exact_logit": current_logit,
                                "label": int(request.label), "request_weight": float(request.request_weight),
                                "prior_30m_same_item": request.prior_30m_same_item,
                                "latest_item": request.latest_item, "is_organic": request.is_organic,
                            })
                    for index in selected_indices:
                        positions[index] += 1

    expected_requests = sum(len(value) for value in grouped.values())
    if len(rows) != expected_requests * len(ACTIONS):
        raise RuntimeError("P9.9 request/action row conservation failed")
    output.mkdir(parents=True)
    raw_path = output / "quality_actions.parquet"
    pq.write_table(pa.Table.from_pylist(rows), raw_path, compression="zstd")
    r0_max = None
    if args.release == "r0":
        r0_max = max(
            rolling.bernoulli_js(float(row["action_logit"]), float(row["current_exact_logit"]))
            for row in rows
        )
    payload = {
        "status": "passed" if r0_max is None or r0_max <= float(contract["canary"]["r0_all_action_JS_max"]) else "failed",
        "release": args.release, "model": args.model, "seed": args.seed,
        "scope": "canary" if args.max_users is not None else "full",
        "users": len(grouped), "requests": expected_requests, "actions": len(ACTIONS),
        "manifest_quality_requests_before_sealed_P8_filter": len(manifest_requests),
        "sealed_P8_quality_requests": len(sealed_request_ids),
        "cold_start_users_without_cutover_state": len(cold_start_users),
        "cold_start_requests_without_cutover_state": cold_start_requests,
        "migration_invocations": migration_invocations,
        "expected_migration_invocations": len(grouped) * len(ACTIONS),
        "appended_action_events": appended_action_events,
        "wall_seconds": time.perf_counter() - started,
        "r0_all_action_JS_max": r0_max,
        "contract_hash": p7.sha256_file(CONTRACT),
        "checkpoint_hash": p7.sha256_file(checkpoint),
        "parent_checkpoint_hash": p7.sha256_file(ROOT / child["parent_checkpoint"]),
        "raw_path": str(raw_path.relative_to(ROOT)), "raw_sha256": p7.sha256_file(raw_path),
        "metrics_computed": False,
    }
    (output / "raw_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if not key.endswith("hash")}, indent=2))
    if payload["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
