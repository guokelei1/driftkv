#!/usr/bin/env python3
"""Evaluate one admitted 8L release under true per-uid rolling-cache lineage.

Scores are generated from the label-free F fidelity view.  Quality metadata is
joined only after every score has been computed.  The two persistent states are
materialized once per uid at cutover: Exact uses the current model and Reuse
uses the parent model; every intervening listen is then appended with the
current model while evicting before append at the frozen 1024-token cap.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
import yaml

import eval_p7_h_raw as p7_h
import scale_8l_common as scale
import train_p7_theta0 as p7
from hstu_kvcache.data.p7_training import P7Request, load_p7_requests
from hstu_kvcache.models import HSTU, HSTUConfig, HSTUKVCache, append_with_rolling_cap

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/scale_8l_hs_v1.yaml"
MANIFEST = ROOT / "data/manifests/p8_release_v1"
LISTENS = ROOT / "data/raw/yambda/flat/50m/listens.parquet"
OUTPUT_ROOT = ROOT / "results/scale_8l_v1/hs_raw"
RELEASES = {
    "r1_edge1": ("edge1_evaluation", 231 * 86_400),
    "r1_edge2": ("edge2_evaluation", 245 * 86_400),
    "r2": ("edge1_evaluation", 231 * 86_400),
}
CAP = 1024
SNAPSHOT_BATCH = 4
DIRECT_BATCH = 8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class RangeReader:
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
            count = min(end, group_end) - cursor
            pieces.append(self.cache[group].slice(cursor - group_start, count))
            cursor += count
        if not pieces:
            return pa.table({
                "uid": pa.array([], type=pa.int64()),
                "timestamp": pa.array([], type=pa.int64()),
                "item_id": pa.array([], type=pa.int64()),
                "is_organic": pa.array([], type=pa.int64()),
            })
        return pa.concat_tables(pieces)


def load_contract() -> dict[str, Any]:
    value = yaml.safe_load(CONTRACT.read_text())
    paths = {
        "scale_contract_sha256": scale.CONTRACT,
        "p8_manifest_summary_sha256": MANIFEST / "materialization_summary.json",
        "frozen_base_bundle_sha256": scale.BASE_ROOT / "bundle_manifest.json",
        "state_transition_source_sha256": ROOT / "src/hstu_kvcache/models/state_transition.py",
        "raw_evaluator_sha256": Path(__file__),
    }
    for release in RELEASES:
        paths[f"{release}_checkpoint_sha256"] = (
            scale.OUTPUT / "releases" / release / "m0_f_seed17" / "selected.pt"
        )
    for key, path in paths.items():
        if value["sealed_inputs"][key] != sha256_file(path):
            raise RuntimeError(f"scale H/S contract hash mismatch: {key}")
    if value["data_access"]["qualification_or_theta3"] is not False:
        raise RuntimeError("scale H/S contract illegally authorizes sealed data")
    return value


def checkpoint_path(release: str) -> Path:
    return scale.OUTPUT / "releases" / release / "m0_f_seed17" / "selected.pt"


def load_model(path: Path, device: torch.device) -> tuple[HSTU, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.pop("model_state_dict")
    model = HSTU(HSTUConfig(**payload["config"]))
    model.load_state_dict(state, strict=True)
    del state
    return model.to(device).eval(), payload


def request_table(split: str, view: str) -> pa.Table:
    index_path = MANIFEST / split / "manifest.index.json"
    index = json.loads(index_path.read_text())
    paths = [index_path.parent / shard["path"] for shard in index["request_shards"]]
    table = pq.read_table(paths)
    return table.filter(
        pc.and_(pc.equal(table["workload"], "F"), pc.equal(table["manifest_kind"], view))
    )


def view_audit(requests: list[P7Request], view: str) -> dict[str, Any]:
    if len({row.request_id for row in requests}) != len(requests):
        raise RuntimeError(f"duplicate F {view} request id")
    if any(len(row.candidate_ids) != 1 for row in requests):
        raise RuntimeError(f"F {view} must contain one observed candidate per request")
    return {
        "requests": len(requests),
        "users": len({row.uid for row in requests}),
        "unique_request_ids": len(requests),
        "one_candidate_per_request": True,
        "view": view,
    }


def request_pointers(split: str, view: str) -> dict[str, tuple[int, int]]:
    table = request_table(split, view)
    return {
        str(request_id): (int(start), int(end))
        for request_id, start, end in zip(
            table["request_id"].to_pylist(),
            table["raw_user_row_start"].to_pylist(),
            table["raw_prefix_end_exclusive"].to_pylist(),
            strict=True,
        )
    }


def arrays(table: pa.Table):
    return (
        table["timestamp"].to_numpy(zero_copy_only=False).astype(np.int64),
        table["item_id"].to_numpy(zero_copy_only=False).astype(np.int64),
        table["is_organic"].to_numpy(zero_copy_only=False).astype(np.int64),
    )


def build_records(
    requests: list[P7Request], pointers: dict[str, tuple[int, int]], cutover: int, max_users: int | None
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    grouped: dict[int, list[P7Request]] = defaultdict(list)
    for request in requests:
        grouped[request.uid].append(request)
    for rows in grouped.values():
        rows.sort(key=lambda row: (row.query_timestamp, row.request_id))
    if max_users is not None:
        chosen = sorted(grouped, key=lambda uid: hashlib.sha256(f"scale-hs:{uid}".encode()).digest())[:max_users]
        grouped = {uid: grouped[uid] for uid in chosen}
    reader = RangeReader(LISTENS)
    records = []
    cold_users = cold_requests = 0
    for uid, rows in sorted(grouped.items(), key=lambda item: pointers[item[1][0].request_id][0]):
        starts = {pointers[row.request_id][0] for row in rows}
        if len(starts) != 1:
            raise RuntimeError("uid requests disagree on raw user start")
        user_start = starts.pop()
        maximum_end = max(pointers[row.request_id][1] for row in rows)
        table = reader.rows(user_start, maximum_end)
        if not np.all(table["uid"].to_numpy(zero_copy_only=False) == uid):
            raise RuntimeError("raw range crossed uid boundary")
        timestamps, items, organic = arrays(table)
        snapshot_end = int(np.searchsorted(timestamps, cutover, side="left"))
        if snapshot_end < 1:
            cold_users += 1
            cold_requests += len(rows)
            continue
        snapshot_start = max(0, snapshot_end - CAP)
        records.append(
            {
                "uid": uid,
                "user_start": user_start,
                "snapshot_end": snapshot_end,
                "snapshot_length": snapshot_end - snapshot_start,
                "snapshot_timestamps": timestamps[snapshot_start:snapshot_end],
                "snapshot_items": items[snapshot_start:snapshot_end],
                "snapshot_organic": organic[snapshot_start:snapshot_end],
                "suffix_timestamps": timestamps[snapshot_end:],
                "suffix_items": items[snapshot_end:],
                "suffix_organic": organic[snapshot_end:],
                "requests": rows,
            }
        )
    return records, {
        "cold_start_users_without_cutover_state": cold_users,
        "cold_start_requests_without_cutover_state": cold_requests,
    }


def snapshot_tensors(records: list[dict[str, Any]], device: torch.device):
    timestamps = np.stack([row["snapshot_timestamps"] for row in records])
    deltas = np.zeros(timestamps.shape, dtype=np.float32)
    if timestamps.shape[1] > 1:
        deltas[:, 1:] = np.diff(timestamps, axis=1).clip(0, 7 * 86_400)
    items = np.stack([row["snapshot_items"] for row in records])
    organic = np.stack([row["snapshot_organic"] for row in records])
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


def paired_state(exact: HSTUKVCache, reuse: HSTUKVCache) -> HSTUKVCache:
    return HSTUKVCache(
        k=torch.cat((exact.k, reuse.k), dim=1),
        v=torch.cat((exact.v, reuse.v), dim=1),
        seq_len=exact.seq_len,
    )


def append_events(
    current: HSTU,
    state: HSTUKVCache,
    timestamps: np.ndarray,
    items: np.ndarray,
    organic: np.ndarray,
    previous_timestamp: int,
    device: torch.device,
) -> tuple[HSTUKVCache, int]:
    if not len(timestamps):
        return state, previous_timestamp
    deltas = np.zeros(len(timestamps), dtype=np.float32)
    deltas[0] = np.clip(int(timestamps[0]) - previous_timestamp, 0, 7 * 86_400)
    if len(timestamps) > 1:
        deltas[1:] = np.diff(timestamps).clip(0, 7 * 86_400)
    state = append_with_rolling_cap(
        current,
        state,
        torch.tensor(items[None, :], dtype=torch.long, device=device).repeat(2, 1),
        torch.tensor((1 + (1 - organic))[None, :], dtype=torch.long, device=device).repeat(2, 1),
        torch.tensor(deltas[None, :], dtype=torch.float32, device=device).repeat(2, 1),
        max_length=CAP,
    )
    return state, int(timestamps[-1])


def score_pair(current: HSTU, base, state: HSTUKVCache, request: P7Request, last_timestamp: int, device):
    candidate = torch.tensor(request.candidate_ids[None, :], dtype=torch.long, device=device).repeat(2, 1)
    query_delta = torch.full(
        (2,), float(np.clip(request.query_timestamp - last_timestamp, 0, 7 * 86_400)),
        dtype=torch.float32, device=device,
    )
    residual = current.score_cc_reuse(
        state,
        candidate,
        query_delta,
        prefix_lengths=torch.full((2,), state.seq_len, dtype=torch.long, device=device),
        query_type_ids=torch.full((2,), 2, dtype=torch.long, device=device),
    )[:, 0]
    features = torch.tensor(request.base_features[None, :, :], dtype=torch.float32, device=device)
    base_logit = float(base(features)[0, 0])
    return base_logit, float(base_logit + residual[0]), float(base_logit + residual[1])


@torch.no_grad()
def rolling_scores(current, parent, base, records, pointers, device):
    output: dict[str, dict[str, float | int]] = {}
    materializations = appended_events = 0
    by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_length[record["snapshot_length"]].append(record)
    for length in sorted(by_length):
        values = by_length[length]
        for begin in range(0, len(values), SNAPSHOT_BATCH):
            batch = values[begin : begin + SNAPSHOT_BATCH]
            tensors = snapshot_tensors(batch, device)
            current_batch = current.compute_kv(*tensors)
            parent_batch = parent.compute_kv(*tensors)
            materializations += len(batch) * 2
            for index, record in enumerate(batch):
                state = paired_state(select_cache(current_batch, index), select_cache(parent_batch, index))
                cursor = 0
                previous_timestamp = int(record["snapshot_timestamps"][-1])
                for request in record["requests"]:
                    absolute_end = pointers[request.request_id][1]
                    target = absolute_end - (record["user_start"] + record["snapshot_end"])
                    if not cursor <= target <= len(record["suffix_timestamps"]):
                        raise RuntimeError("request raw pointer is inconsistent with rolling suffix")
                    timestamps = record["suffix_timestamps"][cursor:target]
                    if len(timestamps) and np.any(timestamps >= request.query_timestamp):
                        raise RuntimeError("rolling lineage appended coincident/future event")
                    state, previous_timestamp = append_events(
                        current,
                        state,
                        timestamps,
                        record["suffix_items"][cursor:target],
                        record["suffix_organic"][cursor:target],
                        previous_timestamp,
                        device,
                    )
                    appended_events += len(timestamps) * 2
                    cursor = target
                    base_logit, exact, reuse = score_pair(current, base, state, request, previous_timestamp, device)
                    output[request.request_id] = {
                        "base_logit": base_logit,
                        "current_exact_rolling_logit": exact,
                        "reuse_parent_rolling_logit": reuse,
                        "snapshot_tokens": record["snapshot_length"],
                        "suffix_events_before_query": target,
                    }
            del current_batch, parent_batch
    return output, {
        "cutover_state_materializations": materializations,
        "appended_state_events": appended_events,
        "lineage_transition_once_per_uid_per_version": True,
        "query_mutates_state": False,
        "rolling_cap": CAP,
    }


@torch.no_grad()
def direct_scores(current, parent, base, requests, device):
    output = {}
    max_base_delta = 0.0
    for start in range(0, len(requests), DIRECT_BATCH):
        batch = requests[start : start + DIRECT_BATCH]
        full_tensors = p7_h.collate(batch, device, history_tokens=1024)
        recent_tensors = p7_h.collate(batch, device, history_tokens=32)
        current_full = p7_h.score_path(current, full_tensors, device, workload="F", chunk_size=1)
        current_recent = p7_h.score_path(current, recent_tensors, device, workload="F", chunk_size=1)
        previous_full = p7_h.score_path(parent, full_tensors, device, workload="F", chunk_size=1)
        base_full = base(full_tensors["features"].float()).float()
        base_recent = base(recent_tensors["features"].float()).float()
        max_base_delta = max(max_base_delta, float((base_full - base_recent).abs().max()))
        for index, request in enumerate(batch):
            base_logit = float(base_full[index, 0])
            output[request.request_id] = {
                "base_logit": base_logit,
                "previous_full1024_logit": base_logit + float(previous_full[index, 0]),
                "current_recent32_logit": base_logit + float(current_recent[index, 0]),
                "request_local_current_full1024_logit": base_logit + float(current_full[index, 0]),
            }
    if max_base_delta != 0.0:
        raise RuntimeError("Frozen Base differs between Full1024 and Recent32")
    return output, max_base_delta


def fidelity_schema() -> pa.Schema:
    return pa.schema([
        ("request_id", pa.string()), ("uid", pa.int64()), ("query_timestamp", pa.int64()),
        ("candidate_id", pa.int64()), ("release", pa.string()), ("model", pa.string()),
        ("seed", pa.int32()), ("base_logit", pa.float32()),
        ("previous_full1024_logit", pa.float32()), ("current_recent32_logit", pa.float32()),
        ("request_local_current_full1024_logit", pa.float32()), ("current_exact_rolling_logit", pa.float32()),
        ("reuse_parent_rolling_logit", pa.float32()), ("history_length", pa.int32()),
        ("snapshot_tokens", pa.int32()), ("suffix_events_before_query", pa.int32()),
        ("request_weight", pa.float64()),
    ])


def quality_schema() -> pa.Schema:
    return pa.schema(list(fidelity_schema()) + [
        pa.field("label", pa.int8()), pa.field("is_organic", pa.int8()),
        pa.field("prior_30m_same_item", pa.bool_()), pa.field("latest_item", pa.bool_()),
        pa.field("long_gap_at_least_3d", pa.bool_()), pa.field("feedback_history_stratum_v2", pa.string()),
    ])


def score_rows(release, requests, rolling, direct, quality_by_id=None):
    rows = []
    exact_delta = 0.0
    for request in requests:
        if request.request_id not in rolling:
            continue
        values = {
            "request_id": request.request_id, "uid": request.uid,
            "query_timestamp": request.query_timestamp, "candidate_id": int(request.candidate_ids[0]),
            "release": release, "model": "m0_f", "seed": 17,
            "history_length": len(request.history_items), "request_weight": request.request_weight,
            **direct[request.request_id], **rolling[request.request_id],
        }
        exact_delta = max(
            exact_delta,
            abs(float(values["request_local_current_full1024_logit"]) - float(values["current_exact_rolling_logit"])),
        )
        if quality_by_id is not None:
            quality = quality_by_id[request.request_id]
            values.update({
                "label": int(quality.label), "is_organic": quality.is_organic,
                "prior_30m_same_item": quality.prior_30m_same_item,
                "latest_item": quality.latest_item,
                "long_gap_at_least_3d": (
                    quality.query_timestamp - int(quality.history_timestamps[-1]) >= 3 * 86_400
                ),
                "feedback_history_stratum_v2": quality.target_stratum,
            })
        rows.append(values)
    return rows, exact_delta


@torch.no_grad()
def evaluate(release: str, device: torch.device, output: Path, max_users: int | None):
    contract = load_contract()
    split, cutover = RELEASES[release]
    fidelity = load_p7_requests(
        MANIFEST, LISTENS, split, "F", manifest_kind="fidelity", history_limit=1024
    )
    quality = load_p7_requests(
        MANIFEST, LISTENS, split, "F", manifest_kind="quality", history_limit=1024
    )
    if max_users is not None:
        fidelity_users = set(
            sorted({row.uid for row in fidelity}, key=lambda uid: hashlib.sha256(f"scale-hs:{uid}".encode()).digest())[:max_users]
        )
        quality_users = set(
            sorted({row.uid for row in quality}, key=lambda uid: hashlib.sha256(f"scale-hs:{uid}".encode()).digest())[:max_users]
        )
        fidelity = [row for row in fidelity if row.uid in fidelity_users]
        quality = [row for row in quality if row.uid in quality_users]
    view_audits = {"fidelity": view_audit(fidelity, "fidelity"), "quality": view_audit(quality, "quality")}
    quality_by_id = {row.request_id: row for row in quality}
    # Strip every future/quality field before the scoring path sees quality-population rows.
    quality_score_requests = [
        replace(
            row,
            manifest_kind="fidelity_quality_population",
            target_index=None,
            label=None,
            is_organic=None,
            prior_30m_same_item=None,
            latest_item=None,
            target_stratum=None,
        )
        for row in quality
    ]

    checkpoint = checkpoint_path(release)
    current, child = load_model(checkpoint, device)
    if child.get("admitted") is not True:
        raise RuntimeError("scale H/S refuses a non-admitted release")
    parent_path = ROOT / child["parent_checkpoint"]
    if sha256_file(parent_path) != child["parent_checkpoint_hash"]:
        raise RuntimeError("release parent checkpoint hash differs")
    parent, _ = load_model(parent_path, device)
    bases, _ = p7.load_bases(("F",), device)
    started = time.perf_counter()
    rows_by_view = {}
    populations = {}
    lineages = {}
    base_deltas = {}
    exact_deltas = {}
    for view, requests in (("fidelity", fidelity), ("quality", quality_score_requests)):
        pointers = request_pointers(split, view)
        records, populations[view] = build_records(requests, pointers, cutover, None)
        rolling, lineages[view] = rolling_scores(current, parent, bases["F"], records, pointers, device)
        retained = [row for row in requests if row.request_id in rolling]
        direct, base_deltas[view] = direct_scores(current, parent, bases["F"], retained, device)
        rows_by_view[view], exact_deltas[view] = score_rows(
            release,
            retained,
            rolling,
            direct,
            quality_by_id if view == "quality" else None,
        )
    output.mkdir(parents=True, exist_ok=False)
    artifacts = []
    for view, schema in (("fidelity", fidelity_schema()), ("quality", quality_schema())):
        path = output / f"F_{view}.parquet"
        pq.write_table(pa.Table.from_pylist(rows_by_view[view], schema=schema), path, compression="zstd")
        artifacts.append({
            "view": view, "path": str(path.relative_to(ROOT)), "sha256": sha256_file(path),
            "requests": len(rows_by_view[view]),
            "users": len({row["uid"] for row in rows_by_view[view]}), "schema": schema.names,
        })
    status = "raw_scores_written_before_metrics"
    payload = {
        "status": status,
        "scope": "canary" if max_users is not None else "full_development_edge",
        "release": release, "model": "m0_f", "seed": 17,
        "comparison_paths": ["PreviousFull1024", "CurrentRecent32", "RequestLocalCurrentFull1024", "CurrentExactRolling", "ReuseParentRolling", "BaseOnly"],
        "qualification_or_theta3_read": False, "metrics_computed": False,
        "contract_sha256": sha256_file(CONTRACT),
        "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": sha256_file(checkpoint),
        "parent_checkpoint": str(parent_path.relative_to(ROOT)), "parent_checkpoint_sha256": sha256_file(parent_path),
        "view_audits": view_audits, "population": populations,
        "lineage": lineages, "base_full_recent_max_abs_delta": max(base_deltas.values()),
        "exact_rolling_vs_request_local_full_max_abs_logit_companion": max(exact_deltas.values()),
        "wall_seconds": time.perf_counter() - started, "artifacts": artifacts,
    }
    (output / "raw_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in (
        "status", "scope", "release", "population", "lineage",
        "exact_rolling_vs_request_local_full_max_abs_logit_companion", "wall_seconds",
    )}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", choices=tuple(RELEASES), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--max-users", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = (args.output or OUTPUT_ROOT / args.release / "m0_f_seed17").resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    evaluate(args.release, torch.device(args.device), output, args.max_users)


if __name__ == "__main__":
    main()
