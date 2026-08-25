#!/usr/bin/env python3
"""Build causal Small foundation request and cutover snapshot manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from hstu_kvcache.data.foundation_manifests import (
    BASE_FEATURE_NAMES,
    CausalFeatureState,
    DAY_SECONDS,
    foundation_request_id,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/yambda500m_small_foundation_chain_v1.yaml"
DEFAULT_OUTPUT = ROOT / "data/manifests/yambda500m_small_foundation_v1"
ARTIST_MAP = ROOT / "data/raw/yambda/artist_item_mapping.parquet"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_contract(contract_path: Path) -> tuple[dict, Path, Path]:
    contract = yaml.safe_load(contract_path.read_text())
    inputs = contract.get("inputs", contract.get("frozen_inputs"))
    if contract.get("status") == "prospective_release_recipe_matrix_v1":
        required = ("dataset_manifest", "item_mapping")
    elif "inputs" in contract:
        required = ("discussion_spec", "unified_scale_contract", "dataset_manifest", "item_mapping")
    else:
        required = ("foundation_contract", "original_launch_contract", "dataset_manifest", "item_mapping", "frozen_base", "v0_checkpoint", "v1_checkpoint")
    for key in required:
        path = ROOT / inputs[key]
        if sha256_file(path) != inputs[f"{key}_sha256"]:
            raise RuntimeError(f"foundation input hash mismatch: {key}")
    return contract, ROOT / inputs["dataset_manifest"], ROOT / inputs["item_mapping"]


def contract_calendar(contract: dict) -> tuple[tuple[tuple[str, int, int], ...], tuple[int, ...], int]:
    if contract.get("status") == "prospective_release_recipe_matrix_v1":
        values = contract["windows_days_half_open"]
        maximum = int(values["maximum_timestamp_exclusive_day"])
        return (
            ("foundation", 0, int(values["foundation_end_day"]) * DAY_SECONDS),
            ("matrix_horizon", int(values["foundation_end_day"]) * DAY_SECONDS, maximum * DAY_SECONDS),
        ), tuple(values["snapshot_days"]), maximum * DAY_SECONDS
    if "windows_days_half_open" in contract and "training_windows" in contract["windows_days_half_open"]:
        values = contract["windows_days_half_open"]
        names = ("foundation", "update1", "update2", "update3", "update4", "evaluation4")
        windows = tuple((name, int(values[name][0]) * DAY_SECONDS, int(values[name][1]) * DAY_SECONDS) for name in names)
        snapshots = tuple(sorted({int(values[name][0]) for name in names if name != "foundation"} | {int(values["foundation"][1])}))
        return windows, snapshots, int(values["maximum_timestamp_exclusive_day"]) * DAY_SECONDS
    values = contract["windows_days_half_open"]
    windows = (
        ("foundation", int(values["base_fit_and_v0_train"][0]) * DAY_SECONDS, int(values["base_fit_and_v0_train"][1]) * DAY_SECONDS),
        ("update1", int(values["v0_observation_and_v1_R0_train"][0]) * DAY_SECONDS, int(values["v0_observation_and_v1_R0_train"][1]) * DAY_SECONDS),
        ("update2", int(values["edge1_evaluation_and_v2_train"][0]) * DAY_SECONDS, int(values["edge1_evaluation_and_v2_train"][1]) * DAY_SECONDS),
        ("evaluation2", int(values["edge2_evaluation"][0]) * DAY_SECONDS, int(values["edge2_evaluation"][1]) * DAY_SECONDS),
    )
    return windows, (217, 224, 231), 238 * DAY_SECONDS


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def mappings(item_mapping: Path) -> tuple[dict[int, int], dict[int, int]]:
    table = pq.read_table(item_mapping, columns=["raw_item_id", "item_idx"])
    item = dict(zip(table["raw_item_id"].to_pylist(), table["item_idx"].to_pylist(), strict=True))
    artist_table = pq.read_table(ARTIST_MAP, columns=["item_id", "artist_id"])
    artist = dict(zip(artist_table["item_id"].to_pylist(), artist_table["artist_id"].to_pylist(), strict=True))
    return item, artist


def schemas() -> tuple[pa.Schema, pa.Schema, pa.Schema]:
    common = [
        ("request_id", pa.string()), ("uid", pa.uint64()), ("query_timestamp", pa.uint64()),
        ("time_block", pa.string()), ("raw_item_id", pa.uint64()), ("item_idx", pa.uint64()),
        ("target_known", pa.bool_()), ("is_organic", pa.uint8()),
        ("history_length", pa.uint16()), ("history_oov_tokens", pa.uint16()),
        ("history_oov_fraction", pa.float32()), ("base_features", pa.list_(pa.float32(), 7)),
    ]
    fidelity = pa.schema(common)
    quality = pa.schema([*common, ("label", pa.uint8())])
    snapshot = pa.schema([
        ("uid", pa.uint64()), ("cutover", pa.uint64()), ("history_length", pa.uint16()),
        ("last_timestamp", pa.uint64()), ("history_oov_tokens", pa.uint16()),
        ("history_oov_fraction", pa.float32()),
    ])
    return fidelity, quality, snapshot


class Writers:
    def __init__(self, output: Path) -> None:
        fidelity, quality, snapshot = schemas()
        self.fidelity_path = output / "requests_fidelity.parquet"
        self.quality_path = output / "requests_quality.parquet"
        self.snapshot_path = output / "snapshots.parquet"
        self.fidelity = pq.ParquetWriter(self.fidelity_path, fidelity, compression="zstd")
        self.quality = pq.ParquetWriter(self.quality_path, quality, compression="zstd")
        self.snapshot = pq.ParquetWriter(self.snapshot_path, snapshot, compression="zstd")
        self.fidelity_schema, self.quality_schema, self.snapshot_schema = fidelity, quality, snapshot

    def write_requests(self, rows: list[dict]) -> None:
        if not rows:
            return
        self.quality.write_table(pa.Table.from_pylist(rows, schema=self.quality_schema))
        self.fidelity.write_table(pa.Table.from_pylist(
            [{key: value for key, value in row.items() if key != "label"} for row in rows],
            schema=self.fidelity_schema,
        ))

    def write_snapshots(self, rows: list[dict]) -> None:
        if rows:
            self.snapshot.write_table(pa.Table.from_pylist(rows, schema=self.snapshot_schema))

    def close(self) -> None:
        self.fidelity.close(); self.quality.close(); self.snapshot.close()


def build(*, output: Path, max_users: int, threads: int, contract_path: Path = CONTRACT) -> dict:
    contract, dataset_path, item_mapping_path = validate_contract(contract_path)
    windows, snapshot_days, maximum_timestamp = contract_calendar(contract)
    dataset = json.loads(dataset_path.read_text())
    dataset_root = dataset_path.parent
    listens = (dataset_root / dataset["shared_listens_glob"]).resolve()
    feedback = (dataset_root / dataset["shared_feedback_glob"]).resolve()
    item_map, artist_map = mappings(item_mapping_path)
    query = f"""
        SELECT * FROM (
          SELECT 1::UTINYINT AS kind, uid, timestamp, raw_item_id,
                 behavior::UTINYINT AS behavior, is_organic::UTINYINT AS is_organic,
                 NULL::UTINYINT AS label
          FROM read_parquet('{sql_path(listens)}', hive_partitioning=true)
          WHERE selector_rank <= {int(max_users)} AND timestamp < {maximum_timestamp}
          UNION ALL
          SELECT 0::UTINYINT AS kind, uid, timestamp, raw_item_id,
                 NULL::UTINYINT AS behavior, is_organic::UTINYINT AS is_organic,
                 label::UTINYINT AS label
          FROM read_parquet('{sql_path(feedback)}', hive_partitioning=true)
          WHERE selector_rank <= {int(max_users)} AND timestamp < {maximum_timestamp}
        )
        ORDER BY timestamp, kind, uid, raw_item_id, behavior, label
    """
    connection = duckdb.connect()
    connection.execute(f"PRAGMA threads={int(threads)}")
    reader = connection.execute(query).fetch_record_batch(rows_per_batch=131_072)
    state = CausalFeatureState(max_history=512)
    writers = Writers(output)
    request_buffer: list[dict] = []
    audit = {
        "raw_feedback_rows": 0, "request_groups": 0, "eligible_requests": 0,
        "exact_duplicate_rows_removed": 0, "conflicting_rows_excluded": 0,
        "empty_prefix_excluded": 0, "target_oov_requests": 0,
        "history_oov_tokens": 0, "history_tokens": 0,
    }
    snapshot_rows = {day: [] for day in snapshot_days}
    emitted_snapshots: set[int] = set()

    def emit_snapshot(day: int) -> None:
        cutover = day * DAY_SECONDS
        for uid in sorted(state.histories):
            summary = state.history_summary(uid)
            if not summary["history_length"]:
                continue
            snapshot_rows[day].append({
                "uid": uid, "cutover": cutover,
                "history_length": summary["history_length"],
                "last_timestamp": summary["last_timestamp"],
                "history_oov_tokens": summary["history_oov_tokens"],
                "history_oov_fraction": summary["history_oov_fraction"],
            })
        writers.write_snapshots(snapshot_rows[day])
        emitted_snapshots.add(day)

    def process_group(timestamp: int, rows: list[dict]) -> None:
        nonlocal request_buffer
        for day in snapshot_days:
            if day not in emitted_snapshots and timestamp >= day * DAY_SECONDS:
                emit_snapshot(day)
        feedback_groups: dict[tuple[int, int], list[dict]] = {}
        listens_at_time = []
        for row in rows:
            if int(row["kind"]) == 0:
                audit["raw_feedback_rows"] += 1
                feedback_groups.setdefault((int(row["uid"]), int(row["raw_item_id"])), []).append(row)
            else:
                listens_at_time.append(row)
        for (uid, raw_item), values in sorted(feedback_groups.items()):
            audit["request_groups"] += 1
            labels = {int(value["label"]) for value in values}
            if len(labels) != 1:
                audit["conflicting_rows_excluded"] += len(values)
                continue
            audit["exact_duplicate_rows_removed"] += len(values) - 1
            block = next((name for name, start, end in windows if start <= timestamp < end), None)
            if block is None:
                continue
            summary = state.history_summary(uid)
            if not summary["history_length"]:
                audit["empty_prefix_excluded"] += 1
                continue
            item_idx = int(item_map.get(raw_item, 0))
            target_known = item_idx != 0
            if not target_known:
                audit["target_oov_requests"] += 1
            artist = int(artist_map.get(raw_item, -1))
            features = state.request_features(
                uid=uid, timestamp=timestamp, raw_item_id=raw_item, artist_id=artist
            )
            row = {
                "request_id": foundation_request_id(uid, timestamp, raw_item),
                "uid": uid, "query_timestamp": timestamp, "time_block": block,
                "raw_item_id": raw_item, "item_idx": item_idx,
                "target_known": target_known,
                "is_organic": min(int(value["is_organic"]) for value in values),
                "history_length": summary["history_length"],
                "history_oov_tokens": summary["history_oov_tokens"],
                "history_oov_fraction": summary["history_oov_fraction"],
                "base_features": [float(value) for value in features],
                "label": labels.pop(),
            }
            request_buffer.append(row)
            audit["eligible_requests"] += int(target_known)
            audit["history_oov_tokens"] += int(summary["history_oov_tokens"])
            audit["history_tokens"] += int(summary["history_length"])
            if len(request_buffer) >= 65_536:
                writers.write_requests(request_buffer); request_buffer = []
        for row in sorted(listens_at_time, key=lambda value: (int(value["uid"]), int(value["raw_item_id"]), int(value["behavior"]))):
            raw_item = int(row["raw_item_id"])
            state.append_listen(
                uid=int(row["uid"]), timestamp=timestamp, raw_item_id=raw_item,
                item_idx=int(item_map.get(raw_item, 0)), artist_id=int(artist_map.get(raw_item, -1)),
            )

    current_timestamp = None
    group: list[dict] = []
    try:
        for batch in reader:
            for row in batch.to_pylist():
                timestamp = int(row["timestamp"])
                if current_timestamp is not None and timestamp != current_timestamp:
                    process_group(current_timestamp, group); group = []
                current_timestamp = timestamp
                group.append(row)
        if current_timestamp is not None:
            process_group(current_timestamp, group)
        for day in snapshot_days:
            if day not in emitted_snapshots:
                emit_snapshot(day)
        writers.write_requests(request_buffer)
    finally:
        writers.close(); connection.close()
    artifacts = {}
    for path in (writers.fidelity_path, writers.quality_path, writers.snapshot_path):
        artifacts[path.name] = {"rows": pq.read_metadata(path).num_rows, "sha256": sha256_file(path)}
    audit["history_oov_fraction"] = (
        audit["history_oov_tokens"] / audit["history_tokens"] if audit["history_tokens"] else None
    )
    payload = {
        "status": "foundation_manifests_materialized",
        "contract": str(contract_path.relative_to(ROOT)), "contract_sha256": sha256_file(contract_path),
        "dataset_manifest_sha256": sha256_file(dataset_path), "max_users": max_users,
        "base_feature_names": list(BASE_FEATURE_NAMES), "timestamp_group_atomic": True,
        "audit": audit, "snapshots": {str(day): len(rows) for day, rows in snapshot_rows.items()},
        "artifacts": artifacts,
    }
    (output / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--max-users", type=int, default=10_000)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if not 1 <= args.max_users <= 10_000 or not 1 <= args.threads <= 32:
        raise ValueError("max-users or threads outside contract bounds")
    output.mkdir(parents=True)
    contract_path = args.contract.resolve()
    print(json.dumps(build(output=output, max_users=args.max_users, threads=args.threads, contract_path=contract_path), indent=2))


if __name__ == "__main__":
    main()
