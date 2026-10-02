#!/usr/bin/env python3
"""One-time CPU-only history packs for the fixed Design 1 development panel.

This helper lives outside active execution-source globs. A typed UID semi-join
preserves the legacy SQL's bounded selection, mapping, and ordering. A fixed
128-user equivalence check precedes the full 10000-user materialization.
No checkpoint, GPU, label selection, or new model prediction is involved.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path("/home/gkl/work/evokv")
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import numpy as np
import duckdb
import pyarrow as pa

from design_one.benchmark_data import PANEL, checked, check_history, load_panel, select_uids
from evaluate_yambda500m_foundation_raw import load_histories
from hstu_kvcache.training.foundation import FoundationHistoryIndex
from hstu_kvcache.data.oov import apply_stable_oov_buckets
from read_correction_v4.evaluate_full import signature
from unified_reuse_2026_09.common import sha256, write_json


def load_histories_semijoin(uids, *, dataset_path, known_vocab_size, oov_buckets,
                           start_timestamp, end_timestamp, max_history, threads=4):
    """Legacy bounded query with a typed UID hash join instead of large IN."""
    dataset = json.loads(dataset_path.read_text())
    listens = (dataset_path.parent / dataset["shared_listens_glob"]).resolve()
    mapping = (dataset_path.parent / dataset["item_mapping_path"]).resolve()
    quoted = lambda path: str(path).replace("'", "''")
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={int(threads)}")
    con.register("selected_uids", pa.table({"uid": pa.array(uids, type=pa.uint64())}))
    # Keep both timestamp inequalities and the pre-map raw-item tie selection
    # exactly as in yambda_history.py. No partition-pruning assumption is used.
    query = f"""
        WITH pre AS (
          SELECT l.uid,l.timestamp,l.raw_item_id,l.behavior,l.is_organic
          FROM read_parquet('{quoted(listens)}', hive_partitioning=true) l
          SEMI JOIN selected_uids u USING(uid)
          WHERE l.timestamp < ?
          QUALIFY row_number() OVER (
            PARTITION BY l.uid ORDER BY l.timestamp DESC,l.raw_item_id DESC,l.behavior DESC
          ) <= ?
        ), post AS (
          SELECT l.uid,l.timestamp,l.raw_item_id,l.behavior,l.is_organic
          FROM read_parquet('{quoted(listens)}', hive_partitioning=true) l
          SEMI JOIN selected_uids u USING(uid)
          WHERE l.timestamp >= ? AND l.timestamp < ?
        ), bounded AS (
          SELECT * FROM pre UNION ALL SELECT * FROM post
        )
        SELECT l.uid,l.timestamp,l.raw_item_id,coalesce(m.item_idx,0) AS item_idx,l.behavior
        FROM bounded l LEFT JOIN read_parquet('{quoted(mapping)}') m USING(raw_item_id)
        ORDER BY l.uid,l.timestamp,l.raw_item_id,l.behavior
    """
    try:
        table = con.execute(query, [int(start_timestamp), int(max_history),
            int(start_timestamp), int(end_timestamp)]).fetch_arrow_table()
    finally:
        con.close()
    item_ids = apply_stable_oov_buckets(table["raw_item_id"].to_numpy(), table["item_idx"].to_numpy(),
        known_vocab_size=int(known_vocab_size), buckets=int(oov_buckets),
        bucket_start=dataset.get("oov_bucket_start"))
    return FoundationHistoryIndex.from_columns(table["uid"].to_numpy(),
        table["timestamp"].to_numpy(), np.asarray(item_ids), table["behavior"].to_numpy(),
        presorted=dataset.get("history_tie_order") == "timestamp_raw_item_behavior")


def equivalent(left, right, uids):
    for uid in uids:
        if any(not np.array_equal(a, b) for a, b in zip(left.rows[uid], right.rows[uid], strict=True)):
            raise RuntimeError(f"legacy/semi-join history tuple or timestamp tie order differs: {uid}")


def restore_pack(path):
    """Restore already-mapped tuples without changing any timestamp tie order."""
    with np.load(path, allow_pickle=False) as arrays:
        uids, offsets = arrays["uids"], arrays["offsets"]
        times, items, behaviors = arrays["timestamps"], arrays["item_ids"], arrays["behaviors"]
    return FoundationHistoryIndex({int(uid): (
        times[int(offsets[i]):int(offsets[i + 1])],
        items[int(offsets[i]):int(offsets[i + 1])],
        behaviors[int(offsets[i]):int(offsets[i + 1])]) for i, uid in enumerate(uids)})


def load_pack(directory, shard_index, *, requested_uids=None):
    """Narrow reader for a future worker; the caller also checks the cache key."""
    binding = json.loads((directory / "binding.json").read_text())
    if signature(binding["inputs"]) != binding["cache_key"]:
        raise RuntimeError("history pack input binding differs")
    pack = binding["packs"][shard_index]
    path = directory / pack["filename"]
    if sha256(path) != pack["sha256"]:
        raise RuntimeError("history pack changed")
    history = restore_pack(path)
    if requested_uids is not None:
        if not set(requested_uids).issubset(history.rows):
            raise RuntimeError("requested users are outside this history shard")
        history = FoundationHistoryIndex({uid: history.rows[uid] for uid in requested_uids})
    return history, binding


def main(args):
    started = time.perf_counter()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"choose a fresh output directory: {output}")
    panel, groups, by_user = load_panel(args.panel)
    dataset_path = checked(panel["sources"]["dataset"])
    dataset = json.loads(dataset_path.read_text())
    # Retained calibration records bind the same frozen endpoints and model
    # dimensions as the reference evaluator; no model weights are loaded here.
    calibration = json.loads(checked(panel["policies"][0]["calibration"]).read_text())
    if calibration["checkpoint_hashes"] != {name: panel["sources"][name]["sha256"]
                                            for name in ("parent", "current")}:
        raise RuntimeError("calibration endpoint binding differs")
    known = int(dataset["foundation_items"])
    num_items = int(calibration["model_config"]["num_items"])
    oov_buckets = num_items - known
    config = json.loads(checked(panel["configuration"]).read_text())
    context = int(config["model"]["context"])
    cutover, end = int(panel["cutover"]), int(panel["days"][1]) * 86400
    uids = sorted(groups["design1_evaluation"])
    shards = [select_uids(groups, 0, rank, 4) for rank in range(4)]
    kwargs = dict(dataset_path=dataset_path, known_vocab_size=known, oov_buckets=oov_buckets,
        start_timestamp=cutover, end_timestamp=end, max_history=context, threads=args.threads)
    pilot = groups["design1_pilot"]
    begin = time.perf_counter()
    legacy_pilot = load_histories(pilot, **kwargs)
    legacy_seconds = time.perf_counter() - begin
    begin = time.perf_counter()
    fast_pilot = load_histories_semijoin(pilot, **kwargs)
    fast_seconds = time.perf_counter() - begin
    equivalent(legacy_pilot, fast_pilot, pilot)
    canary = {"users": len(pilot), "events": sum(len(legacy_pilot.rows[u][0]) for u in pilot),
        "legacy_seconds": legacy_seconds, "semijoin_seconds": fast_seconds,
        "exact_tuple_and_timestamp_tie_equality": True,
        "causal_check": check_history(fast_pilot, by_user, pilot, cutover, context)}
    print(json.dumps({"status": "canary_passed", **canary}), flush=True)
    if args.canary_only:
        write_json(args.canary_output, {**canary, "source_sha256": sha256(Path(__file__)),
                                       "panel_sha256": sha256(args.panel / "binding.json")})
        return

    # Bind the processed manifest and verify its actual history/mapping inputs.
    # This hashes source bytes; it does not interpret any event as history.
    processed_root = dataset_path.parents[2]
    processed_manifest = processed_root / "manifest.json"
    manifest = json.loads(processed_manifest.read_text())
    mapping = (dataset_path.parent / dataset["item_mapping_path"]).resolve()
    source_records = [record for record in manifest["files"]
        if record["path"].startswith("shared/listens/") or processed_root / record["path"] == mapping]
    check_start = time.perf_counter()
    for record in source_records:
        path = processed_root / record["path"]
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"processed history/mapping input differs: {path}")
    source_check_seconds = time.perf_counter() - check_start
    source_paths = (ROOT / "src/hstu_kvcache/data/yambda_history.py",
                    ROOT / "src/hstu_kvcache/data/oov.py",
                    ROOT / "src/hstu_kvcache/training/foundation.py",
                    ROOT / "scripts/evaluate_yambda500m_foundation_raw.py")
    inputs = {"dataset": panel["sources"]["dataset"],
        "processed_manifest": {"path": str(processed_manifest), "sha256": sha256(processed_manifest)},
        "history_and_mapping_sha256": {record["path"]: record["sha256"] for record in source_records},
        "configuration": panel["configuration"], "users_file": panel["users_file"],
        "uids": uids, "uid_shards": shards, "cutover": cutover, "end_timestamp": end,
        "history_length": context, "known_vocab_size": known, "oov_buckets": oov_buckets,
        "history_tie_order": "exact FoundationHistoryIndex.rows output from legacy loader; no resort on restore",
        "loader": "typed_uint64_uid_semijoin; identical legacy pre/post/map/OOV/sort; no week pruning",
        "preparation_source_sha256": sha256(Path(__file__)),
        "loader_sources": {str(path.relative_to(ROOT)): sha256(path) for path in source_paths}}
    cache_key = signature(inputs)
    print(json.dumps({"status": "loading", "users": len(uids), "cache_key": cache_key,
                      "threads": args.threads, "source_check_seconds": source_check_seconds}), flush=True)
    begin = time.perf_counter()
    history = load_histories_semijoin(uids, **kwargs)
    load_seconds = time.perf_counter() - begin
    causal = check_history(history, by_user, uids, cutover, context)
    output.mkdir(parents=True)
    packs = []
    for rank, selected in enumerate(shards):
        lengths = np.asarray([len(history.rows[uid][0]) for uid in selected], dtype=np.int64)
        offsets = np.r_[np.int64(0), lengths.cumsum()]
        path = output / f"shard_{rank:02d}.npz"
        np.savez(path, uids=np.asarray(selected, dtype=np.int64), offsets=offsets,
            timestamps=np.concatenate([history.rows[uid][0] for uid in selected]),
            item_ids=np.concatenate([history.rows[uid][1] for uid in selected]),
            behaviors=np.concatenate([history.rows[uid][2] for uid in selected]))
        restored = restore_pack(path)
        for uid in selected:
            if any(not np.array_equal(left, right)
                   for left, right in zip(history.rows[uid], restored.rows[uid], strict=True)):
                raise RuntimeError(f"history tuple or timestamp tie order changed: {uid}")
        packs.append({"filename": path.name, "sha256": sha256(path), "users": len(selected),
            "events": int(offsets[-1]), "bytes": path.stat().st_size,
            "requests": sum(len(by_user[uid]) for uid in selected)})
        print(json.dumps({"status": "pack_written", "shard": rank, **packs[-1]}), flush=True)

    # Reuse the independent legacy pilot to check that large-batch execution
    # also preserves every tuple/tie, in addition to every saved-array roundtrip.
    begin = time.perf_counter()
    equivalent(history, legacy_pilot, pilot)
    parity_seconds = time.perf_counter() - begin
    binding = {"status": "complete", "cache_key": cache_key, "inputs": inputs, "packs": packs,
        "panel_sha256": sha256(args.panel / "binding.json"), "causal_check": causal,
        "canary": canary,
        "checks": {"all_users_roundtrip_tuple_equality": True, "independent_legacy_pilot_uids": pilot,
            "independent_legacy_tuple_and_tie_equality": True,
            "actual_processed_history_and_mapping_sha256_verified": True},
        "source_check_seconds": source_check_seconds, "semijoin_load_seconds": load_seconds,
        "independent_legacy_check_seconds": parity_seconds, "elapsed_seconds": time.perf_counter() - started,
        "preparation_source": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))}}
    write_json(output / "binding.json", binding)
    print(json.dumps({"status": "complete", "output": str(output), "cache_key": cache_key,
        "events": sum(p["events"] for p in packs), "bytes": sum(p["bytes"] for p in packs),
        "semijoin_load_seconds": load_seconds, "elapsed_seconds": binding["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=PANEL)
    parser.add_argument("--output", type=Path,
        default=ROOT / "results/design_one_2026_10/history_packs_fast")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--canary-only", action="store_true")
    parser.add_argument("--canary-output", type=Path, default=Path("/tmp/design1_history_semijoin_canary.json"))
    args = parser.parse_args()
    main(args)
