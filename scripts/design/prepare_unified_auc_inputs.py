#!/usr/bin/env python3
"""Prepare the fixed unified-AUC snapshots once, on CPU, as mmap arrays."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import glob
import hashlib
import json
from pathlib import Path
import sys
import time

import duckdb
import numpy as np
import pyarrow as pa

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

from design.competitor_data import history_arrays
from design.competitor_models import load_model_pair, sha256_file
from hstu_kvcache.data.oov import apply_stable_oov_buckets
from hstu_kvcache.data.yambda_history import load_yambda_histories
from hstu_kvcache.training.foundation import FoundationHistoryIndex

ARRAY_NAMES = ("timestamps", "items", "behaviors", "deltas", "query_deltas")


def sql_path(path):
    return str(Path(path).resolve()).replace("'", "''")


def fast_history(data, uids, start, end, length, threads):
    """Same bounded selection/tie ordering as load_yambda_histories."""
    manifest = Path(data["manifest"])
    dataset = json.loads(manifest.read_text())
    listens = (manifest.parent / dataset["shared_listens_glob"]).resolve()
    mapping = (manifest.parent / dataset["item_mapping_path"]).resolve()
    connection = duckdb.connect()
    connection.execute(f"SET threads={int(threads)}")
    connection.register("selected_uids", pa.table({"uid": pa.array(uids, type=pa.int64())}))
    # A small registered relation avoids planning two huge parameterized INs.
    # Materialization makes the selected historical scan common to pre/post.
    query = f"""
        WITH filtered AS MATERIALIZED (
          SELECT l.uid,l.timestamp,l.raw_item_id,l.behavior
          FROM read_parquet('{sql_path(listens)}', hive_partitioning=true) l
          SEMI JOIN selected_uids s ON l.uid=s.uid
          WHERE l.week BETWEEN 0 AND {(end - 1) // (7 * 86400)}
            AND l.timestamp < {end}
        ), pre AS (
          SELECT * FROM filtered WHERE timestamp < {start}
          QUALIFY row_number() OVER (
            PARTITION BY uid ORDER BY timestamp DESC,raw_item_id DESC,behavior DESC
          ) <= {length}
        ), bounded AS (
          SELECT * FROM pre UNION ALL SELECT * FROM filtered WHERE timestamp >= {start}
        )
        SELECT l.uid,l.timestamp,l.raw_item_id,coalesce(m.item_idx,0) AS item_idx,l.behavior
        FROM bounded l LEFT JOIN read_parquet('{sql_path(mapping)}') m USING(raw_item_id)
        ORDER BY l.uid,l.timestamp,l.raw_item_id,l.behavior
    """
    started = time.perf_counter()
    try:
        table = connection.execute(query).fetch_arrow_table()
    finally:
        connection.close()
    query_seconds = time.perf_counter() - started
    item_ids = apply_stable_oov_buckets(
        table["raw_item_id"].to_numpy(), table["item_idx"].to_numpy(),
        known_vocab_size=data["known_items"], buckets=data["oov_buckets"],
        bucket_start=dataset.get("oov_bucket_start"),
    )
    history = FoundationHistoryIndex.from_columns(
        table["uid"].to_numpy(), table["timestamp"].to_numpy(), item_ids,
        table["behavior"].to_numpy(),
        presorted=dataset.get("history_tie_order") == "timestamp_raw_item_behavior",
    )
    return history, dict(rows=len(table), query_seconds=query_seconds,
                        total_seconds=time.perf_counter() - started)


def reference_check(data, groups, pairs, length, threads):
    uids = [group[0] for group in groups.values()]
    start, end = min(p["cutover"] for p in pairs) - 1, max(p["cutover"] for p in pairs) + 1
    began = time.perf_counter()
    old = load_yambda_histories(
        Path(data["manifest"]), uids, known_vocab_size=data["known_items"],
        oov_buckets=data["oov_buckets"], start_timestamp=start,
        end_timestamp=end, max_pre_events=length, threads=threads,
    )
    fast, timing = fast_history(data, uids, start, end, length, threads)
    for pair in pairs:
        actual = history_arrays(fast, np.asarray(uids), pair["cutover"], length)
        expected = history_arrays(old, np.asarray(uids), pair["cutover"], length)
        for name, a, b in zip(ARRAY_NAMES, actual, expected, strict=True):
            np.testing.assert_array_equal(a, b, err_msg=f"{pair['edge']} {name}")
            assert a.dtype == b.dtype
    return dict(status="passed", uids=uids, edges=[p["edge"] for p in pairs],
                comparison="all five arrays and dtypes exactly equal to unchanged old loader",
                elapsed_seconds=time.perf_counter() - began, fast_loader=timing)


def write_group(directory, history, uids, cutover, length):
    directory.mkdir(parents=True)
    arrays = history_arrays(history, np.asarray(uids, dtype=np.int64), cutover, length)
    files = {}
    for name, value in zip((*ARRAY_NAMES, "uids"), (*arrays, np.asarray(uids, dtype=np.int64)), strict=True):
        path = directory / f"{name}.npy"
        np.save(path, value, allow_pickle=False)
        files[name] = dict(shape=list(value.shape), dtype=str(value.dtype),
                           bytes=path.stat().st_size, sha256=sha256_file(path))
    return dict(users=len(uids), files=files)


def run(args):
    config = json.loads(args.config.read_text())
    assert config["edges"] == [0, 1, 2, 3] and config["history_length"] == 1024
    groups = {"fit": config["calibration_uids"], "pilot": config["canary"]["mature_pilot_uids"],
              "evaluation": config["evaluation_uids"]}
    uids = [uid for group in groups.values() for uid in group]
    assert len(uids) == len(set(uids)) and len(groups["fit"]) == 256 and len(groups["evaluation"]) == 10000
    pairs = [load_model_pair("medium", edge, verify_hashes=False) for edge in config["edges"]]
    assert all(pair["admission"]["reuse_eligible"] for pair in pairs)
    assert sha256_file(ROOT / config["model_chain"]) == config["model_chain_sha256"]
    data = pairs[0]["dataset"]
    assert sha256_file(data["manifest"]) == data["manifest_sha256"]
    assert sha256_file(data["item_mapping"]) == data["item_mapping_sha256"]
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    print(f"Reference against old loader, {args.threads} CPU threads", flush=True)
    reference = reference_check(data, groups, pairs, config["history_length"], args.threads)
    (args.output / "reference.json").write_text(json.dumps(reference, indent=2) + "\n")
    print(f"Reference passed in {reference['elapsed_seconds']:.2f}s; load {len(uids)} users once", flush=True)
    start, end = min(p["cutover"] for p in pairs) - 1, max(p["cutover"] for p in pairs) + 1
    history, timing = fast_history(data, uids, start, end, config["history_length"], args.threads)
    print(f"History ready: {timing}", flush=True)
    outputs = {}
    for pair in pairs:
        outputs[pair["edge"]] = {}
        for group, selected in groups.items():
            outputs[pair["edge"]][group] = write_group(
                args.output / pair["edge"] / group, history, selected, pair["cutover"], config["history_length"])
        print(f"Prepared {pair['edge']}: fit/pilot/evaluation", flush=True)
    dataset = json.loads(Path(data["manifest"]).read_text())
    files = sorted(Path(p) for p in glob.glob(str(Path(data["manifest"]).parent / dataset["shared_listens_glob"]), recursive=True)
                   if 0 <= int(Path(p).parent.name.split("=")[-1]) <= (end - 1) // (7 * 86400))
    with ThreadPoolExecutor(max_workers=min(args.threads, 8)) as pool:
        hashes = list(pool.map(sha256_file, files))
    sources = [dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=digest)
               for path, digest in zip(files, hashes, strict=True)]
    metadata = dict(status="completed", configuration=str(args.config.resolve()),
        config_sha256=sha256_file(args.config), source_sha256=sha256_file(__file__),
        data=data, model_chain_sha256=config["model_chain_sha256"], threads=args.threads,
        array_order=list(ARRAY_NAMES), uid_order="exact configured order within fit/evaluation/mature_pilot_uids",
        history_length=config["history_length"], cutover_days=config["cutover_days"],
        strict_history_window=[start, end], source_listen_files=sources,
        source_listen_hash_manifest_sha256=hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
        reference=reference, loading=timing, outputs=outputs, elapsed_seconds=time.perf_counter() - started,
        labels_read=False, model_weights_loaded=False)
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Completed in {metadata['elapsed_seconds']:.2f}s: {args.output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/insight/unified_auc_10k_20260920.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/insight/unified_auc_10k_20260920/prepared")
    parser.add_argument("--threads", type=int, default=48)
    run(parser.parse_args())
