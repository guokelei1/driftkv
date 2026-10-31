#!/usr/bin/env python3
"""Freeze the label-free Design 2 V2--V5 panel and one continuous history pack."""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from hstu_kvcache.evaluation.history import load_histories_semijoin, restore_pack
from hstu_kvcache.data.yambda_history import load_yambda_histories
from hstu_kvcache.training.foundation import FoundationHistoryIndex
from hstu_kvcache.common.artifacts import DAY, sha256, write_json

CONFIG = ROOT / "configs/design/medium_v0_v5_development.json"
PANEL = ROOT / "results/design_two_2026_10/benchmark_panel"
START, END = 259 * DAY, 301 * DAY
EDGES = ("v2_to_v3", "v3_to_v4", "v4_to_v5")
NAMESPACE = "design_two_medium_v2_v5_2026_10_02"


def checked(source):
    path = ROOT / source["path"]
    if sha256(path) != source["sha256"]:
        raise RuntimeError(f"frozen input changed: {path}")
    return path


def record(path):
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def uid_hash(uid, namespace=NAMESPACE):
    return hashlib.sha256(f"{namespace}/{int(uid)}".encode()).hexdigest()


def profile_users(dataset_path, uids, threads=8):
    """Counts are computed before D259; the recent count is not capped at 1024."""
    dataset = json.loads(dataset_path.read_text())
    listens = str((dataset_path.parent / dataset["shared_listens_glob"]).resolve()).replace("'", "''")
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={int(threads)}")
    con.register("selected_uids", pa.table({"uid": pa.array(uids, type=pa.uint64())}))
    try:
        table = con.execute(f"""
          SELECT l.uid, count(*) AS pre_events,
                 count(*) FILTER (WHERE timestamp >= ?) AS recent_events
          FROM read_parquet('{listens}', hive_partitioning=true) l
          SEMI JOIN selected_uids u USING(uid)
          WHERE timestamp < ? GROUP BY l.uid ORDER BY l.uid
        """, [(259 - 14) * DAY, START]).fetch_arrow_table()
    finally:
        con.close()
    found = {int(row["uid"]): row for row in table.to_pylist()}
    result = []
    for uid in sorted(uids):
        row = found.get(uid, {"uid": uid, "pre_events": 0, "recent_events": 0})
        row["initial_length"] = min(1024, int(row["pre_events"]))
        # Fixed meaningful cache-length/activity bins, chosen before any scores.
        length_bin = int(np.searchsorted([1, 32, 256, 1024], row["pre_events"], side="right"))
        activity_bin = int(np.searchsorted([1, 32, 128], row["recent_events"], side="right"))
        row["stratum"] = f"h{length_bin}_a{activity_bin}"
        result.append(row)
    return result


def stratified_sample(profiles, count, namespace=NAMESPACE):
    """Proportional allocation with deterministic largest-remainder rounding."""
    if not 0 < count <= len(profiles):
        raise ValueError("sample must be a nonempty subset of the candidate pool")
    strata = defaultdict(list)
    for row in profiles:
        strata[row["stratum"]].append(int(row["uid"]))
    quotas = {key: count * len(users) // len(profiles) for key, users in strata.items()}
    order = sorted(strata, key=lambda key: (-(count * len(strata[key]) % len(profiles)), key))
    for key in order[:count - sum(quotas.values())]:
        quotas[key] += 1
    chosen = []
    for key in sorted(strata):
        users = sorted(strata[key], key=lambda uid: (uid_hash(uid, namespace), uid))
        chosen.extend(users[:quotas[key]])
    return sorted(chosen, key=lambda uid: (uid_hash(uid, namespace), uid)), quotas


def select_uids(groups, group="evaluation", max_users=0, shard_index=0, num_shards=1):
    if group not in ("evaluation", "pilot", "fit", "validation"):
        raise ValueError("unknown Design 2 group")
    chosen = groups[group]
    if max_users:
        if not 0 < max_users <= len(chosen):
            raise ValueError("max_users exceeds the fixed group")
        chosen = chosen[:max_users]
    if not 0 <= shard_index < num_shards:
        raise ValueError("invalid UID shard")
    return sorted(chosen)[shard_index::num_shards]


def load_panel(panel_dir=PANEL):
    panel_dir = Path(panel_dir)
    binding = json.loads((panel_dir / "binding.json").read_text())
    groups = json.loads(checked(binding["users_file"]).read_text())
    table = pq.read_table(checked(binding["files"]["requests"]))
    by_user = {uid: [] for group in ("evaluation", "fit", "validation") for uid in groups[group]}
    for row in table.to_pylist():
        by_user[int(row["uid"])].append(row)
    return binding, groups, by_user


def load_history(panel_dir=PANEL, requested_uids=None):
    binding = json.loads((Path(panel_dir) / "binding.json").read_text())
    history = restore_pack(checked(binding["files"]["history"]))
    if requested_uids is not None:
        history = FoundationHistoryIndex({int(uid): history.rows[int(uid)] for uid in requested_uids})
    return history


def prepare_requests(config, all_uids):
    """Use retained request identities/candidates and labels; never load outcomes."""
    tables, sources, windows = [], [], []
    uid_set = pa.array(all_uids, type=pa.uint64())
    labels_source = None
    for window in config["profile"]["continuous_windows"]:
        if window["edge"] not in EDGES:
            continue
        edge = window["edge"]
        directory = ROOT / "results/unified_reuse_2026_09/medium" / edge
        binding_path = directory / "binding.json"
        bound = json.loads(binding_path.read_text())
        if labels_source is not None and labels_source != bound["sources"]["labels"]:
            raise RuntimeError("the three windows must share the same request labels")
        labels_source = bound["sources"]["labels"]
        for source in window["request_sources"]:
            table = pq.read_table(checked(source))
            table = table.filter(pc.is_in(table["uid"], value_set=uid_set))
            table = table.append_column("edge", pa.array([edge] * len(table), type=pa.string()))
            table = table.append_column("version", pa.array([window["target"]] * len(table), type=pa.string()))
            tables.append(table)
            sources.append(source)
        sources.append(record(binding_path))
        windows.append({**{key: window[key] for key in ("edge", "target", "cutover_seconds", "days_half_open")},
            "parent": edge.split("_to_")[0], "current": window["target"],
            "cutover": window["cutover_seconds"], "end_timestamp": window["days_half_open"][1] * DAY})
    labels = pq.read_table(checked(labels_source), columns=["request_id", "uid", "query_timestamp", "label"],
        filters=[("time_block", "=", "matrix_horizon"), ("target_known", "=", True),
                 ("query_timestamp", ">=", START), ("query_timestamp", "<", END)])
    labels = labels.filter(pc.is_in(labels["uid"], value_set=uid_set))
    table = pa.concat_tables(tables)
    ids = table["request_id"].to_pylist()
    if len(ids) != len(set(ids)):
        raise RuntimeError("request identity repeats in the selected version chain")
    table = table.join(labels, keys=["request_id", "uid", "query_timestamp"], join_type="inner")
    if len(table) != len(ids):
        raise RuntimeError("not every retained request has exactly one matched label")
    table = table.sort_by([("uid", "ascending"), ("query_timestamp", "ascending"), ("request_id", "ascending")])
    return table, {"request_sources": sources, "labels": labels_source}, windows


def main(args):
    began = time.perf_counter()
    if args.output.exists():
        raise FileExistsError(f"choose a fresh output directory: {args.output}")
    config = json.loads(args.config.read_text())
    original = json.loads(checked(config["users"]).read_text())
    dataset_path = checked(config["data"]["dataset"])
    dataset = json.loads(dataset_path.read_text())
    manifest_path = checked(config["model"]["manifest"])
    versions = [row for row in config["model"]["versions"] if row["version"] in ("v2", "v3", "v4", "v5")]
    model_configs = [row["config"] for row in json.loads(manifest_path.read_text())["versions"]
                     if row["version"] in ("v2", "v3", "v4", "v5")]
    if any(cfg != model_configs[0] for cfg in model_configs):
        raise RuntimeError("chain model shapes differ")
    known = int(dataset["foundation_items"])
    oov = int(model_configs[0]["num_items"]) - known
    stage_start = time.perf_counter()
    profiles = profile_users(dataset_path, original["continuous_evaluation"], args.threads)
    evaluation, strata = stratified_sample(profiles, args.users)
    evaluation_set = set(evaluation)
    selected_profiles = [row for row in profiles if row["uid"] in evaluation_set]
    pilot, pilot_strata = stratified_sample(selected_profiles, args.pilot_users, NAMESPACE + "/pilot")
    groups = {"evaluation": evaluation, "pilot": pilot,
              "fit": original["continuous_fit"], "validation": original["continuous_validation"]}
    all_uids = sorted(set(evaluation + groups["fit"] + groups["validation"]))
    if len(all_uids) != len(evaluation) + len(groups["fit"]) + len(groups["validation"]):
        raise RuntimeError("fitting, validation and evaluation users overlap")
    print(json.dumps({"status": "selected", "evaluation_users": len(evaluation),
        "pilot_users": len(pilot), "all_users": len(all_uids),
        "profile_seconds": time.perf_counter() - stage_start}), flush=True)
    stage_start = time.perf_counter()
    requests, request_sources, windows = prepare_requests(config, all_uids)
    print(json.dumps({"status": "requests_loaded", "requests": len(requests),
                     "seconds": time.perf_counter() - stage_start}), flush=True)
    kwargs = dict(dataset_path=dataset_path, known_vocab_size=known, oov_buckets=oov,
                  start_timestamp=START, end_timestamp=END, max_history=1024, threads=args.threads)
    stage_start = time.perf_counter()
    history = load_histories_semijoin(all_uids, **kwargs)
    for uid in all_uids:
        history.rows.setdefault(uid, (np.empty(0, dtype=np.int64),) * 3)
    history_seconds = time.perf_counter() - stage_start
    # One tiny independent reference verifies mapping, OOV and timestamp ties.
    canary_uids = pilot[:8]
    reference = load_yambda_histories(dataset_path, canary_uids, known_vocab_size=known,
        oov_buckets=oov, start_timestamp=START, end_timestamp=END, max_pre_events=1024,
        threads=args.threads)
    for uid in canary_uids:
        other = reference.rows.get(uid, (np.empty(0, dtype=np.int64),) * 3)
        if any(not np.array_equal(a, b) for a, b in zip(history.rows[uid], other, strict=True)):
            raise RuntimeError(f"typed join differs from the independent history loader: {uid}")
    initial_empty, excluded_same_time = [], 0
    for uid, (times, _, _) in history.rows.items():
        if np.any(times[1:] < times[:-1]) or (len(times) and int(times[-1]) >= END):
            raise RuntimeError(f"invalid chronological history: {uid}")
        prefix_count = int(np.searchsorted(times, START, side="left"))
        if prefix_count > 1024:
            raise RuntimeError("initial history exceeds the model context")
        if prefix_count == 0:
            initial_empty.append(uid)
    for row in requests.to_pylist():
        times = history.rows[int(row["uid"])][0]
        timestamp = int(row["query_timestamp"])
        if not START <= timestamp < END:
            raise RuntimeError("request outside the continuous chain")
        stop = int(np.searchsorted(times, timestamp, side="left"))
        excluded_same_time += int(np.searchsorted(times, timestamp, side="right")) - stop
    args.output.mkdir(parents=True)
    write_json(args.output / "users.json", groups)
    pq.write_table(pa.Table.from_pylist(profiles), args.output / "pool_profiles.parquet", compression="zstd")
    pq.write_table(requests, args.output / "requests.parquet", compression="zstd")
    lengths = np.asarray([len(history.rows[uid][0]) for uid in all_uids], dtype=np.int64)
    np.savez(args.output / "histories.npz", uids=np.asarray(all_uids, dtype=np.int64),
        offsets=np.r_[0, lengths.cumsum()],
        timestamps=np.concatenate([history.rows[uid][0] for uid in all_uids]),
        item_ids=np.concatenate([history.rows[uid][1] for uid in all_uids]),
        behaviors=np.concatenate([history.rows[uid][2] for uid in all_uids]))
    # Bind the actual processed bytes, not only the JSON manifest filename.
    processed_manifest = dataset_path.parents[2] / "manifest.json"
    processed = json.loads(processed_manifest.read_text())
    mapping_path = (dataset_path.parent / dataset["item_mapping_path"]).resolve()
    raw_records = [row for row in processed["files"] if row["path"].startswith("shared/listens/")
                   or (processed_manifest.parent / row["path"]).resolve() == mapping_path]
    for row in raw_records:
        if sha256(processed_manifest.parent / row["path"]) != row["sha256"]:
            raise RuntimeError(f"processed history source differs: {row['path']}")
    window_counts = []
    request_rows = requests.to_pylist()
    for window in windows:
        counts = {}
        for group, uids in groups.items():
            group_set = set(uids)
            rows = [row for row in request_rows if row["edge"] == window["edge"] and row["uid"] in group_set]
            counts[group] = {"users": len(uids), "active_users": len({row["uid"] for row in rows}),
                             "requests": len(rows)}
        window_counts.append({**window, "groups": counts})
    source_paths = [Path(__file__), ROOT / "src/hstu_kvcache/evaluation/history.py",
        ROOT / "src/hstu_kvcache/data/yambda_history.py", ROOT / "src/hstu_kvcache/data/oov.py",
        ROOT / "src/hstu_kvcache/training/foundation.py"]
    binding = {"status": "complete", "scale": "medium", "model_chain": ["v2", "v3", "v4", "v5"],
        "edges": list(EDGES), "initial_version": "v2", "cutover": START, "end_timestamp": END,
        "days": [259, 301], "context": 1024, "known_vocab_size": known, "oov_buckets": oov,
        "model_config": model_configs[0], "versions": versions,
        "models": {row["version"]: row for row in versions}, "windows": window_counts,
        "configuration": record(args.config), "users_file": record(args.output / "users.json"),
        "sources": {"dataset": config["data"]["dataset"], "original_users": config["users"],
                    "model_manifest": config["model"]["manifest"], **request_sources,
                    "processed_manifest": record(processed_manifest), "history_and_mapping": raw_records},
        "selection": {"pool": "continuous_evaluation", "pool_users": len(profiles),
            "rule": "Proportional strata; largest-remainder quotas; stable SHA256 UID order within strata.",
            "history_boundaries": [1, 32, 256, 1024], "activity_boundaries": [1, 32, 128],
            "activity_days_half_open": [245, 259], "history_end_exclusive": START,
            "namespace": NAMESPACE, "pilot_namespace": NAMESPACE + "/pilot",
            "evaluation_quotas": strata, "pilot_quotas": pilot_strata,
            "uses_labels_or_model_outcomes": False, "retain_inactive_users": True,
            "pool_scope": "Existing retrospective development pool includes outcome-conditioned Motivation users. "
                          "This new sampling is label-free within that pool, not a population confirmation sample."},
        "state_rule": "Materialize V2 once before D259; inherit actual K/V through releases D259,D273,D287; "
                      "append chronological history and evict normally, including users without requests.",
        "request_rule": "Queries precede all events at the same timestamp; retained labels only score requests.",
        "files": {key: record(args.output / name) for key, name in (
            ("requests", "requests.parquet"), ("history", "histories.npz"), ("profiles", "pool_profiles.parquet"))},
        "checks": {"reference_loader_uids": canary_uids, "exact_reference_tuple_equality": True,
            "initial_empty_users": initial_empty, "contemporaneous_events_excluded": excluded_same_time,
            "all_history_mapping_bytes_verified": True, "fit_validation_evaluation_disjoint": True},
        "cost": {"history_load_seconds": history_seconds, "history_events": int(lengths.sum()),
                 "history_bytes": (args.output / "histories.npz").stat().st_size,
                 "preparation_seconds": time.perf_counter() - began},
        "preparation_sources": {str(path.relative_to(ROOT)): sha256(path) for path in source_paths}}
    write_json(args.output / "binding.json", binding)
    print(json.dumps({"status": "complete", "output": str(args.output), **binding["cost"],
                     "windows": window_counts}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=PANEL)
    parser.add_argument("--users", type=int, default=4096)
    parser.add_argument("--pilot-users", type=int, default=256)
    parser.add_argument("--threads", type=int, default=8)
    main(parser.parse_args())
