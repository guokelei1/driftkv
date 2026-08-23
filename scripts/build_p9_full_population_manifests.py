#!/usr/bin/env python3
"""Materialize P9.7 cutover populations and label-free profiler probes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

import train_p7_theta0 as p7
from hstu_kvcache.data.p7_training import load_p7_requests


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/p9_7_full_population_contract_v1.yaml"
LISTENS = ROOT / "data/raw/yambda/flat/50m/listens.parquet"
P8_MANIFEST = ROOT / "data/manifests/p8_release_v1"
OUTPUT_ROOT = ROOT / "data/manifests/p9_full_population_v1"


def validate_contract() -> dict:
    contract = yaml.safe_load(CONTRACT.read_text())
    paths = {
        "p9_5_result_sha256": ROOT / "configs/contracts/p9_5_rolling_validation_result_v1.yaml",
        "p9_6_result_sha256": ROOT / "configs/contracts/p9_6_transition_cost_result_v1.yaml",
        "p9_6_cost_artifact_sha256": ROOT / "results/p9/p9_6_transition_costs_v1.json",
        "raw_listens_sha256": LISTENS,
        "p8_materialization_summary_sha256": P8_MANIFEST / "materialization_summary.json",
    }
    for key, path in paths.items():
        if p7.sha256_file(path) != contract["inputs"][key]:
            raise RuntimeError(f"P9.7 input hash mismatch: {key}")
    return contract


def candidate_multiset(splits: list[str]) -> list[int]:
    values = []
    for split in splits:
        requests = load_p7_requests(P8_MANIFEST, LISTENS, split, "F", manifest_kind="quality")
        for request in requests:
            values.extend(int(item) for item in request.candidate_ids)
    if len(set(values)) < 16:
        raise RuntimeError("pre-release F candidate pool is too small")
    return values


def deterministic_panel(edge: str, uid: int, pool: list[int], count: int) -> list[int]:
    selected = []
    attempt = 0
    while len(selected) < count:
        digest = hashlib.sha256(f"p9.7|{edge}|{uid}|{attempt}".encode()).digest()
        candidate = int(pool[int.from_bytes(digest[:8], "little") % len(pool)])
        if candidate not in selected:
            selected.append(candidate)
        attempt += 1
        if attempt > 10000:
            raise RuntimeError("candidate collision resolution did not terminate")
    return selected


def build_edge(edge: str, spec: dict, contract: dict, threads: int) -> dict:
    cutover = int(spec["cutover"])
    connection = duckdb.connect()
    connection.execute(f"PRAGMA threads={threads:d}")
    table = connection.execute(
        """
        WITH raw AS (
          SELECT row_number() OVER () - 1 AS raw_index, uid, timestamp, item_id, is_organic
          FROM read_parquet(?)
        ), pre AS (
          SELECT *, row_number() OVER (PARTITION BY uid ORDER BY raw_index DESC) AS reverse_rank
          FROM raw WHERE timestamp < ?
        ), state AS (
          SELECT uid,
                 max(raw_index) + 1 AS raw_prefix_end_exclusive,
                 count(*) AS raw_prefix_length,
                 least(512, count(*)) AS effective_prefix_length,
                 max(timestamp) AS last_activity_timestamp,
                 count(*) FILTER (WHERE timestamp >= ? - 86400) AS events_last_1d,
                 count(*) FILTER (WHERE timestamp >= ? - 604800) AS events_last_7d,
                 count(*) FILTER (WHERE timestamp >= ? - 2592000) AS events_last_30d,
                 count(DISTINCT item_id) FILTER (WHERE timestamp >= ? - 604800) AS unique_items_last_7d,
                 coalesce(avg(CASE WHEN is_organic=1 THEN 1.0 ELSE 0.0 END)
                   FILTER (WHERE timestamp >= ? - 604800), 0.0) AS organic_ratio_last_7d
          FROM pre GROUP BY uid
        ), recent AS (
          SELECT uid, count(*) AS events, count(DISTINCT item_id) AS unique_items
          FROM pre WHERE timestamp >= ? - 604800 GROUP BY uid
        )
        SELECT s.*,
               ? - s.last_activity_timestamp AS last_activity_age_seconds,
               CASE WHEN coalesce(r.events,0)=0 THEN 0.0
                    ELSE 1.0-r.unique_items::DOUBLE/r.events END AS repeat_ratio_last_7d
        FROM state s LEFT JOIN recent r USING(uid) ORDER BY uid
        """,
        [str(LISTENS), cutover, cutover, cutover, cutover, cutover, cutover, cutover, cutover],
    ).fetch_arrow_table()
    pool = candidate_multiset(list(spec["pre_release_candidate_sources"]))
    rows = table.to_pylist()
    probe_rows = []
    for row in rows:
        uid = int(row["uid"])
        candidates = deterministic_panel(
            edge, uid, pool, int(contract["cutover_probe"]["candidate_count"])
        )
        probe_rows.append({
            "request_id": f"p9.7-cutover-{edge}-{uid}",
            "edge": edge,
            "uid": uid,
            "query_timestamp": cutover,
            "query_type_id": 2,
            "candidate_ids": candidates,
            "candidate_hash": hashlib.sha256(
                json.dumps(candidates, separators=(",", ":")).encode()
            ).hexdigest(),
        })
    edge_root = OUTPUT_ROOT / edge
    edge_root.mkdir(parents=True, exist_ok=False)
    state_path = edge_root / "states.parquet"
    probe_path = edge_root / "cutover_probes.parquet"
    pq.write_table(table, state_path, compression="zstd")
    pq.write_table(pa.Table.from_pylist(probe_rows), probe_path, compression="zstd")
    lengths = np.asarray(table["effective_prefix_length"].to_numpy(), dtype=np.int64)
    meta = {
        "edge": edge,
        "cutover": cutover,
        "states": len(rows),
        "probes": len(probe_rows),
        "candidate_count": int(contract["cutover_probe"]["candidate_count"]),
        "candidate_pool_presentations": len(pool),
        "candidate_pool_unique": len(set(pool)),
        "future_request_required": False,
        "future_label_materialized": False,
        "effective_prefix_length": {
            "p50": int(np.quantile(lengths, 0.50)),
            "p90": int(np.quantile(lengths, 0.90)),
            "p99": int(np.quantile(lengths, 0.99)),
            "saturated_512": int(np.sum(lengths == 512)),
        },
        "state_path": str(state_path.relative_to(ROOT)),
        "state_sha256": p7.sha256_file(state_path),
        "probe_path": str(probe_path.relative_to(ROOT)),
        "probe_sha256": p7.sha256_file(probe_path),
    }
    (edge_root / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=24)
    args = parser.parse_args()
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT_ROOT}")
    contract = validate_contract()
    results = [
        build_edge(edge, spec, contract, args.threads)
        for edge, spec in contract["edges"].items()
    ]
    payload = {
        "status": "P9_7_full_cutover_population_and_label_free_probes_materialized",
        "contract_hash": p7.sha256_file(CONTRACT),
        "edges": results,
    }
    (OUTPUT_ROOT / "materialization_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
