#!/usr/bin/env python3
"""Build contract-driven HSTU-native request manifests with DuckDB."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import pyarrow.parquet as pq
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/yambda500m_small_hstu_native_rolling_recipe_matrix_v2.yaml"
OUTPUT = ROOT / "data/manifests/yambda500m_small_hstu_native_rolling_matrix_fast_v2"
DAY = 86_400


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def block_plan(contract: dict) -> tuple[list[tuple[str, int, int]], int, int]:
    if "manifest" in contract and "blocks" in contract["manifest"]:
        blocks = [
            (str(name), int(days[0]), int(days[1]))
            for name, days in contract["manifest"]["blocks"].items()
        ]
    else:
        windows = contract["windows_days_half_open"]
        blocks = [(
            "matrix_horizon",
            int(windows["foundation_end_day"]),
            int(windows["maximum_timestamp_exclusive_day"]),
        )]
    blocks.sort(key=lambda row: row[1])
    for index, (_, start, end) in enumerate(blocks):
        if start < 0 or end <= start:
            raise ValueError("manifest blocks must be nonempty half-open windows")
        if index and start < blocks[index - 1][2]:
            raise ValueError("manifest blocks must not overlap")
    return blocks, min(row[1] for row in blocks), max(row[2] for row in blocks)


def case_expression(blocks: list[tuple[str, int, int]]) -> str:
    clauses = " ".join(
        f"WHEN g.timestamp >= {start * DAY} AND g.timestamp < {end * DAY} THEN '{name}'"
        for name, start, end in blocks
    )
    return f"CASE {clauses} END"


def validate_inputs(contract: dict) -> tuple[Path, Path]:
    frozen = contract["frozen_inputs"]
    for key in ("dataset_manifest", "item_mapping"):
        path = ROOT / frozen[key]
        if digest(path) != frozen[f"{key}_sha256"]:
            raise RuntimeError(f"matrix input hash mismatch: {key}")
    return ROOT / frozen["dataset_manifest"], ROOT / frozen["item_mapping"]


def build(contract_path: Path, output: Path, threads: int) -> dict:
    contract_path = contract_path.resolve()
    output = output.resolve()
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    dataset_path, mapping = validate_inputs(contract)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset_root = dataset_path.parent
    feedback = (dataset_root / dataset["shared_feedback_glob"]).resolve()
    listens = (dataset_root / dataset["shared_listens_glob"]).resolve()
    rank_limit = int(dataset["rank_limit"])
    blocks, start_day, end_day = block_plan(contract)
    start, end = start_day * DAY, end_day * DAY
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    quality = output / "requests_quality.parquet"
    fidelity = output / "requests_fidelity.parquet"
    request_namespace = str(contract["contract"]).replace("'", "''")
    block_case = case_expression(blocks)

    connection = duckdb.connect()
    connection.execute(f"PRAGMA threads={int(threads)}")
    grouped_cte = f"""
      WITH first_listen AS (
        SELECT uid,min(timestamp) AS first_timestamp
        FROM read_parquet('{sql_path(listens)}', hive_partitioning=true)
        WHERE selector_rank <= {rank_limit} AND timestamp < {end}
        GROUP BY uid
      ), grouped AS (
        SELECT uid,timestamp,raw_item_id,min(is_organic)::UTINYINT AS is_organic,
               min(label)::UTINYINT AS label,count(*) AS duplicate_rows,
               count(DISTINCT label) AS label_count
        FROM read_parquet('{sql_path(feedback)}', hive_partitioning=true)
        WHERE selector_rank <= {rank_limit} AND timestamp >= {start} AND timestamp < {end}
        GROUP BY uid,timestamp,raw_item_id
      )
    """
    query = grouped_cte + f"""
      SELECT concat('{request_namespace}:',g.uid,':',g.timestamp,':',g.raw_item_id) AS request_id,
             g.uid::UBIGINT AS uid,g.timestamp::UBIGINT AS query_timestamp,
             {block_case} AS time_block,g.raw_item_id::UBIGINT AS raw_item_id,
             coalesce(m.item_idx,0)::UBIGINT AS item_idx,
             (coalesce(m.item_idx,0) <> 0) AS target_known,g.is_organic,g.label
      FROM grouped g
      JOIN first_listen f USING(uid)
      LEFT JOIN read_parquet('{sql_path(mapping)}') m USING(raw_item_id)
      WHERE g.label_count=1 AND f.first_timestamp < g.timestamp
      ORDER BY query_timestamp,uid,request_id
    """
    try:
        connection.execute(
            "COPY (" + query + ") TO '" + sql_path(quality)
            + "' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        connection.execute(
            "COPY (SELECT request_id,uid,query_timestamp,time_block,raw_item_id,item_idx,"
            "target_known,is_organic FROM read_parquet(?)) TO '"
            + sql_path(fidelity) + "' (FORMAT PARQUET, COMPRESSION ZSTD)",
            [str(quality)],
        )
        audit = connection.execute(
            grouped_cte
            + """
              SELECT count(*) AS grouped_feedback_requests,
                     coalesce(sum(duplicate_rows-1),0) AS exact_duplicate_rows_removed,
                     coalesce(sum(CASE WHEN label_count<>1 THEN duplicate_rows ELSE 0 END),0)
                       AS conflicting_rows_excluded
              FROM grouped
            """
        ).fetchone()
        block_rows = connection.execute(
            """
              SELECT time_block,count(*) AS requests,
                     count(*) FILTER (WHERE target_known) AS known_requests,
                     count(DISTINCT uid) FILTER (WHERE target_known) AS known_users
              FROM read_parquet(?) GROUP BY time_block ORDER BY time_block
            """,
            [str(quality)],
        ).fetchall()
    finally:
        connection.close()
    payload = {
        "status": "hstu_native_contract_matrix_manifest",
        "contract": str(contract_path.relative_to(ROOT)),
        "contract_sha256": digest(contract_path),
        "dataset_manifest_sha256": digest(dataset_path),
        "item_mapping_sha256": digest(mapping),
        "threads": int(threads),
        "rank_limit": rank_limit,
        "window_days_half_open": [start_day, end_day],
        "blocks_days_half_open": {
            name: [block_start, block_end] for name, block_start, block_end in blocks
        },
        "base_features_materialized": False,
        "label_metrics_computed": False,
        "audit": {
            "grouped_feedback_requests": int(audit[0]),
            "exact_duplicate_rows_removed": int(audit[1]),
            "conflicting_rows_excluded": int(audit[2]),
        },
        "block_counts": {
            str(name): {
                "requests": int(requests),
                "known_requests": int(known),
                "known_users": int(users),
            }
            for name, requests, known, users in block_rows
        },
        "artifacts": {
            path.name: {"rows": pq.read_metadata(path).num_rows, "sha256": digest(path)}
            for path in (quality, fidelity)
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--threads", type=int, default=24)
    args = parser.parse_args()
    print(json.dumps(build(args.contract, args.output, args.threads), indent=2))


if __name__ == "__main__":
    main()
