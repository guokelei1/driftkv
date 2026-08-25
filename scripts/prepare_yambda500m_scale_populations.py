#!/usr/bin/env python3
"""Audit Yambda-500M and materialize frozen Medium/Large scale populations.

This is a CPU-only, label-free T1/T2 data preparation entry point. It writes
compact UID and foundation-catalog maps, never expanded event copies, and does
not train or score a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hstu_kvcache.data.scale_population import (  # noqa: E402
    UID_SELECTOR_NAMESPACE,
    select_medium_uids,
    uid_selector_digest,
)


CONTRACT = ROOT / "configs/contracts/yambda500m_scale_population_v1.yaml"
DEFAULT_RAW = ROOT / "data/raw/yambda/flat/500m"
DEFAULT_MANIFEST_ROOT = ROOT / "data/manifests/yambda500m_scale_v1"
DEFAULT_AUDIT = ROOT / "results/data_audit/yambda500m_scale_v1/population_audit.json"
WINDOWS = {
    "foundation_train": (0, 17_539_200),
    "development": (17_539_200, 18_144_000),
    "qualification": (18_144_000, 18_748_800),
    "update1_train": (18_748_800, 19_785_600),
    "update1_admission_dev": (19_785_600, 19_958_400),
    "edge1_evaluation": (19_958_400, 20_563_200),
    "update2_train": (19_958_400, 20_995_200),
    "update2_admission_dev": (20_995_200, 21_168_000),
    "edge2_evaluation": (21_168_000, 21_772_800),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_parquet(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    pq.write_table(
        table,
        temporary,
        compression="zstd",
        use_dictionary=False,
        write_statistics=True,
        version="2.6",
    )
    os.replace(temporary, path)


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def sorted_table(table: pa.Table, column: str) -> pa.Table:
    return table.take(pa.compute.sort_indices(table, sort_keys=[(column, "ascending")]))


def validate_event_file(path: Path, *, listens: bool) -> dict[str, int]:
    required = ["uid", "timestamp", "item_id", "is_organic"]
    if listens:
        required += ["played_ratio_pct", "track_length_seconds"]
    parquet = pq.ParquetFile(path)
    missing = sorted(set(required) - set(parquet.schema_arrow.names))
    if missing:
        raise RuntimeError(f"{path} is missing required columns: {missing}")
    rows = null_values = invalid_organic = timestamp_precision_errors = 0
    sort_violations = 0
    previous_uid: int | None = None
    previous_timestamp: int | None = None
    for batch in parquet.iter_batches(batch_size=1_048_576, columns=required):
        rows += batch.num_rows
        null_values += sum(batch.column(name).null_count for name in required)
        uid = batch.column("uid").to_numpy(zero_copy_only=False).astype(np.int64)
        timestamp = batch.column("timestamp").to_numpy(zero_copy_only=False).astype(np.int64)
        organic = batch.column("is_organic").to_numpy(zero_copy_only=False).astype(np.int64)
        invalid_organic += int(np.sum((organic != 0) & (organic != 1)))
        timestamp_precision_errors += int(np.sum(timestamp % 5 != 0))
        if len(uid):
            if previous_uid is not None:
                sort_violations += int(
                    uid[0] < previous_uid
                    or (uid[0] == previous_uid and timestamp[0] < int(previous_timestamp))
                )
            if len(uid) > 1:
                sort_violations += int(
                    np.sum(
                        (uid[1:] < uid[:-1])
                        | ((uid[1:] == uid[:-1]) & (timestamp[1:] < timestamp[:-1]))
                    )
                )
            previous_uid = int(uid[-1])
            previous_timestamp = int(timestamp[-1])
    result = {
        "rows": rows,
        "required_value_nulls": null_values,
        "invalid_is_organic": invalid_organic,
        "timestamps_not_multiple_of_5": timestamp_precision_errors,
        "uid_timestamp_sort_violations": sort_violations,
    }
    if any(result[key] for key in result if key != "rows"):
        raise RuntimeError(f"raw event validity gate failed for {path}: {result}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--threads", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--memory-limit", default="16GB")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing unsealed materialization (default: refuse overwrite)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.threads < 1:
        raise SystemExit("--threads must be positive")
    contract = yaml.safe_load(CONTRACT.read_text())
    if contract["medium_selector"]["namespace"] != UID_SELECTOR_NAMESPACE:
        raise RuntimeError("selector namespace differs between code and contract")
    theta0_cutoff = int(contract["frozen_time_contract_seconds"]["theta0_cutoff"])
    medium_count = int(contract["medium_selector"]["target_users"])
    paths = {name: args.raw_root / f"{name}.parquet" for name in ("listens", "likes", "dislikes")}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise SystemExit(f"missing raw inputs: {missing}")
    manifest_path = args.manifest_root / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise SystemExit(
            f"refusing to overwrite existing population manifest: {manifest_path}; "
            "use --overwrite only before the artifacts are sealed"
        )

    raw_validity = {
        name: validate_event_file(path, listens=name == "listens")
        for name, path in paths.items()
    }

    connection = duckdb.connect()
    connection.execute(f"PRAGMA threads={int(args.threads)}")
    connection.execute(f"PRAGMA memory_limit='{args.memory_limit}'")
    listens_sql = str(paths["listens"].resolve()).replace("'", "''")
    count_expressions = [
        f"count(*) FILTER (WHERE timestamp < {theta0_cutoff})::BIGINT AS n_theta0",
        *[
        f"count(*) FILTER (WHERE timestamp >= {start} AND timestamp < {end})::BIGINT AS n_{name}"
        for name, (start, end) in WINDOWS.items()
        ],
    ]
    connection.execute(
        f"""
        CREATE TEMP TABLE user_statistics AS
        SELECT uid::UBIGINT AS uid,
               count(*)::BIGINT AS n_listens,
               {', '.join(count_expressions)},
               count(DISTINCT item_id)::BIGINT AS n_unique_items,
               min(timestamp)::BIGINT AS first_timestamp,
               max(timestamp)::BIGINT AS last_timestamp,
               count(*) FILTER (WHERE played_ratio_pct > 50)::BIGINT AS n_positive_listens,
               count(*) FILTER (WHERE is_organic = 1)::BIGINT AS n_organic_listens
        FROM read_parquet('{listens_sql}')
        GROUP BY uid
        """
    )
    stats = connection.execute("SELECT * FROM user_statistics ORDER BY uid").fetch_arrow_table()
    eligible_uids = [
        int(uid)
        for uid, count in zip(stats["uid"].to_pylist(), stats["n_theta0"].to_pylist())
        if int(count) >= 1
    ]
    medium_uids = select_medium_uids(eligible_uids, count=medium_count)
    medium_ranked = sorted(
        ((uid_selector_digest(uid), uid) for uid in medium_uids),
        key=lambda pair: (pair[0], pair[1]),
    )
    large_table = pa.table({"uid": pa.array(sorted(eligible_uids), type=pa.uint64())})
    medium_table = pa.table(
        {
            "uid": pa.array([uid for _, uid in medium_ranked], type=pa.uint64()),
            "selector_rank": pa.array(range(1, medium_count + 1), type=pa.uint32()),
            "selector_sha256": pa.array([digest for digest, _ in medium_ranked]),
        }
    )

    args.manifest_root.mkdir(parents=True, exist_ok=True)
    artifact_paths = {
        "user_statistics": args.manifest_root / "user_statistics.parquet",
        "medium_users": args.manifest_root / "medium_users.parquet",
        "large_users": args.manifest_root / "large_users.parquet",
        "medium_item_mapping": args.manifest_root / "medium_item_mapping.parquet",
        "large_item_mapping": args.manifest_root / "large_item_mapping.parquet",
    }
    atomic_parquet(artifact_paths["user_statistics"], stats)
    atomic_parquet(artifact_paths["medium_users"], medium_table)
    atomic_parquet(artifact_paths["large_users"], large_table)

    connection.register("medium_users", medium_table.select(["uid"]))
    connection.register("large_users", large_table)

    scale_summaries: dict[str, Any] = {}
    for scale in ("medium", "large"):
        user_relation = f"{scale}_users"
        history_profile_row = connection.execute(
            f"""
            SELECT count(*)::BIGINT AS users,
                   count(*) FILTER (WHERE s.n_theta0 >= 32)::BIGINT AS at_least_32,
                   count(*) FILTER (WHERE s.n_theta0 >= 256)::BIGINT AS at_least_256,
                   count(*) FILTER (WHERE s.n_theta0 >= 512)::BIGINT AS at_least_512,
                   count(*) FILTER (WHERE s.n_theta0 >= 1024)::BIGINT AS at_least_1024,
                   quantile_cont(s.n_theta0, [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0]) AS quantiles
            FROM user_statistics s JOIN {user_relation} u USING (uid)
            """
        ).fetchone()
        history_profile = {
            "users": history_profile_row[0],
            "at_least_32": history_profile_row[1],
            "at_least_256": history_profile_row[2],
            "at_least_512": history_profile_row[3],
            "at_least_1024": history_profile_row[4],
            "quantiles_p0_p1_p5_p50_p95_p99_p100": list(history_profile_row[5]),
        }
        mapping = connection.execute(
            f"""
            SELECT row_number() OVER (ORDER BY l.item_id)::UBIGINT AS item_idx,
                   l.item_id::UBIGINT AS raw_item_id
            FROM (
                SELECT DISTINCT item_id
                FROM read_parquet('{listens_sql}') l
                JOIN {user_relation} u USING (uid)
                WHERE timestamp < {theta0_cutoff}
            ) l
            ORDER BY raw_item_id
            """
        ).fetch_arrow_table()
        atomic_parquet(artifact_paths[f"{scale}_item_mapping"], mapping)
        events: dict[str, Any] = {}
        for event_name, event_path in paths.items():
            event_sql = str(event_path.resolve()).replace("'", "''")
            row = connection.execute(
                f"""
                SELECT count(*)::BIGINT AS rows,
                       count(DISTINCT e.uid)::BIGINT AS users,
                       count(DISTINCT e.item_id)::BIGINT AS items,
                       min(timestamp)::BIGINT AS min_timestamp,
                       max(timestamp)::BIGINT AS max_timestamp
                FROM read_parquet('{event_sql}') e
                JOIN {user_relation} u USING (uid)
                """
            ).fetchone()
            events[event_name] = dict(zip(("rows", "users", "items", "min_timestamp", "max_timestamp"), row))
        connection.register(f"{scale}_item_mapping", mapping.select(["raw_item_id"]))
        feedback_sql = " UNION ALL ".join(
            f"SELECT uid, timestamp, item_id, '{name}' AS label FROM read_parquet('{str(paths[name].resolve()).replace(chr(39), chr(39) * 2)}')"
            for name in ("likes", "dislikes")
        )
        request_coverage: dict[str, Any] = {}
        for window, (start, end) in WINDOWS.items():
            rows = connection.execute(
                f"""
                SELECT f.label,
                       count(*)::BIGINT AS raw_requests,
                       count(*) FILTER (WHERE f.timestamp > s.first_timestamp)::BIGINT AS causal_history_requests,
                       count(*) FILTER (WHERE f.timestamp > s.first_timestamp AND m.raw_item_id IS NOT NULL)::BIGINT AS executable_requests,
                       count(DISTINCT f.uid) FILTER (WHERE f.timestamp > s.first_timestamp AND m.raw_item_id IS NOT NULL)::BIGINT AS executable_users
                FROM ({feedback_sql}) f
                JOIN {user_relation} u USING (uid)
                JOIN user_statistics s USING (uid)
                LEFT JOIN {scale}_item_mapping m ON f.item_id = m.raw_item_id
                WHERE f.timestamp >= {start} AND f.timestamp < {end}
                GROUP BY f.label ORDER BY f.label
                """
            ).fetchall()
            by_label = {
                label: {
                    "raw_requests": raw,
                    "causal_history_requests": causal,
                    "executable_requests": executable,
                    "executable_users": users,
                }
                for label, raw, causal, executable, users in rows
            }
            request_coverage[window] = by_label
        scale_summaries[scale] = {
            "users": medium_count if scale == "medium" else len(eligible_uids),
            "foundation_catalog_items": mapping.num_rows,
            "pre_theta0_history_profile": history_profile,
            "events_complete_history": events,
            "f_request_coverage": request_coverage,
        }

    validity = connection.execute(
        f"""
        SELECT count(*)::BIGINT,
               count(*) FILTER (WHERE uid IS NULL OR timestamp IS NULL OR item_id IS NULL OR
                                      is_organic IS NULL OR played_ratio_pct IS NULL OR
                                      track_length_seconds IS NULL)::BIGINT,
               count(*) FILTER (WHERE is_organic NOT IN (0, 1))::BIGINT,
               count(*) FILTER (WHERE timestamp % 5 <> 0)::BIGINT,
               min(timestamp)::BIGINT, max(timestamp)::BIGINT,
               count(DISTINCT uid)::BIGINT, count(DISTINCT item_id)::BIGINT
        FROM read_parquet('{listens_sql}')
        """
    ).fetchone()
    if validity[1] or validity[2] or validity[3]:
        raise RuntimeError(f"raw listens validity gate failed: {validity}")

    length_counts = {
        str(threshold): sum(int(value) >= threshold for value in stats["n_theta0"].to_pylist())
        for threshold in contract["population_eligibility"]["history_length_cohorts_reported_not_selected"]
    }
    quantiles: dict[str, list[float]] = {}
    for column in ("n_listens", "n_theta0", "n_unique_items"):
        quantiles[column] = list(
            connection.execute(
                f"SELECT quantile_cont({column}, [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0]) FROM user_statistics"
            ).fetchone()[0]
        )

    source_hashes = {relative(path): sha256_file(path) for path in paths.values()}
    download_manifest_path = ROOT / contract["raw_inputs"]["download_manifest"]
    download_manifest = json.loads(download_manifest_path.read_text())
    expected_hashes = {
        row["logical_path"]: row.get("observed_sha256") or row.get("sha256")
        for row in download_manifest["files"]
    }
    for name, path in paths.items():
        logical_path = f"flat/500m/{name}.parquet"
        if expected_hashes.get(logical_path) != source_hashes[relative(path)]:
            raise RuntimeError(f"raw hash differs from download manifest: {logical_path}")
    audit = {
        "audit": "yambda500m_scale_population_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": relative(CONTRACT),
        "contract_sha256": sha256_file(CONTRACT),
        "preparation_script_sha256": sha256_file(Path(__file__).resolve()),
        "raw_source_sha256": source_hashes,
        "raw_table_validity": raw_validity,
        "raw_listens": {
            "rows": validity[0],
            "null_or_missing_required_values": validity[1],
            "invalid_is_organic": validity[2],
            "timestamps_not_multiple_of_5": validity[3],
            "min_timestamp": validity[4],
            "max_timestamp": validity[5],
            "users": validity[6],
            "items": validity[7],
        },
        "population": {
            "users_with_listens": stats.num_rows,
            "excluded_no_pre_theta0_state": stats.num_rows - len(eligible_uids),
            "large_eligible_users": len(eligible_uids),
            "medium_selected_users": len(medium_uids),
            "medium_is_subset_of_large": set(medium_uids).issubset(eligible_uids),
            "pre_theta0_history_at_least": length_counts,
            "user_quantiles_p0_p1_p5_p50_p95_p99_p100": quantiles,
        },
        "scales": scale_summaries,
        "interpretation": {
            "short_histories_are_reported_not_filtered": True,
            "future_activity_or_feedback_was_not_used_for_selection": True,
            "selected_users_retain_complete_raw_history_by_uid": True,
            "small_is_existing_frozen_yambda50m_reference": True,
        },
    }
    atomic_json(args.audit_output, audit)

    artifact_hashes = {
        relative(path): sha256_file(path) for path in artifact_paths.values()
    }
    artifact_hashes[relative(args.audit_output)] = sha256_file(args.audit_output)
    manifest = {
        "manifest": "yambda500m_scale_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": relative(CONTRACT),
        "contract_sha256": sha256_file(CONTRACT),
        "preparation_script": relative(Path(__file__).resolve()),
        "preparation_script_sha256": sha256_file(Path(__file__).resolve()),
        "raw_source_sha256": source_hashes,
        "selector": {
            "namespace": UID_SELECTOR_NAMESPACE,
            "algorithm": "ascending_sha256_then_ascending_uid",
            "medium_users": medium_count,
            "large_users": len(eligible_uids),
        },
        "artifacts_sha256": artifact_hashes,
        "training_authorized": False,
        "theta3_access_authorized": False,
    }
    atomic_json(manifest_path, manifest)
    print(json.dumps({"manifest": relative(manifest_path), "audit": relative(args.audit_output), "population": audit["population"], "scales": scale_summaries}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
