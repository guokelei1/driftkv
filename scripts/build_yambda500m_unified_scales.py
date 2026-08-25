#!/usr/bin/env python3
"""Build one shared Yambda-500M store and train-ready S/M/L logical datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hstu_kvcache.data.scale_population import (  # noqa: E402
    UID_SELECTOR_NAMESPACE,
    uid_selector_digest,
)


CONTRACT = ROOT / "configs/contracts/yambda500m_unified_scales_v1.yaml"
DAY = 86_400
BUCKET_SECONDS = 7 * DAY
WINDOWS = {
    "foundation_train": (0, 17_539_200),
    "development": (17_539_200, 18_144_000),
    "qualification": (18_144_000, 18_748_800),
    "update1_train": (18_748_800, 19_785_600),
    "update1_admission": (19_785_600, 19_958_400),
    "edge1_evaluation": (19_958_400, 20_563_200),
    "update2_train": (19_958_400, 20_995_200),
    "update2_admission": (20_995_200, 21_168_000),
    "edge2_evaluation": (21_168_000, 21_772_800),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_parquet(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        table,
        path,
        compression="zstd",
        use_dictionary=False,
        write_statistics=True,
        version="2.6",
    )


def qpath(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data/raw/yambda/flat/500m")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--memory-limit", default="16GB")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Atomically replace the exact generated output after a successful rebuild",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.threads < 1:
        raise SystemExit("--threads must be positive")
    contract = yaml.safe_load(CONTRACT.read_text())
    if contract["population"]["selector_namespace"] != UID_SELECTOR_NAMESPACE:
        raise RuntimeError("selector namespace differs between code and contract")
    output = args.output or ROOT / contract["physical_store"]["root"]
    output = output.resolve()
    if output.exists() and not args.replace:
        raise SystemExit(f"refusing to replace existing unified dataset: {output}")
    raw = {name: args.raw_root / f"{name}.parquet" for name in ("listens", "likes", "dislikes")}
    missing = [str(path) for path in raw.values() if not path.exists()]
    if missing:
        raise SystemExit(f"missing raw inputs: {missing}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + ".staging.", dir=output.parent))
    backup = output.with_name(output.name + f".backup.{os.getpid()}")

    connection = duckdb.connect()
    connection.execute(f"PRAGMA threads={int(args.threads)}")
    connection.execute(f"PRAGMA memory_limit='{args.memory_limit}'")
    listens = qpath(raw["listens"])
    likes = qpath(raw["likes"])
    dislikes = qpath(raw["dislikes"])
    theta0 = int(contract["population"]["theta0_cutoff"])

    try:
        window_counts = [
            f"count(*) FILTER (WHERE timestamp >= {start} AND timestamp < {end})::BIGINT AS n_{name}"
            for name, (start, end) in WINDOWS.items()
        ]
        connection.execute(
            f"""
            CREATE TEMP TABLE listen_stats AS
            SELECT uid::UBIGINT AS uid,
                   count(*)::BIGINT AS n_listens,
                   count(*) FILTER (WHERE timestamp < {theta0})::BIGINT AS n_theta0,
                   {', '.join(window_counts)},
                   count(DISTINCT item_id)::BIGINT AS n_unique_items,
                   min(timestamp)::BIGINT AS first_timestamp,
                   max(timestamp)::BIGINT AS last_timestamp,
                   count(*) FILTER (WHERE played_ratio_pct > 50)::BIGINT AS n_positive_listens,
                   count(*) FILTER (WHERE is_organic = 1)::BIGINT AS n_organic_listens
            FROM read_parquet('{listens}') GROUP BY uid
            """
        )
        eligible = [
            int(row[0])
            for row in connection.execute(
                "SELECT uid FROM listen_stats WHERE n_theta0 >= 1 ORDER BY uid"
            ).fetchall()
        ]
        ranked = sorted(eligible, key=lambda uid: (uid_selector_digest(uid), uid))
        rank_by_uid = {uid: rank for rank, uid in enumerate(ranked, start=1)}
        ranked_users = pa.table(
            {
                "uid": pa.array(sorted(eligible), type=pa.uint64()),
                "selector_rank": pa.array(
                    [rank_by_uid[uid] for uid in sorted(eligible)], type=pa.uint32()
                ),
            }
        )
        connection.register("ranked_users", ranked_users)
        connection.execute(
            f"""
            CREATE TEMP TABLE feedback_stats AS
            SELECT uid::UBIGINT AS uid,
                   count(*) FILTER (WHERE label = 1)::BIGINT AS n_likes,
                   count(*) FILTER (WHERE label = 0)::BIGINT AS n_dislikes,
                   min(timestamp)::BIGINT AS first_feedback_timestamp,
                   max(timestamp)::BIGINT AS last_feedback_timestamp
            FROM (
                SELECT uid, timestamp, 1 AS label FROM read_parquet('{likes}')
                UNION ALL
                SELECT uid, timestamp, 0 AS label FROM read_parquet('{dislikes}')
            ) GROUP BY uid
            """
        )
        user_statistics = connection.execute(
            """
            SELECT s.*, r.selector_rank,
                   coalesce(f.n_likes, 0)::BIGINT AS n_likes,
                   coalesce(f.n_dislikes, 0)::BIGINT AS n_dislikes,
                   f.first_feedback_timestamp, f.last_feedback_timestamp
            FROM listen_stats s JOIN ranked_users r USING (uid)
            LEFT JOIN feedback_stats f USING (uid)
            ORDER BY s.uid
            """
        ).fetch_arrow_table()
        write_parquet(staging / "shared/user_statistics.parquet", user_statistics)
        connection.register("eligible_user_statistics", user_statistics)

        shared_listens = staging / "shared/listens"
        shared_feedback = staging / "shared/feedback"
        connection.execute(
            f"""
            COPY (
                SELECT (e.timestamp // {BUCKET_SECONDS})::INTEGER AS week,
                       e.uid::UBIGINT AS uid, r.selector_rank,
                       e.timestamp::UBIGINT AS timestamp,
                       e.item_id::UBIGINT AS raw_item_id,
                       (1 + (1 - e.is_organic))::UTINYINT AS behavior,
                       e.is_organic::UTINYINT AS is_organic,
                       e.played_ratio_pct::USMALLINT AS played_ratio_pct,
                       e.track_length_seconds::UINTEGER AS track_length_seconds
                FROM read_parquet('{listens}') e JOIN ranked_users r USING (uid)
            ) TO '{qpath(shared_listens)}'
            (FORMAT PARQUET, PARTITION_BY (week), COMPRESSION ZSTD, ROW_GROUP_SIZE 262144)
            """
        )
        connection.execute(
            f"""
            COPY (
                SELECT (e.timestamp // {BUCKET_SECONDS})::INTEGER AS week,
                       e.uid::UBIGINT AS uid, r.selector_rank,
                       e.timestamp::UBIGINT AS timestamp,
                       e.item_id::UBIGINT AS raw_item_id,
                       e.label::UTINYINT AS label,
                       e.is_organic::UTINYINT AS is_organic
                FROM (
                    SELECT uid, timestamp, item_id, is_organic, 1 AS label
                    FROM read_parquet('{likes}')
                    UNION ALL
                    SELECT uid, timestamp, item_id, is_organic, 0 AS label
                    FROM read_parquet('{dislikes}')
                ) e JOIN ranked_users r USING (uid)
            ) TO '{qpath(shared_feedback)}'
            (FORMAT PARQUET, PARTITION_BY (week), COMPRESSION ZSTD, ROW_GROUP_SIZE 262144)
            """
        )

        scale_contracts = contract["population"]["scales"]
        scale_summaries: dict[str, Any] = {}
        for scale, values in scale_contracts.items():
            limit = len(eligible) if values["rank_limit"] == "all_eligible" else int(values["rank_limit"])
            scale_root = staging / "scales" / scale
            users = connection.execute(
                f"SELECT * FROM eligible_user_statistics "
                f"WHERE selector_rank <= {limit} ORDER BY uid"
            ).fetch_arrow_table()
            write_parquet(scale_root / "users.parquet", users)
            mapping = connection.execute(
                f"""
                SELECT row_number() OVER (ORDER BY item_id)::UBIGINT AS item_idx,
                       item_id::UBIGINT AS raw_item_id
                FROM (
                    SELECT DISTINCT e.item_id
                    FROM read_parquet('{listens}') e
                    JOIN ranked_users r USING (uid)
                    WHERE r.selector_rank <= {limit} AND e.timestamp < {theta0}
                ) ORDER BY raw_item_id
                """
            ).fetch_arrow_table()
            write_parquet(scale_root / "item_mapping.parquet", mapping)
            dataset = {
                "dataset": f"yambda500m_unified_{scale}_v1",
                "contract": str(CONTRACT.relative_to(ROOT)),
                "scale": scale,
                "rank_limit": limit,
                "users": users.num_rows,
                "foundation_items": mapping.num_rows,
                "model": values["model"],
                "context": int(values["context"]),
                "shared_listens_glob": "../../shared/listens/**/*.parquet",
                "shared_feedback_glob": "../../shared/feedback/**/*.parquet",
                "users_path": "users.parquet",
                "item_mapping_path": "item_mapping.parquet",
                "timestamp_windows": "half_open",
                "oov_item_idx": 0,
                "feedback_access": contract["feedback_access"],
                "training_authorized": False,
            }
            write_json(scale_root / "dataset.json", dataset)
            scale_summaries[scale] = {
                "users": users.num_rows,
                "foundation_items": mapping.num_rows,
                "rank_limit": limit,
            }

        small_uids = set(ranked[:10_000])
        medium_uids = set(ranked[:30_000])
        if not small_uids.issubset(medium_uids) or not medium_uids.issubset(eligible):
            raise RuntimeError("S/M/L nesting invariant failed")
        old_medium = ROOT / "data/manifests/yambda500m_scale_v1/medium_users.parquet"
        if old_medium.exists():
            old = set(pq.read_table(old_medium, columns=["uid"])["uid"].to_pylist())
            if old != medium_uids:
                raise RuntimeError("unified Medium differs from frozen v1 Medium")

        file_rows = []
        for path in sorted(staging.rglob("*")):
            if not path.is_file():
                continue
            file_rows.append(
                {
                    "path": str(path.relative_to(staging)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        source_manifest = json.loads(
            (ROOT / "data/manifests/yambda500m_scale_v1/manifest.json").read_text()
        )
        manifest = {
            "manifest": "yambda500m_unified_scales_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "contract": str(CONTRACT.relative_to(ROOT)),
            "contract_sha256": sha256_file(CONTRACT),
            "builder": str(Path(__file__).resolve().relative_to(ROOT)),
            "builder_sha256": sha256_file(Path(__file__).resolve()),
            "source_sha256": source_manifest["raw_source_sha256"],
            "selector_namespace": UID_SELECTOR_NAMESPACE,
            "eligible_users": len(eligible),
            "scales": scale_summaries,
            "files": file_rows,
            "logical_content_deterministic": True,
            "parquet_byte_hashes_may_change_with_writer_version": True,
            "training_authorized": False,
            "theta3_training_or_result_access": False,
        }
        write_json(staging / "manifest.json", manifest)

        if output.exists():
            if backup.exists():
                raise RuntimeError(f"backup path already exists: {backup}")
            os.replace(output, backup)
        connection.close()
        os.replace(staging, output)
        if backup.exists():
            shutil.rmtree(backup)
        total_bytes = sum(row["bytes"] for row in file_rows)
        print(json.dumps({
            "output": str(output.relative_to(ROOT)),
            "eligible_users": len(eligible),
            "scales": scale_summaries,
            "files": len(file_rows),
            "bytes": total_bytes,
        }, indent=2))
        return 0
    except BaseException:
        connection.close()
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        if backup.exists() and not output.exists():
            os.replace(backup, output)
        raise


if __name__ == "__main__":
    sys.exit(main())
