#!/usr/bin/env python3
"""Audit whether P8/P9 request-local reuse matches a cutover-materialized cache.

This audit is label-free.  It measures how often post-cutover appends force a
512-token persistent state to evict tokens that were present at release.  When
eviction occurs, recomputing the retained parent tokens per request is not the
same operation as retaining their already-materialized upper-layer K/V.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

import train_p7_theta0 as p7


ROOT = Path(__file__).resolve().parents[1]
LISTENS = ROOT / "data/raw/yambda/flat/50m/listens.parquet"
OUTPUT = ROOT / "results/p9/p9_4_release_lineage_audit_v1.json"
REQUEST_SOURCES = {
    "r0": ROOT / "results/p8/staleness_raw/r0/m0_f_seed17/F_fidelity.parquet",
    "r1_edge1": ROOT / "results/p8/staleness_raw/r1_edge1/m0_f_seed17/F_fidelity.parquet",
    "r1_edge2": ROOT / "results/p8/staleness_raw/r1_edge2/m0_f_seed17/F_fidelity.parquet",
    "r2": ROOT / "results/p8/staleness_raw/r2/m0_f_seed17/F_fidelity.parquet",
}
CUTOVERS = {
    "r0": 231 * 86_400,
    "r1_edge1": 231 * 86_400,
    "r1_edge2": 245 * 86_400,
    "r2": 231 * 86_400,
}


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        raise ValueError("empty percentile input")
    return sorted(values)[round((len(values) - 1) * fraction)]


def audit_release(connection: duckdb.DuckDBPyConnection, release: str) -> dict:
    source = REQUEST_SOURCES[release]
    cutover = CUTOVERS[release]
    rows = connection.execute(
        """
        WITH requests AS (
          SELECT DISTINCT request_id, uid, query_timestamp,
                 prefix_tokens_at_cutover AS reported_retained,
                 suffix_tokens_after_cutover AS reported_suffix
          FROM read_parquet(?)
        ), counts AS (
          SELECT r.*,
                 least(512, count(*) FILTER (WHERE l.timestamp < ?))::INTEGER
                   AS release_snapshot_tokens,
                 count(*) FILTER (
                   WHERE l.timestamp >= ? AND l.timestamp < r.query_timestamp
                 )::INTEGER AS raw_suffix_tokens
          FROM requests r
          JOIN read_parquet(?) l ON l.uid = r.uid AND l.timestamp < r.query_timestamp
          GROUP BY ALL
        )
        SELECT *,
               greatest(0, least(release_snapshot_tokens, 512 - raw_suffix_tokens))::INTEGER
                 AS expected_retained,
               (release_snapshot_tokens + raw_suffix_tokens > 512) AS eviction_required
        FROM counts
        ORDER BY request_id
        """,
        [str(source), cutover, cutover, str(LISTENS)],
    ).fetchall()
    names = [column[0] for column in connection.description]
    records = [dict(zip(names, row, strict=True)) for row in rows]
    suffix = [int(row["raw_suffix_tokens"]) for row in records]
    snapshots = [int(row["release_snapshot_tokens"]) for row in records]
    eviction = [row for row in records if row["eviction_required"]]
    retained_mismatch = [
        row for row in records
        if int(row["reported_retained"]) != int(row["expected_retained"])
    ]
    suffix_mismatch = [
        row for row in records
        if int(row["reported_suffix"]) != int(row["raw_suffix_tokens"])
    ]
    no_parent = [row for row in records if int(row["expected_retained"]) == 0]
    return {
        "release": release,
        "cutover": cutover,
        "source": str(source.relative_to(ROOT)),
        "source_sha256": p7.sha256_file(source),
        "requests": len(records),
        "unique_users": len({int(row["uid"]) for row in records}),
        "release_snapshot_tokens": {
            "p50": percentile(snapshots, 0.50),
            "p90": percentile(snapshots, 0.90),
            "p99": percentile(snapshots, 0.99),
            "saturated_512_requests": sum(value == 512 for value in snapshots),
            "saturated_512_fraction": sum(value == 512 for value in snapshots) / len(records),
        },
        "post_cutover_suffix_tokens": {
            "p50": percentile(suffix, 0.50),
            "p90": percentile(suffix, 0.90),
            "p95": percentile(suffix, 0.95),
            "p99": percentile(suffix, 0.99),
            "zero_suffix_requests": sum(value == 0 for value in suffix),
        },
        "eviction_required_requests": len(eviction),
        "eviction_required_fraction": len(eviction) / len(records),
        "no_eviction_requests": len(records) - len(eviction),
        "no_eviction_fraction": 1.0 - len(eviction) / len(records),
        "expected_no_parent_requests": len(no_parent),
        "reported_retained_length_mismatches": len(retained_mismatch),
        "reported_suffix_length_mismatches": len(suffix_mismatch),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    connection = duckdb.connect()
    connection.execute(f"PRAGMA threads={args.threads:d}")
    releases = [audit_release(connection, release) for release in REQUEST_SOURCES]
    payload = {
        "status": "P9_4_request_local_lineage_not_equivalent_to_cutover_materialization",
        "scope": "label_free_F_fidelity_requests_already_evaluated_by_P8",
        "raw_listens": str(LISTENS.relative_to(ROOT)),
        "raw_listens_sha256": p7.sha256_file(LISTENS),
        "finding": {
            "retained_length_math_matches": all(
                row["reported_retained_length_mismatches"] == 0 for row in releases
            ),
            "suffix_count_math_matches": all(
                row["reported_suffix_length_mismatches"] == 0 for row in releases
            ),
            "semantic_mismatch": (
                "When eviction is required, recomputing retained tokens under the parent "
                "per request discards causal contributions from release-snapshot tokens "
                "that authentic persistent upper-layer K/V still encodes."
            ),
            "P8_P9_interpretation": "request_conditioned_retained_prefix_diagnostic",
            "release_frontier_authorized": False,
            "required_next_gate": "materialized_cutover_append_evict_lineage_canary",
        },
        "releases": releases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
