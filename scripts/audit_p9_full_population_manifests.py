#!/usr/bin/env python3
"""Audit P9.7 full cutover-state and label-free probe manifests."""

from __future__ import annotations

import hashlib
import json
import bisect
from pathlib import Path

import duckdb
import numpy as np
import pyarrow.parquet as pq

import train_p7_theta0 as p7


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/manifests/p9_full_population_v1"
LISTENS = ROOT / "data/raw/yambda/flat/50m/listens.parquet"
OUTPUT = ROOT / "results/p9/p9_7_full_population_audit_v1.json"
EDGES = {
    "edge1": (19958400, "r2"),
    "edge2": (21168000, "r1_edge2"),
}


class RawPointerReader:
    def __init__(self, path: Path) -> None:
        self.parquet = pq.ParquetFile(path)
        self.ends = np.cumsum([
            self.parquet.metadata.row_group(index).num_rows
            for index in range(self.parquet.num_row_groups)
        ]).tolist()
        self.cached_group = -1
        self.cached_table = None

    def rows(self, start: int, end: int):
        pieces = []
        cursor = start
        while cursor < end:
            group = bisect.bisect_right(self.ends, cursor)
            group_start = 0 if group == 0 else self.ends[group - 1]
            group_end = self.ends[group]
            if group != self.cached_group:
                self.cached_table = self.parquet.read_row_group(
                    group, columns=["uid", "timestamp"]
                )
                self.cached_group = group
            count = min(end, group_end) - cursor
            pieces.append(self.cached_table.slice(cursor - group_start, count))
            cursor += count
        import pyarrow as pa
        return pieces[0] if len(pieces) == 1 else pa.concat_tables(pieces)


def pointer_audit(state_table, cutover: int) -> dict:
    reader = RawPointerReader(LISTENS)
    reconstructed_rows = uid_mismatches = noncausal_rows = last_present = 0
    expected_rows = 0
    for row in state_table.to_pylist():
        length = int(row["effective_prefix_length"])
        end = int(row["raw_prefix_end_exclusive"])
        raw = reader.rows(end - length, end)
        uids = raw["uid"].to_numpy(zero_copy_only=False).astype(np.int64)
        timestamps = raw["timestamp"].to_numpy(zero_copy_only=False).astype(np.int64)
        reconstructed_rows += len(raw)
        expected_rows += length
        uid_mismatches += int(np.sum(uids != int(row["uid"])))
        noncausal_rows += int(np.sum(timestamps >= cutover))
        last_present += int(int(row["last_activity_timestamp"]) in timestamps)
    return {
        "states_reconstructed": len(state_table),
        "reconstructed_rows": reconstructed_rows,
        "expected_rows": expected_rows,
        "uid_mismatches": uid_mismatches,
        "noncausal_rows": noncausal_rows,
        "states_with_last_timestamp_present": last_present,
    }


def audit_edge(connection: duckdb.DuckDBPyConnection, edge: str, cutover: int, release: str) -> dict:
    state_path = MANIFEST / edge / "states.parquet"
    probe_path = MANIFEST / edge / "cutover_probes.parquet"
    meta_path = MANIFEST / edge / "manifest.json"
    meta = json.loads(meta_path.read_text())
    state_table = pq.read_table(state_path)
    pointer = pointer_audit(state_table, cutover)
    future_source = ROOT / "results/p8/staleness_raw" / release / "m0_f_seed17/F_fidelity.parquet"
    coverage = connection.execute(
        """
        WITH states AS (SELECT uid FROM read_parquet(?)),
             served AS (SELECT DISTINCT uid FROM read_parquet(?))
        SELECT (SELECT count(*) FROM states), (SELECT count(*) FROM served),
               count(*) FILTER (WHERE served.uid IS NOT NULL),
               count(*) FILTER (WHERE served.uid IS NULL)
        FROM states LEFT JOIN served USING(uid)
        """,
        [str(state_path), str(future_source)],
    ).fetchone()
    probe_table = pq.read_table(probe_path)
    banned = {"label", "target", "future_activity", "future_request_count", "rankability"}
    if banned & set(state_table.schema.names) or banned & set(probe_table.schema.names):
        raise RuntimeError("future/label field entered P9.7 manifests")
    probes = probe_table.to_pylist()
    duplicate_candidate_rows = 0
    candidate_hash_mismatches = 0
    for row in probes:
        candidates = [int(value) for value in row["candidate_ids"]]
        duplicate_candidate_rows += len(candidates) != len(set(candidates))
        digest = hashlib.sha256(json.dumps(candidates, separators=(",", ":")).encode()).hexdigest()
        candidate_hash_mismatches += digest != row["candidate_hash"]
    result = {
        "edge": edge,
        "cutover": cutover,
        "states": int(coverage[0]),
        "served_future_request_users": int(coverage[1]),
        "served_users_in_state_population": int(coverage[2]),
        "unserved_state_users": int(coverage[3]),
        "served_population_fraction": float(coverage[2] / coverage[0]),
        "pointer_audit": {
            **pointer,
        },
        "probe_audit": {
            "rows": len(probes),
            "uid_unique": len(set(int(row["uid"]) for row in probes)),
            "candidate_count_values": sorted(set(len(row["candidate_ids"]) for row in probes)),
            "duplicate_candidate_rows": int(duplicate_candidate_rows),
            "candidate_hash_mismatches": int(candidate_hash_mismatches),
            "future_or_label_fields": sorted(banned & (set(state_table.schema.names) | set(probe_table.schema.names))),
        },
        "artifacts": {
            "state_sha256": p7.sha256_file(state_path),
            "probe_sha256": p7.sha256_file(probe_path),
            "meta_sha256": p7.sha256_file(meta_path),
        },
    }
    if (
        result["pointer_audit"]["states_reconstructed"] != result["states"]
        or result["pointer_audit"]["reconstructed_rows"] != result["pointer_audit"]["expected_rows"]
        or result["pointer_audit"]["uid_mismatches"]
        or result["pointer_audit"]["noncausal_rows"]
        or result["pointer_audit"]["states_with_last_timestamp_present"] != result["states"]
        or result["served_users_in_state_population"] != result["served_future_request_users"]
        or result["probe_audit"]["rows"] != result["states"]
        or result["probe_audit"]["uid_unique"] != result["states"]
        or result["probe_audit"]["candidate_count_values"] != [16]
        or result["probe_audit"]["duplicate_candidate_rows"]
        or result["probe_audit"]["candidate_hash_mismatches"]
        or result["probe_audit"]["future_or_label_fields"]
    ):
        raise RuntimeError(f"P9.7 population audit failed: {result}")
    return result


def main() -> None:
    connection = duckdb.connect()
    connection.execute("PRAGMA threads=24")
    edges = [audit_edge(connection, edge, *spec) for edge, spec in EDGES.items()]
    summary = MANIFEST / "materialization_summary.json"
    payload = {
        "status": "P9_7_full_population_and_probe_audit_passed",
        "materialization_summary": str(summary.relative_to(ROOT)),
        "materialization_summary_sha256": p7.sha256_file(summary),
        "edges": edges,
        "migration_population_uses_future_requests": False,
        "heldout_served_subset_role": "evaluation_only",
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
