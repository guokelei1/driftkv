"""Raw-first six-path schema validation and deterministic sealing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


PATHS = (
    "parent_full_request_local", "current_full_request_local",
    "parent_exact_rolling", "current_exact_rolling",
    "one_hop_reuse_rolling", "recursive_reuse_rolling",
)
REQUIRED_COLUMNS = (
    "request_id", "uid", "query_timestamp", "edge", "path",
    "base_logit", "residual_logit", "append_count_since_cutover",
    "seconds_since_cutover", "history_length", "cache_length",
    "checkpoint_sha256", "manifest_sha256",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_raw_table(table: pa.Table) -> dict:
    missing = sorted(set(REQUIRED_COLUMNS) - set(table.column_names))
    if missing:
        raise ValueError(f"raw table missing columns: {missing}")
    if "label" in table.column_names:
        raise ValueError("raw score artifact must be sealed before label join")
    paths = set(pc.unique(table["path"]).to_pylist())
    unknown = sorted(paths - set(PATHS))
    if unknown:
        raise ValueError(f"unknown evaluation paths: {unknown}")
    keys = list(zip(
        table["request_id"].to_pylist(), table["edge"].to_pylist(),
        table["path"].to_pylist(), strict=True,
    ))
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate request/edge/path rows")
    groups: dict[tuple[str, str], set[str]] = {}
    for request, edge, path in keys:
        groups.setdefault((request, edge), set()).add(path)
    incomplete = sum(value != set(PATHS) for value in groups.values())
    if incomplete:
        raise ValueError(f"{incomplete} request/edge groups do not contain all six paths")
    return {"rows": table.num_rows, "request_edge_groups": len(groups), "paths": list(PATHS)}


def seal_raw(path: Path, output: Path) -> dict:
    table = pq.read_table(path)
    audit = validate_raw_table(table)
    payload = {
        "status": "sealed_raw_before_label_join", "raw_path": str(path),
        "raw_sha256": sha256_file(path), **audit,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    return payload
