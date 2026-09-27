#!/usr/bin/env python3
"""Bind one Reuse edge to exactly the requests in its retained Full evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEFAULT_OUTPUT, PLAN, binding, sha256, write_json


def label_ids(b: dict) -> set[str]:
    start, end = b["days"]
    labels = pq.read_table(b["labels"], filters=[
        ("time_block", "=", "matrix_horizon"), ("target_known", "=", True),
        ("query_timestamp", ">=", start * 86400), ("query_timestamp", "<", end * 86400),
    ], columns=["request_id"])["request_id"].to_pylist()
    if len(labels) != len(set(labels)):
        raise RuntimeError("quality labels have duplicate request IDs")
    return set(labels)


def prepare(scale: str, edge: int, output: Path) -> dict:
    b = binding(scale, edge)
    if (output / "binding.json").exists():
        record = json.loads((output / "binding.json").read_text())
        if record["scale"] != scale or record["edge"] != b["edge"] or record["days"] != b["days"]:
            raise RuntimeError("existing edge has another version or evaluation window")
        if record["full_current_name"] != b["full_current_name"]:
            raise RuntimeError("existing edge has another Full endpoint")
        for name in ("parent", "current", "full_raw", "requests", "labels", "dataset"):
            if record["sources"][name]["sha256"] != sha256(b[name]):
                raise RuntimeError(f"existing edge source changed: {name}")
        if not record.get("labels_panel_checked"):
            existing_ids = set()
            for rank in range(4):
                path = output / f"requests_rank{rank}.parquet"
                if sha256(path) != record["ranks"][rank]["sha256"]:
                    raise RuntimeError(f"prepared request shard changed: {path}")
                existing_ids.update(pq.read_table(path, columns=["request_id"])["request_id"].to_pylist())
            if existing_ids != label_ids(b):
                raise RuntimeError("quality labels differ from the frozen Full request panel")
            record["labels_panel_checked"] = True
            record["checks"].append("quality labels use the same request IDs")
        if record["plan_sha256"] != sha256(PLAN):
            record["plan_sha256"] = sha256(PLAN)
        write_json(output / "binding.json", record)
        return record
    for key in ("parent", "current", "full_raw", "requests", "labels", "dataset"):
        if not b[key].is_file():
            raise FileNotFoundError(b[key])
    full_seal = json.loads(b["full_raw"].with_name("raw.seal.json").read_text())
    if full_seal["raw_sha256"] != sha256(b["full_raw"]):
        raise RuntimeError("Full raw scores differ from their original seal")
    if full_seal["evaluation_day_range"] != b["days"]:
        raise RuntimeError("Full evaluation day range differs from this Reuse edge")
    full = pq.read_table(b["full_raw"], columns=["request_id", "uid", "query_timestamp", "model_name"])
    full = full.filter(pc.equal(full["model_name"], b["full_current_name"]))
    full_ids = full["request_id"].to_pylist()
    if len(full_ids) != len(set(full_ids)):
        raise RuntimeError("Full selected-current scores have duplicate request IDs")
    start, end = b["days"]
    requests = pq.read_table(b["requests"], filters=[
        ("time_block", "=", "matrix_horizon"), ("target_known", "=", True),
        ("query_timestamp", ">=", start * 86400), ("query_timestamp", "<", end * 86400),
    ], columns=["request_id", "uid", "query_timestamp", "item_idx"])
    request_ids = requests["request_id"].to_pylist()
    if len(request_ids) != len(full_ids) or set(request_ids) != set(full_ids):
        raise RuntimeError("Reuse request panel differs from retained Full evaluation")
    if set(request_ids) != label_ids(b):
        raise RuntimeError("quality labels differ from the frozen Full request panel")
    full_identity = dict(zip(full_ids, zip(full["uid"].to_pylist(), full["query_timestamp"].to_pylist())))
    if any(full_identity[row_id] != (uid, timestamp) for row_id, uid, timestamp in zip(
        request_ids, requests["uid"].to_pylist(), requests["query_timestamp"].to_pylist(), strict=True
    )):
        raise RuntimeError("Full and Reuse request identities disagree")
    rows = sorted(requests.to_pylist(), key=lambda row: (row["uid"], row["query_timestamp"], row["request_id"]))
    counts: dict[int, int] = {}
    for row in rows:
        counts[int(row["uid"])] = counts.get(int(row["uid"]), 0) + 1
    loads = [0] * len(b["gpus"]); owners: dict[int, int] = {}
    for uid, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        rank = min(range(len(loads)), key=lambda value: (loads[value], value))
        owners[uid] = rank; loads[rank] += count
    output.mkdir(parents=True, exist_ok=True)
    rank_records = []
    for rank in range(len(loads)):
        path = output / f"requests_rank{rank}.parquet"
        selected = [row for row in rows if owners[int(row["uid"])] == rank]
        partial = path.with_suffix(".parquet.partial")
        pq.write_table(pa.Table.from_pylist(selected, schema=requests.schema), partial, compression="zstd")
        partial.replace(path)
        rank_records.append({"rank": rank, "users": len({row["uid"] for row in selected}),
                             "requests": len(selected), "sha256": sha256(path)})
    record = {
        "status": "prepared", "scale": scale, "edge": b["edge"], "days": b["days"],
        "plan_sha256": sha256(PLAN), "full_current_name": b["full_current_name"],
        "sources": {name: {"path": str(b[name].relative_to(Path(__file__).resolve().parents[2])),
                           "sha256": sha256(b[name])}
                    for name in ("parent", "current", "full_raw", "requests", "labels", "dataset")},
        "requests": len(rows), "users": len(counts), "ranks": rank_records,
        "labels_panel_checked": True,
        "full_raw_seal_sha256": sha256(b["full_raw"].with_name("raw.seal.json")),
        "checks": ["selected Full request IDs equal Reuse request IDs",
                   "UID and request timestamps equal", "Full raw seal verified",
                   "quality labels use the same request IDs"],
    }
    write_json(output / "binding.json", record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", required=True, choices=["medium", "large", "max"])
    parser.add_argument("--edge", type=int, required=True, choices=range(1, 6))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    path = args.output_root / args.scale / f"v{args.edge-1}_to_v{args.edge}"
    print(json.dumps(prepare(args.scale, args.edge, path), ensure_ascii=False, indent=2))
