#!/usr/bin/env python3
"""Freeze outcome-conditioned diagnostic panels from retained Full/Reuse scores."""

from __future__ import annotations

import argparse
from functools import lru_cache
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from hstu_kvcache.evaluation.binary_metrics import binary_metrics, sigmoid
from scripts.selective_recompute_2026_09.cohort import HASH_NAMESPACE, RULE, select_users
from scripts.unified_reuse_2026_09.common import DAY, sha256, write_json

DEFAULT_OUTPUT = ROOT / "results/selective_recompute_2026_09/panels"
REUSE_ROOT = ROOT / "results/unified_reuse_2026_09"


@lru_cache(maxsize=None)
def checked_source(path: Path, expected: str) -> str:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"retained source changed: {path}")
    return actual


def sorted_requests(table: pa.Table) -> pa.Table:
    table = table.sort_by([("request_id", "ascending")]).combine_chunks()
    ids = table["request_id"].to_pylist()
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate request IDs in retained source")
    return table


def paired_population(scale: str, edge: int) -> tuple[pa.Table, dict, dict]:
    directory = REUSE_ROOT / scale / f"v{edge-1}_to_v{edge}"
    bound = json.loads((directory / "binding.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    if summary["status"] != "complete":
        raise RuntimeError("the source Reuse evaluation must be complete")
    for name in ("full_raw", "labels", "dataset"):
        source = bound["sources"][name]
        checked_source(ROOT / source["path"], source["sha256"])
    full_path = ROOT / bound["sources"]["full_raw"]["path"]
    full_seal_path = full_path.with_name("raw.seal.json")
    checked_source(full_seal_path, bound["full_raw_seal_sha256"])
    if json.loads(full_seal_path.read_text())["raw_sha256"] != bound["sources"]["full_raw"]["sha256"]:
        raise RuntimeError("Full raw seal disagrees with the retained binding")
    if summary["full_raw_sha256"] != bound["sources"]["full_raw"]["sha256"]:
        raise RuntimeError("Full raw differs between source binding and summary")
    full = pq.read_table(full_path, columns=["request_id", "uid", "query_timestamp", "model_name", "hstu_logit"])
    full = sorted_requests(full.filter(pc.equal(full["model_name"], bound["full_current_name"])))

    reuse_tables = []
    for shard in summary["reuse_shards"]:
        path = ROOT / shard["path"]
        checked_source(path, shard["sha256"])
        seal = json.loads(path.with_suffix(".seal.json").read_text())
        if seal["sha256"] != shard["sha256"]:
            raise RuntimeError(f"Reuse seal differs from adjudicated shard: {path}")
        table = pq.read_table(path)
        if len(table) != shard["requests"]:
            raise RuntimeError(f"Reuse shard request count changed: {path}")
        reuse_tables.append(table)
    reuse = sorted_requests(pa.concat_tables(reuse_tables))

    request_tables = []
    for rank in bound["ranks"]:
        path = directory / f"requests_rank{rank['rank']}.parquet"
        checked_source(path, rank["sha256"])
        request_tables.append(pq.read_table(path))
    requests = sorted_requests(pa.concat_tables(request_tables))
    start, end = bound["days"]
    labels = sorted_requests(pq.read_table(ROOT / bound["sources"]["labels"]["path"], filters=[
        ("time_block", "=", "matrix_horizon"), ("target_known", "=", True),
        ("query_timestamp", ">=", start * DAY), ("query_timestamp", "<", end * DAY),
    ], columns=["request_id", "uid", "query_timestamp", "label"]))
    for name, table in (("full", full), ("reuse", reuse), ("labels", labels)):
        for column in ("request_id", "uid", "query_timestamp"):
            if not np.array_equal(table[column].to_numpy(), requests[column].to_numpy()):
                raise RuntimeError(f"{name} {column} differs from the frozen request panel")
    if len(requests) != summary["requests"]:
        raise RuntimeError("source population request count differs from its summary")
    result = pa.table({
        "request_id": requests["request_id"], "uid": requests["uid"].cast(pa.int64()),
        "query_timestamp": requests["query_timestamp"].cast(pa.int64()),
        "item_idx": requests["item_idx"].cast(pa.int64()), "label": labels["label"].cast(pa.int64()),
        "full_logit": full["hstu_logit"], "reuse_logit": reuse["hstu_logit"],
        **{name: reuse[name] for name in (
            "history_length", "cache_length", "append_count_since_cutover", "rolling_evictions",
        )},
    })
    sources = {
        "reuse_binding": {"path": str((directory / "binding.json").relative_to(ROOT)),
                          "sha256": sha256(directory / "binding.json")},
        "reuse_summary": {"path": str((directory / "summary.json").relative_to(ROOT)),
                          "sha256": sha256(directory / "summary.json")},
        **bound["sources"],
    }
    return result, bound, sources


def metrics(table: pa.Table) -> dict:
    labels = table["label"].to_numpy()
    full = binary_metrics(labels, table["full_logit"].to_numpy())
    reuse = binary_metrics(labels, table["reuse_logit"].to_numpy())
    gap = None if full["ROC_AUC"] is None else 100 * (full["ROC_AUC"] - reuse["ROC_AUC"])
    history, appends = table["history_length"].to_numpy(), table["append_count_since_cutover"].to_numpy()
    return {
        "users": len(np.unique(table["uid"].to_numpy())), "requests": len(table),
        "positive_requests": int(labels.sum()), "negative_requests": int(len(labels) - labels.sum()),
        "current_full": full, "current_reuse": reuse, "full_minus_reuse_auc_pp": gap,
        "first_query_timestamp": int(pc.min(table["query_timestamp"]).as_py()),
        "last_query_timestamp": int(pc.max(table["query_timestamp"]).as_py()),
        "history_length": {key: float(np.quantile(history, q)) for key, q in (("min", 0), ("p50", .5), ("p90", .9), ("max", 1))},
        "append_count_since_cutover": {key: float(np.quantile(appends, q)) for key, q in (("min", 0), ("p50", .5), ("p90", .9), ("max", 1))},
    }


def user_metadata(table: pa.Table, scores: dict[int, float]) -> pa.Table:
    rows = []
    uids = table["uid"].to_numpy()
    unique, starts, counts = np.unique(uids, return_index=True, return_counts=True)
    for uid, start, count in zip(unique, starts, counts, strict=True):
        user = table.slice(int(start), int(count))
        positives = int(pc.sum(user["label"]).as_py())
        rows.append({
            "uid": int(uid), "requests": int(count), "positive_requests": positives,
            "negative_requests": int(count) - positives, "selection_score": scores.get(int(uid)),
            "first_query_timestamp": int(pc.min(user["query_timestamp"]).as_py()),
            "last_query_timestamp": int(pc.max(user["query_timestamp"]).as_py()),
            "history_length_min": int(pc.min(user["history_length"]).as_py()),
            "history_length_max": int(pc.max(user["history_length"]).as_py()),
            "append_count_since_cutover_max": int(pc.max(user["append_count_since_cutover"]).as_py()),
            "rolling_evictions_max": int(pc.max(user["rolling_evictions"]).as_py()),
        })
    return pa.Table.from_pylist(rows)


def write_parquet(path: Path, table: pa.Table) -> dict:
    partial = path.with_suffix(".parquet.partial")
    pq.write_table(table, partial, compression="zstd")
    partial.replace(path)
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "rows": len(table)}


def prepare(
    scale: str, edge: int, output_root: Path = DEFAULT_OUTPUT, *, users: int = 3000,
    calibration_users: int = 32, canary_users: int = 16,
) -> dict:
    if scale not in {"medium", "large", "max"} or edge not in range(1, 6):
        raise ValueError("scale must be medium/large/max and edge 1..5")
    output = output_root / scale / f"v{edge-1}_to_v{edge}"
    selection = {
        "rule": RULE, "evaluation_users": users, "calibration_users": calibration_users,
        "canary_users": canary_users, "uid_hash_namespace": HASH_NAMESPACE,
        "reserve_controls_first": True,
        "reference_pool": "all remaining source users after hash reservation",
        "user_score": "mean of N/(2*N_class) times the Full-minus-Reuse opposite-label pairwise concordance",
        "sort": "descending user score; SHA256(namespace/uid), then uid for ties",
        "score_transform": "sigmoid (same ROC AUC definition as retained binary_metrics)",
        "selection_passes": 1, "all_selected_user_requests_retained": True,
        "scope": "outcome-conditioned diagnostic subset; not a representative population estimate",
    }
    existing = output / "binding.json"
    if existing.exists():
        record = json.loads(existing.read_text())
        if record["selection"] != selection:
            raise RuntimeError("a different cohort recipe cannot overwrite an already frozen panel")
        for key in ("reuse_binding", "reuse_summary"):
            source = record["sources"][key]
            checked_source(ROOT / source["path"], source["sha256"])
        for source in [record["users_file"], *record["files"].values()]:
            checked_source(ROOT / source["path"], source["sha256"])
        return record

    table, bound, sources = paired_population(scale, edge)
    selected, scores = select_users(
        table["uid"].to_numpy(), table["label"].to_numpy(),
        sigmoid(table["full_logit"].to_numpy()), sigmoid(table["reuse_logit"].to_numpy()),
        evaluation_users=users, calibration_users=calibration_users, canary_users=canary_users,
    )
    population = metrics(table)
    source_summary = json.loads((ROOT / sources["reuse_summary"]["path"]).read_text())
    for name in ("current_full", "current_reuse"):
        if abs(population[name]["ROC_AUC"] - source_summary[name]["ROC_AUC"]) > 1e-10:
            raise RuntimeError(f"source population {name} AUC does not reproduce its adjudication")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "users.json", selected)
    split_metrics, files = {}, {}
    for split, uids in selected.items():
        subset = table.filter(pc.is_in(table["uid"], value_set=pa.array(uids, type=pa.int64())))
        subset = subset.sort_by([("uid", "ascending"), ("query_timestamp", "ascending"), ("request_id", "ascending")])
        split_metrics[split] = metrics(subset)
        split_metrics[split]["population_user_fraction"] = len(uids) / population["users"]
        split_metrics[split]["population_request_fraction"] = len(subset) / population["requests"]
        files[f"{split}_requests"] = write_parquet(output / f"{split}_requests.parquet", subset)
        files[f"{split}_users"] = write_parquet(output / f"{split}_users.parquet", user_metadata(subset, scores))
    chosen_scores = [scores[uid] for uid in selected["evaluation"]]
    record = {
        "status": "frozen", "scale": scale, "edge": bound["edge"], "edge_index": edge,
        "days": bound["days"], "cutover": bound["days"][0] * DAY,
        "full_current_name": bound["full_current_name"], "selection": selection,
        "sources": sources, "source_checkpoint_hashes_verified_here": False,
        "source_checkpoint_note": "hashes inherited from completed Reuse binding; execution preflight verifies weights",
        "population": population, "panels": split_metrics,
        "evaluation_selection_score": {"min": min(chosen_scores), "max": max(chosen_scores),
                                       "mean": float(np.mean(chosen_scores))},
        "users_file": {"path": str((output / "users.json").relative_to(ROOT)), "sha256": sha256(output / "users.json")},
        "files": files,
        "preparation_sources": {str(path.relative_to(ROOT)): sha256(path) for path in (
            Path(__file__), Path(__file__).with_name("cohort.py"),
        )},
        "checks": ["source Full raw seal and Reuse shard hashes verified",
                   "Full, Reuse, requests and labels share every request ID, UID and timestamp",
                   "source population Full and Reuse AUC reproduce their recorded summary",
                   "evaluation, calibration and canary UID sets are disjoint",
                   "each subset preserves every scored request for its selected users",
                   "no baseline outcome used and no resampling based on subset gap"],
    }
    write_json(output / "binding.json", record)
    write_json(output / "summary.json", record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=["medium", "large", "max"])
    parser.add_argument("--edge", type=int, choices=range(1, 6))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--users", type=int, default=3000)
    parser.add_argument("--calibration-users", type=int, default=32)
    parser.add_argument("--canary-users", type=int, default=16)
    args = parser.parse_args()
    if args.all:
        edges = [(scale, edge) for scale in ("medium", "large", "max") for edge in range(1, 6)]
    elif args.scale and args.edge:
        edges = [(args.scale, args.edge)]
    else:
        parser.error("provide --all, or both --scale and --edge")
    for scale, edge in edges:
        record = prepare(scale, edge, args.output_root, users=args.users,
                         calibration_users=args.calibration_users, canary_users=args.canary_users)
        print(json.dumps({"scale": scale, "edge": record["edge"], **record["panels"]["evaluation"]}), flush=True)


if __name__ == "__main__":
    main()
