#!/usr/bin/env python3
"""Compare immutable static/no-cache encoder costs with actual GPU counters."""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[4]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]

import numpy as np

from hstu_kvcache.evaluation.release_context import RELEASES
from hstu_kvcache.evaluation.panels import load_history, load_panel
from experiments.archive.partial_refresh.cost import summary_flops
from experiments.archive.partial_refresh.round2.factored_cost import (
    FOLD_FLOPS, ITEM_PROJECTION_FLOPS, ROW_KV_FLOPS, simulate_factored,
)
from hstu_kvcache.common.artifacts import sha256, write_json


def verify(summary_path, output):
    summary_path = Path(summary_path)
    summary = json.loads(summary_path.read_text())
    configuration_path = summary_path.parent / "configuration.json"
    configuration = json.loads(configuration_path.read_text())
    strategy = summary["exploration_strategy"]
    capacity = int(strategy.get("cache_capacity", 0) or 0)
    static_file = strategy.get("static_item_file")
    if strategy["encoder"] != "factored" or (capacity and not static_file):
        raise ValueError("this exact check covers static item tables and no persistent cache")
    if strategy.get("selection", "tail") != "tail" or strategy.get("min_partial_length"):
        raise ValueError("this check currently covers fixed per-release tail ratios")
    panel_path = Path(configuration["quality_panel"])
    panel, _, requests = load_panel(panel_path)
    histories = load_history(panel_path, requested_uids=[unit["uid"] for unit in summary["units"]])
    rhos = {version: float(strategy.get("long_rho", {}).get(version, 1.)) for version, _ in RELEASES}
    warm_masks, table_sizes = {}, {}
    sources = [summary_path, configuration_path, panel_path / "binding.json", Path(__file__)]
    tables = np.load(static_file) if static_file else None
    if static_file:
        sources.append(Path(static_file))
    for version, _ in RELEASES:
        mask = np.zeros(panel["model_config"]["num_items"], dtype=bool)
        ids = tables[f"{version}_{capacity}"] if tables is not None else np.array([], dtype=np.int64)
        mask[ids] = True
        warm_masks[version], table_sizes[version] = {capacity: mask}, len(ids)
    if tables is not None:
        tables.close()
    origin_by_uid = {}
    source_runs = summary.get("sources", {}).get("runs", [])
    for source_run in source_runs:
        record = source_run["summary"]
        path = Path(record["path"])
        if sha256(path) != record["sha256"]:
            raise RuntimeError("aggregate constituent summary changed")
        source_summary = json.loads(path.read_text())
        for unit in source_summary["units"]:
            if unit["uid"] in origin_by_uid:
                raise RuntimeError("aggregate constituent UID repeats")
            origin_by_uid[unit["uid"]] = str(path)
        sources.append(path)
    prepared = set()
    comparisons = 0
    for unit in summary["units"]:
        uid = int(unit["uid"])
        simulated = simulate_factored(histories.rows[uid], requests[uid], rhos, warm_masks)
        for version, row in unit["releases"].items():
            c = simulated[version]["counts"]
            cache = simulated[version]["cache_counts"][capacity]
            first = c.get("first_read_encode_rows", 0)
            append = c.get("append_encode_rows", 0)
            rows = first + append
            preparation_key = origin_by_uid.get(uid, "single_run"), version
            first_constructor = rows > 0 and preparation_key not in prepared
            static_items = table_sizes[version] if first_constructor else 0
            misses = cache.get("unique_item_misses", 0)
            hits = cache.get("unique_item_hits", 0)
            masks = rows * 288 if summary["storage_mode"] == "masked" else 0
            missing = c.get("initial_encode_rows", 0)
            shaped = dict(c, refreshed_rows=first, append_encoded_rows=append)
            work = summary_flops(shaped, storage_mode=summary["storage_mode"])
            expected = dict(initial_encode_rows=missing,
                refresh_encode_rows=c.get("refresh_encode_rows", 0), append_encode_rows=append,
                initial_encode_flops=missing * ROW_KV_FLOPS,
                refresh_encode_flops=c.get("refresh_encode_rows", 0) * ROW_KV_FLOPS,
                append_encode_flops=append * ROW_KV_FLOPS,
                encoder_fold_preparation_flops=FOLD_FLOPS if first_constructor else 0,
                static_precomputed_items=static_items,
                item_projection_flops=(static_items + misses) * ITEM_PROJECTION_FLOPS,
                item_lookup_bytes=(static_items + misses) * 192 * 4,
                hits=hits, misses=misses,
                support_mask_flops=masks,
                support_mask_transfer_bytes=rows * 6 if summary["storage_mode"] == "masked" else 0,
                support_token_layers=c.get("support_token_layers", 0),
                forced_missing_rows=c.get("forced_missing_rows", 0), forced_older_rows=c.get("forced_older_rows", 0),
                retained_previous_rows=c.get("retained_previous_rows", 0),
                corrected_requests=c.get("corrected_requests", 0), native_requests=c.get("native_requests", 0),
                extra_history_read_flops=work["extra_query_flops"] - work["components_flops"].get("query_support_mask", 0))
            if summary["storage_mode"] == "raw":
                expected.update(read_support_mask_flops=work["components_flops"]["query_support_mask"],
                    read_support_mask_transfer_bytes=c.get("corrected_token_read_batches", 0) * 6,
                    raw_masked_read_batches=c.get("corrected_read_batches", 0))
            actual = row["counters"]
            errors = {key: (value, actual.get(key, 0)) for key, value in expected.items()
                if value != actual.get(key, 0)}
            if errors:
                raise RuntimeError(f"factored GPU counter mismatch {uid}/{version}: {errors}")
            if first_constructor:
                prepared.add(preparation_key)
            comparisons += len(expected)
    result = dict(status="complete", role="Exact factored metadata/GPU arithmetic check",
        users=summary["users"], user_releases=3 * summary["users"], scalar_counter_comparisons=comparisons,
        all_row_item_miss_static_preparation_mask_query_counters_exact=True,
        static_items_requested=capacity, static_table=static_file, rho_by_release=rhos,
        storage_mode=summary["storage_mode"],
        independent_preparation_runs=len(source_runs) if source_runs else 1,
        sources={str(path.resolve()): sha256(path) for path in sources})
    write_json(Path(output), result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.summary, args.output)), flush=True)
