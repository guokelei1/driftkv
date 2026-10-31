#!/usr/bin/env python3
"""Exact static-item affine preparation/maintenance cost on fixed30k events.

Frozen item tables use pre-release fit histories only. Uncached IDs are
projected once per encode call and never admitted, so costs are independent
of cross-user replay order. Every static projection and normalization fold
is charged; persistent shared memory is separate from each user's codes.
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[4]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[4]

import numpy as np

from hstu_kvcache.evaluation.release_context import END, RELEASES
from experiments.archive.partial_refresh.cost import (
    CALIBRATION_FLOPS, REQUEST_MANIFEST, SOURCE_RECONSTRUCTION_SENSITIVITY_FLOPS, summary_flops,
)

START = RELEASES[0][1]
ROW_KV_FLOPS = 37_440
ITEM_PROJECTION_FLOPS = 110_880
FOLD_FLOPS = 221_184
REFERENCE = ROOT / "results/partial_refresh_2026_10/population_cost_raw/summary.json"
ITEM_TABLES = ROOT / "results/partial_refresh_2026_10/round2/item_tables/item_ids.npz"
TABLE_BINDING = ITEM_TABLES.with_name("summary.json")


def add_nocache(summary_path):
    """Derive no-cache costs from already retained per-call unique counts."""
    from copy import deepcopy
    from hstu_kvcache.common.artifacts import sha256, write_json
    summary_path = Path(summary_path)
    report = json.loads(summary_path.read_text())
    reference = json.loads(REFERENCE.read_text())
    baseline = reference["policies"]["original_masked_full_reference"]
    result = deepcopy(report["policies"]["full_static4096"])
    result["static_items_requested"] = 0
    for row in result["by_release"].values():
        cache = row["cache_counts"]
        first_unique = cache["first_read_unique_item_hits"] + cache["first_read_unique_item_misses"]
        append_unique = cache["append_unique_item_hits"] + cache["append_unique_item_misses"]
        row["cache_counts"] = dict(first_read_unique_item_hits=0, first_read_unique_item_misses=first_unique,
            append_unique_item_hits=0, append_unique_item_misses=append_unique,
            unique_item_hits=0, unique_item_misses=first_unique + append_unique, row_item_hits=0)
        components = row["maintenance_components_flops"]
        components["first_read_uncached_items"] = first_unique * ITEM_PROJECTION_FLOPS
        components["append_uncached_items"] = append_unique * ITEM_PROJECTION_FLOPS
        components["static_table_preparation"] = 0
        row["shared_projection_table_bytes"] = 0
        row["maintenance_flops"] = sum(components.values())
    maintenance = sum(row["maintenance_flops"] for row in result["by_release"].values())
    masks = sum(row["maintenance_components_flops"]["encoder_support_mask"] for row in result["by_release"].values())
    query = sum(row["query_extra_flops"] for row in result["by_release"].values())
    complete = CALIBRATION_FLOPS + maintenance + query
    result.update(maintenance_flops=maintenance,
        maintenance_saving_percent=100 * (report["baseline_maintenance_flops"] - maintenance) / report["baseline_maintenance_flops"],
        encoder_only_saving_percent=100 * (report["baseline_encoder_flops"] - (maintenance - masks)) / report["baseline_encoder_flops"],
        complete_added_flops=complete,
        complete_percent_of_once_rebuild=100 * complete / reference["once_rebuild_flops"],
        complete_saving_percent_vs_original_masked=100 * (baseline["complete_added_flops"] - complete) / baseline["complete_added_flops"],
        source_reconstruction_sensitivity_percent=100 * (complete + SOURCE_RECONSTRUCTION_SENSITIVITY_FLOPS) / reference["once_rebuild_flops"],
        cache_scope="no persistent item table; projection shared only among equal IDs within each actual encode call",
        evidence="Derived from exact fixed30k full-rho1 encode call counts already present in this summary, without rereading population data.")
    report["policies"]["full_static0"] = result
    report["nocache_derivation_sources"] = {str(Path(__file__).resolve()): sha256(Path(__file__))}
    write_json(summary_path, report)
    return {key: result[key] for key in ("maintenance_saving_percent", "encoder_only_saving_percent",
        "complete_percent_of_once_rebuild", "complete_saving_percent_vs_original_masked")}


def add_replica_sensitivity(summary_path, replicas=4):
    """Repeat only immutable table/fold construction, never D2 or row work."""
    from hstu_kvcache.common.artifacts import sha256, write_json
    summary_path = Path(summary_path)
    report = json.loads(summary_path.read_text())
    reference = json.loads(REFERENCE.read_text())
    baseline = reference["policies"]["original_masked_full_reference"]
    baseline_complete = baseline["complete_added_flops"]
    denominator = reference["once_rebuild_flops"]
    audit_path = ROOT / "results/partial_refresh_2026_10/validation_raw_strict/cost_audit.json"
    audit = json.loads(audit_path.read_text())
    for result in report["policies"].values():
        preparation = sum(row["maintenance_components_flops"]["static_table_preparation"] +
            row["maintenance_components_flops"]["fixed_normalization_fold"]
            for row in result["by_release"].values())
        extra = (replicas - 1) * preparation
        maintenance = result["maintenance_flops"] + extra
        complete = result["complete_added_flops"] + extra
        fp_mask = sum(row["maintenance_components_flops"]["encoder_support_mask"]
            for row in result["by_release"].values())
        per_release_memory = {version: dict(
            item_table_payload_bytes_per_replica=row["shared_projection_table_bytes"],
            folded_parameter_payload_bytes_per_replica=row["folded_parameter_bytes"],
            item_table_payload_bytes_all_replicas=row["shared_projection_table_bytes"] * replicas,
            folded_parameter_payload_bytes_all_replicas=row["folded_parameter_bytes"] * replicas)
            for version, row in result["by_release"].items()}
        sensitivity = dict(replicas=replicas,
            repeated_construction_components="fixed_normalization_fold + static_table_preparation only",
            one_replica_preparation_flops=preparation,
            extra_independent_preparation_flops=extra,
            maintenance_flops=maintenance,
            maintenance_saving_percent=100 * (report["baseline_maintenance_flops"] - maintenance) / report["baseline_maintenance_flops"],
            encoder_only_saving_percent=100 * (report["baseline_encoder_flops"] - (maintenance - fp_mask)) / report["baseline_encoder_flops"],
            complete_added_flops=complete,
            complete_percent_of_once_rebuild=100 * complete / denominator,
            complete_saving_percent_vs_original_masked=100 * (baseline_complete - complete) / baseline_complete,
            source_reconstruction_sensitivity_percent=100 * (complete + SOURCE_RECONSTRUCTION_SENSITIVITY_FLOPS) / denominator,
            shared_memory_payload_by_release=per_release_memory,
            all_three_releases_item_table_payload_bytes_per_replica=sum(row["shared_projection_table_bytes"] for row in result["by_release"].values()),
            all_three_releases_folded_parameter_payload_bytes_per_replica=sum(row["folded_parameter_bytes"] for row in result["by_release"].values()),
            memory_scope="Logical persistent tensor payload, not loaded-serving peak measurement. CPU ID dictionaries, per-call projection/gather scratch and allocator overhead are separate. A serving Current release needs its own table; the offline user-major model-chain prototype retains three release objects concurrently.")
        if result["storage_mode"] == "raw":
            sensitivity["complete_with_all_sources_no_reuse_percent"] = 100 * (
                complete + SOURCE_RECONSTRUCTION_SENSITIVITY_FLOPS + audit["publication_native_source_flops"]) / denominator
        result["independent_preparation_sensitivity"] = {str(replicas): sensitivity}
    report["primary_reporting_preparation_replicas"] = replicas
    report["replica_sensitivity_derivation_sources"] = {
        str(Path(__file__).resolve()): sha256(Path(__file__)), str(audit_path): sha256(audit_path)}
    write_json(summary_path, report)
    return {name: {key: result["independent_preparation_sensitivity"][str(replicas)][key]
        for key in ("maintenance_saving_percent", "encoder_only_saving_percent",
            "complete_percent_of_once_rebuild", "source_reconstruction_sensitivity_percent")}
        for name, result in report["policies"].items()}


def simulate_factored(events, requests, rhos, warm_masks, *, cap=1024, layers=6):
    """Count the actual first-read and native-band encode calls."""
    timestamps, item_ids = events[:2]
    cursor = int(np.searchsorted(timestamps, START, side="left"))
    length = min(cursor, cap)
    tags = None
    by_version = defaultdict(Counter)
    for request in requests:
        by_version[request["version"]][int(request["query_timestamp"])] += 1
    releases = {}
    for index, (version, at) in enumerate(RELEASES):
        current = int(version[1:])
        until = RELEASES[index + 1][1] if index + 1 < len(RELEASES) else END
        pending, active = length > 0, False
        remaining = layers * (cap - 1) + 1 if pending else 0
        counters = Counter()
        cache_counts = {size: Counter() for size in warm_masks[version]}
        if tags is None and pending:
            tags = np.full(length, -1, dtype=np.int32)

        def encode(ids, role):
            unique = np.unique(ids)
            counters[role + "_encode_rows"] += len(ids)
            counters[role + "_encode_calls"] += 1
            counters["unique_items_in_calls"] += len(unique)
            for size, mask in warm_masks[version].items():
                hits = int(mask[unique].sum())
                misses = len(unique) - hits
                cache_counts[size][role + "_unique_item_hits"] += hits
                cache_counts[size][role + "_unique_item_misses"] += misses
                cache_counts[size]["unique_item_hits"] += hits
                cache_counts[size]["unique_item_misses"] += misses
                cache_counts[size]["row_item_hits"] += int(mask[ids].sum())

        def advance(timestamp):
            nonlocal cursor, length, pending, active, remaining, tags
            stop = int(np.searchsorted(timestamps, timestamp, side="left"))
            while cursor < stop:
                width = 1 << (min(32, stop - cursor).bit_length() - 1)
                cursor += width
                length = min(length + width, cap)
                counters["support_token_layers"] += width * layers
                if pending or active:
                    remaining -= width
                    if remaining <= 0:
                        counters["causal_retirements"] += 1
                        pending = active = False
                        tags = None
                    elif active:
                        encode(item_ids[cursor - width:cursor], "append")
                        tags = np.concatenate((tags, np.full(width, current, dtype=np.int32)))[-length:]
                    else:
                        tags = np.concatenate((tags, np.full(width, -1, dtype=np.int32)))[-length:]

        for timestamp, count in sorted(by_version[version].items()):
            if not at <= timestamp < until:
                raise RuntimeError("request outside release")
            advance(timestamp)
            if pending:
                missing = tags < 0
                older = (tags >= 0) & (tags < current - 1)
                tail = np.arange(length) >= length - int(np.ceil(float(rhos[version]) * length))
                selected = (missing | older | tail) & (tags != current)
                counters["first_reads"] += 1
                counters["first_read_rows"] += length
                counters["forced_missing_rows"] += int(missing.sum())
                counters["forced_older_rows"] += int(older.sum())
                counters["retained_previous_rows"] += int(((tags == current - 1) & ~selected).sum())
                ids = item_ids[cursor - length:cursor][selected]
                if len(ids):
                    # Deduplication occurs across the whole refresh call,
                    # including missing and nonmissing selected rows together.
                    encode(ids, "first_read")
                counters["initial_encode_rows"] += int(missing.sum())
                counters["refresh_encode_rows"] += int(selected.sum()) - int(missing.sum())
                tags[selected] = current
                pending, active = False, True
            if active:
                counters["corrected_requests"] += count
                counters["corrected_token_query_pairs"] += count * length
                batches = (count + 15) // 16
                counters["corrected_read_batches"] += batches
                counters["corrected_token_read_batches"] += batches * length
            else:
                counters["native_requests"] += count
        advance(until)
        releases[version] = dict(counts=dict(counters), cache_counts={size: dict(c) for size, c in cache_counts.items()})
    return releases


def run(args):
    import pyarrow.parquet as pq
    import torch
    from hstu_kvcache.evaluation.history import load_histories_semijoin
    from hstu_kvcache.common.artifacts import sha256, write_json

    began = time.perf_counter()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(args.threads)
    reference = json.loads(REFERENCE.read_text())
    baseline = reference["policies"]["original_masked_full_reference"]
    table_binding = json.loads(TABLE_BINDING.read_text())
    if sha256(ITEM_TABLES) != table_binding["item_ids"]["sha256"]:
        raise RuntimeError("frozen static admission table changed")
    config_path = ROOT / "configs/design/medium_v0_v5_development.json"
    configuration = json.loads(config_path.read_text())
    dataset_path = ROOT / configuration["data"]["dataset"]["path"]
    population_path = ROOT / configuration["data"]["population"]["path"]
    dataset = json.loads(dataset_path.read_text())
    panel_path = ROOT / "results/design_two_2026_10/benchmark_panel/binding.json"
    panel = json.loads(panel_path.read_text())
    for role, path in (("dataset", dataset_path), ("population", population_path)):
        if sha256(path) != configuration["data"][role]["sha256"]:
            raise RuntimeError("frozen population input changed")
    native_manifest = json.loads(REQUEST_MANIFEST.with_name("manifest.json").read_text())
    if sha256(REQUEST_MANIFEST) != native_manifest["artifacts"][REQUEST_MANIFEST.name]["sha256"]:
        raise RuntimeError("request manifest changed")
    uids = sorted(pq.read_table(population_path, columns=["uid"])["uid"].to_pylist())
    if len(uids) != 30_000 or len(set(uids)) != 30_000:
        raise RuntimeError("fixed30k membership changed")
    sizes = (4096, 16384, 32768)
    warm_masks, table_sizes = {}, {}
    with np.load(ITEM_TABLES) as arrays:
        for version, _ in RELEASES:
            warm_masks[version], table_sizes[version] = {}, {}
            for size in sizes:
                ids = arrays[f"{version}_{size}"]
                mask = np.zeros(panel["model_config"]["num_items"], dtype=bool)
                mask[ids] = True
                warm_masks[version][size] = mask
                table_sizes[version][size] = len(ids)
    selected = set(uids)
    requests, seen = defaultdict(list), set()
    table = pq.read_table(REQUEST_MANIFEST,
        filters=[("time_block", "=", "matrix_horizon"), ("target_known", "=", True),
            ("query_timestamp", ">=", START), ("query_timestamp", "<", END)],
        columns=["request_id", "uid", "query_timestamp"])
    for row in table.to_pylist():
        if row["uid"] not in selected:
            continue
        if row["request_id"] in seen:
            raise RuntimeError("request identity repeats")
        seen.add(row["request_id"])
        version = next(name for name, at in reversed(RELEASES) if row["query_timestamp"] >= at)
        requests[row["uid"]].append({**row, "version": version})
    if len(seen) != reference["requests"]:
        raise RuntimeError("fixed30k request count changed")
    active = len(requests)
    histories = load_histories_semijoin(uids, dataset_path=dataset_path,
        known_vocab_size=dataset["foundation_items"],
        oov_buckets=panel["model_config"]["num_items"] - dataset["foundation_items"],
        start_timestamp=START, end_timestamp=END, max_history=1024, threads=args.threads)
    print(json.dumps(dict(status="metadata_loaded", users=len(uids), requests=len(seen),
        elapsed_seconds=time.perf_counter() - began)), flush=True)
    rhos = dict(full={v: 1.0 for v, _ in RELEASES}, strict={"v3": 0., "v4": .875, "v5": .75})
    counts = {name: {v: Counter() for v, _ in RELEASES} for name in rhos}
    cache_counts = {name: {v: {size: Counter() for size in (sizes if name == "full" else (16384,))}
        for v, _ in RELEASES} for name in rhos}
    for number, uid in enumerate(uids):
        events = histories.rows.get(uid, (np.array([], dtype=np.int64), np.array([], dtype=np.int64)))
        for name, fractions in rhos.items():
            masks = warm_masks if name == "full" else {v: {16384: masks[16384]} for v, masks in warm_masks.items()}
            simulated = simulate_factored(events, requests.get(uid, []), fractions, masks)
            for version, row in simulated.items():
                counts[name][version].update(row["counts"])
                for size, values in row["cache_counts"].items():
                    cache_counts[name][version][size].update(values)
        if (number + 1) % 5000 == 0:
            print(json.dumps(dict(status="counting", users=number + 1,
                elapsed_seconds=time.perf_counter() - began)), flush=True)
    for name, baseline_name in (("full", "original_masked_full_reference"), ("strict", "raw_strict")):
        for version, c in counts[name].items():
            old = reference["policies"][baseline_name]["by_release"][version]["counts"]
            mapping = dict(first_read_encode_rows="refreshed_rows", append_encode_rows="append_encoded_rows",
                corrected_requests="corrected_requests", corrected_token_query_pairs="corrected_token_query_pairs",
                forced_missing_rows="forced_missing_rows", forced_older_rows="forced_older_rows")
            for new_key, old_key in mapping.items():
                if c.get(new_key, 0) != old.get(old_key, 0):
                    raise RuntimeError(f"factored event shapes differ from retained reference: {name}/{version}/{new_key}")
    baseline_encoder = sum(row["total_encoder_flops"] for row in baseline["by_release"].values())
    baseline_mask = sum(row["components_flops"]["support_mask"] for row in baseline["by_release"].values())
    baseline_maintenance = baseline_encoder + baseline_mask
    results = {}
    strict_audit_path = ROOT / "results/partial_refresh_2026_10/validation_raw_strict/cost_audit.json"
    strict_audit = json.loads(strict_audit_path.read_text())
    for name, fractions in rhos.items():
        mode = "masked" if name == "full" else "raw"
        for size in (sizes if name == "full" else (16384,)):
            releases = {}
            for version, c in counts[name].items():
                cache = cache_counts[name][version][size]
                encoder_rows = c["first_read_encode_rows"] + c["append_encode_rows"]
                components = dict(initial_row_kv=c["initial_encode_rows"] * ROW_KV_FLOPS,
                    refresh_row_kv=c["refresh_encode_rows"] * ROW_KV_FLOPS,
                    append_row_kv=c["append_encode_rows"] * ROW_KV_FLOPS,
                    first_read_uncached_items=cache["first_read_unique_item_misses"] * ITEM_PROJECTION_FLOPS,
                    append_uncached_items=cache["append_unique_item_misses"] * ITEM_PROJECTION_FLOPS,
                    static_table_preparation=table_sizes[version][size] * ITEM_PROJECTION_FLOPS,
                    fixed_normalization_fold=FOLD_FLOPS,
                    encoder_support_mask=encoder_rows * 288 if mode == "masked" else 0)
                shaped = dict(c, refreshed_rows=c["first_read_encode_rows"], append_encoded_rows=c["append_encode_rows"])
                query = summary_flops(shaped, storage_mode=mode)["extra_query_flops"]
                releases[version] = dict(counts=dict(c), cache_counts=dict(cache),
                    maintenance_components_flops=components, maintenance_flops=sum(components.values()),
                    query_extra_flops=query,
                    shared_projection_table_bytes=table_sizes[version][size] * 1152,
                    folded_parameter_bytes=73_728 * 4 + 288 * 4,
                    flag_transfer_bytes=(encoder_rows if mode == "masked" else c["corrected_token_read_batches"]) * 6)
            maintenance = sum(row["maintenance_flops"] for row in releases.values())
            query = sum(row["query_extra_flops"] for row in releases.values())
            publication = 0 if name == "full" else strict_audit["complete_shared_validation_flops"]
            complete = CALIBRATION_FLOPS + maintenance + query + publication
            results[f"{name}_static{size}"] = dict(rho_by_release=fractions, storage_mode=mode,
                static_items_requested=size, by_release=releases,
                baseline_maintenance_flops=baseline_maintenance,
                maintenance_flops=maintenance,
                maintenance_saving_percent=100 * (baseline_maintenance - maintenance) / baseline_maintenance,
                encoder_only_saving_percent=100 * (baseline_encoder - (maintenance - sum(r["maintenance_components_flops"]["encoder_support_mask"] for r in releases.values()))) / baseline_encoder,
                shared_calibration_flops=CALIBRATION_FLOPS,
                publication_validation_flops=publication,
                complete_added_flops=complete,
                complete_percent_of_once_rebuild=100 * complete / reference["once_rebuild_flops"],
                complete_saving_percent_vs_original_masked=100 * (baseline["complete_added_flops"] - complete) / baseline["complete_added_flops"],
                source_reconstruction_sensitivity_percent=100 * (complete + SOURCE_RECONSTRUCTION_SENSITIVITY_FLOPS) / reference["once_rebuild_flops"],
                cache_scope="one release-shared static table; uncached unique items recomputed once per actual encode call and never admitted",
                quality_scope="same encoder function up to floating-point ordering; existing full/strict quality is a reference, new GPU equality/quality evidence separate")
            if name == "strict":
                results[f"{name}_static{size}"]["complete_with_all_sources_no_reuse_percent"] = 100 * (
                    complete + SOURCE_RECONSTRUCTION_SENSITIVITY_FLOPS + strict_audit["publication_native_source_flops"]) / reference["once_rebuild_flops"]
    report = dict(status="complete", role="Exact fixed30k static Current-item factorization maintenance cost",
        users=len(uids), active_request_users=active, requests=len(seen),
        baseline_maintenance_flops=baseline_maintenance, baseline_encoder_flops=baseline_encoder,
        baseline_encoder_support_mask_flops=baseline_mask,
        cost_convention=dict(row_kv_flops=ROW_KV_FLOPS, unique_item_projection_flops=ITEM_PROJECTION_FLOPS,
            per_release_fold_flops=FOLD_FLOPS, fp32_projection_bytes_per_item=1152),
        policies=results,
        verification=dict(full_and_strict_row_shapes_match_retained_fixed30k_ledgers=True,
            reads_future_labels_or_scores=False, static_admission_from_fit_only_prerelease_histories=True),
        sources={str(path.resolve()): sha256(path) for path in (Path(__file__), REFERENCE,
            ITEM_TABLES, TABLE_BINDING, config_path, panel_path, dataset_path, population_path,
            REQUEST_MANIFEST, strict_audit_path)},
        elapsed_seconds=time.perf_counter() - began,
        notes=["Maintenance denominator includes V3 initialization, every first-read refresh, active appends and original encode support masks.",
            "One-time static preparation, normalization folding and every uncached unique projection are charged in maintenance.",
            "Shared Current-item table memory and folded parameters are additional to unchanged per-user rank8 codes/flags/version tags.",
            "Default cost assumes one shared table per release. Independent serving replicas repeat table preparation and memory; uncached computations depend on their actual encode calls.",
            "Fixed D2 is fully charged. Strict also retains its original complete publication-validation cost; source reconstruction sensitivities remain separate.",
            "Static IDs are selected before release from fit-only histories. No user-major LRU-hit rate is claimed."])
    write_json(args.output / "summary.json", report)
    print(json.dumps(dict(status="complete", output=str(args.output), policies={name: {
        key: result[key] for key in ("maintenance_saving_percent", "encoder_only_saving_percent",
            "complete_percent_of_once_rebuild", "complete_saving_percent_vs_original_masked")}
        for name, result in results.items()}, elapsed_seconds=report["elapsed_seconds"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/partial_refresh_2026_10/round2/factored_cost")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--add-nocache-to-summary", type=Path)
    parser.add_argument("--add-replica-sensitivity", type=Path)
    parser.add_argument("--replicas", type=int, default=4)
    args = parser.parse_args()
    if args.add_replica_sensitivity:
        print(json.dumps(add_replica_sensitivity(args.add_replica_sensitivity, args.replicas)), flush=True)
    elif args.add_nocache_to_summary:
        print(json.dumps(add_nocache(args.add_nocache_to_summary)), flush=True)
    else:
        run(args)
