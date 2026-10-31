#!/usr/bin/env python3
"""Metadata-only fixed30k cost screen for per-layer support-sparse encoding.

No models, GPU, labels or scores are read. The exact native event trace and
actual first-read/FIFO/version policy determine the encoder rows. New partial
policies' publication validation is unknown, and is never silently free.
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
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[4]

import numpy as np

from experiments.archive.partial_refresh.cost import (
    CALIBRATION_FLOPS, REQUEST_MANIFEST, SOURCE_RECONSTRUCTION_SENSITIVITY_FLOPS, summary_flops,
)
from hstu_kvcache.evaluation.release_context import END, RELEASES
from hstu_kvcache.methods.cachetide.lifecycle.lifecycle import InheritedSupport

START = RELEASES[0][1]
LAYER_ROW_FLOPS = 27_744
REFERENCE = ROOT / "results/partial_refresh_2026_10/population_cost_raw/summary.json"
POLICIES = {
    "sparse_full": dict(threshold=1024, v4=1.0, v5=1.0),
    "short768_long50_25": dict(threshold=768, v4=.5, v5=.25),
    "short768_long25_25": dict(threshold=768, v4=.25, v5=.25),
    "short768_long50_0": dict(threshold=768, v4=.5, v5=0.0),
    "onlyfullcap_long25_25": dict(threshold=1023, v4=.25, v5=.25),
}


def support_flags(length, appended, *, cap=1024, layers=6, inherited=True):
    """The row's append index g must not exceed layer*(cap-1)."""
    if not inherited:
        return np.zeros((layers, length), dtype=bool)
    g = appended - length + np.arange(1, length + 1)
    return g[None, :] <= (np.arange(layers) * (cap - 1))[:, None]


def verify_support_formula():
    checked = 0
    for cap, layers in ((4, 1), (4, 3), (7, 2), (7, 6)):
        for initial in (0, 1, cap):
            tracker = InheritedSupport(layers, initial, cap, inherited=bool(initial))
            appended = 0
            for width in (0, 1, 2, 1, 3, 2, 7, 11, 13):
                tracker.append(width)
                appended += width
                wanted = support_flags(tracker.flags.shape[1], appended, cap=cap,
                    layers=layers, inherited=bool(initial))
                if not np.array_equal(wanted, tracker.flags):
                    raise RuntimeError(f"support formula differs at {cap}/{layers}/{initial}/{appended}")
                checked += 1
    return checked


def selected_rho(policy, version, length):
    if version == "v3" or length <= policy["threshold"]:
        return 1.0
    return policy[version]


def simulate_sparse(timestamps, requests, policy, *, cap=1024, layers=6):
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
        inherited = length > 0
        pending, active = inherited, False
        appended = 0
        counts = Counter()
        if tags is None and inherited:
            tags = np.full(length, -1, dtype=np.int32)

        def advance(timestamp):
            nonlocal cursor, length, pending, active, appended, tags
            stop = int(np.searchsorted(timestamps, timestamp, side="left"))
            while cursor < stop:
                width = 1 << (min(32, stop - cursor).bit_length() - 1)
                old_appended = appended
                appended += width
                cursor += width
                length = min(length + width, cap)
                counts["support_token_layers"] += width * layers
                if pending or active:
                    if appended >= layers * (cap - 1) + 1:
                        counts["causal_retirements"] += 1
                        pending = active = False
                        tags = None
                    elif active:
                        true_rows = sum(max(0, min(width, layer * (cap - 1) - old_appended))
                            for layer in range(layers))
                        counts["append_encoded_rows"] += width
                        counts["sparse_append_layer_rows"] += true_rows
                        counts["eligible_append_layer_rows"] += width * layers
                        tags = np.concatenate((tags, np.full(width, current, dtype=np.int32)))[-length:]
                    else:
                        tags = np.concatenate((tags, np.full(width, -1, dtype=np.int32)))[-length:]

        for timestamp, count in sorted(by_version[version].items()):
            if not at <= timestamp < until:
                raise RuntimeError("request outside release")
            advance(timestamp)
            if pending:
                rho = selected_rho(policy, version, length)
                missing = tags < 0
                older = (tags >= 0) & (tags < current - 1)
                tail = np.arange(length) >= length - math.ceil(rho * length)
                selected = (missing | older | tail) & (tags != current)
                flags = support_flags(length, appended, cap=cap, layers=layers, inherited=inherited)
                true_missing = int(flags[:, selected & missing].sum())
                true_all = int(flags[:, selected].sum())
                counts["first_reads"] += 1
                counts["first_read_rows"] += length
                counts["refreshed_rows"] += int(selected.sum())
                counts["forced_missing_rows"] += int(missing.sum())
                counts["forced_older_rows"] += int(older.sum())
                counts["retained_previous_rows"] += int(((tags == current - 1) & ~selected).sum())
                counts["sparse_initial_layer_rows"] += true_missing
                counts["sparse_refresh_layer_rows"] += true_all - true_missing
                counts["eligible_first_read_layer_rows"] += int(selected.sum()) * layers
                tags[selected] = current
                pending, active = False, True
            if active:
                counts["corrected_requests"] += count
                counts["corrected_token_query_pairs"] += count * length
            else:
                counts["native_requests"] += count
        advance(until)
        releases[version] = dict(counts)
    return releases


def compare_gpu(summary_path, output):
    from hstu_kvcache.evaluation.panels import load_history, load_panel
    from hstu_kvcache.common.artifacts import sha256, write_json
    summary_path = Path(summary_path)
    summary = json.loads(summary_path.read_text())
    config_path = summary_path.parent / "configuration.json"
    config = json.loads(config_path.read_text())
    strategy = summary["exploration_strategy"]
    if strategy["encoder"] != "sparse" or summary["storage_mode"] != "masked":
        raise ValueError("this check covers masked support-sparse encoding")
    if strategy.get("selection", "tail") != "tail":
        raise ValueError("metadata cannot reconstruct numerical importance-selected row positions")
    policy = dict(threshold=int(strategy.get("min_partial_length", 1025)) - 1,
        v4=float(strategy["long_rho"]["v4"]), v5=float(strategy["long_rho"]["v5"]))
    panel_path = Path(config["quality_panel"])
    _, _, requests = load_panel(panel_path)
    histories = load_history(panel_path, requested_uids=[unit["uid"] for unit in summary["units"]])
    comparisons = 0
    for unit in summary["units"]:
        uid = int(unit["uid"])
        simulated = simulate_sparse(histories.rows[uid][0], requests[uid], policy)
        for version, row in unit["releases"].items():
            counts = simulated[version]
            initial = counts.get("sparse_initial_layer_rows", 0)
            refresh = counts.get("sparse_refresh_layer_rows", 0)
            append = counts.get("sparse_append_layer_rows", 0)
            eligible = counts.get("eligible_first_read_layer_rows", 0) + counts.get("eligible_append_layer_rows", 0)
            actual = initial + refresh + append
            missing = counts.get("forced_missing_rows", 0)
            expected = dict(initial_encoded_layer_rows=initial, refresh_encoded_layer_rows=refresh,
                append_encoded_layer_rows=append, actual_encoded_layer_rows=actual,
                eligible_layer_rows=eligible, skipped_encoder_layer_rows=eligible - actual,
                masked_clear_layer_rows=eligible - actual,
                initial_encode_flops=initial * LAYER_ROW_FLOPS,
                refresh_encode_flops=refresh * LAYER_ROW_FLOPS,
                append_encode_flops=append * LAYER_ROW_FLOPS,
                initial_encode_rows=missing, refresh_encode_rows=counts.get("refreshed_rows", 0) - missing,
                append_encode_rows=counts.get("append_encoded_rows", 0),
                support_mask_flops=0, support_mask_transfer_bytes=0,
                support_token_layers=counts.get("support_token_layers", 0),
                forced_missing_rows=missing, forced_older_rows=counts.get("forced_older_rows", 0),
                retained_previous_rows=counts.get("retained_previous_rows", 0),
                corrected_requests=counts.get("corrected_requests", 0),
                native_requests=counts.get("native_requests", 0),
                extra_history_read_flops=summary_flops(counts)["extra_query_flops"])
            errors = {key: (value, row["counters"].get(key, 0)) for key, value in expected.items()
                if value != row["counters"].get(key, 0)}
            if errors:
                raise RuntimeError(f"sparse GPU counter mismatch {uid}/{version}: {errors}")
            comparisons += len(expected)
    result = dict(status="complete", role="Exact masked support-sparse metadata/GPU arithmetic check",
        users=summary["users"], user_releases=3 * summary["users"], scalar_counter_comparisons=comparisons,
        all_encoder_and_query_cost_counters_exact=True, policy=policy,
        sources={str(path.resolve()): sha256(path) for path in (summary_path, config_path,
            panel_path / "binding.json", Path(__file__))})
    write_json(Path(output), result)
    return result


def run(args):
    import pyarrow.parquet as pq
    import torch
    from hstu_kvcache.evaluation.history import load_histories_semijoin
    from hstu_kvcache.common.artifacts import sha256, write_json

    began = time.perf_counter()
    formula_checks = verify_support_formula()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(args.threads)
    reference = json.loads(REFERENCE.read_text())
    baseline = reference["policies"]["original_masked_full_reference"]
    config_path = ROOT / "configs/design/medium_v0_v5_development.json"
    configuration = json.loads(config_path.read_text())
    dataset_path = ROOT / configuration["data"]["dataset"]["path"]
    population_path = ROOT / configuration["data"]["population"]["path"]
    dataset = json.loads(dataset_path.read_text())
    panel_path = ROOT / "results/design_two_2026_10/benchmark_panel/binding.json"
    panel = json.loads(panel_path.read_text())
    for role, path in (("dataset", dataset_path), ("population", population_path)):
        if sha256(path) != configuration["data"][role]["sha256"]:
            raise RuntimeError("frozen input changed")
    request_manifest = json.loads(REQUEST_MANIFEST.with_name("manifest.json").read_text())
    if sha256(REQUEST_MANIFEST) != request_manifest["artifacts"][REQUEST_MANIFEST.name]["sha256"]:
        raise RuntimeError("request manifest changed")
    uids = sorted(pq.read_table(population_path, columns=["uid"])["uid"].to_pylist())
    if len(uids) != 30_000 or len(set(uids)) != 30_000:
        raise RuntimeError("fixed30k population changed")
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
        raise RuntimeError("fixed30k requests differ")
    active = len(requests)
    histories = load_histories_semijoin(uids, dataset_path=dataset_path,
        known_vocab_size=dataset["foundation_items"],
        oov_buckets=panel["model_config"]["num_items"] - dataset["foundation_items"],
        start_timestamp=START, end_timestamp=END, max_history=1024, threads=args.threads)
    print(json.dumps(dict(status="metadata_loaded", users=len(uids), requests=len(seen),
        elapsed_seconds=time.perf_counter() - began)), flush=True)
    totals = {name: {version: Counter() for version, _ in RELEASES} for name in POLICIES}
    for number, uid in enumerate(uids):
        times = histories.rows.get(uid, (np.array([], dtype=np.int64),))[0]
        for name, policy in POLICIES.items():
            for version, counts in simulate_sparse(times, requests.get(uid, []), policy).items():
                totals[name][version].update(counts)
        if (number + 1) % 5000 == 0:
            print(json.dumps(dict(status="counting", users=number + 1,
                elapsed_seconds=time.perf_counter() - began)), flush=True)
    compare = ("first_read_rows", "refreshed_rows", "append_encoded_rows", "corrected_requests",
        "corrected_token_query_pairs", "forced_missing_rows", "forced_older_rows", "support_token_layers")
    for version in totals["sparse_full"]:
        old = baseline["by_release"][version]["counts"]
        for key in compare:
            if totals["sparse_full"][version].get(key, 0) != old.get(key, 0):
                raise RuntimeError(f"full sparse trace differs from original ledger: {version}/{key}")
    results = {}
    complete_baseline = baseline["complete_added_flops"]
    for name, releases in totals.items():
        shaped = {}
        for version, counts in releases.items():
            query = summary_flops(counts)["extra_query_flops"]
            initial = counts["sparse_initial_layer_rows"] * LAYER_ROW_FLOPS
            refresh = counts["sparse_refresh_layer_rows"] * LAYER_ROW_FLOPS
            append = counts["sparse_append_layer_rows"] * LAYER_ROW_FLOPS
            eligible = counts["eligible_first_read_layer_rows"] + counts["eligible_append_layer_rows"]
            actual = counts["sparse_initial_layer_rows"] + counts["sparse_refresh_layer_rows"] + counts["sparse_append_layer_rows"]
            shaped[version] = dict(counts=dict(counts), initial_encoder_flops=initial,
                refresh_encoder_flops=refresh, append_encoder_flops=append,
                all_first_read_encoder_flops=initial + refresh,
                total_encoder_flops=initial + refresh + append,
                extra_query_flops=query, variable_flops=initial + refresh + append + query,
                eligible_layer_rows=eligible, actually_encoded_layer_rows=actual,
                skipped_encoder_layer_rows=eligible - actual)
        variable = sum(row["variable_flops"] for row in shaped.values())
        encoded = sum(row["total_encoder_flops"] for row in shaped.values())
        original_encoded = sum(row["total_encoder_flops"] for row in baseline["by_release"].values())
        first_read = sum(row["all_first_read_encoder_flops"] for row in shaped.values())
        added_before_validation = CALIBRATION_FLOPS + variable
        known = name == "sparse_full"
        results[name] = dict(settings=POLICIES[name], by_release=shaped,
            exact_service_added_flops=variable, shared_calibration_flops=CALIBRATION_FLOPS,
            new_publication_validation_flops=0 if known else None,
            complete_added_flops=added_before_validation if known else None,
            complete_added_flops_before_new_publication_validation=added_before_validation,
            complete_saving_percent_before_new_publication_validation=100 * (complete_baseline - added_before_validation) / complete_baseline,
            total_encoder_saving_percent=100 * (original_encoded - encoded) / original_encoded,
            all_first_read_encoder_saving_percent=100 * (baseline["refresh_encoder_flops"] - first_read) / baseline["refresh_encoder_flops"],
            allowed_publication_validation_flops_for_30pct_complete_saving=.7 * complete_baseline - added_before_validation,
            quality_status="all-rho1 is mathematically equal to original masked full codes; GPU evidence reported separately" if known else "prospective cost-only screen; no quality validation or selected policy claim")
    report = dict(status="complete", role="Exact analytical fixed30k support-sparse cost screen",
        users=len(uids), active_request_users=active, requests=len(seen),
        baseline_complete_added_flops=complete_baseline,
        once_rebuild_flops=reference["once_rebuild_flops"],
        encoder_flops_per_true_layer_row=LAYER_ROW_FLOPS,
        policies=results, verification=dict(support_formula_checks=formula_checks,
            full_trace_counts_match_retained_original_ledger=True, reads_labels_or_scores=False),
        baseline_components_flops=dict(shared_D2=CALIBRATION_FLOPS,
            v3_first_read=baseline["by_release"]["v3"]["refresh_encoder_flops"],
            v4_v5_first_read=sum(baseline["by_release"][version]["refresh_encoder_flops"] for version in ("v4", "v5")),
            append_encoding=sum(row["append_encoder_flops"] for row in baseline["by_release"].values()),
            query=baseline["extra_query_flops"],
            support_mask=sum(row["components_flops"]["support_mask"] for row in baseline["by_release"].values())),
        source_reconstruction_sensitivity_flops=SOURCE_RECONSTRUCTION_SENSITIVITY_FLOPS,
        sources={str(path.resolve()): sha256(path) for path in (Path(__file__), REFERENCE,
            dataset_path, population_path, REQUEST_MANIFEST, config_path, panel_path)},
        elapsed_seconds=time.perf_counter() - began,
        notes=["rho is selected from actual first-read length; short users get full refresh. V3 initializes every missing row.",
            "Cleared layer/rows produce zero codes directly, eliminating encoder work and float masks; dense query correction is unchanged.",
            "Partial policies' additional publication validation is unknown. Their complete saving is an upper bound until it is charged.",
            "D2 is fixed at the original three-release cost. Even zero service cannot save30% of complete added FLOPs."])
    write_json(args.output / "summary.json", report)
    print(json.dumps(dict(status="complete", output=str(args.output), policies={name: {
        key: result[key] for key in ("total_encoder_saving_percent", "all_first_read_encoder_saving_percent",
            "complete_saving_percent_before_new_publication_validation")}
        for name, result in results.items()}, elapsed_seconds=report["elapsed_seconds"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/partial_refresh_2026_10/round2/sparse_cost")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--check-formula", action="store_true")
    parser.add_argument("--compare-gpu-summary", type=Path)
    args = parser.parse_args()
    if args.compare_gpu_summary:
        print(json.dumps(compare_gpu(args.compare_gpu_summary, args.output)), flush=True)
    elif args.check_formula:
        print(json.dumps(dict(checked=verify_support_formula())), flush=True)
    else:
        run(args)
