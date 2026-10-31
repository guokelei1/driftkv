#!/usr/bin/env python3
"""Exact row/version arithmetic for the 4.4 partial-refresh prototype.

The simulator reads event and request timestamps only. It uses the native
trajectory's power-of-two append bands, preserves real previous-release tags,
and charges every actually generated row. Population costs use the retained
fixed30k request manifest, never extrapolated quality-panel costs.
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
from collections import Counter, defaultdict, deque
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
from hstu_kvcache.evaluation.release_context import END, RELEASES

START = RELEASES[0][1]
ENCODER_FLOPS_PER_ROW = 166_464
SUPPORT_MASK_FLOPS_PER_ROW = 288
CALIBRATION_FLOPS = 27_313_027_944_710
SOURCE_RECONSTRUCTION_SENSITIVITY_FLOPS = 7_058_675_727_738
REBUILD_FLOPS = 201_640_840_864_172
REFERENCE = ROOT / "results/design_system_2026_10/phase2/population_cost/summary_complete.json"
LAZY_REFERENCE = ROOT / "results/design_system_2026_10/lightweight_read/lazy_codes/population_cost/summary.json"
REQUEST_MANIFEST = ROOT / "data/manifests/yambda500m_medium_hstu_native_d7_d14_v1/requests_fidelity.parquet"


def simulate(timestamps, requests, rho_by_release, *, cap=1024, layers=6, query_batch=16):
    """Return exact release row counters without constructing native tensors.

    A missing tag is -1. At the first read, refresh missing/older-than-previous
    rows and the newest ceil(rho*N) rows, excluding rows already encoded in the
    current version. Current rows after active appends need no later refresh.
    """
    cursor = int(np.searchsorted(timestamps, START, side="left"))
    length = min(cursor, cap)
    tags = deque([-1] * length, maxlen=cap)
    by_version = defaultdict(Counter)
    for request in requests:
        by_version[request["version"]][int(request["query_timestamp"])] += 1
    releases = {}
    for index, (version, at) in enumerate(RELEASES):
        current = int(version[1:])
        rho = float(rho_by_release[version])
        if not 0 <= rho <= 1:
            raise ValueError("rho must lie in [0,1]")
        until = RELEASES[index + 1][1] if index + 1 < len(RELEASES) else END
        superseded_pending = bool(index and pending)
        pending, active = length > 0, False
        remaining = layers * (cap - 1) + 1 if pending else 0
        counts = Counter(publication_events=1)
        counts["superseded_pending_releases"] = int(superseded_pending)

        def advance(timestamp):
            nonlocal cursor, length, pending, active, remaining, tags
            stop = int(np.searchsorted(timestamps, timestamp, side="left"))
            while cursor < stop:
                width = 1 << (min(32, stop - cursor).bit_length() - 1)
                cursor += width
                length = min(length + width, cap)
                counts["support_token_layers"] += width * layers
                tags.extend([-1] * width)
                if pending or active:
                    remaining -= width
                    if remaining <= 0:
                        counts["causal_retirements"] += 1
                        if pending:
                            counts["unmaterialized_retirements"] += 1
                        pending = active = False
                        tags = deque([-1] * length, maxlen=cap)
                    elif active:
                        for offset in range(1, width + 1):
                            tags[-offset] = current
                        counts["append_encoded_rows"] += width

        for timestamp, count in sorted(by_version[version].items()):
            if not at <= timestamp < until:
                raise RuntimeError("request timestamp is outside its release")
            advance(timestamp)
            if pending:
                tail = math.ceil(rho * length)
                updated = []
                counts["first_reads"] += 1
                counts["first_read_rows"] += length
                for position, tag in enumerate(tags):
                    missing = tag < 0
                    older = 0 <= tag < current - 1
                    forced = missing or older
                    selected_tail = position >= length - tail
                    selected = tag != current and (forced or selected_tail)
                    counts["forced_missing_rows"] += int(missing)
                    counts["forced_older_rows"] += int(older)
                    counts["tail_rows"] += int(selected_tail)
                    counts["tail_only_refreshed_rows"] += int(selected and not forced)
                    counts["refreshed_rows"] += int(selected)
                    counts["retained_previous_rows"] += int(not selected and tag == current - 1)
                    counts["current_reused_rows"] += int(tag == current)
                    updated.append(current if selected else tag)
                tags = deque(updated, maxlen=cap)
                pending, active = False, True
            if active:
                counts["corrected_requests"] += count
                counts["corrected_token_query_pairs"] += count * length
                batches = (count + query_batch - 1) // query_batch
                counts["corrected_read_batches"] += batches
                counts["corrected_token_read_batches"] += batches * length
            else:
                counts["native_requests"] += count
        advance(until)
        counts["pending_at_release_end"] = int(pending)
        counts["active_at_release_end"] = int(active)
        counts["final_history_length"] = length
        releases[version] = dict(counts)
    return releases


def summary_flops(counts, *, layers=6, heads=6, head_dim=32, rank=8, storage_mode="masked"):
    """Same scalar-FLOP convention as the retained fullitem KV candidate."""
    rows = counts.get("refreshed_rows", 0)
    appended = counts.get("append_encoded_rows", 0)
    queries = counts.get("corrected_requests", 0)
    pairs = counts.get("corrected_token_query_pairs", 0)
    components = dict(
        first_read_refresh=rows * ENCODER_FLOPS_PER_ROW,
        append_encoding=appended * ENCODER_FLOPS_PER_ROW,
        support_mask=(rows + appended) * layers * heads * rank if storage_mode == "masked" else 0,
        query_weighted_code=layers * 2 * heads * rank * pairs,
        query_value_projection=layers * queries * 2 * heads * rank * head_dim,
        query_value_residual_add=layers * queries * heads * head_dim,
        query_key_projection=layers * queries * 2 * heads * head_dim * rank,
        query_key_dot=layers * pairs * 2 * heads * rank,
        query_key_logit_add=layers * pairs * heads,
    )
    if storage_mode == "raw":
        components["query_support_mask"] = counts.get("corrected_token_read_batches", 0) * layers * heads * rank
    elif storage_mode != "masked":
        raise ValueError("storage_mode must be masked or raw")
    return dict(components_flops=components,
        refresh_encoder_flops=components["first_read_refresh"],
        append_encoder_flops=components["append_encoding"],
        total_encoder_flops=components["first_read_refresh"] + components["append_encoding"],
        encoder_support_mask_transfer_bytes=(rows + appended) * layers if storage_mode == "masked" else 0,
        read_support_mask_transfer_bytes=counts.get("corrected_token_read_batches", 0) * layers if storage_mode == "raw" else 0,
        extra_query_flops=sum(value for name, value in components.items() if name.startswith("query_")),
        variable_flops=sum(components.values()))


def executed_counters(counts, *, storage_mode="masked"):
    """Map pure shape counters to PartialRefreshSession's serving ledger."""
    missing = counts.get("forced_missing_rows", 0)
    refreshed = counts.get("refreshed_rows", 0)
    appended = counts.get("append_encoded_rows", 0)
    costs = summary_flops(counts, storage_mode=storage_mode)
    counters = dict(
        initial_encode_rows=missing, refresh_encode_rows=refreshed - missing,
        append_encode_rows=appended,
        initial_encode_flops=missing * ENCODER_FLOPS_PER_ROW,
        refresh_encode_flops=(refreshed - missing) * ENCODER_FLOPS_PER_ROW,
        append_encode_flops=appended * ENCODER_FLOPS_PER_ROW,
        support_mask_flops=(refreshed + appended) * SUPPORT_MASK_FLOPS_PER_ROW if storage_mode == "masked" else 0,
        support_mask_transfer_bytes=(refreshed + appended) * 6 if storage_mode == "masked" else 0,
        forced_refresh_rows=missing + counts.get("forced_older_rows", 0),
        forced_missing_rows=missing, forced_older_rows=counts.get("forced_older_rows", 0),
        retained_previous_rows=counts.get("retained_previous_rows", 0),
        support_token_layers=counts.get("support_token_layers", 0),
        first_read_refresh_events=counts.get("first_reads", 0),
        publication_events=counts.get("publication_events", 0),
        superseded_pending_releases=counts.get("superseded_pending_releases", 0),
        retirement_events=counts.get("causal_retirements", 0),
        unmaterialized_retirement_events=counts.get("unmaterialized_retirements", 0),
        corrected_requests=counts.get("corrected_requests", 0),
        native_requests=counts.get("native_requests", 0),
        extra_history_read_flops=costs["extra_query_flops"] - costs["components_flops"].get("query_support_mask", 0), response_flops=0)
    if storage_mode == "raw":
        counters.update(read_support_mask_flops=costs["components_flops"]["query_support_mask"],
            read_support_mask_transfer_bytes=counts.get("corrected_token_read_batches", 0) * 6,
            raw_masked_read_batches=counts.get("corrected_read_batches", 0))
    return counters


def compare_gpu_summary(summary_path, *, panel_dir=None, output=None):
    """Check every GPU user/release encoding and corrected-read counter exactly."""
    from hstu_kvcache.evaluation.panels import load_history, load_panel
    from hstu_kvcache.common.artifacts import sha256, write_json
    summary_path = Path(summary_path).resolve()
    summary = json.loads(summary_path.read_text())
    config_path = summary_path.parent / "configuration.json"
    configuration = json.loads(config_path.read_text())
    storage_mode = summary.get("storage_mode", configuration.get("storage_mode", "masked"))
    panel_dir = Path(panel_dir or configuration["quality_panel"])
    _, _, requests = load_panel(panel_dir)
    uids = [int(unit["uid"]) for unit in summary["units"]]
    histories = load_history(panel_dir, requested_uids=uids)
    matched, matched_components = 0, 0
    for unit in summary["units"]:
        uid = int(unit["uid"])
        releases = simulate(histories.rows[uid][0], requests[uid], summary["policy"])
        for version, row in unit["releases"].items():
            expected = executed_counters(releases[version], storage_mode=storage_mode)
            actual = row["counters"]
            errors = {key: (value, actual.get(key, 0)) for key, value in expected.items()
                if value != actual.get(key, 0)}
            if errors:
                raise RuntimeError(f"GPU-vs-shape ledger mismatch {uid}/{version}: {errors}")
            matched += 1
            matched_components += len(expected)
    result = dict(status="complete", role="Exact analytical-vs-executed partial-refresh cost check",
        users=len(uids), user_releases=matched, scalar_counter_comparisons=matched_components,
        all_encoding_and_added_read_counters_exact=True, policy=summary["policy"],
        storage_mode=storage_mode,
        sources={str(path.resolve()): sha256(path) for path in
            (summary_path, config_path, panel_dir / "binding.json", Path(__file__))})
    if output is not None:
        write_json(Path(output), result)
    return result


def native_source_shapes(timestamps, config, *, requests=None):
    """Replay native source costs for fit-scale or validation preparation.

    Fit-scale preparation switches at the three cutovers and stops at V5's
    cutover. Validation also visits its real first/read times and every window
    end, exactly as the GPU validator's native trajectory ledger does.
    """
    from hstu_kvcache.evaluation.cost import CostModel
    cap = config["max_seq_len"]
    costs = {backend: CostModel(config["hidden_size"], config["num_layers"], config["num_heads"], backend)
        for backend in ("torch", "triton")}
    cursor = int(np.searchsorted(timestamps, START, side="left"))
    length = min(cursor, cap)
    current = "v2"
    components = Counter()
    components["v2/init"] = costs["triton" if length >= cap else "torch"].full_cache(length)
    by_version = defaultdict(set)
    for request in requests or []:
        by_version[request["version"]].add(int(request["query_timestamp"]))

    def advance(timestamp):
        nonlocal cursor, length
        stop = int(np.searchsorted(timestamps, timestamp, side="left"))
        while cursor < stop:
            width = 1 << (min(32, stop - cursor).bit_length() - 1)
            cm = costs["triton" if length >= cap else "torch"]
            components[current + "/append"] += cm.band_append(length, width, window_size=cap)
            cursor += width
            length = min(length + width, cap)

    for index, (version, at) in enumerate(RELEASES):
        advance(at)
        current = version
        if requests is not None:
            for timestamp in sorted(by_version[version]):
                advance(timestamp)
            until = RELEASES[index + 1][1] if index + 1 < len(RELEASES) else END
            advance(until)
    return dict(components)


def audit_validation(summary_path, *, output):
    """Charge actual old-code preparation missing from the forward-only report."""
    from hstu_kvcache.evaluation.panels import load_history, load_panel
    from hstu_kvcache.common.artifacts import sha256, write_json
    summary_path = Path(summary_path).resolve()
    summary = json.loads(summary_path.read_text())
    config_path = summary_path.parent / "configuration.json"
    configuration = json.loads(config_path.read_text())
    panel_dir = Path(configuration["calibration_panel"])
    storage_mode = summary.get("storage_mode", configuration.get("storage_mode", "masked"))
    panel, _, requests = load_panel(panel_dir)
    histories = load_history(panel_dir, requested_uids=summary["fit_uids"] + summary["validation_uids"])
    components, releases = Counter(), {version: Counter() for version, _ in RELEASES}
    for uid in summary["validation_uids"]:
        for version, counts in simulate(histories.rows[uid][0], requests[uid], summary["policy"]).items():
            work = summary_flops(counts, storage_mode=storage_mode)["components_flops"]
            # The validator prepares/advances state but does not score live
            # request items. Its artificial candidate queries are already in
            # the original forward report, and must not be charged twice.
            work = {key: value for key, value in work.items() if not key.startswith("query_")}
            releases[version].update(work)
            components.update(work)
    preparation = sum(components.values())
    fit_native, validation_native = Counter(), Counter()
    for uid in summary["fit_uids"]:
        fit_native.update(native_source_shapes(histories.rows[uid][0], panel["model_config"]))
    for uid in summary["validation_uids"]:
        validation_native.update(native_source_shapes(histories.rows[uid][0], panel["model_config"], requests=requests[uid]))
    if sum(validation_native.values()) != summary["validation_history_resource_flops"]:
        raise RuntimeError("validation native source shape costs differ from the executed GPU ledger")
    publication_sources = sum(fit_native.values()) + sum(validation_native.values())
    result = dict(status="complete", role="Supplemental accounting for publication validation",
        policy=summary["policy"], validation_users=len(summary["validation_uids"]),
        storage_mode=storage_mode,
        original_shared_forward_flops=summary["shared_validation_flops"],
        old_code_preparation_flops=preparation,
        old_code_preparation_components_flops=dict(components),
        old_code_preparation_by_release={version: dict(row) for version, row in releases.items()},
        complete_shared_validation_flops=summary["shared_validation_flops"] + preparation,
        observed_validation_native_source_flops=summary["validation_history_resource_flops"],
        observed_fit_native_source_flops=sum(fit_native.values()),
        publication_native_source_flops=publication_sources,
        publication_native_source_components_flops=dict(fit=dict(fit_native), validation=dict(validation_native)),
        validation_native_source_ledger_matches_gpu=True,
        native_source_cost_scope="Primary cost assumes retained native source inputs are available. The conservative nonreuse sensitivity charges original calibration source reconstruction plus fit-scale and validation native replay separately; their causal prefixes overlap, so this is an upper bound without reuse, not a claim of independent required reconstructions.",
        verification="Old-code preparation uses the same real first-read times/native bands as validation, without scoring future request items or labels.",
        sources={str(path.resolve()): sha256(path) for path in
            (summary_path, config_path, panel_dir / "binding.json", Path(__file__))})
    write_json(Path(output), result)
    return result


def finalize_cost(summary_path, validation_audit_path, *, output):
    """Add audited publication work to exact counts, without replaying events."""
    from hstu_kvcache.common.artifacts import sha256, write_json
    summary_path, validation_audit_path = Path(summary_path), Path(validation_audit_path)
    summary = json.loads(summary_path.read_text())
    audit = json.loads(validation_audit_path.read_text())
    if audit.get("storage_mode", "masked") != summary.get("storage_mode", "masked"):
        raise RuntimeError("validation audit and population code storage modes differ")
    lazy_reference = json.loads(LAZY_REFERENCE.read_text())
    eager = sum(row["exact_rebuild_flops"] for row in summary["native_by_release"].values())
    native_attention = sum(row["lazy_value_cost"]["native_history_attention_flops"]
        for row in lazy_reference["by_release"].values())
    denominator = summary["once_rebuild_flops"]
    baseline = summary["policies"]["full_refresh_reference"]["complete_added_flops"]
    for name, result in summary["policies"].items():
        if name != "full_refresh_reference":
            if result["rho_by_release"] != audit["policy"]:
                raise RuntimeError("publication cost audit belongs to a different refresh policy")
            result["publication_validation_flops"] = audit["complete_shared_validation_flops"]
            result["publication_validation_components_flops"] = dict(
                original_shared_forward=audit["original_shared_forward_flops"],
                old_code_preparation=audit["old_code_preparation_flops"])
        complete = result["variable_flops"] + result["shared_calibration_flops"] + result["publication_validation_flops"]
        result.update(complete_added_flops=complete,
            complete_percent_of_once_rebuild=100 * complete / denominator,
            complete_saving_percent=100 * (baseline - complete) / baseline,
            complete_with_source_reconstruction_percent=100 * (complete + SOURCE_RECONSTRUCTION_SENSITIVITY_FLOPS) / denominator,
            complete_percent_of_all_resident_eager_rebuild=100 * complete / eager,
            native_history_attention_flops=native_attention,
            extra_percent_of_native_history_attention=100 * result["extra_query_flops"] / native_attention)
        if name != "full_refresh_reference":
            result["publication_native_source_sensitivity_flops"] = audit["publication_native_source_flops"]
            result["complete_with_publication_sources_percent"] = 100 * (complete + audit["publication_native_source_flops"]) / denominator
            result["complete_with_all_sources_no_reuse_percent"] = 100 * (
                complete + SOURCE_RECONSTRUCTION_SENSITIVITY_FLOPS + audit["publication_native_source_flops"]) / denominator
        for version, release in result["by_release"].items():
            full = summary["policies"]["full_refresh_reference"]["by_release"][version]
            release["refresh_encoder_saving_percent"] = 100 * (full["refresh_encoder_flops"] - release["refresh_encoder_flops"]) / full["refresh_encoder_flops"]
    summary["all_resident_eager_rebuild_flops"] = eager
    summary["publication_validation_audit"] = str(validation_audit_path.resolve())
    summary["publication_native_source_execution"] = dict(
        validation=audit["observed_validation_native_source_flops"],
        fit=audit["observed_fit_native_source_flops"],
        scope=audit["native_source_cost_scope"])
    summary["sources"].update({str(path.resolve()): sha256(path)
        for path in (summary_path, validation_audit_path, Path(__file__))})
    summary["notes"].append("Complete publication validation includes its previously omitted old-code preparation, supplied by the separately retained cost_audit.json. The generated validation summary is unchanged.")
    write_json(Path(output), summary)
    return {name: {key: result[key] for key in ("complete_percent_of_once_rebuild",
        "refresh_encoder_saving_percent", "complete_saving_percent", "complete_with_source_reconstruction_percent")}
        for name, result in summary["policies"].items()}


def _native_comparison(shaped, reference):
    keys = ("requests", "initial_history_length", "appends", "native_append_flops",
        "exact_rebuild_flops", "first_request_rebuild_flops", "fresh_per_request_history_flops",
        "native_query_flops", "support_token_layers", "final_history_length")
    errors = {key: (shaped.get(key, 0), reference.get(key, 0)) for key in keys
              if shaped.get(key, 0) != reference.get(key, 0)}
    if errors:
        raise RuntimeError(f"native event ledger no longer matches retained fixed30k evidence: {errors}")


def run_population(args):
    import pyarrow.parquet as pq
    import torch
    from hstu_kvcache.evaluation.history import load_histories_semijoin
    from experiments.archive.compact_read.profile_costs import ShapeCosts, profile_user
    from hstu_kvcache.methods.cachetide.compact.lightweight_fullitem_kv import LightweightFullItemKVAdapter
    from hstu_kvcache.common.artifacts import sha256, write_json

    began = time.perf_counter()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(args.threads)
    reference = json.loads(REFERENCE.read_text())
    lazy_reference = json.loads(LAZY_REFERENCE.read_text())
    configuration = json.loads(args.configuration.read_text())
    panel = json.loads((args.panel / "binding.json").read_text())
    dataset_path = ROOT / configuration["data"]["dataset"]["path"]
    dataset = json.loads(dataset_path.read_text())
    population_path = ROOT / configuration["data"]["population"]["path"]
    uids = sorted(pq.read_table(population_path, columns=["uid"])["uid"].to_pylist())
    if len(uids) != 30_000 or len(set(uids)) != 30_000:
        raise RuntimeError("fixed30k membership changed")
    for source in ("dataset", "population"):
        record = configuration["data"][source]
        if sha256(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"frozen {source} binding changed")
    request_manifest = args.request_manifest.resolve()
    manifest_path = request_manifest.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if sha256(request_manifest) != manifest["artifacts"][request_manifest.name]["sha256"]:
        raise RuntimeError("request manifest payload differs from its retained hash")
    selected = set(uids)
    requests, seen = defaultdict(list), set()
    table = pq.read_table(request_manifest,
        filters=[("time_block", "=", "matrix_horizon"), ("target_known", "=", True),
            ("query_timestamp", ">=", START), ("query_timestamp", "<", END)],
        columns=["request_id", "uid", "query_timestamp"])
    for row in table.to_pylist():
        if row["uid"] not in selected:
            continue
        if row["request_id"] in seen:
            raise RuntimeError("request identity repeats")
        seen.add(row["request_id"])
        version = next(name for name, start in reversed(RELEASES) if row["query_timestamp"] >= start)
        requests[row["uid"]].append({**row, "version": version})
    if len(seen) != reference["requests"]:
        raise RuntimeError("fixed30k request total differs from original ledger")
    active_users = len(requests)
    histories = load_histories_semijoin(uids, dataset_path=dataset_path,
        known_vocab_size=dataset["foundation_items"],
        oov_buckets=panel["model_config"]["num_items"] - dataset["foundation_items"],
        start_timestamp=START, end_timestamp=END, max_history=1024, threads=args.threads)
    print(json.dumps(dict(status="population_metadata_loaded", users=len(uids), requests=len(seen),
        active_request_users=active_users, elapsed_seconds=time.perf_counter() - began)), flush=True)
    adapters = {version: LightweightFullItemKVAdapter(6, 6, 32, rank=8) for version, _ in RELEASES}
    native_costs = ShapeCosts(panel["model_config"], adapters)
    native_totals = {version: Counter() for version, _ in RELEASES}
    policies = json.loads(args.policies.read_text()) if args.policies else {"selected": {
        version: float(args.rho) for version, _ in RELEASES}}
    validation_record = None
    validation_records, validation_audits = {}, {}
    validation_paths, audit_paths = [], []
    if args.validation_summary:
        validation_record = json.loads(args.validation_summary.read_text())
        policies = {"selected": validation_record["policy"], **(policies if args.policies else {})}
        args.validation_flops = validation_record["shared_validation_flops"]
    if args.validation_summaries:
        mapping = json.loads(args.validation_summaries.read_text())
        policies = {}
        for name, summary_path in mapping.items():
            path = Path(summary_path)
            audit_path = path.parent / "cost_audit.json"
            record, audit = json.loads(path.read_text()), json.loads(audit_path.read_text())
            if record["policy"] != audit["policy"] or record.get("storage_mode", "masked") != args.storage_mode:
                raise RuntimeError("validation policy/storage and its cost audit disagree")
            validation_records[name], validation_audits[name] = record, audit
            validation_paths.append(path)
            audit_paths.append(audit_path)
            policies[name] = record["policy"]
    policies["full_refresh_reference"] = {version: 1.0 for version, _ in RELEASES}
    totals = {name: {version: Counter() for version, _ in RELEASES} for name in policies}
    for user_index, uid in enumerate(uids):
        timestamps = histories.rows.get(uid, (np.array([], dtype=np.int64),))[0]
        user_requests = requests.get(uid, [])
        shaped = profile_user(uid, timestamps, user_requests, native_costs)
        for version, row in shaped["releases"].items():
            native_totals[version].update({key: value for key, value in row.items() if key != "uid"})
        for name, rhos in policies.items():
            for version, counts in simulate(timestamps, user_requests, rhos).items():
                totals[name][version].update(counts)
        if (user_index + 1) % 5000 == 0:
            print(json.dumps(dict(status="population_counting", users=user_index + 1,
                elapsed_seconds=time.perf_counter() - began)), flush=True)
    for version, shaped in native_totals.items():
        _native_comparison(shaped, reference["by_release"][version]["counts_and_flops"])
        counts = totals["full_refresh_reference"][version]
        old = lazy_reference["by_release"][version]["counts_and_flops"]
        names = {"refreshed_rows": "lazy_initial_encode_tokens", "append_encoded_rows": "lazy_append_encode_tokens",
            "corrected_requests": "lazy_corrected_requests", "corrected_token_query_pairs": "lazy_corrected_token_query_pairs",
            "support_token_layers": "lazy_support_token_layers", "first_reads": "lazy_initializations"}
        for new_name, old_name in names.items():
            if counts.get(new_name, 0) != old.get(old_name, 0):
                raise RuntimeError(f"full-refresh count differs from retained lazy reference: {version}/{new_name}")
    rebuild = sum(value["first_request_rebuild_flops"] for value in native_totals.values())
    if rebuild != REBUILD_FLOPS:
        raise RuntimeError("first-request Rebuild denominator changed")
    full_variable = sum(summary_flops(counts)["variable_flops"]
        for counts in totals["full_refresh_reference"].values())
    original_masked_variable = full_variable
    full_variable = sum(summary_flops(counts, storage_mode=args.storage_mode)["variable_flops"]
        for counts in totals["full_refresh_reference"].values())
    full_refresh = sum(summary_flops(counts, storage_mode=args.storage_mode)["refresh_encoder_flops"]
        for counts in totals["full_refresh_reference"].values())
    results = {}
    cost_policies = dict(totals)
    if args.storage_mode == "raw":
        cost_policies["original_masked_full_reference"] = totals["full_refresh_reference"]
    native_attention = sum(row["lazy_value_cost"]["native_history_attention_flops"]
        for row in lazy_reference["by_release"].values())
    eager = sum(row["exact_rebuild_flops"] for row in native_totals.values())
    for name, by_release in cost_policies.items():
        mode = "masked" if name == "original_masked_full_reference" else args.storage_mode
        releases = {version: {"counts": dict(counts), **summary_flops(counts, storage_mode=mode)}
            for version, counts in by_release.items()}
        variable = sum(row["variable_flops"] for row in releases.values())
        refresh = sum(row["refresh_encoder_flops"] for row in releases.values())
        baseline = name in ("full_refresh_reference", "original_masked_full_reference")
        publication_validation = 0 if baseline else (
            validation_audits[name]["complete_shared_validation_flops"] if name in validation_audits else args.validation_flops)
        complete = variable + CALIBRATION_FLOPS + publication_validation
        query_extra = sum(row["extra_query_flops"] for row in releases.values())
        native_query = sum(row["native_query_flops"] for row in native_totals.values())
        results[name] = dict(rho_by_release=policies.get(name, policies["full_refresh_reference"]), by_release=releases,
            storage_mode=mode,
            variable_flops=variable, shared_calibration_flops=CALIBRATION_FLOPS,
            publication_validation_flops=publication_validation,
            complete_added_flops=complete, complete_percent_of_once_rebuild=100 * complete / rebuild,
            source_reconstruction_sensitivity_flops=SOURCE_RECONSTRUCTION_SENSITIVITY_FLOPS,
            complete_with_source_reconstruction_percent=100 * (complete + SOURCE_RECONSTRUCTION_SENSITIVITY_FLOPS) / rebuild,
            refresh_encoder_flops=refresh, refresh_encoder_saving_percent=100 * (full_refresh - refresh) / full_refresh,
            variable_saving_percent=100 * (full_variable - variable) / full_variable,
            complete_saving_percent=100 * (full_variable + CALIBRATION_FLOPS - complete) / (full_variable + CALIBRATION_FLOPS),
            complete_saving_percent_vs_original_masked=100 * (original_masked_variable + CALIBRATION_FLOPS - complete) / (original_masked_variable + CALIBRATION_FLOPS),
            complete_percent_of_all_resident_eager_rebuild=100 * complete / eager,
            extra_query_flops=query_extra, native_query_flops=native_query,
            native_history_attention_flops=native_attention,
            extra_percent_of_native_whole_query=100 * query_extra / native_query,
            extra_percent_of_native_history_attention=100 * query_extra / native_attention,
            encoder_support_mask_transfer_bytes=sum(row["encoder_support_mask_transfer_bytes"] for row in releases.values()),
            read_support_mask_transfer_bytes=sum(row["read_support_mask_transfer_bytes"] for row in releases.values()))
        if name in validation_audits:
            audit = validation_audits[name]
            results[name].update(
                publication_validation_components_flops=dict(original_shared_forward=audit["original_shared_forward_flops"],
                    old_code_preparation=audit["old_code_preparation_flops"]),
                publication_validation_audit=str((Path(mapping[name]).parent / "cost_audit.json").resolve()),
                publication_native_source_sensitivity_flops=audit["publication_native_source_flops"],
                complete_with_publication_sources_percent=100 * (complete + audit["publication_native_source_flops"]) / rebuild,
                complete_with_all_sources_no_reuse_percent=100 * (complete + SOURCE_RECONSTRUCTION_SENSITIVITY_FLOPS + audit["publication_native_source_flops"]) / rebuild,
                native_source_cost_scope=audit["native_source_cost_scope"])
        for version, release in releases.items():
            full = summary_flops(totals["full_refresh_reference"][version], storage_mode=args.storage_mode)
            release["refresh_encoder_saving_percent"] = 100 * (full["refresh_encoder_flops"] - release["refresh_encoder_flops"]) / full["refresh_encoder_flops"]
    summary = dict(status="complete", role="Exact analytical fixed30k event ledger; no GPU timing or population quality claim",
        storage_mode=args.storage_mode,
        users=len(uids), requests=len(seen), active_request_users=active_users,
        days=[259, 301], model_chain=["v2", "v3", "v4", "v5"], context=1024,
        once_rebuild_flops=rebuild, policies=results,
        all_resident_eager_rebuild_flops=eager,
        state_bytes_per_retained_row=dict(native_fp32_kv=9216, rank8_fp32_codes=1152,
            existing_support_flags=6, added_int32_version_tag=4),
        version_tag_percent_of_native_kv=100 * 4 / 9216,
        original_masked_full_reference_cost=dict(variable_flops=original_masked_variable,
            complete_added_flops=original_masked_variable + CALIBRATION_FLOPS,
            complete_percent_of_once_rebuild=100 * (original_masked_variable + CALIBRATION_FLOPS) / rebuild),
        native_by_release={version: dict(row) for version, row in native_totals.items()},
        verification=dict(native_release_totals_match_retained_ledger=True,
            full_refresh_counts_match_retained_lazy_ledger=True,
            reads_feedback_labels=False, gpu_execution_count_match="supplied separately by prototype evaluation"),
        publication_validation_source=None if args.validation_summary is None else str(args.validation_summary.resolve()),
        sources={str(path.resolve()): sha256(path) for path in (REFERENCE, LAZY_REFERENCE,
            args.configuration, args.panel / "binding.json", dataset_path, population_path,
            request_manifest, manifest_path, Path(__file__),
            *([args.validation_summary] if args.validation_summary else []),
            *([args.policies] if args.policies else []),
            *([args.validation_summaries] if args.validation_summaries else []),
            *validation_paths, *audit_paths)},
        elapsed_seconds=time.perf_counter() - began,
        notes=["Original raw request shards and per-user cost rows were retired; retained native manifest identities/timestamps reproduce all recorded native release totals.",
            "Every calibration and service operation is charged. Publication validation FLOPs are provided by its separately retained execution ledger.",
            "Refresh set is the union of tail, missing and older tags, excluding current tags. Integer tag comparisons are metadata work, not floating-point arithmetic.",
            "Query overhead is unchanged by refresh ratio. No loaded-throughput or p99 claim."])
    write_json(output / "summary.json", summary)
    print(json.dumps(dict(status="complete", output=str(output), policies={name: {
        key: result[key] for key in ("complete_percent_of_once_rebuild", "refresh_encoder_saving_percent", "complete_saving_percent")}
        for name, result in results.items()}, elapsed_seconds=summary["elapsed_seconds"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", type=Path, default=ROOT / "configs/design/medium_v0_v5_development.json")
    parser.add_argument("--panel", type=Path, default=ROOT / "results/design_two_2026_10/benchmark_panel")
    parser.add_argument("--request-manifest", type=Path, default=REQUEST_MANIFEST)
    parser.add_argument("--policies", type=Path, help="JSON mapping policy names to v3/v4/v5 refresh ratios")
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--storage-mode", choices=("masked", "raw"), default="masked")
    parser.add_argument("--validation-flops", type=int, default=0)
    parser.add_argument("--validation-summary", type=Path, help="Publication summary supplying selected policy and complete validation FLOPs")
    parser.add_argument("--validation-summaries", type=Path, help="JSON mapping policy names to completed validation summaries with sibling cost_audit.json files")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare-gpu-summary", type=Path)
    parser.add_argument("--audit-validation-summary", type=Path)
    parser.add_argument("--complete-summary", type=Path)
    parser.add_argument("--validation-audit", type=Path)
    parser.add_argument("--quality-panel", type=Path)
    args = parser.parse_args()
    if args.audit_validation_summary:
        if args.output is None:
            parser.error("--output JSON path is required for validation audit")
        print(json.dumps(audit_validation(args.audit_validation_summary, output=args.output)), flush=True)
    elif args.complete_summary:
        if args.output is None or args.validation_audit is None:
            parser.error("--output JSON path and --validation-audit are required for completion")
        print(json.dumps(finalize_cost(args.complete_summary, args.validation_audit, output=args.output)), flush=True)
    elif args.compare_gpu_summary:
        print(json.dumps(compare_gpu_summary(args.compare_gpu_summary,
            panel_dir=args.quality_panel, output=args.output)), flush=True)
    else:
        if args.output is None:
            parser.error("--output is required for population cost")
        run_population(args)
