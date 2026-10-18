#!/usr/bin/env python3
"""CPU event ledger and arithmetic prefilter for the query/relay candidate.

No labels, model fitting or GPU work. A new ledger is necessary: behavior
writes now consume corrected history, and current-born rows have zero codes.
Dense matrix shapes are charged, even where a rolling mask rejects pairs.
Calibration supplied to ``price`` is explicit; historical local32/joint16 is
only a sensitivity, never evidence of a completed new candidate's cost.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

REFERENCE = ROOT / "results/partial_refresh_2026_10/population_cost_raw/summary.json"
SHARED_COST = ROOT / "results/code_structure_2026_10/shared12_private6/cost.json"
PANEL = ROOT / "results/design_two_2026_10/benchmark_panel/binding.json"
CONFIG = ROOT / "configs/design/medium_v0_v5_development.json"
REQUESTS = ROOT / "data/manifests/yambda500m_medium_hstu_native_d7_d14_v1/requests_fidelity.parquet"


def read(path):
    return json.loads(Path(path).read_text())


def _hash(path):
    import hashlib
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def simulate_user(events, requests, releases, end, *, cap=1024, band_cap=32):
    """Count literal dense and exact prefix-only correction shapes.

    Each release inherits all rows; current-born suffix codes are zero. Codes
    are prepared on first history consumption, including a behavior append.
    Prefix masking within a band rejects OLDEST inherited columns first. Short
    histories retain their inherited prefix until the cache becomes full.
    """
    timestamps = events[0]
    cursor = int(np.searchsorted(timestamps, releases[0][1], side="left"))
    length = min(cursor, cap)
    by_version = defaultdict(Counter)
    for request in requests:
        by_version[request["version"]][int(request["query_timestamp"])] += int(request.get("candidates", 1))
    previous_materialized = False
    result = {}
    for number, (version, at) in enumerate(releases):
        until = releases[number + 1][1] if number + 1 < len(releases) else end
        c = Counter(initial_history_rows=length)
        # At release start the prior current buffer must become an anchor.
        if previous_materialized:
            c["current_to_anchor_rows"] += length
        coded = length
        materialized = False

        def consume(role):
            nonlocal materialized
            if not materialized:
                c["first_consumptions"] += 1
                c["first_consumption_" + role] += 1
                c["anchor_to_current_rows"] += length
                materialized = True

        def advance(timestamp):
            nonlocal cursor, length, coded
            stop = int(np.searchsorted(timestamps, timestamp, side="left"))
            while cursor < stop:
                width = 1 << (min(band_cap, stop - cursor).bit_length() - 1)
                consume("writer")
                c["writer_bands"] += 1
                c["writer_queries"] += width
                c["writer_dense_pairs"] += width * (length + width)
                c["writer_prefix_pairs"] += width * coded
                if coded:
                    c["writer_nonzero_prefix_queries"] += width
                # Useful pairs are diagnostic. They are not the dense cost.
                dropped = np.maximum(0, length + np.arange(1, width + 1) - cap)
                c["writer_valid_prefix_pairs"] += int(np.maximum(0, coded - dropped).sum())
                c["native_append_shape_" + str(length) + ":" + str(width)] += 1
                removed = max(0, length + width - cap)
                coded = max(0, coded - removed)
                length = min(cap, length + width)
                cursor += width
                c["appends"] += width

        for request_index, (timestamp, queries) in enumerate(sorted(by_version[version].items())):
            advance(timestamp)
            consume("query")
            c["ranking_queries"] += queries
            c["ranking_dense_pairs"] += queries * length
            c["ranking_prefix_pairs"] += queries * coded
            if coded:
                c["ranking_nonzero_prefix_queries"] += queries
            if request_index == 0:
                c["once_rebuild_length_" + str(length)] += 1
            for offset in range(0, queries, 32):
                candidates = min(32, queries - offset)
                c["native_query_shape_" + str(length) + ":" + str(candidates)] += 1
        advance(until)
        c["final_history_rows"] = length
        c["final_coded_rows"] = coded
        c["materialized_users"] = int(materialized)
        previous_materialized = materialized
        result[version] = dict(c)
    return result


def generate_ledger(output, *, threads=4):
    import pyarrow.parquet as pq
    from design_one.prepare_history import load_histories_semijoin
    from design_two.common import RELEASES, END
    from read_correction_2026_09.cost import CostModel, eager_read

    began = time.perf_counter()
    reference, configuration, panel = read(REFERENCE), read(CONFIG), read(PANEL)
    dataset_path = ROOT / configuration["data"]["dataset"]["path"]
    population_path = ROOT / configuration["data"]["population"]["path"]
    dataset = read(dataset_path)
    uids = sorted(pq.read_table(population_path, columns=["uid"])["uid"].to_pylist())
    if len(uids) != 30000 or len(set(uids)) != 30000:
        raise RuntimeError("fixed30k population changed")
    start = RELEASES[0][1]
    requests = defaultdict(list)
    selected, seen = set(uids), set()
    table = pq.read_table(REQUESTS,
        filters=[("time_block", "=", "matrix_horizon"), ("target_known", "=", True),
                 ("query_timestamp", ">=", start), ("query_timestamp", "<", END)],
        columns=["request_id", "uid", "query_timestamp"])
    for row in table.to_pylist():
        if row["uid"] not in selected:
            continue
        if row["request_id"] in seen:
            raise RuntimeError("duplicate request identity")
        seen.add(row["request_id"])
        version = next(name for name, at in reversed(RELEASES) if row["query_timestamp"] >= at)
        requests[row["uid"]].append({**row, "version": version})
    if len(seen) != reference["requests"]:
        raise RuntimeError("fixed30k requests changed")
    print(json.dumps(dict(status="loading_cpu_histories", users=len(uids), requests=len(seen))), flush=True)
    histories = load_histories_semijoin(uids, dataset_path=dataset_path,
        known_vocab_size=dataset["foundation_items"],
        oov_buckets=panel["model_config"]["num_items"] - dataset["foundation_items"],
        start_timestamp=start, end_timestamp=END, max_history=1024, threads=threads)
    costs = {backend: CostModel(192, 6, 6, backend) for backend in ("torch", "triton")}
    counts = {version: Counter() for version, _ in RELEASES}
    resident_items = {version: np.zeros(panel["model_config"]["num_items"], dtype=bool)
                      for version, _ in RELEASES}
    all_events_items = np.zeros(panel["model_config"]["num_items"], dtype=bool)
    for number, uid in enumerate(uids):
        events = histories.rows.get(uid, (np.array([], dtype=np.int64), np.array([], dtype=np.int64)))
        for version, at in RELEASES:
            at_cursor = int(np.searchsorted(events[0], at, side="left"))
            resident_items[version][events[1][max(0, at_cursor - 1024):at_cursor]] = True
        all_events_items[events[1]] = True
        shape = simulate_user(events, requests.get(uid, []), RELEASES, END)
        for version, values in shape.items():
            counts[version].update(values)
        if (number + 1) % 5000 == 0:
            print(json.dumps(dict(status="counting", users=number + 1,
                                  elapsed_seconds=time.perf_counter() - began)), flush=True)
    native = {}
    for version, c in counts.items():
        native[version] = dict(native_append_flops=0, native_query_flops=0,
                               once_rebuild_flops=0, all_resident_rebuild_flops=0)
        for key in tuple(c):
            if key.startswith("native_append_shape_"):
                length, width = map(int, key.removeprefix("native_append_shape_").split(":"))
                cm = costs["triton" if length >= 1024 else "torch"]
                native[version]["native_append_flops"] += c[key] * cm.band_append(length, width, window_size=1024)
            elif key.startswith("native_query_shape_"):
                length, candidates = map(int, key.removeprefix("native_query_shape_").split(":"))
                native[version]["native_query_flops"] += c[key] * eager_read(costs["torch"], length, queries=candidates)
            elif key.startswith("once_rebuild_length_"):
                length = int(key.removeprefix("once_rebuild_length_"))
                cm = costs["triton" if length >= 1024 else "torch"]
                native[version]["once_rebuild_flops"] += c[key] * cm.full_cache(length)
        # The retained reference has already counted the actual first request
        # lengths and original eager resident rebuilds. Preserve these totals.
        old = reference["native_by_release"][version]
        native[version]["all_resident_rebuild_flops"] = old["exact_rebuild_flops"]
        for new_key, old_key in (("appends", "appends"), ("ranking_queries", "requests"),
                                 ("initial_history_rows", "initial_history_length"),
                                 ("final_history_rows", "final_history_length")):
            if c[new_key] != old[old_key]:
                raise RuntimeError(f"native event shape differs: {version}/{new_key}")
        for key in ("native_append_flops", "native_query_flops"):
            if native[version][key] != old[key]:
                raise RuntimeError(f"native cost differs: {version}/{key}")
        if native[version]["once_rebuild_flops"] != old["first_request_rebuild_flops"]:
            raise RuntimeError(f"first-request rebuild differs: {version}")
        # Keep compact aggregate histograms, not per-user copies.
    report = dict(status="complete_cpu_shapes", users=len(uids), requests=len(seen),
        model_config=panel["model_config"], by_release={v: dict(c) for v, c in counts.items()},
        native_by_release=native, once_rebuild_flops=reference["once_rebuild_flops"],
        all_resident_eager_rebuild_flops=reference["all_resident_eager_rebuild_flops"],
        catalogue_items=panel["model_config"]["num_items"],
        prerelease_resident_distinct_items={v: int(mask.sum()) for v, mask in resident_items.items()},
        observed_distinct_items_whole_ledger=int(all_events_items.sum()),
        sources={str(path.relative_to(ROOT)): _hash(path) for path in (
            Path(__file__), REFERENCE, SHARED_COST, PANEL, CONFIG, dataset_path, population_path, REQUESTS)},
        elapsed_seconds=time.perf_counter() - began,
        scope="Native values do not affect shapes. Current-born rows are zero-code. Dense and prefix-only execution shapes are separate; masked pairs do not receive free FLOPs. Observed item union is a diagnostic, not prospective admission.")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2) + "\n")
    return report


def read_flops(*, pairs, queries, head_rank=18, layers=6, heads=6, head_dim=32,
               gates=True, gate_normalization="raw"):
    """Literal split prefix QK/code AV + query projection + latent decode."""
    pair_work = layers * (4 * heads * head_rank + heads) * pairs
    fixed_work = layers * (4 * heads * head_dim * head_rank + heads * head_dim) * queries
    # K/V gate q@W, tanh, +1 and two Hadamard multiplies.
    gate_work = layers * (4 * heads * head_dim * head_rank + 6 * heads * head_rank) * queries if gates else 0
    normalization = layers * heads * (4 * head_dim + 2) * queries \
        if gates and gate_normalization == "rms" else 0
    return dict(code_pair_flops=pair_work, code_fixed_query_flops=fixed_work,
                nonlinear_gate_flops=gate_work, gate_query_normalization_flops=normalization,
                total=pair_work + fixed_work + gate_work + normalization)


def row_update_flops(branch="B", xi_width=3, *, layers=6, block_sparse=False,
                     feature_mode="native"):
    """Implemented shared12/private6 geometry; fixed RMS features are transient."""
    code = 48
    branch_squares = 12**2 + 6 * 6**2 if block_sparse else code**2
    native_inputs = 384 + 6 * 64
    rms = 4 * native_inputs + 2 * 7
    if feature_mode == "origin_encoder" and xi_width == 1:
        rms = 0
    eta = 2 * (384 * 12 + 6 * 64 * 6) + (5 if feature_mode == "origin_encoder" else 3) * code
    xi = (384 + 7) * (xi_width - 1)  # Means on K, optionally V, per branch.
    if branch == "C":
        # Native KV projection + precomputed complete item projection + prior
        # block-code term, then add and SiLU. Nonlinear item call cost separate.
        return layers * (2 * 6912 + 2 * code**2 + 6 * code)
    interaction = 2 * code * xi_width if branch == "B" else 0
    # P anchor and Q eta block matmuls; inverse conversion has same arithmetic.
    return layers * (rms + eta + 4 * branch_squares + xi + interaction + 3 * code)


def origin_feature_bank_cost(catalogue_items, *, layers=6, code_width=48,
                             embedding_width=192, replicas=4,
                             preparation_mode="independent"):
    """One immutable origin bank per chain, reused across every release.

    Item table stores e_origin@W only. Bias and native KV weights are separate
    to preserve the original encode's addition order and rounding behavior.
    """
    copies = 1 if preparation_mode == "compile_once_broadcast" else replicas
    one_preparation = layers * catalogue_items * 2 * embedding_width * code_width
    item_bytes = layers * catalogue_items * code_width * 4
    native_weights_bytes = layers * 6912 * 4
    bias_bytes = layers * code_width * 4
    payload = item_bytes + native_weights_bytes + bias_bytes
    return dict(preparation_flops=one_preparation * copies,
        one_replica_preparation_flops=one_preparation, preparation_copies=copies,
        item_table_payload_bytes_per_replica=item_bytes,
        frozen_native_weights_and_bias_bytes_per_replica=native_weights_bytes + bias_bytes,
        payload_bytes_per_replica=payload, resident_replicas=replicas,
        all_replicas_payload_bytes=payload * replicas,
        publication_transfer_bytes=(replicas - 1) * payload
            if preparation_mode == "compile_once_broadcast" else 0,
        scope="Exactly one [layers,catalogue,48] item table per replica for the entire continuous chain; no per-row feature tensor and no per-release duplication.")


def item_prepare_flops(branch="B", xi_width=3, *, layers=6, embedding=192,
                       block_sparse=False):
    """New H plus A@D prefix composition per item; three-pointwise SiLU."""
    code = 48
    branch_squares = 12**2 + 6 * 6**2 if block_sparse else code**2
    r = xi_width if branch == "B" else 1
    nonlinear = 0  # Current item_coefficients is affine; no invented SiLU.
    return layers * (2 * embedding * code * r + code * r + nonlinear
                     + 2 * branch_squares * r + code * r)


def price(ledger, *, branch="B", xi_width=3, calibration_flops=0,
          source_reconstruction_flops=0, replicas=4, prefix_only=True,
          table_scope="catalogue", block_sparse=True, head_rank=18,
          gate_normalization="raw", preparation_mode="independent",
          feature_mode="native", publication_metadata=None,
          gate_source="layer", stable_anchor_inverse=False):
    """Capacity prefilter, requiring actual new calibration for final evidence."""
    releases = {}
    # The retained ledger records cfg.num_items. ItemEmbedding actually owns
    # cfg.num_items+1 rows; full publication prepares every model ID, including
    # padding and the last possibly unused row. Keep provenance untouched.
    catalogue = ledger["model_config"]["num_items"] + 1
    immutable_bank = origin_feature_bank_cost(catalogue, replicas=replicas,
        preparation_mode=preparation_mode) if feature_mode == "origin_encoder" else None
    for release_index, (version, c) in enumerate(ledger["by_release"].items()):
        # No correction work is necessary after all inherited rows are evicted.
        q = c["ranking_nonzero_prefix_queries"] if prefix_only else c["ranking_queries"]
        w = c["writer_nonzero_prefix_queries"] if prefix_only else c["writer_queries"]
        qpairs = c["ranking_prefix_pairs"] if prefix_only else c["ranking_dense_pairs"]
        wpairs = c["writer_prefix_pairs"] if prefix_only else c["writer_dense_pairs"]
        rank = read_flops(pairs=qpairs, queries=q, head_rank=head_rank,
                          gate_normalization=gate_normalization)
        writer = read_flops(pairs=wpairs, queries=w, head_rank=head_rank,
                            gate_normalization=gate_normalization)
        rows = c.get("anchor_to_current_rows", 0) + c.get("current_to_anchor_rows", 0)
        items = catalogue if table_scope == "catalogue" else ledger["prerelease_resident_distinct_items"][version]
        prep_copies = 1 if preparation_mode == "compile_once_broadcast" else replicas
        prep = prep_copies * items * item_prepare_flops(branch, xi_width, block_sparse=block_sparse)
        # Shared block-matrix publication arithmetic is tiny, but charged.
        block_cube = 12**3 + 6 * 6**3 if block_sparse else 48**3
        matrix_compilation = 6 * (4 * block_cube + (360 if block_sparse else 48**2))
        if publication_metadata is not None:
            published = publication_metadata[version]
            if table_scope != "catalogue" or published["catalogue_table_rows"] != catalogue:
                raise ValueError("actual publication and priced catalogue rows differ")
            config = published["adapter_config"]
            if config["mode"] != branch or config["xi_width"] != xi_width:
                raise ValueError("actual publication and priced update shape differ")
            if config.get("feature_mode", "native") != feature_mode:
                raise ValueError("actual publication and priced fixed feature mode differ")
            if config.get("read_shared", 12) + config.get("read_private", 6) != head_rank \
                    or config.get("gate_normalization", "raw") != gate_normalization:
                raise ValueError("actual publication and priced query reader differ")
            work = published["actual_compilation_components_flops"]
            prep = prep_copies * (work["item_projection_flops"] + work["history_composition_flops"])
            matrix_compilation = prep_copies * (work["matrix_composition_flops"]
                                               + work["cayley_preparation_flops_estimate"])
        maintenance = dict(anchor_current_conversions=rows * row_update_flops(branch, xi_width,
                               block_sparse=block_sparse, feature_mode=feature_mode),
                           shared_item_projection_and_composition=prep,
                           shared_matrix_compilation=matrix_compilation)
        # Opt-in precision.py inverts the actual saved FP32 blocks once per
        # release in FP64. Row multiply/state geometry is unchanged. This is
        # the helper's disclosed LU estimate, not a measured kernel count.
        inverse_bytes = 0
        if stable_anchor_inverse:
            inverse_flops = 6 * sum((8 * width**3 + 2) // 3
                                    for width in (12, 6, 6, 6, 6, 6, 6))
            maintenance["actual_block_inverse_preparation_estimate"] = prep_copies * inverse_flops
            inverse_bytes = 6 * (12**2 + 6 * 6**2) * 4
        if immutable_bank is not None and release_index == 0:
            maintenance["immutable_origin_feature_bank_preparation"] = immutable_bank["preparation_flops"]
        table_bytes = items * 6 * 48 * (xi_width if branch == "B" else 1) * 4
        matrix_bytes = 2 * 6 * 48**2 * 4  # Actual dense P/Q payload, including zeros.
        broadcast_bytes = (replicas - 1) * (table_bytes + matrix_bytes + inverse_bytes) \
            if preparation_mode == "compile_once_broadcast" else 0
        if immutable_bank is not None and release_index == 0:
            broadcast_bytes += immutable_bank["publication_transfer_bytes"]
        releases[version] = dict(counts={k: c.get(k, 0) for k in (
            "anchor_to_current_rows", "current_to_anchor_rows", "writer_queries",
            "writer_prefix_pairs", "writer_dense_pairs", "ranking_queries",
            "ranking_prefix_pairs", "ranking_dense_pairs")},
            ranking_read_flops=rank, writer_read_flops=writer,
            maintenance_components_flops=maintenance,
            shared_item_table_payload_bytes_per_replica=table_bytes,
            shared_composition_matrix_payload_bytes_per_replica=matrix_bytes,
            global_anchor_inverse_payload_bytes_per_replica=inverse_bytes,
            # Item lookup is executed once per operation and shared by layers,
            # including operations whose inherited code prefix is empty.
            gate_item_lookup_bytes=(c["ranking_queries"] + c["writer_queries"]) * ledger["model_config"]["hidden_size"] * 4
                if gate_source == "item" else 0,
            shared_state_publication_transfer_bytes=broadcast_bytes)
    maintenance = sum(sum(r["maintenance_components_flops"].values()) for r in releases.values())
    query = sum(r["ranking_read_flops"]["total"] for r in releases.values())
    writer = sum(r["writer_read_flops"]["total"] for r in releases.values())
    complete = calibration_flops + maintenance + query + writer
    denominator = ledger["once_rebuild_flops"]
    native_query = sum(r["native_query_flops"] for r in ledger["native_by_release"].values())
    native_append = sum(r["native_append_flops"] for r in ledger["native_by_release"].values())
    # The20% denominator is ONLY first-recommendation KV reconstruction.
    # Ordinary native writes and complete query service remain in both paths.
    common_native = native_append + native_query
    full_service, relay_service = common_native + denominator, common_native + complete
    net_saved = denominator - complete
    source_net_saved = net_saved - source_reconstruction_flops
    cfg = ledger["model_config"]
    layers, hidden, heads = cfg["num_layers"], cfg["hidden_size"], cfg["num_heads"]
    # Independent candidate rows read the native history and their own self
    # entry. No cross-candidate attention is executed in the common reader.
    native_history_attention = layers * (4 * hidden + 3 * heads) * sum(
        c["ranking_dense_pairs"] for c in ledger["by_release"].values())
    native_self_attention = layers * (4 * hidden + 2 * heads) * sum(
        c["ranking_queries"] for c in ledger["by_release"].values())
    native_attention = native_history_attention + native_self_attention
    return dict(status="analytical_prefilter_not_completed_candidate", branch=branch, xi_width=xi_width,
        prefix_only=prefix_only, block_sparse=block_sparse, read_head_rank=head_rank,
        gate_normalization=gate_normalization, table_scope=table_scope,
        preparation_replicas=replicas, preparation_mode=preparation_mode,
        feature_mode=feature_mode, immutable_origin_feature_bank=immutable_bank,
        gate_source=gate_source, stable_anchor_inverse=stable_anchor_inverse,
        extra_gate_item_lookup_bytes=sum(r["gate_item_lookup_bytes"] for r in releases.values()),
        gate_item_lookup_scope="Duplicate transient Current item embedding: one192-dimensional FP32 payload per candidate or appended behavior, including zero-code operations; no added arithmetic or persistent row feature. Population is sharded, not multiplied by replicas.",
        anchor_inverse_scope="FP64 LU inverse estimate approximately8/3 n^3 for actual saved12/6 blocks, stored FP32; no extra per-row FLOPs or state." if stable_anchor_inverse else None,
        catalogue_table_rows=catalogue, source_ledger_catalogue_items=ledger["catalogue_items"],
        catalogue_scope="Full ItemEmbedding rows0..cfg.num_items inclusive; retained source ledger is not rewritten.",
        publication_cost_scope="Actual full-catalogue compilation metadata, repeated according to preparation policy" \
            if publication_metadata is not None else "Analytical compilation preview; actual publication metadata can replace these preparation components.",
        by_release=releases, calibration_flops=calibration_flops, maintenance_flops=maintenance,
        extra_ranking_flops=query, extra_writer_flops=writer, complete_added_flops=complete,
        once_rebuild_flops=denominator, complete_percent_of_once_rebuild=100 * complete / denominator,
        source_reconstruction_sensitivity_flops=source_reconstruction_flops,
        complete_with_source_reconstruction_percent=100 * (complete + source_reconstruction_flops) / denominator,
        remaining_calibration_budget_flops=.2 * denominator - maintenance - query - writer,
        extra_percent_of_native_whole_query=100 * query / native_query,
        native_history_attention_flops=native_history_attention,
        native_inclusive_self_attention_flops=native_self_attention,
        native_history_and_self_attention_flops=native_attention,
        extra_percent_of_native_history_attention=100 * query / native_history_attention,
        extra_percent_of_native_history_and_self_attention=100 * query / native_attention,
        native_attention_scope="Actual ranking dense history pairs: L*(4D+3H) per pair for QK, scale/ELU+1 and AV; independent inclusive self: L*(4D+2H) per candidate. Excludes query embeddings, projections, output/gate layers, normalization and final scoring, which remain in native_whole_query.",
        extra_percent_of_native_append=100 * writer / native_append,
        common_native_incremental_append_flops=native_append,
        common_native_whole_query_flops=native_query,
        total_modeled_service_full_flops=full_service,
        total_modeled_service_relay_flops=relay_service,
        total_modeled_service_relay_with_source_reconstruction_flops=relay_service + source_reconstruction_flops,
        net_saved_flops=net_saved,
        net_saved_with_source_reconstruction_flops=source_net_saved,
        net_saved_percent_of_total_full_service=100 * net_saved / full_service,
        net_saved_with_source_percent_of_total_full_service=100 * source_net_saved / full_service,
        net_kv_rebuild_replacement_saving_percent=100 * net_saved / denominator,
        net_kv_rebuild_replacement_with_source_saving_percent=100 * source_net_saved / denominator,
        service_total_scope="Full=native incremental append+native complete query+once-per-user/release first-query KV rebuild. Relay=the same native append/query+all method-added FLOPs, including calibration and shared preparation. Both inherit the initial V2 cache. This modeled phase excludes foundation-model training, retrieval infrastructure, transfer and loaded wall time. The20% budget remains alladded/KVrebuild, not a fraction of total service.",
        code_bytes_per_row=1152, native_kv_bytes_per_row=9216, code_percent_of_native_kv=12.5,
        read_rank_scope=("Current adapter supports this total accessible width through selected shared/private coordinates. This does not imply fitted weights or quality evidence.")
                        if head_rank in (6, 12, 18) else
                        "Architecture capacity estimate only; this width is not implemented.",
        shared_state_publication_transfer_bytes=sum(r["shared_state_publication_transfer_bytes"] for r in releases.values()),
        transfer_scope="D item table plus dense P/Q tensors only; excludes native caches, code buffers, ordinary serving traffic and other checkpoint tensors.",
        source_status="Supplied calibration and source costs; zero means omitted unknown work, not free calibration.",
        table_scope_limit="Catalogue prepares all valid item IDs prospectively. Resident-only is a lower bound: new/missing-item preparation and historical composition are not yet charged.",
        c_branch_limit="C cannot use shared prefix compilation; its conversion/item work shown here is not complete cost, and old updater retention plus skipped-release replay are omitted." if branch == "C" else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", action="store_true")
    parser.add_argument("--ledger", type=Path, default=ROOT / "results/query_relay_2026_10/cost/event_ledger.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/query_relay_2026_10/cost/prefilter.json")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--gate-normalization", choices=("raw", "rms"), default="rms")
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--preparation-mode", choices=("independent", "compile_once_broadcast"), default="independent")
    parser.add_argument("--feature-mode", choices=("native", "origin_encoder"), default="native")
    args = parser.parse_args()
    ledger = generate_ledger(args.ledger, threads=args.threads) if args.count else read(args.ledger)
    historical = read(SHARED_COST)["shared12_private6"]
    reports = {}
    for branch, r in (("A", 1), ("B", 2), ("B", 3)):
        for prefix, block_sparse in ((True, False), (True, True), (False, False)):
            key = f"{branch}_xi{r}_{'prefix' if prefix else 'dense'}_{'block' if block_sparse else 'dense48'}"
            reports[key] = price(ledger, branch=branch, xi_width=r, prefix_only=prefix,
                block_sparse=block_sparse, gate_normalization=args.gate_normalization,
                replicas=args.replicas, preparation_mode=args.preparation_mode,
                feature_mode=args.feature_mode,
                calibration_flops=historical["shared_calibration_flops"],
                source_reconstruction_flops=historical["source_reconstruction_sensitivity_flops"])
    capacity = {str(rank): price(ledger, head_rank=rank,
        gate_normalization=args.gate_normalization, replicas=args.replicas,
        preparation_mode=args.preparation_mode,
        feature_mode=args.feature_mode,
        source_reconstruction_flops=historical["source_reconstruction_sensitivity_flops"])
        for rank in (2, 4, 6, 8, 12, 18)}
    report = dict(status="prefilter", ledger=str(args.ledger),
        calibration_scope="Historical local32/joint16 cost sensitivity, not fitted candidate calibration.",
        reports=reports, read_rank_capacity_with_unknown_calibration=capacity)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: {k: value[k] for k in (
        "complete_percent_of_once_rebuild", "maintenance_flops", "extra_ranking_flops",
        "extra_writer_flops", "remaining_calibration_budget_flops")}
        for key, value in reports.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
