"""CPU-only full-population analytical costs from real histories/request indices.

No model weights or candidate feature matrices are loaded, and no experiment
is launched. The preparation's ordered populations and source/target metadata
are checked semantically. Both native dense and native tiled initialization /
append alternatives are reported; this does not choose a deployment backend.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import numpy as np

from .common import HERE, DATA, write_json
from .cost import RecFlowCostModel, fitting_flops
from experiments.recflow_full_population.code.protocol import day_bounds_ms


def _metadata_cost(checkpoint, backend):
    path = Path(checkpoint).with_suffix(".json")
    metadata = json.loads(path.read_text())["metadata"]
    arch = metadata["proposal"]["architecture"]
    scale = metadata["scale"]
    shape = arch[scale]
    if (arch["block_variant"] != "hstu_reference" or arch["activation"] != "silu" or
            arch["causal_diagonal"] != "inclusive" or arch["attention_dropout"] != 0 or
            shape["heads"] * shape["head_dim"] != shape["hidden_size"]):
        raise ValueError(f"Unsupported actual checkpoint architecture: {path}")
    return RecFlowCostModel(hidden_size=shape["hidden_size"], num_layers=shape["layers"],
        num_heads=shape["heads"], item_embedding_dim=shape["item_embedding_dimension"],
        relative_position_bias=arch["relative_position_bias"], attention_backend=backend,
        max_seq_len=arch["max_seq_len_including_query"])


def _histogram(values, mask=None):
    x = values if mask is None else values[mask]
    counts = np.bincount(np.asarray(x, np.int64), minlength=1025)
    return {str(i): int(count) for i, count in enumerate(counts) if count}


def _sum_history(histogram, fn):
    return sum(count * fn(int(n)) for n, count in histogram.items())


def _add_prefixes(difference, prefix, count):
    """Histogram scalar writes after eviction, without visiting every event."""
    if not count:
        return
    prefix = min(prefix, 1023)
    growing = min(count, 1023 - prefix)
    if growing:
        difference[prefix] += 1
        difference[prefix + growing] -= 1
    difference[1023] += count - growing
    difference[1024] -= count - growing


def _groups(times):
    return int(1 + np.count_nonzero(times[1:] != times[:-1])) if len(times) else 0


def _append_flops(model, histogram, events, groups):
    # Features/time encoding is shared once per real timestamp append block.
    # Every event still executes independent FIFO-aware Transformer blocks.
    return int(model.embedding(events, calls=groups) +
               sum(count * (model.append(int(n), 1) - model.embedding(1))
                   for n, count in histogram.items()))


def _read_flops(model, q, qn, chunks, chunk_size):
    setup = 3 * model.temporal_frequencies
    row = model.read(0, 1, chunk_size=chunk_size) - setup
    pair = model.read(1, 1, chunk_size=chunk_size) - model.read(0, 1, chunk_size=chunk_size)
    return int(q * row + qn * pair + chunks * setup)


def audit_benchmark(folder, data=DATA):
    """Compare real completed GPU ledger artifacts with CPU input re-enumeration.

    Scores are not recomputed or selected. This imports Torch only for reading
    already-written small user result files on CPU, never model checkpoints.
    """
    import torch
    from .inputs import RecFlowInputs
    folder = Path(folder).resolve()
    plan = json.loads((folder / "plan.json").read_text())
    inputs = RecFlowInputs(data)
    cfg, rows = plan["config"], plan["policies"]
    checked = []
    for edge_entry in plan["selected_edges"]:
        edge_id, day = edge_entry["id"], edge_entry["evaluation_day"]
        cm = _metadata_cost(edge_entry["target_checkpoint"], cfg["backend"])
        am = _metadata_cost(edge_entry["target_checkpoint"], cfg["append_backend"])
        out = folder / edge_id
        if not (out / "summary.json").exists():
            raise ValueError(f"Benchmark has no completed summary: {out}")
        direct_read = direct_append = 0
        for payload in inputs.day(day, uids=plan["eval_uids"]):
            saved = torch.load(out / "users" / f"{payload.uid}.pt", map_location="cpu", weights_only=False)
            ledger = saved["ledger"]
            n = len(payload.initial_history)
            consume = bool(payload.requests or len(payload.events))
            repair = [cm.operation_cost(p["method"], n, selected_tokens=p.get("tokens", 0),
                         interval=p.get("interval"), width=p.get("width"))["total_flops"] if consume else 0 for p in rows]
            read = append = nrequests = ncandidates = 0
            for block in payload.timeline():
                if block.kind == "append":
                    append += am.rolling_append(n, len(block.events))
                    if cfg.get("cuda_graphs", False) and n == 1024:
                        append += (len(block.events) - 1) * 3 * am.temporal_frequencies
                    n = min(1024, n + len(block.events))
                else:
                    for request in block.requests:
                        if request.history_length != n:
                            raise ValueError("CPU benchmark timeline history mismatch")
                        effective_q = len(request.labels)
                        if cfg.get("cuda_graphs", False) and n == 1024:
                            effective_q = ((effective_q + cfg["candidate_chunk"] - 1) // cfg["candidate_chunk"]) * cfg["candidate_chunk"]
                        read += cm.read(n, effective_q, chunk_size=cfg["candidate_chunk"])
                        nrequests += 1
                        ncandidates += len(request.labels)
            expected = dict(repair_flops=repair, ordinary_read_flops_per_policy=read,
                ordinary_append_flops_per_policy=append, requests=nrequests, candidates=ncandidates,
                append_events=len(payload.events), initial_rows=len(payload.initial_history), final_rows=n,
                once_rebuild_flops=cm.full_cache(len(payload.initial_history)) if consume else 0,
                source_cache_flops=cm.full_cache(len(payload.initial_history)))
            for name, value in expected.items():
                if ledger[name] != value:
                    raise ValueError(f"Actual benchmark arithmetic differs: {edge_id}/{payload.uid}/{name}")
            direct_read += read
            direct_append += append
        fitted = json.loads((out / "fit_summary.json").read_text())
        fit_lengths = [len(s.events) for s in inputs.fit_snapshots(day)[:cfg["fit_limit"]]]
        teacher = np.cumsum([2 * cm.full_cache(n) for n in fit_lengths])
        for p in fitted["actual_policies"]:
            u, w = p["fit_users"], p["width"]
            expected = fitting_flops(sum(fit_lengths[:u]), w, cfg["fit_epochs"], cfg["fit"]["batch_rows"],
                int(teacher[u - 1]), 2 * cm.num_layers * cm.num_heads, cm.head_dim)
            if fitted["profiles"][p["tag"]]["flop_breakdown"] != expected:
                raise ValueError(f"Actual fitting ledger differs: {edge_id}/{p['tag']}")
        checked.append(dict(edge_id=edge_id, users=len(plan["eval_uids"]), policies=len(rows),
            fit_policies=len(fitted["actual_policies"]), ordinary_read_flops_per_policy=direct_read,
            ordinary_append_flops_per_policy=direct_append, status="exact_integer_agreement"))
    return dict(folder=str(folder), edges=checked, status="passed", gpu_used=False)


def estimate(preparation_path, data=DATA, check_benchmarks=(), candidate_chunk=None):
    started = time.perf_counter()
    preparation_path, data = Path(preparation_path).resolve(), Path(data).resolve()
    prepared = json.loads(preparation_path.read_text())
    cfg, points, edges = prepared["config"], prepared["policies"], prepared["edges"]
    uids, fit_uids = np.asarray(prepared["eval_uids"], np.int64), np.asarray(prepared["fit_uids"], np.int64)
    population = np.load(data / "population.npz", allow_pickle=False)
    if len(uids) != 30000 or len(np.unique(uids)) != len(uids) or set(uids) != set(population["train"]):
        raise ValueError("Preparation must contain the complete original 30000 quality/cost users")
    if len(fit_uids) != 256 or not set(fit_uids).issubset(set(population["cache_fit"])):
        raise ValueError("Fit users must be the original cache_fit role")
    if np.intersect1d(fit_uids, uids).size:
        raise ValueError("Calibration/quality users overlap")
    expected_order = np.random.default_rng(cfg["population_seed"]).permutation(population["cache_fit"])
    if not np.array_equal(fit_uids, expected_order[:256]):
        raise ValueError("Fit population prefix/order changed from the declared seed")
    history = np.load(data / "history/events.npy", mmap_mode="r", allow_pickle=False)
    offsets = np.load(data / "history/user_offsets.npy", mmap_mode="r", allow_pickle=False)
    bounds = np.asarray([day_bounds_ms(e["evaluation_day"]) for e in edges], np.int64)
    request_counts, last_requests, request_work, selection_paths = [], [], [], []
    chunk = cfg["candidate_chunk"] if candidate_chunk is None else int(candidate_chunk)
    if chunk < 1:
        raise ValueError("Candidate chunk must be positive")
    for edge in edges:
        path = data / f"selections/role2_d{edge['evaluation_day']}_d{edge['evaluation_day']}.npy"
        r = np.load(path, mmap_mode="r", allow_pickle=False)
        if not np.isin(r["uid"], uids).all():
            raise ValueError("Daily request index includes users outside quality population")
        if len(r) != edge["coverage"]["requests"]:
            raise ValueError("Daily index request count changed from preparation")
        counts = np.bincount(r["uid"], minlength=len(offsets) - 1)
        last = np.full(len(offsets) - 1, -1, np.int64)
        np.maximum.at(last, r["uid"], r["ts"])
        q = r["stop"] - r["start"]
        if (q <= 0).any() or int(q.sum()) != edge["coverage"]["candidates"]:
            raise ValueError("Full candidate counts differ from preparation")
        n = r["history_length"].astype(np.int64)
        padded_q = np.where(n == 1024, ((q + chunk - 1) // chunk) * chunk, q)
        request_counts.append(counts)
        last_requests.append(last)
        request_work.append(dict(requests=len(r), candidates=int(q.sum()),
            candidate_history_pairs=int((q * n).sum()), query_chunk_calls=int(((q + chunk - 1) // chunk).sum()),
            graph_padded_candidates=int(padded_q.sum()), graph_padding_candidate_rows=int((padded_q - q).sum()),
            graph_candidate_history_pairs=int((padded_q * n).sum()),
            graph_eligible_full_history_requests=int((n == 1024).sum()),
            graph_eligible_full_history_candidates=int(q[n == 1024].sum()),
            sum_request_history_lengths=int(n.sum()), sum_request_history_lengths_squared=int((n * n).sum()),
            candidate_chunk_size=chunk))
        selection_paths.append(str(path))
    n_edges = len(edges)
    lengths = np.zeros((n_edges, len(uids)), np.int16)
    counts = np.zeros((n_edges, len(uids)), np.int32)
    first = np.zeros((n_edges, len(uids)), bool)
    query_active = np.zeros((n_edges, len(uids)), bool)
    diffs = {key: np.zeros((n_edges, 1025), np.int64) for key in ("all", "after_last_query", "no_query")}
    append_rows = {key: np.zeros(n_edges, np.int64) for key in diffs}
    append_groups = {key: np.zeros(n_edges, np.int64) for key in diffs}
    graph_append_extra_calls = {key: np.zeros(n_edges, np.int64) for key in diffs}
    for j, uid in enumerate(uids):
        start, stop = offsets[uid:uid + 2]
        times = history["ts"][start:stop]
        boundaries = np.searchsorted(times, bounds.ravel(), side="left").reshape(n_edges, 2)
        for i, (begin, end) in enumerate(boundaries):
            n, m = min(1024, int(begin)), int(end - begin)
            length = times[begin:end]
            active = bool(request_counts[i][uid])
            lengths[i, j], counts[i, j] = n, m
            query_active[i, j], first[i, j] = active, bool(active or m)
            _add_prefixes(diffs["all"][i], n, m)
            append_rows["all"][i] += m
            append_groups["all"][i] += _groups(length)
            starts = np.r_[0, np.flatnonzero(length[1:] != length[:-1]) + 1] if m else np.empty(0, np.int64)
            block_sizes = np.diff(np.r_[starts, m]) if m else np.empty(0, np.int64)
            # Eligibility is checked once at the start of each timestamp block.
            # A short prefix that reaches capacity inside that block stays eager.
            extra_calls = np.where(n + starts >= 1024, block_sizes - 1, 0)
            graph_append_extra_calls["all"][i] += int(extra_calls.sum())
            if not active:
                _add_prefixes(diffs["no_query"][i], n, m)
                append_rows["no_query"][i] += m
                append_groups["no_query"][i] += _groups(length)
                graph_append_extra_calls["no_query"][i] += int(extra_calls.sum())
            elif m:
                before = int(np.searchsorted(length, last_requests[i][uid], side="left"))
                _add_prefixes(diffs["after_last_query"][i], n + before, m - before)
                append_rows["after_last_query"][i] += m - before
                append_groups["after_last_query"][i] += _groups(length[before:])
                graph_append_extra_calls["after_last_query"][i] += int(extra_calls[starts >= before].sum())
    fit_lengths = np.zeros((n_edges, len(fit_uids)), np.int16)
    for j, uid in enumerate(fit_uids):
        start, stop = offsets[uid:uid + 2]
        times = history["ts"][start:stop]
        fit_lengths[:, j] = np.minimum(1024, np.searchsorted(times, bounds[:, 0], side="left"))
    result_edges = []
    for i, edge in enumerate(edges):
        n, consumer, active = lengths[i].astype(np.int64), first[i], query_active[i]
        histogram = _histogram(n, consumer)
        all_histogram = _histogram(n)
        actual = dict(quality_users=len(uids), first_consumer_users=int(consumer.sum()),
            request_active_users=int(active.sum()), behavior_only_users=int((consumer & ~active).sum()),
            completely_inactive_users=int((~consumer).sum()),
            once_repair_history_rows=int(n[consumer].sum()),
            sum_repair_history_lengths_squared=int((n[consumer] ** 2).sum()),
            initial_history_rows=int(n.sum()), full_initial_history_users=int((n == 1024).sum()),
            empty_initial_history_users=int((n == 0).sum()), appended_history_events=int(counts[i].sum()),
            maintenance_after_last_request_events=int(append_rows["after_last_query"][i] + append_rows["no_query"][i]),
            after_last_query_append_events=int(append_rows["after_last_query"][i]),
            no_query_append_events=int(append_rows["no_query"][i]))
        for key in ("quality_users", "first_consumer_users", "request_active_users", "once_repair_history_rows",
                    "initial_history_rows", "full_initial_history_users", "empty_initial_history_users",
                    "appended_history_events", "maintenance_after_last_request_events"):
            if actual[key] != edge["coverage"][key]:
                raise ValueError(f"Independent shape accounting differs for {edge['id']}: {key}")
        if int(fit_lengths[i].sum()) != edge["fit_history_rows"]:
            raise ValueError("Independent fit prefix lengths differ from preparation")
        append_histograms = {}
        for key in diffs:
            hist = np.cumsum(diffs[key][i])[:1024]
            if int(hist.sum()) != append_rows[key][i] or (hist < 0).any():
                raise ValueError("FIFO prefix histogram does not cover every behavior")
            append_histograms[key] = {str(j): int(count) for j, count in enumerate(hist) if count}
        backend_costs = {}
        for backend in ("torch", "triton"):
            cm = _metadata_cost(edge["target_checkpoint"], backend)
            source = _metadata_cost(edge["source_checkpoint"], backend)
            if asdict(source) != asdict(cm):
                raise ValueError("Source/target shape differs; use separate ledgers")
            denominator = _sum_history(histogram, cm.full_cache)
            fit_prefix_teacher = np.cumsum([2 * cm.full_cache(int(n)) for n in fit_lengths[i]]).tolist()
            point_costs = []
            for p in points:
                work = {key: 0 for key in ("recompute_flops", "selection_flops", "total_flops", "old_boundary_bytes")}
                sort_comparisons = 0.
                for hist_n, population_count in histogram.items():
                    piece = cm.operation_cost(p["method"], int(hist_n), selected_tokens=p.get("tokens", 0),
                                              interval=p.get("interval"), width=p.get("width"))
                    for key in work:
                        work[key] += population_count * piece[key]
                    sort_comparisons += population_count * piece["selection_sort_comparisons_estimate"]
                fit = None
                if p["method"] == "kv_map":
                    users = p["fit_users"]
                    observations = int(fit_lengths[i, :users].sum())
                    fit = fitting_flops(observations, p["width"], cfg["fit"]["epochs"], cfg["fit"]["batch_rows"],
                        fit_prefix_teacher[users - 1], 2 * cm.num_layers * cm.num_heads, cm.head_dim)
                    fit.update(fit_uid_order_reference="top-level fit_uid_order", prefix_stop_exclusive=users,
                        prerelease_cutoff_ms=int(bounds[i, 0]), fit_role="cache_fit",
                        empty_users=int((fit_lengths[i, :users] == 0).sum()),
                        label_or_future_activity_selection=False)
                fit_flops = fit["total_flops"] if fit is not None else 0
                point_costs.append(dict(**p, **work, fit_flops=fit_flops,
                    additional_flops=work["total_flops"] + fit_flops,
                    percent_of_once_rebuild=100 * (work["total_flops"] + fit_flops) / denominator,
                    sort_comparisons_estimate=sort_comparisons, fit=fit,
                    map_weight_bytes=(fit["parameters"] + 4 * 2 * cm.num_layers * cm.num_heads * cm.head_dim) * 4 if fit else 0))
            ordinary_append = {key: _append_flops(cm, append_histograms[key], int(append_rows[key][i]),
                                                int(append_groups[key][i])) for key in diffs}
            rw = request_work[i]
            ordinary_read = _read_flops(cm, rw["candidates"], rw["candidate_history_pairs"],
                                       rw["query_chunk_calls"], chunk)
            graph_read = _read_flops(cm, rw["graph_padded_candidates"], rw["graph_candidate_history_pairs"],
                                    rw["query_chunk_calls"], chunk)
            graph_append = {key: ordinary_append[key] + int(graph_append_extra_calls[key][i]) *
                                 3 * cm.temporal_frequencies for key in diffs}
            starts = {p["interval"][0] for p in points if p["method"] == "layer" and p["interval"][0]}
            backend_costs[backend] = dict(model=asdict(cm), once_rebuild_flops=denominator,
                common_source_sunk_reference_all_population_flops=_sum_history(all_histogram, cm.full_cache),
                actual_first_consumer_source_construction_flops=denominator,
                ordinary_read_flops_per_policy=ordinary_read,
                ordinary_append_flops_per_policy=ordinary_append["all"],
                ordinary_graph_read_flops_per_policy=graph_read,
                ordinary_graph_append_flops_per_policy=graph_append["all"],
                graph_padding_candidate_rows=rw["graph_padding_candidate_rows"],
                after_last_query_graph_append_flops_per_policy=graph_append["after_last_query"],
                no_query_graph_append_flops_per_policy=graph_append["no_query"],
                after_last_query_append_flops_per_policy=ordinary_append["after_last_query"],
                no_query_append_flops_per_policy=ordinary_append["no_query"],
                ordinary_all_29_read_flops_standalone=ordinary_read * len(points),
                ordinary_all_29_append_flops_standalone=ordinary_append["all"] * len(points),
                old_boundary_union_bytes=int(n[consumer].sum()) * cm.hidden_size * 4 * len(starts),
                old_boundary_distinct_starts_zero_based=sorted(starts), policies=point_costs)
            if point_costs[0]["additional_flops"] != 0 or point_costs[1]["additional_flops"] != denominator:
                raise ValueError("Reuse/Rebuild endpoint cost failed")
        result_edges.append(dict(edge_id=edge["id"], evaluation_day=edge["evaluation_day"],
            interval_ms_half_open=bounds[i].tolist(), source_checkpoint=edge["source_checkpoint"],
            target_checkpoint=edge["target_checkpoint"], request_selection=selection_paths[i],
            actual_coverage=actual, request_workload=request_work[i],
            first_consumer_history_length_histogram=histogram,
            all_population_history_length_histogram=all_histogram,
            append_prefix_histograms=append_histograms,
            append_timestamp_blocks={key: int(append_groups[key][i]) for key in diffs},
            graph_append_extra_frequency_calls={key: int(graph_append_extra_calls[key][i]) for key in diffs},
            fit_prerelease_history_lengths_in_uid_order=fit_lengths[i].tolist(), by_backend=backend_costs))
        print(json.dumps(dict(stage="analytic_costs", edge=edge["id"], first_consumers=int(consumer.sum()),
            rebuild_torch_flops=backend_costs["torch"]["once_rebuild_flops"],
            rebuild_triton_flops=backend_costs["triton"]["once_rebuild_flops"])), flush=True)
    benchmark_audits = [audit_benchmark(path, data) for path in check_benchmarks]
    return dict(status="complete_cpu_analytic_preparation", prepared_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        source_preparation=str(preparation_path), data=str(data), policies=points,
        candidate_chunk_size=chunk, preparation_candidate_chunk_size=cfg["candidate_chunk"],
        population_users=len(uids), fit_uid_order=fit_uids.tolist(), edges=result_edges,
        existing_benchmark_ledger_audits=benchmark_audits,
        edge_count=len(result_edges), policy_count=len(points), hash_protection=False,
        measured_gpu_quality=False, formal_started=False, elapsed_seconds=time.perf_counter()-started,
        arithmetic_scope="Analytical equation FLOPs: FMA=2, explicit SiLU/LayerNorm/features/relative bias; fused instruction count and timing are not inferred.",
        standalone_policy_accounting="Each method pays its own preparation/teacher/fit/deployment. Sharing inputs, teacher capture, or native prefixes across the 29-method benchmark is not a standalone-method FLOP discount.",
        once_rebuild_denominator="One current-model history encoding for every fixed-population user with a request or behavior consumer; fully inactive users remain in the population but incur no release repair. Histories are the release-midnight prefix, with cap1024.",
        source_reference="common_source_sunk_reference_all_population_flops is an inherited serving reference, not current experiment work for silent users; actual_first_consumer_source_construction_flops reflects source caches the engine constructs.",
        backend_scope="torch/triton are alternative native preparation and scalar-append executions. Arbitrary selected rows and candidate reads remain dense. Combine one backend's repair/fit with the chosen append backend; no backend is selected by this CPU report.",
        boundary_scope="old_boundary_bytes per non-prefix Layer is one FP32[N,H] actual source-layer input per repaired user. Union bytes deduplicate equal starting layers only. These are summed hypothetical resident bytes across the population, not measured peak GPU memory; capture/copies are not FLOPs.",
        selector_scope="Deviation pays full-history first-layer K/V source difference plus stable sorting work for each partial policy. Full/zero-token endpoints bypass selection. Comparisons/indexing are reported separately, never called FLOPs.",
        fit_scope="Six fixed configurations use prefixes of the top-level ordered 256 cache_fit UIDs, strictly before the release day. Real variable-length row counts, final partial minibatches, source+teacher construction, normalization, diagnostics, backward and AdamW are charged once per edge/policy; no future labels.",
        ordinary_scope="Ordinary reads/append are outside release-repair ratios but fully reported. No-query maintenance and post-last-query maintenance are disjoint. Every timestamp append group encodes features once, then every behavior uses scalar evict-before FIFO.",
        graph_scope="Additional graph ordinary-work fields charge candidate padding only at history1024, and per-event feature frequency setup only when the timestamp block starts full. Short-to-full transition blocks stay eager. Chunk size is recorded in request_workload. One-time warmup/capture arithmetic and copies are excluded from these steady work counts; measured setup time must be reported separately.",
        timing_scope="A work-count basis for later measured-stage extrapolation only. Analytic FLOPs alone do not establish wall time, especially launch-bound scalar append or shared-method batching.")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--preparation", type=Path, default=HERE / "runs/preparation/preparation.json")
    parser.add_argument("--output", type=Path, default=HERE / "runs/preparation/analytic_costs.json")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--check-benchmark", type=Path, action="append", default=[])
    parser.add_argument("--candidate-chunk", type=int,
                        help="Explicit executed candidate chunk override; original preparation is not rewritten")
    args = parser.parse_args()
    result = estimate(args.preparation, args.data, args.check_benchmark, args.candidate_chunk)
    write_json(args.output, result)
    print(json.dumps(dict(output=str(args.output), edges=result["edge_count"], policies=result["policy_count"],
                         seconds=result["elapsed_seconds"], formal_started=False)), flush=True)


if __name__ == "__main__":
    main()
