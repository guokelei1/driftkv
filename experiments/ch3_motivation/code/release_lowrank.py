"""Joint E14 request-instance compression in one fixed publishing-time row space.

Different reads of the same item have distinct matrix rows and real full-H
queries. The historical columns are immutable publishing-time old rows. FIFO
eviction clips columns only during actual read replay, never during centering
or SVD. Candidate reads and compression replays do not write persistent state.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter, defaultdict
import hashlib
import time

import numpy as np
import torch

from experiments.ch3_motivation.code.fg_trajectory_control import (
    make_paired_trajectories, score_fg_timestamp, H_DISABLED_DIAGNOSTIC,
)
from experiments.ch3_motivation.code.analysis import decompose_interaction
from experiments.ch3_motivation.code.lowrank_batch import BatchedReconstruction, decompose_interaction_batch
from experiments.ch3_motivation.code.lowrank_rank_batch import reconstruct_rank_batch
from experiments.ch3_motivation.code.probe_projected_fast import (
    ProjectedFastProbeTrajectory, project_row_features, projected_interaction_correction,
)
from hstu_kvcache.models import HSTUKVCache

VARIANTS = dict(both_norm=dict(only="both", normalize_kv=True),
    both_raw=dict(only="both", normalize_kv=False), key_norm=dict(only="key", normalize_kv=True),
    value_norm=dict(only="value", normalize_kv=True))


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@dataclass
class RequestPanel:
    index: int
    instances: list[dict]
    full_scores: torch.Tensor | None = None
    full_queries: torch.Tensor | None = None

    def positions_at(self, timestamp):
        return [index for index, instance in enumerate(self.instances)
                if instance["timestamp"] == timestamp]


def plan_request_panels(requests_by_timestamp, *, uid, edge, known_items, size=230, seed=20261022):
    """Keep all actual feedback instances; pad only the final chronological panel.

    Padding samples unique (timestamp,item) pairs from that panel's actual
    timestamps and the inclusive known-item catalog. No labels/model gains
    select samples. A repeated item at a different time is a different row.
    """
    if size < 1 or known_items < 1:
        raise ValueError("positive panel size and known-item catalog required")
    actual, ordinals = [], Counter()
    for timestamp, rows in sorted(requests_by_timestamp.items()):
        for row in sorted(rows, key=lambda row: int(row["raw_item_id"])):
            key = int(uid), int(timestamp), int(row["raw_item_id"])
            if int(row["uid"]) != int(uid) or int(row["query_timestamp"]) != int(timestamp):
                raise ValueError("request instances must match the real user/time keys")
            if not 1 <= int(row["item_idx"]) <= known_items:
                raise ValueError("actual request target outside the known-item catalog")
            ordinal = ordinals[key]
            ordinals[key] += 1
            actual.append(dict(timestamp=int(timestamp), item_idx=int(row["item_idx"]),
                actual=dict(row, request_instance_ordinal=ordinal)))
    panels = []
    for start in range(0, len(actual), size):
        instances = actual[start:start+size]
        index = len(panels)
        times = sorted({row["timestamp"] for row in instances})
        occupied = {(row["timestamp"], row["item_idx"]) for row in instances}
        if len(times)*known_items-len(occupied) < size-len(instances):
            raise ValueError("not enough distinct time/item instances to fill the requested panel")
        digest = hashlib.sha256(f"ch3-e6-release/{seed}/{edge}/{uid}/{index}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        while len(instances) < size:
            timestamp, item = times[int(rng.integers(len(times)))], int(rng.integers(1, known_items+1))
            if (timestamp, item) not in occupied:
                occupied.add((timestamp, item))
                instances.append(dict(timestamp=timestamp, item_idx=item, actual=None))
        panels.append(RequestPanel(index, instances))
    return panels


@dataclass
class CPUSnapshot:
    timestamp: int
    cache: HSTUKVCache
    source_offset: int
    source_rows: int
    query_gap: int

    @property
    def bytes(self):
        return sum(value.numel()*value.element_size() for value in (self.cache.k, self.cache.v))


@dataclass
class ReleaseContext:
    initial_source: HSTUKVCache
    initial_ids: torch.Tensor
    initial_features: tuple
    snapshots: dict[int, CPUSnapshot]

    @property
    def initial_rows(self):
        return self.initial_source.seq_len


@torch.no_grad()
def capture_release_user(parent, current, probe, fg_probe, events, requests_by_timestamp, uid, cfg, *,
                         known_items, on_panel_ready, candidate_size=230, execution="projected",
                         fg_pool=None, full_h_pool=None):
    """One causal pass, CPU cache snapshots, one resident h panel at a time.

    on_panel_ready(panel,h[L,H,C,N0,2D],context) runs only transient reads while
    the writer pauses. It returns score rows and a compact numeric report.
    The callback must not advance either persistent trajectory.
    """
    started = time.perf_counter()
    panels = plan_request_panels(requests_by_timestamp, uid=uid, edge=cfg["edge"],
        known_items=known_items, size=candidate_size)
    if not panels:
        raise ValueError("release compression requires actual feedback instances")
    normalized_requests = defaultdict(list)
    for panel in panels:
        for instance in panel.instances:
            if instance["actual"] is not None:
                normalized_requests[instance["timestamp"]].append(instance["actual"])
    pair = make_paired_trajectories(parent, current, fg_probe, probe, events, cfg,
        fg_pool=fg_pool, full_h_pool=full_h_pool, execution=execution)
    trajectory, fg_trajectory = pair.full_h, pair.fg
    trajectory._prepare()
    n0 = trajectory.source.seq_len
    context = ReleaseContext(trajectory.source, trajectory.source_ids, tuple(trajectory.features), {})
    if isinstance(trajectory, ProjectedFastProbeTrajectory):
        trajectory._ensure_padded()
        # Keep the publishing row map independent of the mutable FIFO table.
        initial_projected = tuple(rows[:, :, :n0].clone() for rows in trajectory.padded)
    else:
        initial_projected = project_row_features(probe, trajectory.features)
    device = trajectory.device
    _sync(device)
    timings = dict(initial_prefill_and_prepare=time.perf_counter()-started,
        fg_own_advance_and_read=0., full_h_advance=0., cpu_snapshot_copy=0.,
        full_h_panel_capture=0.)
    output, reports = [], []
    active_panel, h_tensor, filled = None, None, None
    maximum_h_bytes = 0
    for timestamp, actual_rows in sorted(normalized_requests.items()):
        timestamp = int(timestamp)
        stage_started = time.perf_counter()
        output.extend(score_fg_timestamp(fg_trajectory, actual_rows, uid=uid, timestamp=timestamp,
            query_batch=int(cfg.get("query_batch", 16))))
        _sync(device)
        timings["fg_own_advance_and_read"] += time.perf_counter()-stage_started
        stage_started = time.perf_counter()
        trajectory.advance(timestamp)
        _sync(device)
        timings["full_h_advance"] += time.perf_counter()-stage_started
        source_rows = trajectory.source.seq_len
        offset = n0-source_rows
        if not 0 <= offset <= n0:
            raise ValueError("FIFO inherited-row coordinates are inconsistent")
        gap = 0 if trajectory.native.last_timestamp is None else timestamp-trajectory.native.last_timestamp
        stage_started = time.perf_counter()
        snapshot = CPUSnapshot(timestamp,
            HSTUKVCache(trajectory.cache.k.to("cpu", copy=True), trajectory.cache.v.to("cpu", copy=True),
                trajectory.cache.seq_len), offset, source_rows, int(gap))
        context.snapshots[timestamp] = snapshot
        timings["cpu_snapshot_copy"] += time.perf_counter()-stage_started
        delta = torch.as_tensor([gap], device=device, dtype=torch.float32)
        for panel in panels:
            positions = panel.positions_at(timestamp)
            if not positions:
                continue
            if active_panel is None:
                active_panel = panel.index
                h_tensor = trajectory.cache.k.new_empty(probe.cfg.num_layers, probe.cfg.num_heads,
                    candidate_size, n0, 2*probe.cfg.head_dim)
                maximum_h_bytes = max(maximum_h_bytes, h_tensor.numel()*h_tensor.element_size())
                filled = np.zeros(candidate_size, dtype=bool)
                panel.full_scores = torch.empty(candidate_size)
                panel.full_queries = torch.empty(probe.cfg.num_layers, probe.cfg.num_heads,
                    candidate_size, probe.cfg.head_dim)
            if active_panel != panel.index:
                raise ValueError("chronological request panels overlapped unexpectedly")
            stage_started = time.perf_counter()
            ids = torch.as_tensor([[panel.instances[index]["item_idx"] for index in positions]], device=device)
            indices = torch.as_tensor(positions, device=device)
            last_key, last_value = None, None

            def capture(layer, qc, rows, query_start, row_start):
                nonlocal last_key, last_value
                key = layer, query_start
                if last_key != key:
                    last_value = projected_interaction_correction(probe, layer, qc, initial_projected[layer])
                    selected = indices[query_start:query_start+qc.shape[2]]
                    h_tensor[layer].index_copy_(1, selected, last_value[0])
                    last_key = key
                return last_value[:, :, :, offset+row_start:offset+row_start+rows.shape[2]]

            scores, trace = probe.score(current, trajectory.source, trajectory.source_ids, ids, delta,
                prepared_cache=trajectory.cache, read_features=trajectory.features,
                interaction_override=capture, trace=True)
            if source_rows == 0 and n0:
                # The full-H reader no longer calls h after causal clearance.
                # Its query still has an offline h(q,all publishing rows) row;
                # actual rank reads will have zero inherited columns to use.
                for layer, q in enumerate(trace.queries):
                    value = projected_interaction_correction(probe, layer, q, initial_projected[layer])
                    h_tensor[layer].index_copy_(1, indices, value[0])
            panel.full_scores[positions] = scores[0].cpu()
            for layer, q in enumerate(trace.queries):
                panel.full_queries[layer].index_copy_(1, torch.as_tensor(positions), q[0].cpu())
            _sync(device)
            timings["full_h_panel_capture"] += time.perf_counter()-stage_started
            filled[positions] = True
            last_value = None
            if filled.all():
                before = trajectory.native.cursor, fg_trajectory.native.cursor
                rows, report = on_panel_ready(panel, h_tensor, context)
                if before != (trajectory.native.cursor, fg_trajectory.native.cursor):
                    raise ValueError("compression callback advanced a persistent trajectory")
                output.extend(rows)
                reports.append(report)
                active_panel, h_tensor, filled = None, None, None
    if active_panel is not None:
        raise ValueError("a joint release panel has unfilled request-instance rows")
    stop_after_last = bool(cfg.get("stop_after_last_query", False))
    if not stop_after_last:
        trajectory.advance(int(cfg["end_day"])*86400)
        fg_trajectory.advance(int(cfg["end_day"])*86400)
    actual_instances = sum(len(rows) for rows in requests_by_timestamp.values())
    report = dict(uid=uid, panels=reports, actual_feedback_instances=actual_instances,
        panel_instances=len(panels)*candidate_size, snapshot_count=len(context.snapshots),
        cpu_snapshot_bytes=sum(snapshot.bytes for snapshot in context.snapshots.values()),
        maximum_resident_h_panel_bytes=maximum_h_bytes, initial_history_rows=n0,
        cleared_feedback_instances=sum(len(requests_by_timestamp[timestamp]) for timestamp, snapshot
            in context.snapshots.items() if snapshot.source_rows == 0),
        inactive_column_policy="all publishing rows remain in the joint matrix; only actual reads crop FIFO-surviving columns",
        initial_prefill_calls=1, full_h_trajectory=trajectory.report(), trained_fg_trajectory=fg_trajectory.report(),
        unscored_suffix_advanced=not stop_after_last,
        stage_seconds=timings,
        seconds=time.perf_counter()-started,
        scope="hindsight request-instance compression; every rank reads fixed full-H snapshots and does not write its own trajectory")
    return output, report


@torch.no_grad()
def decompose_release_heads(h, *, batch_heads=10, seed=17, **options):
    """Batch across layers while preserving each old layer's random sketch.

    Matrix order is layer-major/head-minor. Changing batch size changes only
    linear-algebra batching, never the matrix definition or sketch seed.
    """
    if h.ndim != 5 or batch_heads < 1:
        raise ValueError("release h must be [layers,heads,candidates,rows,2D]")
    flattened = h.flatten(0, 1)
    decompositions = []
    for first in range(0, len(flattened), batch_heads):
        stop = min(first+batch_heads, len(flattened))
        seeds = [seed+index//h.shape[1] for index in range(first, stop)]
        decompositions.extend(decompose_interaction_batch(flattened[first:stop],
            seed=seeds, **options))
    return decompositions


@torch.no_grad()
def evaluate_release_panel(current, probe, panel, h, context, *, ranks=(0, 1, 2, 4, 6, 8, 16),
        variants=("both_norm",), method="randomized", oversample=8, power_iterations=1,
        seed=17, batch_heads=10, batch_rank_scores=False, verify_exact_heads=2,
        compare_rank_execution=False, energy_guard=True, energy_threshold=.98,
        energy_tolerance=1e-5):
    """One joint decomposition; all actual-time interventions use CPU snapshots.

    Full reconstruction is checked on the exact assigned candidate batch at
    every time. Actual score rows are emitted once, including repeated targets
    and requests whose inherited columns have all been causally cleared.
    """
    device = h.device
    times = sorted({instance["timestamp"] for instance in panel.instances})
    all_cleared = all(context.snapshots[time].source_rows == 0 for time in times)
    decompositions, checks, energies = {}, [], []
    exact_check_seconds = guard_seconds = fallback_seconds = 0.
    guard_records = []
    _sync(device)
    started = time.perf_counter()
    for variant in variants:
        options = VARIANTS[variant]
        decompositions[variant] = {}
        selected = set(np.linspace(0, probe.cfg.num_layers*probe.cfg.num_heads-1,
            min(verify_exact_heads, probe.cfg.num_layers*probe.cfg.num_heads), dtype=int).tolist())
        if not context.initial_rows or all_cleared:
            continue
        flat = decompose_release_heads(h, batch_heads=batch_heads, seed=seed, **options,
            method=method, max_rank=max(ranks), oversample=oversample,
            power_iterations=power_iterations, energy_guard=energy_guard,
            energy_threshold=energy_threshold, energy_tolerance=energy_tolerance)
        for obj in flat:
            if obj.metadata["execution_batch_index"] == 0:
                guard_seconds += obj.metadata["energy_guard_batch_seconds"]
                fallback_seconds += obj.metadata["exact_fallback_batch_seconds"]
        for layer in range(probe.cfg.num_layers):
            heads = flat[layer*probe.cfg.num_heads:(layer+1)*probe.cfg.num_heads]
            decompositions[variant][layer] = BatchedReconstruction(h[layer], heads)
            for head, obj in enumerate(heads):
                if obj.metadata["energy_guard"] is not None:
                    guard_records.append(dict(variant=variant, layer=layer, head=head,
                        **obj.metadata["energy_guard"]))
                energies.append(dict(variant=variant, layer=layer, head=head,
                    retained_at_maximum_rank=obj.metadata["normalized_retained_energy_at_max_rank"]))
                if method == "randomized" and layer*probe.cfg.num_heads+head in selected:
                    _sync(device)
                    check_started = time.perf_counter()
                    exact = decompose_interaction(h[layer, head], **options, method="exact", max_rank=max(ranks))
                    points = []
                    for rank in (2, 6, max(ranks)):
                        count = min(rank, exact.max_rank)
                        wanted = float(exact.singular_values[:count].square().sum())
                        got = float(obj.singular_values[:count].square().sum())
                        ratio = got/wanted if wanted > 1e-20 else 1.
                        approximate_h, exact_h = obj.reconstruct(count), exact.reconstruct(count)
                        selected_block = obj.selected_slice
                        disagreement = ((approximate_h-exact_h)[..., selected_block]/obj.block_scales).square().sum()
                        residual = ((h[layer, head]-approximate_h)[..., selected_block]/obj.block_scales).square().sum()
                        exact_residual = ((h[layer, head]-exact_h)[..., selected_block]/obj.block_scales).square().sum()
                        energy = obj.metadata["normalized_interaction_squared_norm"]
                        guard = obj.metadata["energy_guard"]
                        original_energy = None if guard is None else sum(
                            value**2 for value in guard["all_sketch_singular_values"][:count])
                        points.append(dict(rank=rank, captured_energy_ratio_to_exact=ratio,
                            original_sketch_energy_ratio_to_independent_exact=(original_energy/wanted
                                if original_energy is not None and wanted > 1e-20 else None),
                            normalized_interaction_reconstruction_disagreement=float(disagreement)/energy if energy else 0.,
                            normalized_interaction_residual=float(residual)/energy if energy else 0.,
                            exact_normalized_interaction_residual=float(exact_residual)/energy if energy else 0.))
                        if ratio < .98:
                            raise RuntimeError("release SVD approximation below98% exact energy; improve accuracy before measurement")
                    checks.append(dict(variant=variant, layer=layer, head=head, points=points,
                        normalized_matrix_total_energy=obj.metadata["normalized_interaction_squared_norm"],
                        independent_scalar_total_energy=exact.metadata["normalized_interaction_squared_norm"]))
                    del exact, approximate_h, exact_h
                    _sync(device)
                    exact_check_seconds += time.perf_counter()-check_started
    _sync(device)
    decomposition_seconds = time.perf_counter()-started
    output, frame_reports = [], []
    read_started = time.perf_counter()
    for timestamp in times:
        snapshot = context.snapshots[timestamp]
        offset, inherited = snapshot.source_offset, snapshot.source_rows
        cache = HSTUKVCache(snapshot.cache.k.to(device), snapshot.cache.v.to(device), snapshot.cache.seq_len)
        source = HSTUKVCache(context.initial_source.k[:, :, offset:], context.initial_source.v[:, :, offset:], inherited)
        source_ids = context.initial_ids[:, offset:]
        features = tuple(rows[:, :, offset:] for rows in context.initial_features)
        assigned = panel.positions_at(timestamp)
        actual = [index for index in assigned if panel.instances[index]["actual"] is not None]
        assigned_indices = torch.as_tensor(assigned, device=device)
        assigned_ids = torch.as_tensor([[panel.instances[index]["item_idx"] for index in assigned]], device=device)
        delta = torch.as_tensor([snapshot.query_gap], device=device, dtype=torch.float32)
        def original(indices):
            def fixed(layer, qc, rows, query_start, row_start):
                selected = indices[query_start:query_start+qc.shape[2]]
                return h[layer][:, selected, offset+row_start:offset+row_start+rows.shape[2]][None]
            return fixed
        restored, _ = probe.score(current, source, source_ids, assigned_ids, delta,
            prepared_cache=cache, read_features=features, interaction_override=original(assigned_indices))
        expected = panel.full_scores[assigned].to(device)
        restore_error = float((restored[0]-expected).abs().max())
        torch.testing.assert_close(restored[0], expected, atol=1e-5, rtol=1e-6)
        indices = torch.as_tensor(actual, device=device)
        ids = torch.as_tensor([[panel.instances[index]["item_idx"] for index in actual]], device=device)
        full, _ = probe.score(current, source, source_ids, ids, delta,
            prepared_cache=cache, read_features=features, interaction_override=original(indices))
        actual_shape_error = float((full[0]-panel.full_scores[actual].to(device)).abs().max())
        torch.testing.assert_close(full[0], panel.full_scores[actual].to(device), atol=1e-5, rtol=1e-6)
        def zero_h(layer, qc, rows, query_start, row_start):
            return qc.new_zeros(*qc.shape[:3], rows.shape[2], 2*probe.cfg.head_dim)
        disabled, _ = probe.score(current, source, source_ids, ids, delta,
            prepared_cache=cache, read_features=features, interaction_override=zero_h)
        predictions = dict(full=full[0], **{H_DISABLED_DIAGNOSTIC: disabled[0]})
        rank_execution_checks = []
        for variant in variants:
            if inherited == 0:
                predictions.update({f"{variant}_r{rank}": full[0] for rank in ranks})
                continue
            def rank_scores(batched):
                _sync(device)
                score_started = time.perf_counter()
                if batched:
                    def replacement(layer, qc, rows, query_start, row_start):
                        selected = indices[query_start:query_start+qc.shape[2]]
                        return reconstruct_rank_batch(decompositions[variant][layer], ranks, candidates=selected,
                            rows=slice(offset+row_start, offset+row_start+rows.shape[2]))
                    values, _ = probe.score(current, source, source_ids, ids.expand(len(ranks), -1),
                        delta.expand(len(ranks)), prepared_cache=cache, read_features=features,
                        interaction_override=replacement)
                else:
                    values = []
                    for rank in ranks:
                        def replacement(layer, qc, rows, query_start, row_start):
                            selected = indices[query_start:query_start+qc.shape[2]]
                            obj = decompositions[variant][layer]
                            return obj.reconstruct(min(rank, obj.max_rank), candidates=selected,
                                rows=slice(offset+row_start, offset+row_start+rows.shape[2]))[None]
                        value, _ = probe.score(current, source, source_ids, ids, delta, prepared_cache=cache,
                            read_features=features, interaction_override=replacement)
                        values.append(value[0])
                    values = torch.stack(values)
                _sync(device)
                return values, time.perf_counter()-score_started
            values, score_seconds = rank_scores(batch_rank_scores)
            if compare_rank_execution:
                alternate, alternate_seconds = rank_scores(not batch_rank_scores)
                error = float((values-alternate).abs().max())
                torch.testing.assert_close(values, alternate, atol=1e-5, rtol=1e-6)
                rank_execution_checks.append(dict(variant=variant, logit_max_error=error,
                    rank_logit_max_errors={str(rank): float((values[index]-alternate[index]).abs().max())
                        for index, rank in enumerate(ranks)},
                    serial_seconds=alternate_seconds if batch_rank_scores else score_seconds,
                    batched_seconds=score_seconds if batch_rank_scores else alternate_seconds,
                    alternate_execution_seconds=alternate_seconds,
                    ranks=list(ranks), cache_batch=cache.k.shape[1], rank_query_batch=len(ranks)))
            predictions.update({f"{variant}_r{rank}": values[index] for index, rank in enumerate(ranks)})
        for name, scores in predictions.items():
            for index, score in zip(actual, scores.cpu().tolist(), strict=True):
                output.append(dict(panel.instances[index]["actual"], method=name, score=float(score)))
        frame_reports.append(dict(timestamp=timestamp, source_offset=offset, source_rows=inherited,
            actual_instances=len(actual), assigned_instances=len(assigned),
            full_rank_logit_max_error=restore_error, full_rank_actual_shape_max_error=actual_shape_error,
            rank_execution_checks=rank_execution_checks))
    _sync(device)
    report = dict(panel=panel.index, instances=len(panel.instances),
        actual_feedback_instances=sum(instance["actual"] is not None for instance in panel.instances),
        matrix_columns=context.initial_rows, matrix_coordinate="publishing-time immutable old history rows",
        all_actual_reads_cleared=all_cleared, decomposition_skipped=not context.initial_rows or all_cleared,
        decomposition_seconds=decomposition_seconds, exact_check_seconds=exact_check_seconds,
        energy_guard_seconds=guard_seconds, exact_fallback_seconds=fallback_seconds,
        energy_guard=dict(matrices=len(guard_records),
            passed_without_exact=sum(not item["exact_fallback"] for item in guard_records),
            exact_fallback_matrices=sum(item["exact_fallback"] for item in guard_records),
            worst_ratio_lower_bound=min((min(item["captured_energy_ratio_lower_bound"])
                for item in guard_records), default=None), records=guard_records),
        decomposition_without_exact_checks_seconds=decomposition_seconds-exact_check_seconds,
        read_replay_seconds=time.perf_counter()-read_started,
        frames=frame_reports, svd_checks=checks, maximum_rank_energy=energies,
        full_rank_max_error=max(frame["full_rank_logit_max_error"] for frame in frame_reports),
        rank_score_execution="batched" if batch_rank_scores else "serial")
    return output, report


def evaluate_release_user(parent, current, probe, fg_probe, events, requests_by_timestamp, uid, cfg, *,
        known_items, candidate_size=230, execution="projected", fg_pool=None, full_h_pool=None,
        panel_options=None, verification_state=None):
    options = dict(panel_options or {})
    verified = set() if verification_state is None else verification_state
    def finish(panel, h, context):
        selected = dict(options)
        if all(variant in verified for variant in selected.get("variants", ("both_norm",))):
            selected["verify_exact_heads"] = 0
        rows, report = evaluate_release_panel(current, probe, panel, h, context, **selected)
        verified.update(check["variant"] for check in report["svd_checks"])
        return rows, report
    return capture_release_user(parent, current, probe, fg_probe, events, requests_by_timestamp, uid, cfg,
        known_items=known_items, candidate_size=candidate_size, execution=execution,
        fg_pool=fg_pool, full_h_pool=full_h_pool,
        on_panel_ready=finish)
