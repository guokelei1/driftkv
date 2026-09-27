"""Bounded real-model memory/speed probe; no quality measurements or disk state."""

from __future__ import annotations

import math
import time

import torch

from hstu_kvcache.baselines import layer_recompute as lr
from hstu_kvcache.baselines.deviation_recompute import recompute_deviation
from hstu_kvcache.baselines.query_recompute import recompute_query
from hstu_kvcache.baselines.serving import score_one_query
from hstu_kvcache.baselines.tail_recompute import recompute_tail
from hstu_kvcache.models import HSTUKVCache


@torch.inference_mode()
def stress(parent, current, scale, device, planconfig):
    """Run two bounded append/repair cycles at configured maximum tensor sizes.

    The full serving cohort and its layer-boundary tensors stay resident while
    a query batch is repaired. No empty_cache occurs between methods/repeats.
    OOM propagates to the canary caller; no user or method is silently omitted.
    Timings are synthetic throughput estimates, never quality evidence.
    """
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("stress probe requires the real evaluation GPU")
    parent.eval()
    current.eval()
    length = int(planconfig["history_length"])
    cohort = int(planconfig["cohort_sizes"][scale])
    queries = int(planconfig["query_batches"][scale])
    chunk = int(planconfig["sparse_query_chunk"])
    if length != current.cfg.max_seq_len:
        raise ValueError("stress context must match the selected checkpoint")
    # Bound IDs below the actual item embedding table; no dataset access needed.
    item_high = min(current.cfg.num_items - 1, 4096)
    items = (torch.arange(cohort * length, device=device).reshape(cohort, length)
             % item_high + 1).long()
    behaviors = torch.ones_like(items)
    deltas = torch.ones((cohort, length), device=device)
    deltas[:, 0] = 0
    owners = torch.arange(queries, device=device) % cohort
    candidates = ((owners * 7 + 3) % item_high + 1)[:, None]
    query_deltas = torch.full((queries,), 2.0, device=device)
    interval = (0, min(current.cfg.num_layers - 1,
                       math.ceil(current.cfg.num_layers * max(planconfig["layer_fractions"])) - 1))
    tokens = max(1, math.ceil(length * max(planconfig["token_fractions"])))
    band = min(32, length)
    total_memory = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    before_state_allocated = torch.cuda.memory_allocated(device)

    def timed(call):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        value = call()
        torch.cuda.synchronize(device)
        return value, time.perf_counter() - start

    state, capture_seconds = timed(lambda: lr.capture_state(parent, items, behaviors, deltas))
    baseline_seconds = {name: [] for name in ("layer", "tail", "deviation", "query")}
    read_seconds = {name: [] for name in baseline_seconds}
    full_seconds, append_seconds, retained_allocated, retained_reserved = [], [], [], []
    ordinary_reader_seconds, append_only_reader_seconds = [], []
    reader_max_abs_logit_error = 0.0
    peak_by_component = []
    start = time.perf_counter()
    for repeat in range(2):
        added_items = ((torch.arange(cohort * band, device=device).reshape(cohort, band)
                        + repeat * band + 11) % item_high + 1)
        added_behaviors = torch.ones_like(added_items)
        added_deltas = torch.ones((cohort, band), device=device)

        def append_once():
            return lr.append_band(current, state, added_items, added_behaviors,
                                  added_deltas, max_length=length)

        state, seconds = timed(append_once)
        append_seconds.append(seconds)
        items = torch.cat((items[:, band:], added_items), dim=1)
        behaviors = torch.cat((behaviors[:, band:], added_behaviors), dim=1)
        deltas = torch.cat((deltas[:, band:], added_deltas), dim=1)
        deltas[:, 0] = 0  # Request-local Full input convention.
        if state.cache.seq_len != length or state.cache.k.shape[2] != length:
            raise RuntimeError("stress append exceeded the retained context bound")
        selected_cache = HSTUKVCache(
            state.cache.k.index_select(1, owners), state.cache.v.index_select(1, owners), length,
        )
        selected_state = lr.LayerRecomputeState(selected_cache, {
            layer: values.index_select(0, owners) for layer, values in state.boundary_inputs.items()
        })
        raw = tuple(values.index_select(0, owners) for values in (items, behaviors, deltas))
        ordinary, seconds = timed(lambda: current.score_cc_reuse(selected_cache, candidates, query_deltas))
        ordinary_reader_seconds.append(seconds)
        append_only, seconds = timed(lambda: score_one_query(current, selected_cache, candidates, query_deltas))
        append_only_reader_seconds.append(seconds)
        reader_max_abs_logit_error = max(reader_max_abs_logit_error,
                                         float((ordinary - append_only).abs().max()))
        del ordinary, append_only
        full, seconds = timed(lambda: current.compute_kv(*raw))
        full_seconds.append(seconds)
        # Full scoring uses the same transient query reader as each baseline.
        full_score, _ = timed(lambda: current.score_cc_reuse(full, candidates, query_deltas))
        if not torch.isfinite(full_score).all():
            raise RuntimeError("nonfinite Full output on the bounded stress input")
        del full, full_score
        operations = {
            "layer": lambda: lr.recompute_interval(current, selected_state, *raw, interval).cache,
            "tail": lambda: recompute_tail(current, selected_cache, *raw, n=tokens),
            "deviation": lambda: recompute_deviation(current, selected_cache, *raw, n=tokens,
                                                     query_chunk_size=chunk),
            "query": lambda: recompute_query(current, selected_cache, *raw, candidates, query_deltas,
                                             n=tokens, query_chunk_size=chunk),
        }
        for method, operation in operations.items():
            repaired, seconds = timed(operation)
            baseline_seconds[method].append(seconds)
            scores, seconds = timed(lambda: score_one_query(current, repaired, candidates, query_deltas))
            read_seconds[method].append(seconds)
            if repaired.seq_len != length or not torch.isfinite(scores).all():
                raise RuntimeError(f"invalid {method} cache length or stress output")
            if repeat == 0:
                reference = current.score_cc_reuse(repaired, candidates, query_deltas)
                reader_max_abs_logit_error = max(reader_max_abs_logit_error,
                                                 float((reference - scores).abs().max()))
                del reference
            peak_by_component.append({
                "repeat": repeat, "method": method,
                "cumulative_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / (1 << 30),
                "cumulative_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / (1 << 30),
            })
            del repaired, scores
        del operations, operation, selected_state, selected_cache, raw
        torch.cuda.synchronize(device)
        retained_allocated.append(torch.cuda.memory_allocated(device))
        retained_reserved.append(torch.cuda.memory_reserved(device))
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    if peak_allocated > total_memory * planconfig["memory_fraction"]:
        raise RuntimeError("configured probe exceeded the planned allocation safety line")
    # Same live tensor shapes must not retain an additional request-sized state.
    # A small allowance accounts for lazy backend initialization bookkeeping.
    growth = retained_allocated[-1] - retained_allocated[0]
    if growth > (16 << 20):
        raise RuntimeError(f"stress cycles retained growing CUDA allocations: {growth} bytes")
    if reader_max_abs_logit_error > 2e-5:
        raise RuntimeError(f"append-only reader differs from ordinary reader: {reader_max_abs_logit_error}")
    return {
        "status": "passed", "synthetic_only": True, "scale": scale, "repeats": 2,
        "history_length": length, "cohort_size": cohort, "query_batch": queries,
        "layer_interval": list(interval), "selected_tokens": tokens,
        "append_band_tokens": band,
        "sparse_query_chunk": chunk, "gpu_total_gib": total_memory / (1 << 30),
        "capture_seconds": capture_seconds, "append_seconds": append_seconds,
        "full_cache_batch_seconds": full_seconds,
        "full_cache_seconds_per_request": sum(full_seconds) / (2 * queries),
        "baseline_batch_seconds": baseline_seconds,
        "baseline_seconds_per_request": {name: sum(values) / (2 * queries)
                                          for name, values in baseline_seconds.items()},
        "query_read_batch_seconds": read_seconds,
        "query_read_seconds_per_request": {name: sum(values) / (2 * queries)
                                            for name, values in read_seconds.items()},
        "ordinary_reader_batch_seconds": ordinary_reader_seconds,
        "append_only_reader_batch_seconds": append_only_reader_seconds,
        "reader_speedup": sum(ordinary_reader_seconds) / sum(append_only_reader_seconds),
        "reader_max_abs_logit_error": reader_max_abs_logit_error,
        "before_state_allocated_gib": before_state_allocated / (1 << 30),
        "persistent_state_tensor_bytes": (state.cache.k.numel() + state.cache.v.numel()) * state.cache.k.element_size()
            + sum(values.numel() * values.element_size() for values in state.boundary_inputs.values()),
        "peak_allocated_gib": peak_allocated / (1 << 30),
        "peak_reserved_gib": peak_reserved / (1 << 30),
        "peak_allocated_fraction": peak_allocated / total_memory,
        "peak_reserved_fraction": peak_reserved / total_memory,
        "retained_allocated_gib_after_repeat": [value / (1 << 30) for value in retained_allocated],
        "retained_reserved_gib_after_repeat": [value / (1 << 30) for value in retained_reserved],
        "retained_allocation_growth_bytes": growth,
        "peak_by_component": peak_by_component,
        "elapsed_seconds_excluding_capture": time.perf_counter() - start,
        "scope": "maximum retained length and configured batches; two cycles check transient release, not an uptime guarantee",
    }
