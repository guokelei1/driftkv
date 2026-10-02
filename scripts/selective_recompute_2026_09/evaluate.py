"""Batched causal Reuse timelines with ephemeral request-time repair.

Only observed events mutate the shared serving cache. Each repair sees the same
snapshot, and its temporary K/V is discarded after scoring. Full anchors retain
the original request-local history order; repair inputs align with serving K/V.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, replace
import math

import numpy as np
import torch

from hstu_kvcache.baselines import layer_recompute as lr
from hstu_kvcache.baselines.tail_recompute import recompute_tail
from hstu_kvcache.baselines.deviation_recompute import layer0_deviation_scores
from hstu_kvcache.baselines.query_recompute import query_attention_scores
from hstu_kvcache.baselines.sparse_recompute import recompute_positions, select_top_positions
from hstu_kvcache.baselines.serving import score_one_query
from hstu_kvcache.models import HSTUKVCache
from hstu_kvcache.training import collate_foundation_batch
from selective_recompute_2026_09.common import DAY, METHODS, budgets


@contextmanager
def native_backend(models, backend):
    """Avoid one Triton compilation per growing short-history length."""
    previous = [(block.attn, block.attn.backend) for model in models for block in model.blocks]
    try:
        for attention, _ in previous:
            attention.backend = backend
        yield
    finally:
        for attention, original in previous:
            attention.backend = original


def take_state(state, indices):
    index = torch.as_tensor(indices, dtype=torch.long, device=state.cache.k.device)
    return lr.LayerRecomputeState(
        HSTUKVCache(state.cache.k.index_select(1, index), state.cache.v.index_select(1, index), state.cache.seq_len),
        {layer: values.index_select(0, index) for layer, values in state.boundary_inputs.items()},
    )


def put_state(state, indices, updated):
    if len(indices) == state.cache.k.shape[1] and state.cache.seq_len != updated.cache.seq_len:
        return updated
    index = torch.as_tensor(indices, dtype=torch.long, device=state.cache.k.device)
    state.cache.k.index_copy_(1, index, updated.cache.k)
    state.cache.v.index_copy_(1, index, updated.cache.v)
    for layer in state.boundary_inputs:
        state.boundary_inputs[layer].index_copy_(0, index, updated.boundary_inputs[layer])
    return state


@dataclass
class Snapshot:
    state: lr.LayerRecomputeState
    events: tuple
    candidates: torch.Tensor
    query_deltas: torch.Tensor
    requests: list[dict]
    context: object = None


def prefix_events(raw, cutover, max_length):
    stop = int(np.searchsorted(raw[0], cutover, side="left"))
    values = sorted(zip(*(a[:stop] for a in raw)), key=lambda e: (int(e[0]), int(e[1]), int(e[2])))
    return values[-max_length:]


@torch.inference_mode()
def snapshots(uids, by_user, history, parent, current, cutover, *, max_length,
              query_batch, stats, append_band_size=32, observer=None):
    """Yield equal-length request batches; short histories run as singleton cohorts."""
    device = next(current.parameters()).device
    prefixes = [prefix_events(history.rows[uid], cutover, max_length) for uid in uids]
    n = len(prefixes[0])
    if n < 1 or any(len(p) != n for p in prefixes) or (n < max_length and len(uids) != 1):
        raise ValueError("cohort requires full equal histories or one short-history user")
    times = torch.tensor([[int(e[0]) for e in p] for p in prefixes], device=device)
    items = torch.tensor([[int(e[1]) for e in p] for p in prefixes], device=device)
    behaviors = torch.tensor([[int(e[2]) for e in p] for p in prefixes], device=device)
    deltas = torch.zeros_like(times, dtype=torch.float32)
    deltas[:, 1:] = times[:, 1:] - times[:, :-1]
    state = lr.capture_state(parent, items, behaviors, deltas)
    if observer is not None:
        if getattr(observer, 'needs_raw_events', False):
            observer.initialized(state.cache, raw_events=(items, behaviors, deltas))
        else:
            observer.initialized(state.cache)
    stats["initial_history_hist"].update({n: len(uids)})
    actions = []
    for uid in uids:
        requests_at = defaultdict(list)
        for request in by_user[uid]:
            requests_at[int(request["query_timestamp"])].append(request)
        last_query = max(requests_at)
        raw = history.rows[uid]
        post = defaultdict(list)
        start = int(np.searchsorted(raw[0], cutover, side="left"))
        stop = int(np.searchsorted(raw[0], last_query, side="left"))
        for timestamp, item, behavior in zip(*(a[start:stop] for a in raw)):
            post[int(timestamp)].append((int(timestamp), int(item), int(behavior)))
        timeline = []
        for timestamp in sorted(set(requests_at) | set(post)):
            if timestamp in requests_at:
                timeline.append(("query", timestamp, requests_at[timestamp]))
            timeline.extend(("append", timestamp, event) for event in sorted(post.get(timestamp, ())))
        actions.append(timeline)
    positions = [0] * len(uids)
    last_times = [int(p[-1][0]) for p in prefixes]
    while True:
        active = [i for i in range(len(uids)) if positions[i] < len(actions[i])]
        if not active:
            break
        bands = defaultdict(list)
        for i in active:
            available = 0
            while (available < append_band_size and positions[i] + available < len(actions[i])
                   and actions[i][positions[i] + available][0] == "append"):
                available += 1
            if available:
                # Six reusable shapes rather than compiling every suffix length.
                bands[1 << (available.bit_length() - 1)].append(i)
        for width, append_indices in sorted(bands.items(), reverse=True):
            events = [[actions[i][positions[i] + j][2] for j in range(width)] for i in append_indices]
            new_times = torch.tensor([[e[0] for e in values] for values in events], device=device)
            new_items = torch.tensor([[e[1] for e in values] for values in events], device=device)
            new_behaviors = torch.tensor([[e[2] for e in values] for values in events], device=device)
            previous_times = torch.tensor([[last_times[i]] for i in append_indices], device=device)
            delta_sources = torch.cat((previous_times, new_times[:, :-1]), dim=1)
            new_deltas = (new_times - delta_sources).clamp(0, 7 * DAY).float()
            n = state.cache.seq_len
            selected = take_state(state, append_indices)
            updated = lr.append_band(current, selected, new_items, new_behaviors, new_deltas, max_length)
            if observer is not None:
                if getattr(observer, 'needs_raw_events', False):
                    observer.appended(append_indices, selected.cache, updated.cache, width,
                                      raw_events=(new_items, new_behaviors, new_deltas))
                else:
                    observer.appended(append_indices, selected.cache, updated.cache, width)
            state = put_state(state, append_indices, updated)
            del updated, selected
            stats.setdefault("band_append_hist", Counter()).update({f"{n}:{width}": len(append_indices)})
            for name, new in (("times", new_times), ("items", new_items), ("behaviors", new_behaviors)):
                old = {"times": times, "items": items, "behaviors": behaviors}[name]
                combined = torch.cat((old[append_indices], new), dim=1)[:, -max_length:]
                if n < max_length:
                    if name == "times": times = combined
                    elif name == "items": items = combined
                    else: behaviors = combined
                else:
                    old[append_indices] = combined
            for i, values in zip(append_indices, events):
                last_times[i] = values[-1][0]
                positions[i] += width
        entries = []
        for i in range(len(uids)):
            if positions[i] < len(actions[i]) and actions[i][positions[i]][0] == "query":
                _, timestamp, requests = actions[i][positions[i]]
                entries.extend((i, timestamp, r) for r in requests)
                positions[i] += 1
        for start in range(0, len(entries), query_batch):
            chunk = entries[start:start + query_batch]
            owner = [v[0] for v in chunk]
            candidates = torch.tensor([[int(v[2]["item_idx"])] for v in chunk], device=device)
            query_deltas = torch.tensor([float(v[1] - last_times[v[0]]) for v in chunk], device=device)
            selected_times = times[owner]
            raw_deltas = torch.zeros_like(selected_times, dtype=torch.float32)
            raw_deltas[:, 1:] = selected_times[:, 1:] - selected_times[:, :-1]
            selected = take_state(state, owner)
            stats["full_history_hist"].update({selected.cache.seq_len: len(chunk)})
            yield Snapshot(selected, (items[owner], behaviors[owner], raw_deltas), candidates,
                           query_deltas, [v[2] for v in chunk],
                           observer.reading(owner) if observer is not None else None)


def all_snapshots(cohorts, by_user, history, parent, current, cutover, query_batch, stats, default_cost, append_band_size, observer=None):
    for cohort in cohorts:
        short = len(prefix_events(history.rows[cohort[0]], cutover, current.cfg.max_seq_len)) < current.cfg.max_seq_len
        backend = "torch" if short else default_cost.attention_backend
        cohort_stats = {name: Counter() for name in ("full_history_hist", "append_prefix_hist", "initial_history_hist", "band_append_hist")}
        with native_backend((parent, current), backend):
            for snap in snapshots(cohort, by_user, history, parent, current, cutover,
                                  max_length=current.cfg.max_seq_len, query_batch=query_batch, stats=cohort_stats,
                                  append_band_size=append_band_size, observer=observer):
                yield snap, replace(default_cost, attention_backend=backend)
        for name, values in cohort_stats.items():
            stats[name].update(values)
        if short:
            for name in ("full_history_hist", "append_prefix_hist", "band_append_hist"):
                stats["torch_" + name].update(cohort_stats[name])


@torch.inference_mode()
def score_unit(uids, by_user, history, parent, current, cutover, *, intervals,
               cost_model, cohort_size, query_batch, sparse_query_chunk=64,
               verify=False, methods=METHODS, append_band_size=32):
    configs = budgets(len(current.blocks))
    rows = {method: [] for method in methods}
    stats = {name: Counter() for name in ("full_history_hist", "append_prefix_hist", "initial_history_hist",
                                        "torch_full_history_hist", "torch_append_prefix_hist",
                                        "band_append_hist", "torch_band_append_hist")}
    controls = {"reuse_max_abs_logit_error": 0.0, "full_max_abs_logit_error": 0.0,
                "requests": 0, "repair_full_max_abs_logit_error": 0.0,
                "repair_vs_saved_full_max_abs_logit_difference": 0.0}
    full = [uid for uid in uids if len(prefix_events(history.rows[uid], cutover, current.cfg.max_seq_len)) == current.cfg.max_seq_len]
    short = [uid for uid in uids if uid not in set(full)]
    cohorts = [full[i:i + cohort_size] for i in range(0, len(full), cohort_size)] + [[uid] for uid in short]
    for snap, active_cost in all_snapshots(cohorts, by_user, history, parent, current, cutover,
                                          query_batch, stats, cost_model, append_band_size):
            n = snap.state.cache.seq_len
            if verify:
                actual_reuse = current.observe_cc_reuse(snap.state.cache, snap.candidates, snap.query_deltas)[0][:, 0]
                full_requests = [{**r, "weight": r.get("weight", 1.0)} for r in snap.requests]
                full_batch = collate_foundation_batch(full_requests, history, device=snap.candidates.device,
                                                     max_history=current.cfg.max_seq_len)
                actual_full = current.observe_cc_full(full_batch.item_ids, full_batch.behaviors,
                    full_batch.time_deltas, full_batch.candidate_ids, full_batch.query_time_deltas,
                    lengths=full_batch.lengths)[0][:, 0]
                expected_reuse = torch.tensor([r["reuse_logit"] for r in snap.requests], device=actual_reuse.device)
                expected_full = torch.tensor([r["full_logit"] for r in snap.requests], device=actual_full.device)
                controls["reuse_max_abs_logit_error"] = max(controls["reuse_max_abs_logit_error"], float((actual_reuse - expected_reuse).abs().max()))
                controls["full_max_abs_logit_error"] = max(controls["full_max_abs_logit_error"], float((actual_full - expected_full).abs().max()))
                # The full repair must equal a rebuild of its exact aligned inputs.
                repaired = recompute_tail(current, snap.state.cache, *snap.events, n)
                repair_scores = current.observe_cc_reuse(repaired, snap.candidates, snap.query_deltas)[0][:, 0]
                canonical_full = current.observe_cc_full(*snap.events, snap.candidates, snap.query_deltas)[0][:, 0]
                controls["repair_full_max_abs_logit_error"] = max(controls["repair_full_max_abs_logit_error"], float((repair_scores-canonical_full).abs().max()))
                controls["repair_vs_saved_full_max_abs_logit_difference"] = max(controls["repair_vs_saved_full_max_abs_logit_difference"], float((repair_scores-expected_full).abs().max()))
                controls["requests"] += len(snap.requests)
                del repaired, actual_full, actual_reuse, repair_scores, canonical_full
            selectors = {}
            pending = []
            for method in methods:
                needs_selection = method in ("deviation", "query") and any(
                    max(1, math.ceil(n * c["fraction"])) < n for c in configs[method])
                if method == "deviation" and needs_selection:
                    selectors[method] = layer0_deviation_scores(current, snap.state.cache, *snap.events)
                elif method == "query" and needs_selection:
                    selectors[method] = query_attention_scores(current, snap.state.cache, snap.candidates, snap.query_deltas)
                if needs_selection:
                    ranking = torch.argsort(selectors[method], dim=1, descending=True, stable=True)
                for config in configs[method]:
                    interval = None
                    selected_tokens = 0
                    if method == "layer":
                        interval = tuple(intervals[str(config["layers"])])
                        repaired = lr.recompute_interval(current, snap.state, *snap.events, interval).cache
                    else:
                        selected_tokens = max(1, math.ceil(n * config["fraction"]))
                        if method == "tail":
                            repaired = recompute_tail(current, snap.state.cache, *snap.events, selected_tokens)
                        elif selected_tokens == n:
                            repaired = current.compute_kv(*snap.events)
                        else:
                            positions = ranking[:, :selected_tokens].sort(dim=1).values
                            repaired = recompute_positions(current, snap.state.cache, *snap.events, positions,
                                                           query_chunk_size=sparse_query_chunk)
                    scores = score_one_query(current, repaired, snap.candidates, snap.query_deltas)[:, 0]
                    operation = active_cost.operation_cost(method, n, selected_tokens=selected_tokens, interval=interval)
                    pending.append((method, config["name"], selected_tokens, interval, operation, scores))
                    del repaired
            # One host transfer/synchronization for the whole snapshot's budgets.
            if not pending:
                continue
            score_matrix = torch.stack([value[-1] for value in pending]).float().cpu().numpy()
            if not np.isfinite(score_matrix).all():
                raise RuntimeError("nonfinite baseline logits")
            for (method, budget_name, selected_tokens, interval, operation, _), logits in zip(pending, score_matrix):
                    for request, logit in zip(snap.requests, logits):
                        rows[method].append({"request_id": request["request_id"], "uid": int(request["uid"]),
                            "query_timestamp": int(request["query_timestamp"]), "budget": budget_name,
                            "hstu_logit": float(logit), "history_length": n, "selected_tokens": selected_tokens,
                            "interval_start": interval[0] if interval else -1,
                            "interval_end": interval[1] if interval else -1, **operation})
    if verify:
        for key in ("reuse_max_abs_logit_error", "full_max_abs_logit_error", "repair_full_max_abs_logit_error"):
            if controls[key] > 2e-5:
                raise RuntimeError(f"reference control failed: {key}={controls[key]}")
    return rows, {name: dict(value) for name, value in stats.items()}, controls
