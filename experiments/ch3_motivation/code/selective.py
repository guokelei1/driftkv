"""Selective-repair primitives for the new Chapter 3 study.

The main experiment repairs once, before the first history consumer, and then
keeps each method's own cache.  ``request_local`` is a separate diagnostic of
the older experiment: each read repairs an immutable rolling Reuse snapshot.
No historical quality result or historical layer choice is loaded here.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import copy
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]

from hstu_kvcache.evaluation.baselines import DAY, prepare_policy, raw_tensors
from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.methods.baselines import layer_recompute as lr
from hstu_kvcache.methods.baselines.layer_recompute.core import _capture_boundaries
from hstu_kvcache.evaluation.trajectory import NativeTrajectory, _short_prefix_backend
from hstu_kvcache.evaluation.cost import CostModel, eager_read


def _widths(cfg, layers):
    widths = list(dict.fromkeys(int(x) for x in cfg["partial_points"]["layer"]))
    if not widths or any(width < 1 or width > layers for width in widths):
        raise ValueError("Layer budgets must be actual layer counts within the model")
    return widths


def _cost_for_length(cm, length, cfg):
    return cm if length == cfg["history_length"] else CostModel(
        cm.hidden_size, cm.num_layers, cm.num_heads, "torch")


@torch.no_grad()
def profile_intervals(current, capture_rows, histories, uids, cfg, device):
    """Directly measure every contiguous interval at each declared layer count.

    Captures must contain same-window ``parent``, ``teacher``, and actual old
    ``boundaries``. Candidates are pre-release fitting candidates, not held-out
    feedback. Each interval starts from the same immutable old state; selection
    minimizes equal-user teacher-logit MSE, without adding single-layer gains.
    The returned FLOPs include both cache acquisitions and all actual probes.
    """
    if not uids:
        raise ValueError("interval profiling requires fitting users")
    cm = CostModel(current.cfg.hidden_size, len(current.blocks), current.cfg.num_heads,
                   cfg["attention_backend"])
    widths = _widths(cfg, cm.num_layers)
    intervals = [(start, start + width - 1) for width in widths
                 for start in range(cm.num_layers - width + 1)]
    errors = {interval: 0. for interval in intervals}
    baseline_mse = 0.
    parts = Counter()
    parts_by_width = {width: Counter() for width in widths}
    for uid in uids:
        row = capture_rows[uid]
        events = histories.rows[uid]
        stop = int(np.searchsorted(events[0], cfg["release_day"] * DAY, side="left"))
        retained = tuple(column[max(0, stop - cfg["history_length"]):stop] for column in events)
        raw = raw_tensors(retained, device)
        parent, teacher = row["parent"].to(device), row["teacher"].to(device)
        n = parent.seq_len
        if n != raw[0].shape[1] or teacher.seq_len != n:
            raise ValueError("profile captures and cutover history must use the same tail window")
        # ``raw`` is supplied by the new capture path and binds the values too,
        # not only tensor lengths (a first-window/last-window mismatch can hide).
        if "raw" in row and any(not torch.equal(a.to(device), b)
                                for a, b in zip(row["raw"], raw, strict=True)):
            raise ValueError("profile raw inputs differ from the captured cutover window")
        state = lr.LayerRecomputeState(parent, {
            layer: value.to(device) for layer, value in row["boundaries"].items()})
        candidates = row["candidates"].to(device).reshape(1, -1)
        delta = torch.as_tensor(row["query_delta"], device=device, dtype=torch.float32).reshape(1)
        q = candidates.shape[1]
        cost = _cost_for_length(cm, n, cfg)
        parts["source_cache"] += cost.full_cache(n)
        parts["teacher_cache"] += cost.full_cache(n)
        with _short_prefix_backend(current, n, cfg["history_length"]):
            target = score(current, teacher, candidates, delta)[0]
            uncorrected = score(current, parent, candidates, delta)[0]
            baseline_mse += float((uncorrected - target).square().mean()) / len(uids)
            parts["teacher_and_reuse_reads"] += 2 * eager_read(cost, n, queries=q)
            parts["logit_mse"] += 3 * q
            for interval in intervals:
                repaired = lr.recompute_interval(current, state, *raw, interval)
                predicted = score(current, repaired.cache, candidates, delta)[0]
                errors[interval] += float((predicted - target).square().mean()) / len(uids)
                parts["interval_rebuilds"] += cost.layer_rebuild(n, interval)
                parts["interval_reads"] += eager_read(cost, n, queries=q)
                parts["logit_mse"] += 3 * q
                width_parts = parts_by_width[interval[1] - interval[0] + 1]
                width_parts["interval_rebuilds"] += cost.layer_rebuild(n, interval)
                width_parts["interval_reads"] += eager_read(cost, n, queries=q)
                width_parts["logit_mse"] += 3 * q
    selected = {str(width): list(min(
        (interval for interval in intervals if interval[1] - interval[0] + 1 == width),
        key=lambda interval: (errors[interval], interval[0]))) for width in widths}
    flops = int(sum(parts.values()))
    common_parts = {key: parts[key] for key in ("source_cache", "teacher_cache", "teacher_and_reuse_reads")}
    common_parts["baseline_logit_mse"] = sum(3 * capture_rows[uid]["candidates"].numel() for uid in uids)
    standalone = {str(width): int(sum(common_parts.values()) + sum(parts_by_width[width].values()))
                  for width in widths}
    profiles = dict(
        intervals=[dict(interval=list(interval), layers=interval[1] - interval[0] + 1,
                        mse=errors[interval], baseline_mse=baseline_mse,
                        gain=baseline_mse - errors[interval]) for interval in intervals],
        selected=selected, users=len(uids), fitting_uids=[int(uid) for uid in uids],
        calibration_flops=flops, flop_components=dict(parts),
        common_flop_components=common_parts,
        parts_by_width={str(width): dict(value) for width, value in parts_by_width.items()},
        standalone_calibration_flops_by_width=standalone,
        selection="direct contiguous-interval teacher-logit MSE, equal-user mean; fit users only",
        state_schedule="independent snapshot per interval; no repaired state carried between probes")
    return profiles, flops


def repair_cost(policy, shapes, cm, cfg, *, schedule="persistent"):
    """Standalone selection/repair plus fitting FLOPs on executed shapes.

    ``shapes['first']`` is the first history consumer histogram of event_shapes.
    A request-local diagnostic instead supplies ``shapes['requests']``. These
    are actual replay workloads; the driver's separate 230-read sensitivity
    ledger must not be silently substituted for them.
    """
    if schedule not in ("persistent", "request_local"):
        raise ValueError("unknown selective-repair schedule")
    histogram = shapes["first" if schedule == "persistent" else "requests"]
    total, sorts = int(policy.get("calibration_flops", 0)), 0.
    for (n, kind, q), count in histogram.items():
        cost = _cost_for_length(cm, n, cfg)
        if policy["method"] == "reuse":
            continue
        if policy["method"] == "rebuild":
            total += count * cost.full_cache(n)
            continue
        work = cost.operation_cost(policy["method"], n,
            selected_tokens=int(np.ceil(n * policy.get("fraction", 0))),
            interval=policy.get("interval"), queries=q)
        # Behavior selectors use a behavior token, not the query-token embed.
        if policy["method"] == "query" and kind == "behavior" and 0 < np.ceil(n * policy["fraction"]) < n:
            work["total_flops"] += cost.embedding(q) - cost.embedding(q, query=True)
        total += count * work["total_flops"]
        sorts += count * work["selection_sort_comparisons_estimate"]
    return int(total), float(sorts)


def make_policies(cm, shapes, profiles, cfg):
    """Policies use the existing audited persistent evaluator's native schema."""
    denominator = repair_cost(dict(method="rebuild"), shapes, cm, cfg)[0]
    if denominator <= 0:
        raise ValueError("the paired population has no Rebuild work")
    policies = [dict(tag="reuse", method="reuse", calibration_flops=0),
                dict(tag="rebuild", method="rebuild", calibration_flops=0)]
    for width in _widths(cfg, cm.num_layers):
        interval = tuple(profiles["selected"][str(width)])
        policies.append(dict(tag=f"layer_{width}layers", method="layer", layers=width,
                             interval=interval, calibration_flops=profiles["standalone_calibration_flops_by_width"][str(width)],
                             selection="direct contiguous-interval fitting MSE"))
    for method in ("tail", "deviation", "query"):
        for fraction in cfg["partial_points"][method]:
            policies.append(dict(tag=f"{method}_{100*fraction:g}pct", method=method,
                                 fraction=float(fraction), calibration_flops=0))
    for policy in policies:
        flops, sorts = repair_cost(policy, shapes, cm, cfg)
        policy.update(extra_flops=flops, cost_percent=100 * flops / denominator,
                      selection_sort_comparisons_estimate=sorts)
    return policies, denominator


def _policy_copy(base):
    trajectory = copy.copy(base)
    trajectory.ledger = Counter(base.ledger)
    trajectory.producers = base.producers.copy()
    return trajectory


@torch.no_grad()
def evaluate_reference(parent, current, histories, by_user, uids, policies, cfg, device,
                       *, semantics="persistent"):
    """Independent scalar reference; CPU-capable and intentionally graph-free.

    Returns per-event scores for numerical comparisons, not population claims.
    For fewer than 512 users metadata always marks the run diagnostic.
    """
    if semantics not in ("persistent", "request_local"):
        raise ValueError("semantics must be persistent or request_local")
    allowed = {"reuse", "rebuild", "layer", "tail", "deviation", "query"}
    if any(policy["method"] not in allowed for policy in policies):
        raise ValueError("the selective reference supports only endpoints and selective repairs")
    scores, observed = [], {policy["tag"]: Counter() for policy in policies}
    request_shapes = Counter()
    at, end = cfg["release_day"] * DAY, cfg["end_day"] * DAY

    def append_scores(trajectory, uid, timestamp, requests, policy):
        for position in range(0, len(requests), cfg["query_batch"]):
            rows = requests[position:position + cfg["query_batch"]]
            ids = torch.tensor([[row["item_idx"] for row in rows]], device=device)
            delta = torch.tensor([timestamp - trajectory.last_timestamp], device=device, dtype=torch.float32)
            values = score(current, trajectory.cache, ids, delta)[0][0].cpu().tolist()
            scores.extend(dict(row, method=policy["tag"], score=value)
                          for row, value in zip(rows, values, strict=True))
            observed[policy["tag"]]["candidate_scores"] += len(rows)

    for uid in uids:
        with _capture_boundaries(parent) as boundaries:
            base = NativeTrajectory(parent, cfg["edge"] - 1, histories.rows[uid], at,
                max_length=cfg["history_length"], append_band_size=cfg["append_band_size"])
        base.switch(current, cfg["edge"], at)
        rounds = by_user.get(uid, {})
        if semantics == "persistent":
            for policy in policies:
                trajectory = _policy_copy(base)
                prepared = False

                def prepare(consumer):
                    nonlocal prepared
                    if prepared:
                        return
                    prepare_policy(trajectory, boundaries, policy, current, consumer, cfg, device)
                    observed[policy["tag"]]["prepare_calls"] += 1
                    observed[policy["tag"]]["prepare_rows"] += trajectory.cache.seq_len
                    prepared = True

                def advance(until):
                    stop = int(np.searchsorted(trajectory.events[0], until, side="left"))
                    if not prepared and trajectory.cursor < stop:
                        pos = trajectory.cursor
                        prepare(dict(kind="behavior",
                            items=torch.tensor([[int(trajectory.events[1][pos])]], device=device),
                            behavior=torch.tensor([[int(trajectory.events[2][pos])]], device=device),
                            delta=torch.tensor([[min(7 * DAY, int(trajectory.events[0][pos]) - trajectory.last_timestamp)]],
                                               device=device, dtype=torch.float32)))
                    trajectory.advance(until)

                for timestamp, requests in sorted(rounds.items()):
                    advance(timestamp)
                    selection = requests[:cfg.get("repair_query_batch", 16)]
                    prepare(dict(kind="query", items=torch.tensor([[row["item_idx"] for row in selection]], device=device),
                        delta=torch.tensor([timestamp - trajectory.last_timestamp], device=device, dtype=torch.float32)))
                    append_scores(trajectory, uid, timestamp, requests, policy)
                advance(end)
                observed[policy["tag"]]["appended_rows"] += trajectory.appended_tokens
        else:
            # Keep actual producer inputs for the common rolling Reuse state.
            # Appends use native Current reads; repaired views never enter it.
            added = {}

            @contextmanager
            def capture_append(model, _cache, _events, _version):
                nonlocal added
                with _capture_boundaries(model) as values:
                    yield
                added = values

            def retain_boundaries(_old, new, _events, _version):
                nonlocal boundaries
                boundaries = {layer: torch.cat((boundaries[layer], added[layer]), 1)[:, -new.seq_len:]
                              for layer in boundaries}

            base.append_context, base.on_append = capture_append, retain_boundaries
            for timestamp, requests in sorted(rounds.items()):
                base.advance(timestamp)
                selection = requests[:cfg.get("repair_query_batch", 16)]
                consumer = dict(kind="query",
                    items=torch.tensor([[row["item_idx"] for row in selection]], device=device),
                    delta=torch.tensor([timestamp - base.last_timestamp], device=device, dtype=torch.float32))
                request_shapes[base.cache.seq_len, "query", len(selection)] += 1
                for policy in policies:
                    view = _policy_copy(base)
                    prepare_policy(view, boundaries, policy, current, consumer, cfg, device)
                    observed[policy["tag"]]["prepare_calls"] += 1
                    observed[policy["tag"]]["prepare_rows"] += view.cache.seq_len
                    append_scores(view, uid, timestamp, requests, policy)
            base.advance(end)
            for policy in policies:
                observed[policy["tag"]]["appended_rows"] += base.appended_tokens
    return scores, dict(semantics=semantics, users=len(uids), diagnostic=len(uids) < 512,
        observed={tag: dict(value) for tag, value in observed.items()},
        request_shapes=[dict(shape=list(shape), count=count) for shape, count in request_shapes.items()],
        state_scope=("first-consumer repair persists; each policy appends from its own cache"
                     if semantics == "persistent" else
                     "per-request repair is discarded; native Reuse writes are shared; Rebuild is also per-request"))
