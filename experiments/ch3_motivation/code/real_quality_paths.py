"""Batched causal native trajectories for simple real-feedback controls.

Only the cutover cache differs between persistent paths. All later writes are
independent native model outputs, computed together along the batch axis.
Request-local oracle mixtures are read-only diagnostics and never written back.
"""
from __future__ import annotations

from collections import Counter
import json

import numpy as np
import torch

from hstu_kvcache.adaptation.reader import score as native_score
from hstu_kvcache.evaluation.trajectory import MAX_DELTA_SECONDS, NativeTrajectory, _short_prefix_backend
from hstu_kvcache.models import HSTUKVCache
from hstu_kvcache.models.state_transition import append_with_rolling_band

PERSISTENT_METHODS = ("reuse", "f025", "f05", "f1", "f15", "f2", "f_key", "f_value", "rebuild")
SMALL_METHODS = ("reuse", "f1", "rebuild")
ORACLE_BETAS = (("kv_oracle025", .25), ("kv_oracle05", .5), ("kv_oracle075", .75))
NUMERIC_ATOL, NUMERIC_RTOL = 1e-6, 1e-5
# Real Large FP32 GEMMs select different reduction shapes for batched and
# single-path writes. See real_quality_angles/attempt_1/numeric_diagnostic:
# identical inputs are exactly equal under FP64; two complete FP32 trajectories
# reached 1.049e-5 max K/V difference but <=4.77e-7 score difference.
CACHE_NUMERIC_ATOL = 1e-5


def _check_numeric(actual, expected, kind, methods, *, atol):
    if torch.allclose(actual, expected, atol=atol, rtol=NUMERIC_RTOL):
        return
    difference = (actual-expected).abs()
    tolerance = atol+NUMERIC_RTOL*expected.abs()
    worst = int((difference/tolerance).reshape(-1).argmax())
    location = []
    for size in reversed(difference.shape):
        location.append(worst % size)
        worst //= size
    location.reverse()
    index = tuple(location)
    report = dict(kind=kind, max_abs=float(difference.max()),
        relative_rms=float(difference.square().mean().sqrt()
            /expected.square().mean().sqrt().clamp_min(1e-30)),
        failed_elements=int((difference>tolerance).sum()), elements=difference.numel(),
        reference_max_abs=float(expected.abs().max()), location=location,
        actual_at_worst=float(actual[index]), expected_at_worst=float(expected[index]),
        max_tolerance_ratio=float((difference/tolerance).max()), atol=atol, rtol=NUMERIC_RTOL)
    if actual.ndim == 4:
        report.update(layer=location[0], method=methods[location[1]])
    elif actual.ndim == 2:
        report.update(method=methods[location[0]])
    raise ValueError("batched native numerics differ from independent causal paths: "+json.dumps(report))


def _member(cache, index):
    return HSTUKVCache(cache.k[:, index:index+1], cache.v[:, index:index+1], cache.seq_len)


def _pack(caches):
    return HSTUKVCache(torch.cat([c.k for c in caches], 1),
                       torch.cat([c.v for c in caches], 1), caches[0].seq_len)


def _blend(left, right, beta):
    # Endpoints avoid gratuitous float32 subtraction/addition round trips.
    if beta == 0:
        return left
    if beta == 1:
        return right
    return HSTUKVCache(right.k + (1-beta)*(left.k-right.k),
                       right.v + (1-beta)*(left.v-right.v), left.seq_len)


def _error_table(cache, teacher_index):
    """One compact transfer: [path, K/V/combined], aligned current teacher."""
    if cache.seq_len:
        k = (cache.k-cache.k[:, teacher_index:teacher_index+1]).square().mean((0, 2, 3))
        v = (cache.v-cache.v[:, teacher_index:teacher_index+1]).square().mean((0, 2, 3))
    else:
        k = v = cache.k.new_zeros(cache.k.shape[1])
    return torch.stack((k, v, (k+v)/2), -1)


class BatchedQualityPaths:
    """One user, one append call per band, an independent cache per method.

    ``structural_controls=False`` selects reuse/F1/rebuild for history-cap
    sensitivities. Extra maps are the two pre-fitted source-cache transforms
    supplied by the caller; they execute once and must not mutate the source.
    ``check_batch`` intentionally executes independent reference paths ONLY in
    bounded numerical probes; ordinary population execution has no such loop.
    """

    @torch.no_grad()
    def __init__(self, parent, current, probe, events, cfg, *, history_cap=1024,
                 structural_controls=True, check_batch=False, extra_maps=None):
        if current.training or parent.training:
            raise ValueError("real quality paths require frozen eval-mode models")
        self.model, self.device = current, next(current.parameters()).device
        self.version, self.old_version = int(cfg["edge"]), int(cfg["edge"])-1
        self.max_length, self.band = int(history_cap), int(cfg["append_band_size"])
        self.time, self.end_time = int(cfg["release_day"])*86400, int(cfg["end_day"])*86400
        source = NativeTrajectory(parent, self.old_version, events, self.time,
            max_length=self.max_length, append_band_size=self.band)
        self.events, self.cursor = source.events, source.cursor
        self.origin, self.initial_length = source.cursor, source.cache.seq_len
        self.appended_tokens = self.evicted_tokens = 0
        self.check_batch = bool(check_batch)
        self.max_append_abs_error = self.max_score_abs_error = 0.
        self.append_shapes, self.query_shapes = Counter(), Counter()
        ids = torch.as_tensor(source.retained_events[1], device=self.device, dtype=torch.long)[None]
        mapped = probe.prepare_cache(current, source.cache, ids)
        teacher = source.fresh_cache(current)
        choices = dict(reuse=source.cache, f1=mapped, rebuild=teacher)
        if structural_controls:
            # Source + alpha*(F-source), using exact source and F endpoints.
            for name, alpha in (("f025", .25), ("f05", .5), ("f15", 1.5), ("f2", 2.)):
                choices[name] = HSTUKVCache(source.cache.k+alpha*(mapped.k-source.cache.k),
                    source.cache.v+alpha*(mapped.v-source.cache.v), source.cache.seq_len)
            choices["f_key"] = HSTUKVCache(mapped.k, source.cache.v, source.cache.seq_len)
            choices["f_value"] = HSTUKVCache(source.cache.k, mapped.v, source.cache.seq_len)
        self.methods = list(PERSISTENT_METHODS if structural_controls else SMALL_METHODS)
        extra_maps = extra_maps or {}
        for name, transform in extra_maps.items():
            if name in choices or name in dict(ORACLE_BETAS):
                raise ValueError("extra map method name collides with a native control")
            before = source.cache.k._version, source.cache.v._version
            result = transform(source.cache)
            if before != (source.cache.k._version, source.cache.v._version):
                raise ValueError("extra source-cache transforms must not modify native input")
            if (result.seq_len != source.cache.seq_len or result.k.shape != source.cache.k.shape
                    or result.v.shape != source.cache.v.shape
                    or result.k.device != self.device or result.v.device != self.device
                    or result.k.dtype != source.cache.k.dtype or result.v.dtype != source.cache.v.dtype):
                raise ValueError("extra map output differs from the paired source geometry")
            choices[name] = result
            self.methods.append(name)
        self.index = {name:i for i, name in enumerate(self.methods)}
        self.cache = _pack([choices[m] for m in self.methods])
        self.reference = [choices[m] for m in self.methods] if self.check_batch else None
        self.extra_map_names = list(extra_maps)

        # Each event input is uploaded once. A later query consumes only its
        # causal prefix; future resident inputs never enter an earlier append.
        self.prefetch_end = int(np.searchsorted(self.events[0], self.end_time, side="left"))
        times = self.events[0][self.origin:self.prefetch_end]
        gaps = np.empty(len(times), dtype=np.float32)
        if len(times):
            gaps[0] = 0 if source.last_timestamp is None else int(times[0])-source.last_timestamp
            gaps[1:] = np.diff(times)
            np.clip(gaps, 0, MAX_DELTA_SECONDS, out=gaps)
        self.items = torch.as_tensor(np.asarray(self.events[1][self.origin:self.prefetch_end], dtype=np.int64),
                                     device=self.device)[None]
        self.behaviors = torch.as_tensor(np.asarray(self.events[2][self.origin:self.prefetch_end], dtype=np.int64),
                                         device=self.device)[None]
        self.deltas = torch.as_tensor(gaps, device=self.device)[None]
        names, _, error = self.snapshot()
        self.initial_mse = dict(zip(names, error, strict=True))

    @property
    def last_timestamp(self):
        return int(self.events[0][self.cursor-1]) if self.cursor else None

    @torch.no_grad()
    def advance(self, until):
        until = int(until)
        if until < self.time:
            raise ValueError("quality trajectory cannot move backward")
        stop = int(np.searchsorted(self.events[0], until, side="left"))
        if stop > self.prefetch_end:
            raise ValueError("query exceeds the frozen event-input window")
        count = len(self.methods)
        while self.cursor < stop:
            width = 1 << (min(self.band, stop-self.cursor).bit_length()-1)
            offset = self.cursor-self.origin
            inputs = (self.items[:, offset:offset+width], self.behaviors[:, offset:offset+width],
                      self.deltas[:, offset:offset+width])
            before = self.cache.seq_len
            with _short_prefix_backend(self.model, before, self.max_length):
                new = append_with_rolling_band(self.model, self.cache,
                    *(x.expand(count, -1) for x in inputs), self.max_length)
                if self.check_batch:
                    self.reference = [append_with_rolling_band(self.model, c, *inputs,
                        self.max_length) for c in self.reference]
                    expected = _pack(self.reference)
                    self.max_append_abs_error = max(self.max_append_abs_error,
                        float((new.k-expected.k).abs().max()), float((new.v-expected.v).abs().max()))
                    _check_numeric(new.k,expected.k,"key",self.methods,atol=CACHE_NUMERIC_ATOL)
                    _check_numeric(new.v,expected.v,"value",self.methods,atol=CACHE_NUMERIC_ATOL)
            self.cache = new
            self.cursor += width
            self.appended_tokens += width
            self.evicted_tokens += max(0, before+width-self.max_length)
            self.append_shapes[before, width] += 1
        self.time = until

    @torch.no_grad()
    def snapshot(self):
        f = _member(self.cache, self.index["f1"])
        teacher = _member(self.cache, self.index["rebuild"])
        oracles = [_blend(f, teacher, beta) for _, beta in ORACLE_BETAS]
        merged = HSTUKVCache(torch.cat([self.cache.k]+[c.k for c in oracles], 1),
            torch.cat([self.cache.v]+[c.v for c in oracles], 1), self.cache.seq_len)
        names = self.methods+[name for name, _ in ORACLE_BETAS]
        measured = _error_table(merged, self.index["rebuild"]).cpu().double().numpy()
        if not np.isfinite(measured).all():
            raise ValueError("nonfinite KV state error")
        errors = [dict(k_mse=float(e[0]), v_mse=float(e[1]), kv_mse=float(e[2])) for e in measured]
        return names, merged, errors

    @torch.no_grad()
    def score(self, candidates, query_delta, *, snapshot=None):
        names, cache, errors = self.snapshot() if snapshot is None else snapshot
        values, _ = native_score(self.model, cache, candidates.expand(len(names), -1),
            query_delta.expand(len(names), *query_delta.shape[1:]))
        if not bool(torch.isfinite(values).all()):
            raise ValueError("nonfinite real-feedback score")
        if self.check_batch:
            f, teacher = self.reference[self.index["f1"]], self.reference[self.index["rebuild"]]
            references = self.reference+[_blend(f, teacher, beta) for _, beta in ORACLE_BETAS]
            expected = torch.cat([native_score(self.model, c, candidates, query_delta)[0]
                                  for c in references], 0)
            self.max_score_abs_error = max(self.max_score_abs_error, float((values-expected).abs().max()))
            _check_numeric(values,expected,"score",names,atol=NUMERIC_ATOL)
        self.query_shapes[self.cache.seq_len, candidates.shape[1]] += 1
        return names, values, errors

    def ledger(self):
        return dict(source_native_prefills=1, source_version=self.old_version,
            current_teacher_prefills=1, f_materializations=1, extra_map_materializations={m:1 for m in self.extra_map_names},
            initial_rows=self.initial_length, persistent_methods=self.methods,
            append_shapes=[dict(before_length=n, width=w, calls=c, batch=len(self.methods))
                           for (n,w),c in sorted(self.append_shapes.items())],
            query_shapes=[dict(history_length=n, queries=q, calls=c,
                batch=len(self.methods)+len(ORACLE_BETAS)) for (n,q),c in sorted(self.query_shapes.items())],
            event_input_execution=dict(rows=self.prefetch_end-self.origin,
                array_transfers=3 if self.device.type=="cuda" else 0,
                transient_bytes=sum(t.numel()*t.element_size() for t in (self.items,self.behaviors,self.deltas)),
                per_append_scalar_tensor_constructions=0),
            scope="Actual shared research calls; each persistent path has independent native writes. Oracle cache mixtures only score and require a teacher trajectory.")


@torch.no_grad()
def replay_user(parent, current, F, events, requests, uid, cfg, history_cap=1024,
                structural_controls=True, check_batch=False, extra_maps=None):
    """Return (real event rows, per-UID statistics, native execution ledger)."""
    paths = BatchedQualityPaths(parent, current, F, events, cfg, history_cap=history_cap,
        structural_controls=structural_controls, check_batch=check_batch, extra_maps=extra_maps)
    times = sorted(int(t) for t, rows in requests.items() if rows)
    if any(not paths.time <= t < paths.end_time for t in times):
        raise ValueError("real feedback request outside the frozen release window")
    raw_requests = [row for t in times for row in requests[t]]
    if any(int(row["label"]) not in (0,1) for row in raw_requests):
        raise ValueError("real feedback labels must be binary")
    candidate_ids = torch.as_tensor([int(row["item_idx"]) for row in raw_requests],
        device=paths.device, dtype=torch.long)[None]
    gaps = []
    for timestamp in times:
        cursor = int(np.searchsorted(paths.events[0], timestamp, side="left"))
        gaps.append(0 if cursor==0 else timestamp-int(paths.events[0][cursor-1]))
    query_gaps = torch.as_tensor(gaps, device=paths.device, dtype=torch.float32)
    rows, origin = [], 0
    for t_index, timestamp in enumerate(times):
        paths.advance(timestamp)
        snapshot = paths.snapshot()
        request = requests[timestamp]
        for start in range(0, len(request), int(cfg.get("query_batch",16))):
            selected = request[start:start+int(cfg.get("query_batch",16))]
            candidates = candidate_ids[:, origin+start:origin+start+len(selected)]
            names, scores, errors = paths.score(candidates, query_gaps[t_index:t_index+1], snapshot=snapshot)
            for name, values, error in zip(names, scores.cpu().tolist(), errors, strict=True):
                for request_row, value in zip(selected, values, strict=True):
                    rows.append(dict(uid=int(uid), query_timestamp=timestamp,
                        raw_item_id=int(request_row["raw_item_id"]), item_idx=int(request_row["item_idx"]),
                        label=int(request_row["label"]), method=name, score=float(value),
                        appended_events=paths.appended_tokens, history_rows=paths.cache.seq_len,
                        history_length=paths.cache.seq_len,
                        query_gap=int(gaps[t_index]), history_cap=int(history_cap),
                        initial_kv_mse=paths.initial_mse[name]["kv_mse"],
                        **error, **{"cutover_"+k:v for k,v in paths.initial_mse[name].items()}))
        origin += len(request)
    stats = dict(uid=int(uid), feedback_events=len(raw_requests), request_timestamps=len(times),
        initial_rows=paths.initial_length, initial_history_length=paths.initial_length,
        appended_events=paths.appended_tokens,
        evicted_tokens=paths.evicted_tokens, final_rows=paths.cache.seq_len,
        persistent_methods=paths.methods, methods=paths.methods+[n for n,_ in ORACLE_BETAS],
        initial_mse=paths.initial_mse, history_cap=int(history_cap),
        method_names=paths.methods+[n for n,_ in ORACLE_BETAS],
        max_batch_abs_error=max(paths.max_append_abs_error,paths.max_score_abs_error) if check_batch else None,
        max_append_abs_error=paths.max_append_abs_error if check_batch else None,
        max_score_abs_error=paths.max_score_abs_error if check_batch else None,
        numeric_tolerances=dict(atol=NUMERIC_ATOL,rtol=NUMERIC_RTOL,
            cache_atol=CACHE_NUMERIC_ATOL,score_atol=NUMERIC_ATOL),
        source_state="native old-model cutover prefix; F and all extra maps materialize once",
        teacher="current cutover rebuild followed by its own independent native causal appends",
        oracle_scope="request-local F1-to-rebuild KV blends; never written into persistent paths",
        prediction_input_prefetch=dict(candidate_rows=len(raw_requests), query_gaps=len(times),
            transient_bytes=candidate_ids.numel()*candidate_ids.element_size()+query_gaps.numel()*query_gaps.element_size(),
            array_transfers=2 if paths.device.type=="cuda" else 0))
    return rows, stats, paths.ledger()
