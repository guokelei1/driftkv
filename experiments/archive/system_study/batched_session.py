"""Small synchronous batches over distinct, existing user sessions.

Only independent users at the same model/version and actual tensor geometry
are combined. Each user's next history consumer retains its original logical
time. This is a bulk executor; it does not claim an online arrival scheduler.
"""
from collections import defaultdict
from contextlib import contextmanager
import time

import numpy as np
import torch
import triton
import triton.language as tl

from hstu_kvcache.adaptation.reader import score as read_score
from hstu_kvcache.methods.cachetide.compact.lightweight import LightweightValueState
from hstu_kvcache.evaluation.trajectory import MAX_DELTA_SECONDS, _short_prefix_backend
from hstu_kvcache.models import HSTUKVCache
from hstu_kvcache.models.state_transition import append_with_rolling_band
from hstu_kvcache.methods.cachetide.adapter import operation_gate_query
from hstu_kvcache.methods.cachetide.cost import read_flops
from hstu_kvcache.methods.cachetide.session import RelaySession, zero_state
from experiments.archive.system_study.budgeted_rebuild import native_history_read


@triton.jit(do_not_specialize=["N", "NEW_N", "REMOVED", "S0", "S1", "S2"])
def _shift_code_rows(Source, Output, N, NEW_N, REMOVED, S0, S1, S2,
                     LAYERS: tl.constexpr, WIDTH: tl.constexpr,
                     POINTER_TABLE: tl.constexpr, BLOCK: tl.constexpr):
    owner = tl.program_id(1)
    if POINTER_TABLE:
        address = tl.load(Output + owner)
        destination = address.to(tl.pointer_type(tl.float32))
        active = address != 0
    else:
        destination = Output
        active = True
    if active:
        index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = index < LAYERS * NEW_N * WIDTH
        layer = index // (NEW_N * WIDTH)
        row = index // WIDTH % NEW_N
        coord = index % WIDTH
        value = tl.load(Source + layer*S0 + owner*S1 + (row+REMOVED)*S2 + coord,
                        valid & (row+REMOVED < N), other=0.)
        tl.store(destination + index, value, valid)


@torch.no_grad()
def append_code_buffers(packed, states, coded_rows, width, new_length):
    """Shift retained codes and append zeros once, into independent user storage."""
    source = packed.codes
    if source.dtype != torch.float32:
        raise ValueError("the frozen serving probe uses FP32 codes")
    removed = max(0, packed.seq_len+width-new_length)
    outputs, addresses = [], []
    for state, count in zip(states, coded_rows, strict=True):
        if count == 0 and state.seq_len == new_length:
            # Current format with coded_rows=0 is already an all-zero window.
            outputs.append(state)
            addresses.append(0)
        else:
            codes = source.new_empty((source.shape[0], 1, new_length, *source.shape[3:]))
            outputs.append(LightweightValueState(codes, new_length))
            addresses.append(codes.data_ptr())
    if not any(addresses):
        return outputs
    pointer_table = len(outputs) > 1
    target = (torch.tensor(addresses, dtype=torch.int64, device=source.device)
              if pointer_table else outputs[0].codes)
    _shift_code_rows[(triton.cdiv(source.shape[0]*new_length*48, 1024), len(outputs))](
        source, target, packed.seq_len, new_length, removed, *source.stride()[:3],
        LAYERS=source.shape[0], WIDTH=48, POINTER_TABLE=pointer_table, BLOCK=1024)
    return outputs


def grouped_uids(uids, key, maximum):
    buckets = defaultdict(list)
    for uid in uids:
        buckets[key(uid)].append(uid)
    return [members[start:start+maximum] for members in buckets.values()
            for start in range(0, len(members), maximum)]


def append_pair_counts(old_length, width, coded_rows, maximum, diagonal):
    """Exact causal/window mask cardinalities for 0 <= coded_rows <= old_length.

    Each new query sees a contiguous key interval. Sum its lengths directly;
    accounting must not allocate a width-by-history mask on the service path.
    """
    base, cap = old_length+1+diagonal, maximum+diagonal
    k = min(width, max(0, cap-base))
    native = k*base+k*(k-1)//2+(width-k)*cap
    a = old_length-maximum+1

    def positive_sum(z):
        count = min(width, max(0, z))
        return count*z-count*(count-1)//2

    return native, positive_sum(coded_rows-a)-positive_sum(-a)


def zero_tail_execution(coded_rows, queries, *, layers=6, heads=6, head_dim=32, rank=6):
    """Extra executed arithmetic, separate from every user's logical ledger."""
    maximum = max(coded_rows, default=0)
    extra_rows = len(coded_rows)*maximum-sum(coded_rows)
    active = queries*sum(count == 0 for count in coded_rows) if maximum else 0
    work = read_flops(pairs=queries*extra_rows, queries=active, head_rank=rank,
        layers=layers, heads=heads, head_dim=head_dim, gates=True, gate_normalization="rms")
    # InferenceAdapter computes the item-query RMS once across all layers.
    rms_saved = (layers-1)*heads*(4*head_dim+2)*active
    return dict(coded_rows=maximum, session_coded_rows=list(coded_rows),
        additional_zero_code_rows=extra_rows, additional_zero_code_pairs=queries*extra_rows,
        additional_active_code_queries=active, additional_code_pair_flops=work["code_pair_flops"],
        additional_fixed_query_flops=work["total"]-work["code_pair_flops"]-rms_saved,
        additional_coalescing_flops=work["total"]-rms_saved)


class PhaseRecorder:
    """Diagnostic barriers: individual phase sums are not uninstrumented time."""
    def __init__(self, device):
        self.device, self.rows, self.context = device, [], {}

    def call(self, phase, function, metadata):
        torch.cuda.synchronize(self.device)
        before = torch.cuda.memory_allocated(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        value = function()
        torch.cuda.synchronize(self.device)
        row = dict(**self.context, **metadata, phase=phase,
            wall_seconds=time.perf_counter()-started,
            peak_increment_bytes=torch.cuda.max_memory_allocated(self.device)-before)
        # Arrow infers columns from the first row, which can be a code conversion.
        row["compute_backend"] = metadata.get("compute_backend", "eager")
        self.rows.append(row)
        return value


class BatchedSessions:
    def __init__(self, sessions, *, max_batch=8, recorder=None, coalesce_zero_tail=False, projection_fused=False,
                 native_fused_append=False, fused_code_append=False, graph_compute=None):
        if max_batch < 1:
            raise ValueError("max_batch must be positive")
        self.sessions, self.max_batch, self.recorder = sessions, max_batch, recorder
        self.coalesce_zero_tail = coalesce_zero_tail
        self.projection_fused = projection_fused
        self.native_fused_append = native_fused_append
        self.fused_code_append = fused_code_append
        self.graph_compute = graph_compute
        self.calls, self.request_context = [], {}

    def _key(self, uid):
        session = self.sessions[uid]
        trajectory = session.trajectory
        return trajectory.version, id(trajectory.model), trajectory.cache.seq_len

    def _metadata(self, operation, uids, **extra):
        first = self.sessions[uids[0]]
        return dict(operation=operation, uids=list(uids), batch=len(uids),
            version=f"v{first.trajectory.version}", history_length=first.cache.seq_len,
            coded_rows=getattr(first, "coded_rows", 0),
            logical_times=[self.request_context.get(u, {}).get("logical_time", first.trajectory.time) for u in uids],
            trace_event_ids=[self.request_context.get(u, {}).get("event_sequence") for u in uids], **extra)

    def _read_geometry(self, members, width, metadata):
        counts = [getattr(self.sessions[u], "coded_rows", 0) for u in members]
        metadata.update(zero_tail_execution(counts, width))
        if self.projection_fused and metadata["coded_rows"]:
            queries_padded = width if width <= 4 else ((width+15)//16)*16
            rank_padded = 8 if width <= 4 else 16
            metadata.update(projected_query_useful_flops=2*6*6*len(members)*width*32*6,
                projected_query_explicit_tile_flops=2*6*6*len(members)*queries_padded*32*rank_padded,
                projection_tile_scope="Triton multiply-add tile arithmetic only, not measured instructions or an asserted delta over cuBLAS padding")
        if not self.coalesce_zero_tail and len(set(counts)) != 1:
            raise AssertionError("strict mode mixed code-prefix lengths")
        if self.recorder is not None and self.coalesce_zero_tail:
            # Check only in the untimed correctness/phase run. Production state
            # transitions establish these exact zeros by construction.
            for uid, count in zip(members, counts, strict=True):
                if count < metadata["coded_rows"] and torch.count_nonzero(
                        self.sessions[uid].state.codes[:, :, count:metadata["coded_rows"]]).item():
                    raise AssertionError("zero-tail coalescing encountered a nonzero suffix")
        return metadata["coded_rows"]

    def _call(self, phase, function, metadata):
        return function() if self.recorder is None else self.recorder.call(phase, function, metadata)

    def _pack(self, uids, *, codes=False):
        sessions = [self.sessions[u] for u in uids]
        if len(uids) == 1:
            return sessions[0].cache, sessions[0].state if codes else None
        cache = HSTUKVCache(torch.cat([s.cache.k for s in sessions], 1),
                           torch.cat([s.cache.v for s in sessions], 1), sessions[0].cache.seq_len)
        state = (LightweightValueState(torch.cat([s.state.codes for s in sessions], 1), cache.seq_len)
                 if codes else None)
        return cache, state

    @staticmethod
    def _slice_cache(cache, index, batch):
        # Individual ownership prevents a single surviving user retaining a full
        # obsolete packed batch. Copies are charged to scatter, not hidden.
        if batch == 1 and not getattr(cache, "transient_graph_output", False):
            return cache
        return HSTUKVCache(cache.k[:, index:index+1].clone(),
                          cache.v[:, index:index+1].clone(), cache.seq_len)

    @torch.no_grad()
    def _convert(self, uids, *, inverse):
        groups = grouped_uids(uids, lambda u: (*self._key(u), id(self.sessions[u].prefix)), self.max_batch)
        for members in groups:
            first = self.sessions[members[0]]
            operation = "anchorize" if inverse else "materialize"
            metadata = self._metadata(operation, members, query_width=0, append_width=0)
            def pack():
                cache, state = self._pack(members, codes=True)
                ids = torch.as_tensor(np.stack([self.sessions[u].trajectory.retained_events[1]
                    for u in members]), device=cache.k.device, dtype=torch.long)
                return cache, state, ids
            cache, state, ids = self._call("pack_h2d", pack, metadata)
            function = first.prefix.anchorize if inverse else first.prefix.materialize
            self._call("compute", lambda: function(cache, state, ids), metadata)
            def scatter():
                for i, uid in enumerate(members):
                    session = self.sessions[uid]
                    if len(members) > 1:
                        session.state.codes.copy_(state.codes[:, i:i+1])
                    session._record_update("anchorize" if inverse else "compiled_prepare",
                                           session.trajectory.version)
                    if not inverse:
                        session.state_format = "current"
                        session.coded_rows = session.cache.seq_len
                    session._record_memory()
            self._call("scatter", scatter, metadata)
            self.calls.append(metadata)

    def prepare(self, uids):
        pending = []
        for uid in uids:
            session = self.sessions[uid]
            if not isinstance(session, RelaySession) or session.adapter is None:
                continue
            if session.pending:
                raise ValueError("this experiment uses the retained compiled chain only")
            if session.state is None:
                session.state = zero_state(session.cache, session.adapter)
            if session.state_format == "anchor":
                pending.append(uid)
        self._convert(pending, inverse=False)

    @contextmanager
    def _corrected_append(self, model, adapter, cache, state, ids, width, coded_rows, maximum):
        total = cache.seq_len+width
        positions = torch.arange(cache.seq_len, total, device=cache.k.device)
        keys = torch.arange(total, device=cache.k.device)
        diagonal = -1 if model.cfg.causal_diagonal == "exclusive" else 0
        keep = (keys[None] <= positions[:, None]+diagonal) & (keys[None] > positions[:, None]-maximum)
        if adapter is None:
            from hstu_kvcache.methods.cachetide.read import _mask_geometry
            from experiments.archive.system_study.fused_query_projection_kernel import launch_read
            def reader(index, attention, q, k, v):
                mask, strides = _mask_geometry(keep, *q.shape[:3], k.shape[1], q.device)
                # Empty views/dummy pointers are never read by HAS_CODE=False.
                # Native service allocates no codes, gates or decoder state.
                return launch_read(q, k, v, k[:, :0, :48], q, q, q, q, mask, strides,
                                   code_count=0, scale=float(attention.scale))
        else:
            gate_query = operation_gate_query(adapter, model, ids)
            reader = adapter.make_history_read(state, valid_mask=keep, code_count=coded_rows, gate_query=gate_query)
        restored = []
        try:
            for index, block in enumerate(model.blocks):
                attention = block.attn
                restored.append((attention, "forward_with_cache" in attention.__dict__,
                                 attention.__dict__.get("forward_with_cache")))
                def corrected(x, old_k, old_v, window_size=None, *, layer=index, attn=attention):
                    batch, entering, _ = x.shape
                    q, k_new, v_new = attn._project(x)
                    k = torch.cat((old_k, k_new.transpose(1, 2).reshape(batch, entering, attn.inner)), 1)
                    v = torch.cat((old_v, v_new.transpose(1, 2).reshape(batch, entering, attn.inner)), 1)
                    return attn._finish(reader(layer, attn, q, k, v)), (k, v)
                attention.forward_with_cache = corrected
            yield keep
        finally:
            for attention, existed, previous in restored:
                if existed:
                    attention.forward_with_cache = previous
                else:
                    del attention.forward_with_cache

    @torch.no_grad()
    def advance(self, targets):
        stops = {}
        for uid, until in targets.items():
            trajectory = self.sessions[uid].trajectory
            if until < trajectory.time:
                raise ValueError("batch cannot move a user's logical time backward")
            stops[uid] = int(np.searchsorted(trajectory.events[0], until, side="left"))
        while True:
            pending = [u for u, stop in stops.items() if self.sessions[u].trajectory.cursor < stop]
            if not pending:
                break
            self.prepare(pending)
            def key(uid):
                session = self.sessions[uid]
                trajectory = session.trajectory
                width = 1 << (min(trajectory.append_band_size, stops[uid]-trajectory.cursor).bit_length()-1)
                return (*self._key(uid), width, 0 if self.coalesce_zero_tail else getattr(session, "coded_rows", 0),
                        id(getattr(session, "adapter", None)))
            for members in grouped_uids(pending, key, self.max_batch):
                first = self.sessions[members[0]]
                trajectory, model = first.trajectory, first.trajectory.model
                width, coded = key(members[0])[3:5]
                maximum = trajectory.max_length
                is_relay = isinstance(first, RelaySession) and first.adapter is not None
                metadata = self._metadata("append", members, query_width=width, append_width=width)
                metadata["append_backend"] = ("corrected_fused" if is_relay else
                    "generic_zero_code" if self.native_fused_append else "original_native")
                metadata["code_append_backend"] = "fused_independent_buffers" if is_relay and self.fused_code_append else "per_user_torch"
                coded = self._read_geometry(members, width, metadata)
                events = {u: tuple(c[self.sessions[u].trajectory.cursor:self.sessions[u].trajectory.cursor+width]
                                  for c in self.sessions[u].trajectory.events) for u in members}
                def pack():
                    cache, state = self._pack(members, codes=is_relay)
                    deltas = []
                    for uid in members:
                        times = events[uid][0]
                        last = self.sessions[uid].trajectory.last_timestamp
                        delta = np.empty(width, dtype=np.float32)
                        delta[0] = 0 if last is None else times[0]-last
                        delta[1:] = np.diff(times)
                        deltas.append(np.clip(delta, 0, MAX_DELTA_SECONDS))
                    device = cache.k.device
                    ids = torch.as_tensor(np.stack([events[u][1] for u in members]), device=device, dtype=torch.long)
                    behavior = torch.as_tensor(np.stack([events[u][2] for u in members]), device=device, dtype=torch.long)
                    delta = torch.as_tensor(np.stack(deltas), device=device)
                    return cache, state, ids, behavior, delta
                cache, state, ids, behavior, delta = self._call("pack_h2d", pack, metadata)
                use_graph = self.graph_compute is not None and self.graph_compute.supports("append", cache, ids)
                metadata["compute_backend"] = "cuda_graph" if use_graph else "eager"
                def compute():
                    if use_graph:
                        return self.graph_compute.run("append", model, first.adapter if is_relay else None,
                            cache, state, ids, delta, behavior=behavior, coded_rows=coded)
                    with _short_prefix_backend(model, cache.seq_len, maximum):
                        if not is_relay:
                            if not self.native_fused_append:
                                return append_with_rolling_band(model, cache, ids, behavior, delta, maximum)
                            with self._corrected_append(model, None, cache, None, ids, width, 0, maximum):
                                return append_with_rolling_band(model, cache, ids, behavior, delta, maximum)
                        with self._corrected_append(model, first.adapter, cache, state, ids, width, coded, maximum):
                            return append_with_rolling_band(model, cache, ids, behavior, delta, maximum)
                new = self._call("compute", compute, metadata)
                def scatter():
                    old_n, total = cache.seq_len, cache.seq_len+width
                    updated_codes = None
                    if is_relay and self.fused_code_append:
                        updated_codes = append_code_buffers(state,
                            [self.sessions[u].state for u in members],
                            [self.sessions[u].coded_rows for u in members], width, new.seq_len)
                    diagonal = -1 if model.cfg.causal_diagonal == "exclusive" else 0
                    for i, uid in enumerate(members):
                        session, event = self.sessions[uid], events[uid]
                        t, old = session.trajectory, session.cache
                        t.cache = self._slice_cache(new, i, len(members))
                        t.cursor += width
                        t.producers = np.concatenate((t.producers, np.full(width, t.version, dtype=np.int64)))[-maximum:]
                        t.ledger["append", t.version, old_n, width] += 1
                        t.appended_tokens += width
                        t.evicted_tokens += max(0, total-maximum)
                        if is_relay:
                            actual_coded = session.coded_rows
                            valid_pairs, code_pairs = append_pair_counts(
                                old_n, width, actual_coded, maximum, diagonal)
                            session.costs["writer_gate_item_lookup_bytes"] += width*model.cfg.hidden_size*4
                            session.read_ledger["writer", t.version, old_n, width, total] += 1
                            session.coded_read_ledger["writer", t.version, old_n, width, actual_coded, total] += 1
                            session.costs.update(dict(writer_read_calls=1, writer_queries=width,
                                writer_dense_pairs=width*total, writer_dense_code_pairs=width*actual_coded,
                                writer_valid_pairs=valid_pairs, writer_old_code_valid_pairs=code_pairs))
                            if updated_codes is None:
                                session._appended(old, t.cache, event, t.version)
                            else:
                                session.state = updated_codes[i]
                                session.coded_rows = max(0, actual_coded-max(0, total-new.seq_len))
                                session.costs["zero_code_appended_rows"] += width
                                session._record_memory()
                self._call("scatter", scatter, metadata)
                self.calls.append(metadata)
        for uid, until in targets.items():
            self.sessions[uid].trajectory.time = int(until)

    @torch.no_grad()
    def publish(self, model, version, at, *, adapter=None, prefix=None, rebuild=False, event_sequence=None):
        self.request_context = {u: dict(logical_time=at, event_sequence=event_sequence) for u in self.sessions}
        self.advance({u: at for u in self.sessions})
        if adapter is not None:
            self._convert([u for u, s in self.sessions.items() if s.prefix is not None
                           and s.state_format == "current"], inverse=True)
        for session in self.sessions.values():
            trajectory = session.trajectory
            if isinstance(session, RelaySession):
                if session.prefix is None:
                    session.state = zero_state(session.cache, adapter)
                session.prefix, session.adapter, session.state_format = prefix, adapter, "anchor"
                trajectory.model, trajectory.version = model, version
                session.costs["publication_events"] += 1
                session.events.append(dict(action="publish", version=version, at=at,
                                           cursor=trajectory.cursor, compiled=True))
                session._record_memory()
            else:
                trajectory.model, trajectory.version = model, version
                session.pending_rebuild = rebuild
                session.costs["publication_events"] += 1

    @torch.no_grad()
    def _rebuild(self, uids, requests):
        pending = [u for u in uids if getattr(self.sessions[u], "pending_rebuild", False)]
        for members in grouped_uids(pending, self._key, self.max_batch):
            first = self.sessions[members[0]]
            model, n = first.trajectory.model, first.cache.seq_len
            metadata = self._metadata("rebuild", members, query_width=0, append_width=0)
            def pack():
                retained = [self.sessions[u].trajectory.retained_events for u in members]
                deltas = np.zeros((len(members), n), dtype=np.float32)
                for i, events in enumerate(retained):
                    deltas[i, 1:] = np.diff(events[0])
                device = first.cache.k.device
                return (torch.as_tensor(np.stack([e[1] for e in retained]), device=device, dtype=torch.long),
                        torch.as_tensor(np.stack([e[2] for e in retained]), device=device, dtype=torch.long),
                        torch.as_tensor(deltas, device=device))
            inputs = self._call("pack_h2d", pack, metadata)
            use_graph = self.graph_compute is not None and self.graph_compute.supports("rebuild", first.cache, inputs[0])
            metadata["compute_backend"] = "cuda_graph" if use_graph else "eager"
            def compute():
                if use_graph:
                    return self.graph_compute.run("rebuild", model, None, first.cache, None,
                        inputs[0], inputs[2], behavior=inputs[1])
                with _short_prefix_backend(model, n, first.trajectory.max_length):
                    return model.compute_kv(*inputs)
            cache = self._call("compute", compute, metadata)
            def scatter():
                for i, uid in enumerate(members):
                    session = self.sessions[uid]
                    t = session.trajectory
                    t.cache = self._slice_cache(cache, i, len(members))
                    t.producers = np.full(n, t.version, dtype=np.int64)
                    t.ledger["rebuild", t.version, 0, n] += 1
                    session.pending_rebuild = False
                    session.costs["rebuild_events"] += 1
                    session.events.append(dict(uid=uid, at=requests[uid][0], version=t.version, history_length=n))
            self._call("scatter", scatter, metadata)
            self.calls.append(metadata)

    @torch.no_grad()
    def score(self, requests):
        """uid -> (logical timestamp, candidate row list), one pending round/uid."""
        self.advance({u: at for u, (at, _) in requests.items()})
        self.prepare(list(requests))
        self._rebuild(list(requests), requests)
        def key(uid):
            session = self.sessions[uid]
            return (*self._key(uid), len(requests[uid][1]), 0 if self.coalesce_zero_tail else getattr(session, "coded_rows", 0),
                    id(getattr(session, "adapter", None)))
        result = {}
        for members in grouped_uids(requests, key, self.max_batch):
            first = self.sessions[members[0]]
            is_relay = isinstance(first, RelaySession)
            q = len(requests[members[0]][1])
            metadata = self._metadata("score", members, query_width=q, append_width=0)
            coded = self._read_geometry(members, q, metadata)
            def pack():
                cache, state = self._pack(members, codes=is_relay)
                candidates = torch.tensor([[r["item_idx"] for r in requests[u][1]] for u in members],
                                           device=cache.k.device, dtype=torch.long)
                deltas = torch.tensor([requests[u][0]-self.sessions[u].trajectory.last_timestamp
                    if self.sessions[u].trajectory.last_timestamp is not None else 0 for u in members],
                    device=cache.k.device, dtype=torch.float32)
                return cache, state, candidates, deltas
            cache, state, candidates, deltas = self._call("pack_h2d", pack, metadata)
            use_graph = self.graph_compute is not None and self.graph_compute.supports("score", cache, candidates)
            metadata["compute_backend"] = "cuda_graph" if use_graph else "eager"
            def compute():
                if use_graph:
                    return self.graph_compute.run("score", first.trajectory.model,
                        first.adapter if is_relay else None, cache, state, candidates, deltas, coded_rows=coded)
                if is_relay:
                    gate = operation_gate_query(first.adapter, first.trajectory.model, candidates)
                    reader = first.adapter.make_history_read(state, code_count=coded, gate_query=gate)
                else:
                    reader = native_history_read
                return read_score(first.trajectory.model, cache, candidates, deltas, history_read_fn=reader)[0]
            scores = self._call("compute", compute, metadata)
            values = self._call("scatter_d2h", lambda: scores.detach().cpu().tolist(), metadata)
            for i, uid in enumerate(members):
                session = self.sessions[uid]
                version, n = session.trajectory.version, session.cache.seq_len
                if is_relay:
                    c = session.coded_rows
                    session.costs["ranking_gate_item_lookup_bytes"] += q*session.trajectory.model.cfg.hidden_size*4
                    session.read_ledger["ranking", version, n, q, n] += 1
                    session.coded_read_ledger["ranking", version, n, q, c, n] += 1
                    session.costs.update(dict(ranking_read_calls=1, ranking_queries=q,
                        ranking_dense_pairs=q*n, ranking_dense_code_pairs=q*c))
                else:
                    session.query_ledger[version, n, q] += 1
                    session.costs["candidate_scores"] += q
                for row, value in zip(requests[uid][1], values[i], strict=True):
                    result[row["request_id"]] = value
            self.calls.append(metadata)
        return result
