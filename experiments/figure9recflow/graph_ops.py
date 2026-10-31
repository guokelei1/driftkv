"""Bounded CUDA-graph execution for RecFlow reads and scalar FIFO writes.

There is one full-history resident buffer and at most TWO graphs per executor.
Graphs use separate private pools: neither graph may overwrite the other's live
outputs. Short histories and unsupported batch layouts use the eager oracle.

By default append returns independent storage, like cache_ops.append_behaviors.
``borrow_state=True`` is an explicit single-user fast path: its returned cache
is borrowed resident storage, invalidated by loading a different cache or the
next append. Finish that user's timeline, or snapshot() before switching users.
Score results always own their storage. Candidate queries never write history.
"""
from __future__ import annotations

import time

import torch

from experiments.figure9recflow.cache_ops import append_behaviors, score_candidates
from hstu_kvcache.models.kv_cache import HSTUKVCache


def _clone(cache):
    return HSTUKVCache(cache.k.clone(), cache.v.clone(), cache.seq_len)


class GraphExecutor:
    def __init__(self, model, method_count=29, history_limit=1024, chunk_size=128,
                 *, borrow_state=False, warmup=2):
        if min(method_count, history_limit, chunk_size) < 1 or warmup < 1:
            raise ValueError("Positive graph shapes and at least one warmup are required")
        if history_limit >= model.backbone.cfg.max_seq_len:
            raise ValueError("History must leave room for one transient query")
        if model.training or model.backbone.training:
            raise ValueError("CUDA graph model must be in eval mode")
        self.model = model
        self.method_count, self.history_limit, self.chunk_size = method_count, history_limit, chunk_size
        self.borrow_state, self.warmup = bool(borrow_state), int(warmup)
        self.resident = None
        self.read_graph = self.append_graph = None
        self.read_features = self.read_deltas = self.read_output = None
        self.write_features = self.write_behaviors = self.write_deltas = None
        self.stats = dict(read_captures=0, append_captures=0, read_replays=0,
                          append_replays=0, eager_reads=0, eager_appends=0,
                          capture_seconds=0., resident_input_copies=0)

    def _eligible(self, cache, features):
        # The scalar, inclusive HSTU-reference contract is checked by the eager
        # core during capture. A single shared feature batch avoids duplicating
        # the large external encoder across methods.
        return (cache.k.is_cuda and cache.k.dtype == torch.float32
                and cache.seq_len == self.history_limit
                and cache.k.shape[1] == self.method_count
                and features.ndim == 3 and features.shape[0] == 1 and features.shape[-1] == 5
                and features.device == cache.k.device
                and features.dtype in (torch.int32, torch.int64))

    def _load(self, cache):
        if cache.k.shape != cache.v.shape or cache.k.shape[2] != self.history_limit:
            raise ValueError("Graph cache dimensions differ")
        if self.resident is None:
            self.resident = HSTUKVCache(torch.empty_like(cache.k), torch.empty_like(cache.v), cache.seq_len)
        elif (self.resident.k.device != cache.k.device or self.resident.k.dtype != cache.k.dtype
              or self.resident.k.shape != cache.k.shape):
            raise ValueError("An executor binds one model/device/cache geometry; create another explicitly")
        # The single-user borrowed path can keep the resident in place between
        # read and write calls. Never overwrite it while a previous graph reads.
        if cache.k.data_ptr() != self.resident.k.data_ptr():
            self.resident.k.copy_(cache.k)
        if cache.v.data_ptr() != self.resident.v.data_ptr():
            self.resident.v.copy_(cache.v)
        if cache.k.data_ptr() != self.resident.k.data_ptr() or cache.v.data_ptr() != self.resident.v.data_ptr():
            self.stats["resident_input_copies"] += 1

    def _capture(self, operation, *, warm_operation=None):
        device = self.resident.k.device
        started = time.perf_counter()
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(self.warmup):
                value = (operation if warm_operation is None else warm_operation)()
                del value
        torch.cuda.current_stream(device).wait_stream(stream)
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        # No shared pool: append and read are independently replayable.
        with torch.cuda.graph(graph, stream=stream):
            result = operation()
        torch.cuda.current_stream(device).wait_stream(stream)
        self.stats["capture_seconds"] += time.perf_counter() - started
        return graph, result

    def _fill_read(self, features, delta, start, stop):
        n = stop - start
        self.read_features[:, :n].copy_(features[:, start:stop])
        if n < self.chunk_size:
            self.read_features[:, n:].copy_(features[:, stop - 1:stop].expand(-1, self.chunk_size - n, -1))
        if delta.ndim == 1:
            self.read_deltas.copy_(delta[:, None].expand(1, self.chunk_size))
        else:
            self.read_deltas[:, :n].copy_(delta[:, start:stop])
            if n < self.chunk_size:
                self.read_deltas[:, n:].copy_(delta[:, stop - 1:stop].expand(-1, self.chunk_size - n))

    @torch.no_grad()
    def score(self, cache, features, delta):
        """Score all candidates; final graph chunk duplicates then discards rows."""
        delta_ok = (delta.ndim == 1 and delta.shape == (1,)) or (
            delta.ndim == 2 and delta.shape == features.shape[:2])
        if not self._eligible(cache, features) or not delta_ok or features.shape[1] == 0:
            self.stats["eager_reads"] += 1
            return score_candidates(self.model, cache, features, delta, chunk_size=self.chunk_size)
        if self.model.training or self.model.backbone.training:
            raise ValueError("Captured model must remain in eval mode")
        self._load(cache)
        if self.read_features is None:
            self.read_features = torch.empty((1, self.chunk_size, 5), device=cache.k.device, dtype=torch.long)
            self.read_deltas = torch.empty((1, self.chunk_size), device=cache.k.device, dtype=torch.float32)
        outputs = cache.k.new_empty((self.method_count, features.shape[1]))
        for start in range(0, features.shape[1], self.chunk_size):
            stop = min(start + self.chunk_size, features.shape[1])
            self._fill_read(features, delta, start, stop)
            if self.read_graph is None:
                self.read_graph, self.read_output = self._capture(lambda: score_candidates(
                    self.model, self.resident, self.read_features, self.read_deltas, chunk_size=self.chunk_size))
                self.stats["read_captures"] += 1
            self.read_graph.replay()
            self.stats["read_replays"] += 1
            # Result owns its storage; subsequent graph replays cannot clobber
            # earlier chunks, requests, or output retained by the caller.
            outputs[:, start:stop].copy_(self.read_output[:, :stop - start])
        return outputs

    @torch.no_grad()
    def append(self, cache, features, behavior, delta):
        """Append chronologically, replaying one strict FIFO graph per event."""
        layout_ok = behavior.shape == features.shape[:2] and delta.shape == features.shape[:2]
        if not self._eligible(cache, features) or not layout_ok or features.shape[1] == 0:
            self.stats["eager_appends"] += 1
            return append_behaviors(self.model, cache, features, behavior, delta, window_size=self.history_limit)
        if self.model.training or self.model.backbone.training:
            raise ValueError("Captured model must remain in eval mode")
        self._load(cache)
        if self.write_features is None:
            self.write_features = torch.empty((1, 1, 5), device=cache.k.device, dtype=torch.long)
            self.write_behaviors = torch.empty((1, 1), device=cache.k.device, dtype=torch.long)
            self.write_deltas = torch.empty((1, 1), device=cache.k.device, dtype=torch.float32)
        for index in range(features.shape[1]):
            self.write_features.copy_(features[:, index:index + 1])
            self.write_behaviors.copy_(behavior[:, index:index + 1])
            self.write_deltas.copy_(delta[:, index:index + 1])
            if self.append_graph is None:
                def compute():
                    return append_behaviors(self.model, self.resident, self.write_features,
                                            self.write_behaviors, self.write_deltas, window_size=self.history_limit)

                def persist():
                    complete = compute()
                    # All layer reads and new-KV construction finish before
                    # either resident tensor is overwritten on this stream.
                    self.resident.k.copy_(complete.k)
                    self.resident.v.copy_(complete.v)
                    return self.resident

                saved = _clone(self.resident)
                self.append_graph, _ = self._capture(persist, warm_operation=compute)
                # Capture/warmup is setup, never a consumed behavior event.
                self.resident.k.copy_(saved.k)
                self.resident.v.copy_(saved.v)
                del saved
                self.stats["append_captures"] += 1
            self.append_graph.replay()
            self.stats["append_replays"] += 1
        return self.resident if self.borrow_state else _clone(self.resident)

    @torch.no_grad()
    def snapshot(self, cache=None):
        """Own a resident result before switching users or retaining a version."""
        source = self.resident if cache is None else cache
        if source is None:
            raise ValueError("No resident cache exists yet")
        return _clone(source)

    def clear(self):
        """Release both bounded graphs and their buffers; borrowed state expires."""
        self.read_graph = self.append_graph = None
        self.read_output = self.resident = None
        self.read_features = self.read_deltas = None
        self.write_features = self.write_behaviors = self.write_deltas = None
