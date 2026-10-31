"""Bounded, chronological native cache replay across model releases.

The supplied event order is authoritative, including ties. ``advance(t)`` reads
only events strictly before t, so a query or release at t precedes its events.
Candidate reads never write to this state. Read-correction observers may track
auxiliary state, but cannot replace native writes.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager, nullcontext
from typing import Callable, ContextManager

import numpy as np
import torch

from hstu_kvcache.models import HSTU, HSTUKVCache
from hstu_kvcache.models.state_transition import append_with_rolling_band


EventArrays = tuple[np.ndarray, np.ndarray, np.ndarray]
AppendObserver = Callable[[HSTUKVCache, HSTUKVCache, EventArrays, int], None]
AppendContext = Callable[[HSTU, HSTUKVCache, EventArrays, int], ContextManager]
MAX_DELTA_SECONDS = 7 * 86400


@contextmanager
def _short_prefix_backend(model: HSTU, length: int, max_length: int):
    """Avoid compiling a Triton specialization for each short-history length."""
    if length >= max_length:
        yield
        return
    attention = [block.attn for block in model.blocks]
    previous = [module.backend for module in attention]
    try:
        for module in attention:
            module.backend = "torch"
        yield
    finally:
        for module, backend in zip(attention, previous, strict=True):
            module.backend = backend


class NativeTrajectory:
    """One user's native state; callers orchestrate users, queries and releases.

    ``switch(..., rebuild=False)`` implements continuous Reuse. An independent
    trajectory using ``rebuild=True`` is the release-rebuild Exact control:
    rebuild the retained raw prefix at each switch, then append natively. Exact
    is intentionally distinct from ``fresh_cache()``, which reconstructs the
    current raw prefix on demand and erases its earlier formation context.

    The ledger counts ``(action, version, before_length, width)`` operations.
    Init/rebuild have before_length zero; append records each actual band. This
    lets an external cost model charge the same shapes without GPU timing here.
    No historical K/V snapshots are accumulated. Append callbacks see bounded
    old/new caches and must not mutate either one.
    """

    def __init__(
        self,
        model: HSTU,
        version: int,
        events: EventArrays,
        start_time: int,
        *,
        max_length: int = 1024,
        append_band_size: int = 32,
        on_append: AppendObserver | None = None,
        append_context: AppendContext | None = None,
    ):
        self.events = tuple(np.asarray(column) for column in events)
        if len(self.events) != 3 or any(
            column.ndim != 1 or len(column) != len(self.events[0])
            for column in self.events
        ):
            raise ValueError("events must be equal-length timestamp/item/behavior arrays")
        if np.any(self.events[0][1:] < self.events[0][:-1]):
            raise ValueError("events must already be in chronological order")
        if max_length < 1 or not 1 <= append_band_size <= max_length:
            raise ValueError("append band and cache cap must be positive; band cannot exceed cap")
        self.model, self.version = model, version
        self.max_length, self.append_band_size = max_length, append_band_size
        self.on_append, self.append_context = on_append, append_context
        self.time = int(start_time)
        self.cursor = int(np.searchsorted(self.events[0], self.time, side="left"))
        self.ledger: Counter[tuple[str, int, int, int]] = Counter()
        self.appended_tokens = self.evicted_tokens = 0
        self.cache = self._prefill(model)
        self.producers = np.full(self.cache.seq_len, version, dtype=np.int64)
        self.ledger["init", version, 0, self.cache.seq_len] += 1

    @property
    def retained_events(self) -> EventArrays:
        count = min(self.cursor, self.max_length)
        return tuple(column[self.cursor - count : self.cursor] for column in self.events)

    @property
    def last_timestamp(self) -> int | None:
        return int(self.events[0][self.cursor - 1]) if self.cursor else None

    @torch.no_grad()
    def _prefill(self, model: HSTU) -> HSTUKVCache:
        times, items, behaviors = self.retained_events
        parameter = next(model.parameters())
        if not len(times):
            shape = (len(model.blocks), 1, 0, model.blocks[0].attn.inner)
            return HSTUKVCache(parameter.new_empty(shape), parameter.new_empty(shape), 0)
        deltas = np.zeros(len(times), dtype=np.float32)
        # Canonical prefix reconstruction uses raw gaps. Native appends below
        # retain the existing serving rule that clamps gaps to seven days.
        deltas[1:] = np.diff(times)
        with _short_prefix_backend(model, len(times), self.max_length):
            return model.compute_kv(
                torch.as_tensor(items, device=parameter.device, dtype=torch.long)[None],
                torch.as_tensor(behaviors, device=parameter.device, dtype=torch.long)[None],
                torch.as_tensor(deltas, device=parameter.device)[None],
            )

    @torch.no_grad()
    def fresh_cache(self, model: HSTU | None = None) -> HSTUKVCache:
        """Diagnostic only; leave live state and its execution ledger unchanged."""
        return self._prefill(self.model if model is None else model)

    @torch.no_grad()
    def advance(self, until: int) -> None:
        """Append every event before ``until``, including request-free intervals."""
        until = int(until)
        if until < self.time:
            raise ValueError("trajectory cannot move backward in time")
        stop = int(np.searchsorted(self.events[0], until, side="left"))
        device = next(self.model.parameters()).device
        while self.cursor < stop:
            # Reuse a small set of attention-kernel shapes rather than one for
            # every possible remaining suffix length.
            width = 1 << (min(self.append_band_size, stop - self.cursor).bit_length() - 1)
            end = self.cursor + width
            appended = tuple(column[self.cursor:end] for column in self.events)
            times, items, behaviors = appended
            deltas = np.empty(width, dtype=np.float32)
            deltas[0] = 0 if self.last_timestamp is None else times[0] - self.last_timestamp
            deltas[1:] = np.diff(times)
            np.clip(deltas, 0, MAX_DELTA_SECONDS, out=deltas)
            old = self.cache
            context = (nullcontext() if self.append_context is None else
                       self.append_context(self.model, old, appended, self.version))
            was_training = self.model.training
            if was_training:
                self.model.eval()
            try:
                with _short_prefix_backend(self.model, old.seq_len, self.max_length), context:
                    new = append_with_rolling_band(
                        self.model, old,
                        torch.as_tensor(items, device=device, dtype=torch.long)[None],
                        torch.as_tensor(behaviors, device=device, dtype=torch.long)[None],
                        torch.as_tensor(deltas, device=device)[None], self.max_length,
                    )
            finally:
                if was_training:
                    self.model.train()
            self.cache, self.cursor = new, end
            self.producers = np.concatenate(
                (self.producers, np.full(width, self.version, dtype=np.int64))
            )[-self.max_length:]
            self.ledger["append", self.version, old.seq_len, width] += 1
            self.appended_tokens += width
            self.evicted_tokens += max(0, old.seq_len + width - self.max_length)
            if self.on_append is not None:
                self.on_append(old, new, appended, self.version)
        self.time = until

    def switch(self, model: HSTU, version: int, at: int, *, rebuild: bool = False) -> None:
        """Finish old-version writes, then publish; events at ``at`` use new weights."""
        self.advance(at)
        self.model, self.version = model, version
        if rebuild:
            self.cache = self._prefill(model)
            self.producers = np.full(self.cache.seq_len, version, dtype=np.int64)
            self.ledger["rebuild", version, 0, self.cache.seq_len] += 1
