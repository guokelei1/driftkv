"""Synchronous corrected reads and writes over one retained native trajectory.

The trajectory owns event ordering, rolling eviction and the native operation
ledger. A temporary append context replaces only its attention read: Q/K/V
projections and native block updates remain the checkpoint's operations. Codes
are appended as zero for current-produced rows, not as an assertion of Fresh
equivalence. No inherited-support flags or reconstructed K/V copy are retained.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch

from hstu_kvcache.adaptation.reader import read_embedded, score as read_score
from hstu_kvcache.design_one.lightweight import LightweightValueState
from hstu_kvcache.models import HSTUKVCache
from hstu_kvcache.models.state_transition import retain_latest_cache
from query_relay_2026_10.adapter import operation_gate_query


def zero_state(cache, adapter):
    shape = (adapter.num_layers, cache.k.shape[1], cache.seq_len,
             adapter.heads, adapter.rank)
    return LightweightValueState(cache.k.new_zeros(shape), cache.seq_len)


def append_corrected_event(model, cache, state, adapter, item, behavior, delta,
                           max_length, *, trace=False, code_count=None):
    """Differentiable one-event oracle for calibration and append checks.

    Unlike NativeTrajectory.advance, this helper does not enter no_grad. A
    frozen model therefore permits gradients through codes/reader parameters
    and the newly generated upper-layer K/V. Callers choose when to detach.
    """
    retained = min(cache.seq_len, max_length - 1)
    if code_count is not None:
        code_count = max(0, min(retained, code_count - (cache.seq_len - retained)))
    cache = retain_latest_cache(cache, retained)
    state = LightweightValueState(state.codes[:, :, -retained:] if retained else
                                  state.codes[:, :, :0], retained)
    x = model.embed_inputs(item, behavior, delta)
    result = read_embedded(model, cache, x, trace=trace,
                           history_read_fn=adapter.make_history_read(state, code_count=code_count,
                               gate_query=operation_gate_query(adapter, model, item)))
    new = HSTUKVCache(torch.cat((cache.k, result.new_kv.k), dim=2),
                     torch.cat((cache.v, result.new_kv.v), dim=2), retained + 1)
    zeros = state.codes.new_zeros((*state.codes.shape[:2], 1, *state.codes.shape[3:]))
    codes = LightweightValueState(torch.cat((state.codes, zeros), dim=2), retained + 1)
    return new, codes, result


@dataclass(frozen=True)
class ReleaseUpdate:
    model: object
    adapter: object
    version: int


class RelaySession:
    """One compact buffer, with explicit-update or compiled-anchor preparation.

    Explicit mode is the reference and non-composable branch: missed releases
    are applied in chronological order on first consumption. Compiled mode
    stores anchors while idle and current codes after preparation. At publish,
    a materialized old buffer is anchorized before retiring its old prefix.
    Sessions are synchronous: temporary attention methods are never shared by
    concurrently executing model calls.
    """

    def __init__(self, trajectory):
        if trajectory.on_append is not None or trajectory.append_context is not None:
            raise ValueError("RelaySession owns trajectory append callbacks")
        if trajectory.model.cfg.relative_position_bias:
            raise ValueError("relay prototype uses the retained no-position-bias models")
        self.trajectory = trajectory
        self.adapter = self.prefix = self.state = None
        self.state_format = "current"
        self.pending = []
        self.coded_rows = 0
        self.costs, self.memory, self.read_ledger = Counter(), Counter(), Counter()
        self.coded_read_ledger = Counter()
        self.events = []
        trajectory.append_context = self._append_context
        trajectory.on_append = self._appended
        self._record_memory()

    @property
    def cache(self):
        return self.trajectory.cache

    @property
    def pending_encode(self):
        return bool(self.pending) or self.state_format == "anchor"

    @property
    def native_mode(self):
        return self.adapter is None

    def _record_memory(self):
        native = self.cache.k.numel() * self.cache.k.element_size()
        native += self.cache.v.numel() * self.cache.v.element_size()
        compact = 0 if self.state is None else self.state.storage_bytes()
        self.memory["native_kv_peak_bytes"] = max(self.memory["native_kv_peak_bytes"], native)
        self.memory["compact_peak_bytes"] = max(self.memory["compact_peak_bytes"], compact)

    def _record_update(self, action, version):
        self.costs[action + "_calls"] += 1
        self.costs[action + "_rows"] += self.cache.seq_len
        self.events.append(dict(action=action, version=int(version),
                                rows=self.cache.seq_len, cursor=self.trajectory.cursor))

    @torch.no_grad()
    def prepare_codes(self):
        """Prepare before any history consumer, including event-only intervals."""
        if self.adapter is None:
            return None
        if self.state is None:
            self.state = zero_state(self.cache, self.adapter)
        if self.state_format == "anchor":
            self.state = self.prefix.materialize(
                self.cache, self.state, self.trajectory.retained_events[1])
            self.state_format = "current"
            self.coded_rows = self.cache.seq_len
            self._record_update("compiled_prepare", self.trajectory.version)
        while self.pending:
            release = self.pending.pop(0)
            ids = torch.as_tensor(self.trajectory.retained_events[1],
                                  device=self.cache.k.device, dtype=torch.long)[None]
            features = release.model.lookup_item_embeddings(ids)
            self.state = release.adapter.advance_codes(self.cache, self.state, features,
                                                       item_ids=ids)
            self.coded_rows = self.cache.seq_len
            self.costs["item_lookup_bytes"] += features.numel() * features.element_size()
            self._record_update("explicit_update", release.version)
        self._record_memory()
        return self.state

    @torch.no_grad()
    def publish(self, model, version, at, adapter, *, prefix=None):
        if version == self.trajectory.version:
            raise ValueError("publish needs a new version")
        self.advance(at)  # Events at the release timestamp use the new model.
        if self.prefix is not None and prefix is None:
            raise ValueError("do not mix compiled and explicit modes within one trajectory")
        if prefix is not None:
            if self.pending:
                raise ValueError("a compiled chain cannot replace unexecuted explicit updates")
            if self.prefix is not None and self.state_format == "current":
                self.state = self.prefix.anchorize(
                    self.cache, self.state, self.trajectory.retained_events[1])
                self._record_update("anchorize", self.trajectory.version)
            elif self.prefix is None:
                # The initial native trajectory starts with correction zero.
                self.state = zero_state(self.cache, adapter)
            self.prefix = prefix
            self.state_format = "anchor"
        else:
            self.pending.append(ReleaseUpdate(model, adapter, int(version)))
        self.adapter = adapter
        self.trajectory.switch(model, version, at)
        self.costs["publication_events"] += 1
        self.events.append(dict(action="publish", version=int(version), at=int(at),
                                cursor=self.trajectory.cursor, compiled=prefix is not None))
        self._record_memory()

    @contextmanager
    def _append_context(self, model, old, events, version):
        if self.adapter is None:
            yield
            return
        state = self.prepare_codes()
        width = len(events[0])
        total = old.seq_len + width
        query_positions = torch.arange(old.seq_len, total, device=old.k.device)
        key_positions = torch.arange(total, device=old.k.device)
        diagonal = -1 if model.cfg.causal_diagonal == "exclusive" else 0
        keep = (key_positions[None] <= query_positions[:, None] + diagonal)
        keep &= key_positions[None] > query_positions[:, None] - self.trajectory.max_length
        # The entering band is native-only. The reader accepts a shorter code
        # prefix, so no temporary all-zero band codes need to be constructed.
        gate_query = None
        if self.adapter.gate_source == "item":
            item_ids = torch.as_tensor(events[1], device=old.k.device, dtype=torch.long)[None]
            gate_query = operation_gate_query(self.adapter, model, item_ids)
            self.costs["writer_gate_item_lookup_bytes"] += gate_query.numel() * gate_query.element_size()
        reader = self.adapter.make_history_read(state, valid_mask=keep,
                                                code_count=self.coded_rows, gate_query=gate_query)
        restored = []
        self.read_ledger["writer", int(version), old.seq_len, width, total] += 1
        self.coded_read_ledger["writer", int(version), old.seq_len,
                               width, self.coded_rows, total] += 1
        self.costs["writer_read_calls"] += 1
        self.costs["writer_queries"] += width
        self.costs["writer_dense_pairs"] += width * total
        self.costs["writer_dense_code_pairs"] += width * self.coded_rows
        self.costs["writer_valid_pairs"] += int(keep.sum())
        self.costs["writer_old_code_valid_pairs"] += int(keep[:, :self.coded_rows].sum())
        try:
            for index, block in enumerate(model.blocks):
                attention = block.attn
                present = "forward_with_cache" in attention.__dict__
                previous = attention.__dict__.get("forward_with_cache")
                restored.append((attention, present, previous))

                def corrected(x_new, cached_k, cached_v, window_size=None,
                              *, layer=index, attn=attention):
                    batch, entering, _ = x_new.shape
                    q, k_new, v_new = attn._project(x_new)
                    k_flat = k_new.transpose(1, 2).reshape(batch, entering, attn.inner)
                    v_flat = v_new.transpose(1, 2).reshape(batch, entering, attn.inner)
                    keys = torch.cat((cached_k, k_flat), dim=1)
                    values = torch.cat((cached_v, v_flat), dim=1)
                    heads = reader(layer, attn, q, keys, values)
                    return attn._finish(heads), (keys, values)

                attention.forward_with_cache = corrected
            yield
        finally:
            for attention, present, previous in restored:
                if present:
                    attention.forward_with_cache = previous
                else:
                    del attention.forward_with_cache

    @torch.no_grad()
    def _appended(self, old, new, events, version):
        if self.adapter is not None:
            width = len(events[0])
            zeros = self.state.codes.new_zeros(
                (*self.state.codes.shape[:2], width, *self.state.codes.shape[3:]))
            self.state = LightweightValueState(torch.cat((self.state.codes, zeros), dim=2)
                                                [:, :, -new.seq_len:].contiguous(), new.seq_len)
            removed = max(0, old.seq_len + width - new.seq_len)
            self.coded_rows = max(0, self.coded_rows - removed)
            self.costs["zero_code_appended_rows"] += width
        self._record_memory()

    def advance(self, until):
        self.trajectory.advance(until)

    def prepare_read(self, at):
        self.advance(at)
        self.prepare_codes()

    def read_callback(self, candidates=None):
        if self.adapter is None:
            return None
        gate_query = operation_gate_query(self.adapter, self.trajectory.model, candidates)
        if gate_query is not None:
            self.costs["ranking_gate_item_lookup_bytes"] += gate_query.numel() * gate_query.element_size()
        return self.adapter.make_history_read(self.state, code_count=self.coded_rows,
                                              gate_query=gate_query)

    @torch.no_grad()
    def score(self, at, candidates, *, trace=False):
        self.prepare_read(at)
        delta = 0 if self.trajectory.last_timestamp is None else int(at) - self.trajectory.last_timestamp
        query_delta = torch.tensor([delta], device=self.cache.k.device, dtype=torch.float32)
        result = read_score(self.trajectory.model, self.cache, candidates, query_delta,
                            history_read_fn=self.read_callback(candidates), trace=trace)
        if self.adapter is not None:
            queries = candidates.numel()
            self.read_ledger["ranking", self.trajectory.version, self.cache.seq_len,
                             queries, self.cache.seq_len] += 1
            self.coded_read_ledger["ranking", self.trajectory.version, self.cache.seq_len,
                                   queries, self.coded_rows, self.cache.seq_len] += 1
            self.costs["ranking_read_calls"] += 1
            self.costs["ranking_queries"] += queries
            self.costs["ranking_dense_pairs"] += queries * self.cache.seq_len
            self.costs["ranking_dense_code_pairs"] += queries * self.coded_rows
        return result if trace else result[0]

    def source_snapshot(self, at):
        """Detached calibration source at a causal boundary, with no labels."""
        self.prepare_read(at)
        cache = HSTUKVCache(self.cache.k.detach().clone(), self.cache.v.detach().clone(),
                           self.cache.seq_len)
        state = None if self.state is None else LightweightValueState(
            self.state.codes.detach().clone(), self.state.seq_len)
        return cache, state
