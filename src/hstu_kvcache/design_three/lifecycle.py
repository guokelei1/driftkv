"""One native trajectory, release-bound compact codes, and causal retirement.

Support is a conservative dependency statement, not a predicted quality loss.
Clearing it means independence from the K/V present at the latest release. It
does not mean equality to a fresh reconstruction of the latest raw prefix.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch

from hstu_kvcache.adaptation.reader import score as read_score
from hstu_kvcache.design_one.compact import CompactItemState
from hstu_kvcache.design_two.coordination import make_override
from hstu_kvcache.models import HSTUKVCache

_BOUND_COORDINATOR = object()


class InheritedSupport:
    """Boolean support for each layer's retained K/V rows, in FIFO order.

    A new layer-0 K/V depends only on the current event embedding. A new
    layer-l K/V can depend on the new lower-layer representation and every
    visible lower-layer cached K/V. Consequently lower-layer support propagates
    upward, including after every old producer row has been evicted. Processing
    writes sequentially here has the same sliding window as native band writes.
    """

    def __init__(self, layers, length, cap, *, inherited=False):
        self.cap = int(cap)
        self.flags = np.full((int(layers), int(length)), inherited, dtype=bool)

    @property
    def clear(self):
        return not bool(self.flags.any())

    def reset(self, length, *, inherited):
        self.flags = np.full((self.flags.shape[0], int(length)), inherited, dtype=bool)

    def append(self, width):
        for _ in range(int(width)):
            # The new row reads itself and at most cap-1 preceding rows.
            old = self.flags[:, -(self.cap - 1):] if self.cap > 1 else self.flags[:, :0]
            new = np.zeros((old.shape[0], 1), dtype=bool)
            for layer in range(1, old.shape[0]):
                new[layer, 0] = new[layer - 1, 0] or old[layer - 1].any()
            self.flags = np.concatenate((old, new), axis=1)


class LifecycleSession:
    """Small synchronous lifecycle around an existing NativeTrajectory.

    publish() finishes old-version writes before replacing model/adapter/strengths.
    advance() includes idle intervals. score() takes only time and candidate IDs;
    no label, Full score, or outcome is accepted. Optional reconstruction policies
    are fixed controls: at release or immediately before the first request.

    ``exit_when_clear=False`` retains the historical always-corrected D1 control.
    Its reads/codes match CompactObserver; native writes remain identical under
    either setting until an explicitly requested reconstruction occurs.
    """

    POLICIES = ("correct", "rebuild_at_release", "rebuild_on_first_request")

    def __init__(self, trajectory, *, exit_when_clear=True):
        if trajectory.on_append is not None or trajectory.append_context is not None:
            raise ValueError("lifecycle owns native append observation; corrected writes are unsupported")
        if trajectory.model.cfg.relative_position_bias:
            raise ValueError("causal retirement is checked for native no-position-bias rolling writes")
        self.trajectory = trajectory
        self.exit_when_clear = bool(exit_when_clear)
        self.support = InheritedSupport(len(trajectory.model.blocks), trajectory.cache.seq_len,
                                        trajectory.max_length)
        self.adapter = self.coordinator = self.state = None
        self.policy = "correct"
        self.pending_rebuild = False
        self.costs = Counter()
        self.memory = Counter()
        self.events = []
        trajectory.on_append = self._appended
        self._record_memory()

    @property
    def cache(self):
        return self.trajectory.cache

    @property
    def read_mode(self):
        return "corrected" if self.state is not None else "native"

    @property
    def native_mode(self):
        return self.state is None

    def _record_memory(self):
        native = sum(t.numel() * t.element_size() for t in (self.cache.k, self.cache.v))
        compact = 0 if self.state is None else self.state.storage_bytes()
        support = self.support.flags.nbytes
        for name, value in (("native_kv_peak_bytes", native), ("compact_peak_bytes", compact),
                            ("support_peak_bytes", support), ("extra_peak_bytes", compact + support)):
            self.memory[name] = max(self.memory[name], int(value))

    @torch.no_grad()
    def _encode(self, cache, items, cost_key):
        ids = torch.as_tensor(items, device=cache.k.device, dtype=torch.long)[None]
        features = self.trajectory.model.lookup_item_embeddings(ids)
        self.costs[cost_key] += self.adapter.estimate_flops(tokens=cache.seq_len)["token_transform"]
        self.costs["item_lookup_bytes"] += features.numel() * features.element_size()
        return self.adapter.encode_cache(cache, features)

    def _retire(self, reason, at):
        was_active = self.state is not None or self.pending_rebuild
        self.state = None
        self.pending_rebuild = False
        if was_active:
            self.costs["retirement_events"] += 1
            self.events.append(dict(action="retire", reason=reason, at=int(at),
                                    version=self.trajectory.version,
                                    cursor=self.trajectory.cursor))

    @torch.no_grad()
    def _appended(self, old, new, events, version):
        width = len(events[0])
        self.support.append(width)
        self.costs["support_token_layers"] += width * len(self.trajectory.model.blocks)
        if self.state is not None and self.exit_when_clear and self.support.clear:
            self._retire("inherited_support_cleared", int(events[0][-1]))
        elif self.state is not None:
            entering = HSTUKVCache(new.k[:, :, -width:], new.v[:, :, -width:], width)
            coded = self._encode(entering, events[1], "append_encode_flops")
            hidden = torch.cat((self.state.hidden, coded.hidden), dim=2)[:, :, -new.seq_len:].contiguous()
            self.state = CompactItemState(hidden, new.seq_len)
        self._record_memory()

    def advance(self, until):
        self.trajectory.advance(until)

    @torch.no_grad()
    def publish(self, model, version, at, adapter, coordinator=None, *, policy="correct"):
        if policy not in self.POLICIES:
            raise ValueError(f"unsupported fixed lifecycle policy: {policy}")
        if coordinator is not None and coordinator.mode != "static":
            raise ValueError("this lifecycle binds release-fixed static strengths only")
        if version == self.trajectory.version:
            raise ValueError("publish requires a new version; use rebuild for current-version replacement")
        self.advance(at)
        self.trajectory.switch(model, version, at)
        self.adapter, self.coordinator, self.policy = adapter, coordinator, policy
        self.state = None  # Codes depend on both the target encoder and target item embeddings.
        self.support.reset(self.cache.seq_len, inherited=True)
        self.pending_rebuild = policy == "rebuild_on_first_request"
        self.costs["publication_events"] += 1
        self.events.append(dict(action="publish", at=int(at), version=int(version),
                                cursor=self.trajectory.cursor, policy=policy))
        if policy == "rebuild_at_release":
            self.rebuild(at, reason="release")
        elif policy == "correct" and (not self.exit_when_clear or not self.support.clear):
            self.state = self._encode(self.cache, self.trajectory.retained_events[1],
                                      "initial_encode_flops")
        self._record_memory()

    @torch.no_grad()
    def rebuild(self, at, *, reason="explicit"):
        """Replace live state from the current raw prefix, then continue native writes."""
        self.advance(at)
        self.trajectory.switch(self.trajectory.model, self.trajectory.version, at, rebuild=True)
        self.support.reset(self.cache.seq_len, inherited=False)
        self.state = None
        self.pending_rebuild = False
        self.costs["rebuild_events"] += 1
        self.events.append(dict(action="rebuild", reason=reason, at=int(at),
                                version=self.trajectory.version, cursor=self.trajectory.cursor))
        self._record_memory()

    def prepare_read(self, at):
        self.advance(at)
        if self.pending_rebuild:
            self.rebuild(at, reason="first_request")

    def history_override(self, coordinator=_BOUND_COORDINATOR):
        if self.state is None:
            return None
        model = self.trajectory.model
        coordinator = self.coordinator if coordinator is _BOUND_COORDINATOR else coordinator
        if coordinator is None:
            return self.adapter.make_history_override(model, self.cache, self.state)
        if coordinator.mode != "static":
            raise ValueError("lifecycle comparisons accept release-fixed strengths only")
        return make_override(self.adapter, model, self.cache, self.state, coordinator)

    @torch.no_grad()
    def score(self, at, candidates, *, coordinator=_BOUND_COORDINATOR):
        """Return logits; native update/rebuild and correction work stay in ledgers.

        trajectory.ledger contains all real prefill/append/rebuild shapes. costs
        contains correction FLOPs and logical support work, without charging any
        calibration or shared decoder setup (those are release-wide caller costs).
        """
        self.prepare_read(at)
        coordinator = self.coordinator if coordinator is _BOUND_COORDINATOR else coordinator
        delta = 0 if self.trajectory.last_timestamp is None else int(at) - self.trajectory.last_timestamp
        query_delta = torch.tensor([delta], dtype=torch.float32, device=self.cache.k.device)
        logits = read_score(self.trajectory.model, self.cache, candidates, query_delta,
                            history_override=self.history_override(coordinator))[0]
        if self.state is not None:
            work = self.adapter.estimate_flops(tokens=self.cache.seq_len, candidates=candidates.shape[1])
            self.costs["extra_history_read_flops"] += work["extra_history_read"]
            self.costs["response_flops"] += work["candidate_reads"]
            if coordinator is not None:
                work = coordinator.estimate_added_flops(tokens=self.cache.seq_len,
                    heads=self.trajectory.model.cfg.num_heads, candidates=candidates.shape[1])
                self.costs["coordination_flops"] += work["total"]
            self.costs["corrected_requests"] += candidates.numel()
        else:
            self.costs["native_requests"] += candidates.numel()
        return logits
