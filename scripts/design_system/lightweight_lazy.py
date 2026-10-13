"""Demand-driven code preparation over the unchanged native KV trajectory."""
from __future__ import annotations

import torch

from design_system.evaluate_lightweight import LightweightSession


class LazyLightweightSession(LightweightSession):
    """Bind each release immediately; materialize its codes at the first read.

    Encoders are deterministic, row-local, and fixed during a release. Native
    appends and causal support continue while preparation is pending. This
    defers code generation, never reconstruction of native K/V.
    """

    def __init__(self, trajectory, *, kernel="vector", exit_when_clear=True):
        self.pending_encode = False
        super().__init__(trajectory, kernel=kernel, exit_when_clear=exit_when_clear)

    @property
    def read_mode(self):
        return "pending" if self.pending_encode else super().read_mode

    @property
    def native_mode(self):
        return not self.pending_encode and super().native_mode

    def _retire(self, reason, at):
        pending = self.pending_encode
        self.pending_encode = False
        super()._retire(reason, at)
        if pending:
            self.costs["unmaterialized_retirement_events"] += 1
            self.events.append(dict(action="cancel_pending_encode", reason=reason,
                at=int(at), version=self.trajectory.version, cursor=self.trajectory.cursor))

    @torch.no_grad()
    def _appended(self, old, new, events, version):
        super()._appended(old, new, events, version)
        if self.pending_encode and self.exit_when_clear and self.support.clear:
            self._retire("inherited_support_cleared_before_first_read", int(events[0][-1]))

    @torch.no_grad()
    def publish(self, model, version, at, adapter, coordinator=None, *, policy="correct"):
        if policy != "correct" or coordinator is not None:
            raise ValueError("lazy prototype uses correction with already-folded release parameters")
        if version == self.trajectory.version:
            raise ValueError("publish requires a new version")
        self.advance(at)  # Complete old-version native writes before rebinding.
        if self.pending_encode:
            self.costs["superseded_pending_releases"] += 1
        self.trajectory.switch(model, version, at)
        self.adapter, self.coordinator, self.policy = adapter, None, policy
        self.state = None
        self.pending_rebuild = False
        self.support.reset(self.cache.seq_len, inherited=True)
        self.pending_encode = not self.exit_when_clear or not self.support.clear
        self.costs["publication_events"] += 1
        self.events.append(dict(action="publish", at=int(at), version=int(version),
            cursor=self.trajectory.cursor, policy="encode_on_first_read"))
        self._record_memory()

    @torch.no_grad()
    def rebuild(self, at, *, reason="explicit"):
        super().rebuild(at, reason=reason)
        self.pending_encode = False

    @torch.no_grad()
    def prepare_read(self, at):
        super().prepare_read(at)
        if self.pending_encode:
            if self.exit_when_clear and self.support.clear:
                self._retire("inherited_support_cleared_before_first_read", at)
            else:
                self.state = self._encode(self.cache, self.trajectory.retained_events[1],
                                          "initial_encode_flops")
                self.pending_encode = False
                self.costs["first_read_code_initializations"] += 1
                self.events.append(dict(action="initialize_codes", at=int(at),
                    version=self.trajectory.version, cursor=self.trajectory.cursor,
                    retained_rows=self.cache.seq_len))
                self._record_memory()
