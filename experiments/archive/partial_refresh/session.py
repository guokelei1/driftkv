"""Synchronous section 4.4 prototype over unchanged native K/V writes.

The first read after publication refreshes the selected rows. Pending appends
are marked missing and join that first-read refresh; once active, appends are
encoded immediately. One int32 encoder version belongs to each complete row
of codes, independently of NativeTrajectory.producers. An old row is usable
only from the immediately preceding numeric model version.

No corrected full K/V is retained. The optional CUDA dispatch is the existing
full-item KV fused reader; the default uses the same adapter's Torch reference.
The optional raw storage mode retains unmasked encoder outputs and gates one
temporary layer at read time. It preserves dormant code coordinates when the
existing causal flags reset at the next release, without adding retained codes.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from hstu_kvcache.adaptation.reader import score as native_score
from hstu_kvcache.methods.cachetide.compact.lightweight import LightweightValueState
from hstu_kvcache.methods.cachetide.lifecycle.lifecycle import LifecycleSession
from hstu_kvcache.models import HSTUKVCache

MISSING_VERSION = -1


@dataclass(frozen=True)
class RefreshPlan:
    indices: np.ndarray
    tail_rows: int
    missing_rows: int
    older_rows: int
    forced_rows: int
    current_rows: int
    retained_previous_rows: int

    @property
    def refreshed_rows(self):
        return len(self.indices)

    def counts(self):
        return dict(refreshed_rows=self.refreshed_rows, tail_rows=self.tail_rows,
                    missing_rows=self.missing_rows, older_rows=self.older_rows,
                    forced_rows=self.forced_rows, current_rows=self.current_rows,
                    retained_previous_rows=self.retained_previous_rows)

    def as_dict(self):
        return dict(indices=self.indices, missing=self.missing_rows, older=self.older_rows,
                    tail=self.tail_rows, current=self.current_rows,
                    selected=self.refreshed_rows, retained_old=self.retained_previous_rows)


def refresh_plan(versions, version, rho):
    """Tail union missing/older rows, with current-version rows excluded."""
    if not math.isfinite(rho) or not 0 <= rho <= 1:
        raise ValueError("rho must be a finite fraction in [0, 1]")
    versions = np.asarray(versions)
    n = len(versions)
    tail_rows = math.ceil(rho * n)
    tail = np.arange(n) >= n - tail_rows
    missing = versions == MISSING_VERSION
    older = (versions < int(version) - 1) & ~missing
    current = versions == int(version)
    selected = (tail | missing | older) & ~current
    return RefreshPlan(np.flatnonzero(selected), tail_rows, int(missing.sum()),
                       int(older.sum()), int((missing | older).sum()),
                       int(current.sum()), int(((versions == int(version) - 1) & ~selected).sum()))


@dataclass(frozen=True)
class CodeSnapshot:
    codes: torch.Tensor | None
    versions: np.ndarray
    version: int

    @property
    def seq_len(self):
        return len(self.versions)


def candidate_state(snapshot, rho, reference):
    """Construct one validation candidate using an already encoded reference.

    This helper performs no encoder or service accounting. The caller charges
    the reference encoding once and each complete candidate query separately.
    The snapshot is the actual saved old-code state, never reconstructed old
    codes. Current-version rows are preserved even in the rho=1 candidate.
    """
    if reference.seq_len != snapshot.seq_len:
        raise ValueError("reference and actual old-code snapshot lengths differ")
    plan = refresh_plan(snapshot.versions, snapshot.version, rho)
    if snapshot.codes is None:
        if plan.refreshed_rows != snapshot.seq_len:
            raise ValueError("absent code tensor requires every row to be missing")
        codes = reference.codes.clone()
    else:
        codes = snapshot.codes.clone()
        codes[:, :, plan.indices] = reference.codes[:, :, plan.indices]
    return LightweightValueState(codes, reference.seq_len), plan


class PartialRefreshSession(LifecycleSession):
    """One live user; release-fixed rho controls its first-read refresh."""

    def __init__(self, trajectory, *, kernel="torch", exit_when_clear=True,
                 storage_mode="masked"):
        if storage_mode not in ("masked", "raw"):
            raise ValueError("storage_mode must be masked or raw")
        self.storage_mode = storage_mode
        self.pending_encode = False
        self.rho = 1.0
        self.code_versions = np.empty(0, dtype=np.int32)
        self.last_refresh = None
        self.kernel = kernel
        super().__init__(trajectory, exit_when_clear=exit_when_clear)

    @property
    def read_mode(self):
        return "pending" if self.pending_encode else super().read_mode

    @property
    def native_mode(self):
        return not self.pending_encode and self.state is None

    def _record_memory(self):
        native = sum(t.numel() * t.element_size() for t in (self.cache.k, self.cache.v))
        compact = 0 if self.state is None else self.state.storage_bytes()
        support, versions = self.support.flags.nbytes, self.code_versions.nbytes
        for name, value in (("native_kv_peak_bytes", native), ("compact_peak_bytes", compact),
                            ("support_peak_bytes", support), ("code_version_peak_bytes", versions),
                            ("extra_peak_bytes", compact + support + versions)):
            self.memory[name] = max(self.memory[name], int(value))

    def refresh_plan(self, rho=None):
        return self._refresh_plan(rho).as_dict()

    def _refresh_plan(self, rho=None):
        return refresh_plan(self.code_versions, self.trajectory.version,
                            self.rho if rho is None else rho)

    def _initial_encoder_flops(self, plan):
        return self.adapter.estimate_flops(tokens=plan.missing_rows)["token_transform"]

    def candidate_state(self, rho, full_current_state):
        """Validation candidate without modifying live codes, versions or costs."""
        plan = self._refresh_plan(rho)
        codes = (full_current_state.codes.clone() if self.state is None
                 else self.state.codes.clone())
        codes[:, :, plan.indices] = full_current_state.codes[:, :, plan.indices]
        return LightweightValueState(codes, full_current_state.seq_len)

    def snapshot_codes(self):
        """Call after advance(first_query_time), before prepare_read, for validation."""
        if len(self.code_versions) != self.cache.seq_len:
            if not self.native_mode:
                raise RuntimeError("code versions and native rows differ")
            versions = np.full(self.cache.seq_len, MISSING_VERSION, dtype=np.int32)
        else:
            versions = self.code_versions.copy()
        return CodeSnapshot(None if self.state is None else self.state.codes.clone(),
                            versions, int(self.trajectory.version))

    @torch.no_grad()
    def _encode_cache(self, cache, items, flags, cost_key):
        ids = torch.as_tensor(items, device=cache.k.device, dtype=torch.long)[None]
        features = self.trajectory.model.lookup_item_embeddings(ids)
        state = self.adapter.encode_cache(cache, features)
        if self.storage_mode == "masked":
            mask = torch.as_tensor(flags.copy(), device=state.codes.device)[:, None, :, None, None]
            state.codes.mul_(mask)
        if cost_key is not None:
            self.costs[cost_key] += self.adapter.estimate_flops(tokens=cache.seq_len)["token_transform"]
            self.costs["item_lookup_bytes"] += features.numel() * features.element_size()
            if self.storage_mode == "masked":
                self.costs["support_mask_flops"] += state.codes.numel()
                self.costs["support_mask_transfer_bytes"] += flags.nbytes
                self.costs["masked_clear_layer_rows"] += int((~flags).sum())
        return state

    @torch.no_grad()
    def encode_reference(self, *, cost_key=None):
        """Full current codes in the selected storage mode, without replacement.

        A named cost_key optionally records this offline validation encoding in
        the session ledger; None leaves the serving ledger unchanged.
        """
        return self._encode_cache(self.cache, self.trajectory.retained_events[1],
                                  self.support.flags, cost_key)

    def _retire(self, reason, at):
        active, pending = self.state is not None or self.pending_encode, self.pending_encode
        self.state = None
        self.code_versions = np.empty(0, dtype=np.int32)
        self.pending_encode = self.pending_rebuild = False
        if active:
            self.costs["retirement_events"] += 1
            self.costs["unmaterialized_retirement_events"] += int(pending)
            self.events.append(dict(action="retire", reason=reason, at=int(at),
                                    version=self.trajectory.version, cursor=self.trajectory.cursor))

    @torch.no_grad()
    def _appended(self, old, new, events, version):
        width = len(events[0])
        self.support.append(width)
        self.costs["support_token_layers"] += width * len(self.trajectory.model.blocks)
        if (self.pending_encode or self.state is not None) and self.exit_when_clear and self.support.clear:
            self._retire("inherited_support_cleared", int(events[0][-1]))
        elif self.pending_encode:
            self.code_versions = np.concatenate((self.code_versions,
                np.full(width, MISSING_VERSION, dtype=np.int32)))[-new.seq_len:]
            if self.state is not None:
                shape = list(self.state.codes.shape)
                shape[2] = width
                codes = torch.cat((self.state.codes, self.state.codes.new_zeros(shape)), dim=2)
                self.state = LightweightValueState(codes[:, :, -new.seq_len:].contiguous(), new.seq_len)
        elif self.state is not None:
            entering = HSTUKVCache(new.k[:, :, -width:], new.v[:, :, -width:], width)
            encoded = self._encode_cache(entering, events[1], self.support.flags[:, -width:],
                                         "append_encode_flops")
            codes = torch.cat((self.state.codes, encoded.codes), dim=2)[:, :, -new.seq_len:].contiguous()
            self.state = LightweightValueState(codes, new.seq_len)
            self.code_versions = np.concatenate((self.code_versions,
                np.full(width, version, dtype=np.int32)))[-new.seq_len:]
            self.costs["append_encode_rows"] += width
        self._record_memory()

    @torch.no_grad()
    def publish(self, model, version, at, adapter, *, rho=1.0):
        if int(version) <= int(self.trajectory.version):
            raise ValueError("publish requires a later numeric model version")
        refresh_plan([], version, rho)
        self.advance(at)
        if self.pending_encode:
            self.costs["superseded_pending_releases"] += 1
        self.trajectory.switch(model, int(version), at)
        self.adapter, self.coordinator, self.policy = adapter, None, "correct"
        self.rho = float(rho)
        if self.state is None:
            self.code_versions = np.full(self.cache.seq_len, MISSING_VERSION, dtype=np.int32)
        self.support.reset(self.cache.seq_len, inherited=True)
        self.pending_encode = not self.exit_when_clear or not self.support.clear
        self.pending_rebuild = False
        self.costs["publication_events"] += 1
        self.events.append(dict(action="publish", at=int(at), version=int(version),
                                cursor=self.trajectory.cursor, rho=self.rho,
                                storage_mode=self.storage_mode,
                                policy="partial_encode_on_first_read"))
        self._record_memory()

    @torch.no_grad()
    def rebuild(self, at, *, reason="explicit"):
        super().rebuild(at, reason=reason)
        self.pending_encode = False
        self.code_versions = np.empty(0, dtype=np.int32)

    @torch.no_grad()
    def prepare_read(self, at):
        self.advance(at)
        if not self.pending_encode:
            return
        if self.exit_when_clear and self.support.clear:
            self._retire("inherited_support_cleared_before_first_read", at)
            return
        plan = self._refresh_plan()
        n = self.cache.seq_len
        if self.state is None:
            shape = (self.adapter.num_layers, self.cache.k.shape[1], n,
                     self.adapter.heads, self.adapter.rank)
            self.state = LightweightValueState(self.cache.k.new_zeros(shape), n)
        if plan.refreshed_rows:
            indices = torch.as_tensor(plan.indices, device=self.cache.k.device)
            cache = HSTUKVCache(self.cache.k.index_select(2, indices),
                               self.cache.v.index_select(2, indices), plan.refreshed_rows)
            encoded = self._encode_cache(cache, self.trajectory.retained_events[1][plan.indices],
                                         self.support.flags[:, plan.indices], "refresh_encode_flops")
            self.state.codes[:, :, indices] = encoded.codes
            self.code_versions[plan.indices] = self.trajectory.version
            initial = self._initial_encoder_flops(plan)
            self.costs["initial_encode_flops"] += initial
            self.costs["refresh_encode_flops"] -= initial
        self.pending_encode = False
        self.last_refresh = plan
        self.costs["initial_encode_rows"] += plan.missing_rows
        self.costs["refresh_encode_rows"] += plan.refreshed_rows - plan.missing_rows
        self.costs["forced_refresh_rows"] += plan.forced_rows
        self.costs["forced_missing_rows"] += plan.missing_rows
        self.costs["forced_older_rows"] += plan.older_rows
        self.costs["retained_previous_rows"] += plan.retained_previous_rows
        self.costs["first_read_refresh_events"] += 1
        self.events.append(dict(action="refresh_codes", at=int(at), version=self.trajectory.version,
                                cursor=self.trajectory.cursor, rho=self.rho, retained_rows=n,
                                storage_mode=self.storage_mode,
                                indices=plan.indices.tolist(), **plan.counts()))
        self._record_memory()

    def read_callback(self, candidates, *, force_torch=False):
        if self.state is None:
            return None
        fused = (not force_torch and self.kernel == "vector" and self.cache.k.is_cuda
                and self.cache.k.dtype == torch.float32 and self.adapter.heads == 6
                and self.adapter.head_dim == 32 and self.adapter.rank == 8
                and self.adapter.response_hidden_width == 0
                and 0 < self.cache.seq_len <= 1024 and 0 < candidates.shape[1] <= 16)
        if self.storage_mode == "raw":
            state = self.state

            def read(index, attention, q, k, v):
                if state.seq_len != k.shape[1]:
                    raise ValueError("raw compact codes and history rows differ")
                mask = torch.as_tensor(self.support.flags[index].copy(),
                    device=state.codes.device)[None, :, None, None]
                codes = state.codes[index] * mask
                layer = self.adapter.layers[index]
                if fused:
                    if (attention.training or attention.position_bias is not None
                            or attention.block_variant != "legacy"
                            or attention.cfg.activation != "elu_plus1"):
                        raise ValueError("KV dispatch implements frozen legacy ELU+1 history reads")
                    from hstu_kvcache.methods.cachetide.compact.lightweight_kv_dispatch import fused_kv_read
                    return fused_kv_read(q, k, v, codes, layer.key_decoder,
                                         layer.decoder, scale=attention.scale)
                return layer.read(attention, q, k, v, codes)

            return read
        if fused:
            from hstu_kvcache.methods.cachetide.compact.lightweight_kv_dispatch import make_history_read
            return make_history_read(self.adapter, self.state)
        return self.adapter.make_history_read(self.state)

    @torch.no_grad()
    def score(self, at, candidates):
        self.prepare_read(at)
        delta = 0 if self.trajectory.last_timestamp is None else int(at) - self.trajectory.last_timestamp
        query_delta = torch.tensor([delta], dtype=torch.float32, device=self.cache.k.device)
        logits = native_score(self.trajectory.model, self.cache, candidates, query_delta,
                              history_read_fn=self.read_callback(candidates))[0]
        if self.state is not None:
            work = self.adapter.estimate_flops(tokens=self.cache.seq_len, candidates=candidates.shape[1])
            self.costs["extra_history_read_flops"] += work["extra_history_read"]
            self.costs["response_flops"] += work["candidate_reads"]
            if self.storage_mode == "raw":
                # One mask per complete batched read, shared by its candidates.
                self.costs["read_support_mask_flops"] += self.state.codes.numel()
                self.costs["read_support_mask_transfer_bytes"] += self.support.flags.nbytes
                self.costs["raw_masked_read_batches"] += 1
            self.costs["corrected_requests"] += candidates.numel()
        else:
            self.costs["native_requests"] += candidates.numel()
        return logits
