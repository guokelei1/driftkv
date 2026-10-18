"""Exact masked-code prototype that avoids encoding cleared layer/row pairs.

This is equivalent to the original *masked storage* policy, including across
releases: a false support flag makes the stored masked code exactly zero. It
does not preserve the dormant raw codes used by the separate raw-storage branch.
One row version still commits after all layers have their valid masked values.
"""
from __future__ import annotations

import numpy as np
import torch

from hstu_kvcache.design_one.lightweight import LightweightValueState
from hstu_kvcache.models import HSTUKVCache
from partial_refresh_2026_10.session import MISSING_VERSION, PartialRefreshSession


class SparseMaskedSession(PartialRefreshSession):
    """Only true support layer/rows execute the unchanged full-item encoder."""

    def __init__(self, trajectory, *, kernel="torch", exit_when_clear=True,
                 storage_mode="masked"):
        if storage_mode != "masked":
            raise ValueError("sparse encoding requires masked storage")
        super().__init__(trajectory, kernel=kernel, exit_when_clear=exit_when_clear,
                         storage_mode="masked")

    @property
    def layer_row_encoder_flops(self):
        h, d, r = self.adapter.heads, self.adapter.head_dim, self.adapter.rank
        width = (h + 2) * d
        return h * (2 * width + 2 * width * r + 2 * r)

    @torch.no_grad()
    def _encode_cache(self, cache, items, flags, cost_key):
        if flags.shape != (self.adapter.num_layers, cache.seq_len):
            raise ValueError("support flags must align with encoded layer/rows")
        codes = cache.k.new_zeros((self.adapter.num_layers, cache.k.shape[1],
                                  cache.seq_len, self.adapter.heads, self.adapter.rank))
        union = np.flatnonzero(flags.any(axis=0))
        features = None
        if len(union):
            ids = torch.as_tensor(np.asarray(items)[union], device=cache.k.device,
                                  dtype=torch.long)[None]
            features = self.trajectory.model.lookup_item_embeddings(ids)
            for layer_index, layer in enumerate(self.adapter.layers):
                selected = np.flatnonzero(flags[layer_index])
                if not len(selected):
                    continue
                row_indices = torch.as_tensor(selected, device=cache.k.device)
                feature_indices = torch.as_tensor(np.searchsorted(union, selected),
                                                  device=cache.k.device)
                encoded = layer.encode(cache.k[layer_index].index_select(1, row_indices),
                    cache.v[layer_index].index_select(1, row_indices),
                    features.index_select(1, feature_indices))
                codes[layer_index, :, row_indices] = encoded
        if cost_key is not None:
            actual, eligible = int(flags.sum()), flags.size
            self.costs[cost_key] += actual * self.layer_row_encoder_flops
            self.costs[cost_key.replace("_encode_flops", "_encoded_layer_rows")] += actual
            self.costs["actual_encoded_layer_rows"] += actual
            self.costs["eligible_layer_rows"] += eligible
            self.costs["skipped_encoder_layer_rows"] += eligible - actual
            self.costs["masked_clear_layer_rows"] += eligible - actual
            self.costs["item_lookup_bytes"] += (0 if features is None else
                features.numel() * features.element_size())
        return LightweightValueState(codes, cache.seq_len)

    @torch.no_grad()
    def prepare_read(self, at):
        """Parent selection/lifecycle, with missing/refresh FLOPs split by flags."""
        self.advance(at)
        if not self.pending_encode:
            return
        if self.exit_when_clear and self.support.clear:
            self._retire("inherited_support_cleared_before_first_read", at)
            return
        plan = self._refresh_plan()
        n = self.cache.seq_len
        if self.state is None:
            self.state = LightweightValueState(self.cache.k.new_zeros(
                (self.adapter.num_layers, self.cache.k.shape[1], n,
                 self.adapter.heads, self.adapter.rank)), n)
        if plan.refreshed_rows:
            missing = plan.indices[self.code_versions[plan.indices] == MISSING_VERSION]
            missing_layers = int(self.support.flags[:, missing].sum())
            indices = torch.as_tensor(plan.indices, device=self.cache.k.device)
            cache = HSTUKVCache(self.cache.k.index_select(2, indices),
                               self.cache.v.index_select(2, indices), plan.refreshed_rows)
            encoded = self._encode_cache(cache, self.trajectory.retained_events[1][plan.indices],
                                         self.support.flags[:, plan.indices], "refresh_encode_flops")
            self.state.codes[:, :, indices] = encoded.codes
            self.code_versions[plan.indices] = self.trajectory.version
            initial = missing_layers * self.layer_row_encoder_flops
            self.costs["initial_encode_flops"] += initial
            self.costs["refresh_encode_flops"] -= initial
            self.costs["initial_encoded_layer_rows"] += missing_layers
            self.costs["refresh_encoded_layer_rows"] -= missing_layers
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
                                storage_mode=self.storage_mode, encoder_policy="support_sparse",
                                indices=plan.indices.tolist(), **plan.counts()))
        self._record_memory()
