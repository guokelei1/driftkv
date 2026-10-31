"""Same code function with a release-shared Current-item affine cache.

The user stores the same code tensor and versions. Shared projection cache,
parameter folding, all cache misses and lookups are charged separately from
per-row KV arithmetic; a hit never makes shared preparation free.
"""
from __future__ import annotations

import numpy as np
import torch

from experiments.archive.partial_refresh.session import PartialRefreshSession
from experiments.archive.partial_refresh.round2.factored_encoder import FactoredEncoder


class FactoredRefreshSession(PartialRefreshSession):
    shared_encoders = {}

    def __init__(self, *args, cache_capacity=None, static_item_file=None, **kwargs):
        self.cache_capacity = cache_capacity
        self.static_item_file = static_item_file
        super().__init__(*args, **kwargs)

    def _encoder(self):
        key = (id(self.trajectory.model), id(self.adapter), self.cache_capacity, self.static_item_file)
        if key not in self.shared_encoders:
            static_ids = None
            if self.static_item_file:
                with np.load(self.static_item_file) as tables:
                    static_ids = tables[f"v{self.trajectory.version}_{self.cache_capacity}"].copy()
            encoder = FactoredEncoder(self.adapter, self.trajectory.model,
                                      capacity=self.cache_capacity,static_ids=static_ids)
            self.shared_encoders[key] = encoder
            self.costs["encoder_fold_preparation_flops"] += encoder.counters["parameter_preparation_flops"]
            for name in ("item_projection_flops","item_lookup_bytes","static_precomputed_items"):
                self.costs[name] += encoder.counters.get(name,0)
        return self.shared_encoders[key]

    @property
    def row_encoder_flops(self):
        l,h,d,r = self.adapter.num_layers,self.adapter.heads,self.adapter.head_dim,self.adapter.rank
        return l*h*(4*d*r+2*r)

    def _initial_encoder_flops(self, plan):
        return plan.missing_rows * self.row_encoder_flops

    @torch.no_grad()
    def _encode_cache(self, cache, items, flags, cost_key):
        encoder = self._encoder()
        before = dict(encoder.counters)
        state = encoder.encode(cache, items)
        if self.storage_mode == "masked":
            mask = torch.as_tensor(flags.copy(), device=state.codes.device)[:,None,:,None,None]
            state.codes.mul_(mask)
        if cost_key is not None:
            self.costs[cost_key] += encoder.counters["row_kv_flops"] - before.get("row_kv_flops",0)
            for name in ("item_projection_flops","item_lookup_bytes","hits","misses"):
                self.costs[name] += encoder.counters.get(name,0) - before.get(name,0)
            self.costs["encoder_fold_preparation_flops"] += encoder.counters["parameter_preparation_flops"] - before.get("parameter_preparation_flops",0)
            if self.storage_mode == "masked":
                self.costs["support_mask_flops"] += state.codes.numel()
                self.costs["support_mask_transfer_bytes"] += flags.nbytes
                self.costs["masked_clear_layer_rows"] += int((~flags).sum())
        return state

    @classmethod
    def encoder_inventory(cls):
        return [dict(capacity=encoder.capacity,counters=dict(encoder.counters),memory=dict(encoder.memory))
                for encoder in cls.shared_encoders.values()]
