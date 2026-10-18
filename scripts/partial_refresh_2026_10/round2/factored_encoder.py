"""Exact fixed-normalization folding with shared Current-item affine terms.

This synchronous prototype binds one frozen model/adapter release. It folds
the per-coordinate centers/scales into the encoder weights, memoizes the item
affine term across users, and still evaluates the native K/V term for every
encoded row. Saved per-user codes retain their original shape and meaning.

The cache uses CPU IDs/slot metadata and one GPU/CPU tensor table. Each encode
call looks up a batch of unique misses once; no per-entry scalar device reads
occur. Capacity bounds the persistent tensor, including when one call contains
more unique items than capacity. Batch-local terms remain valid after eviction.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
import sys

import numpy as np
import torch
from torch.nn import functional as F

from hstu_kvcache.design_one.lightweight import LightweightValueState


class FactoredEncoder:
    """`encode(cache, item_ids)` returns a LightweightValueState.

    Constructor counters include actual folding and optional static-table
    preparation. `last_delta` includes them on the first call; a caller charging
    construction separately should use before/after counters for serving.
    `counters` are cumulative, `memory` reports absolute
    tensor peaks and an approximate CPU metadata footprint. These are logical
    arithmetic/payload accounts, not loaded-serving performance measurements.
    """

    @torch.no_grad()
    def __init__(self, adapter, model, capacity=None, static_ids=None):
        if capacity is not None and capacity < 0:
            raise ValueError("item projection capacity must be nonnegative")
        self.static_ids = None if static_ids is None else np.unique(np.asarray(static_ids,dtype=np.int64))
        self.static_mode = static_ids is not None
        self.adapter, self.model = adapter, model
        self.capacity = len(self.static_ids) if self.static_mode else capacity
        self.layers, self.heads = adapter.num_layers, adapter.heads
        self.dim, self.rank = adapter.head_dim, adapter.rank
        self.item_width = self.heads * self.dim
        self.latent_width = self.layers * self.heads * self.rank
        weights = torch.stack([layer.input_weight.detach() for layer in adapter.layers])
        scales = torch.stack([layer.input_scale.detach() for layer in adapter.layers])
        centers = torch.stack([layer.input_center.detach() for layer in adapter.layers])
        biases = torch.stack([layer.input_bias.detach() for layer in adapter.layers])
        if bool((scales <= 0).any()) or not bool(torch.isfinite(scales).all()):
            raise ValueError("fixed encoder scales must be positive and finite")
        effective = weights / scales[..., None]
        self.kv_weight = effective[:, :, :2*self.dim].contiguous()
        self.item_weight = effective[:, :, 2*self.dim:].permute(2,0,1,3).reshape(
            self.item_width,self.latent_width).contiguous()
        self.bias = (biases - (centers[..., None] * effective).sum(-2)).reshape(-1)
        self.device, self.dtype = weights.device, weights.dtype
        self.slots = OrderedDict()  # item ID -> stable tensor-row index
        self._next_slot = 0
        self.table = torch.empty((0,self.latent_width),device=self.device,dtype=self.dtype)
        folded_bytes = sum(t.numel()*t.element_size() for t in (
            self.kv_weight,self.item_weight,self.bias))
        self.parameter_preparation_flops = 3*weights.numel()
        self.counters = Counter(parameter_preparation_flops=self.parameter_preparation_flops)
        self._reported = Counter()
        self.last_delta = {}
        self.memory = dict(folded_parameter_bytes=folded_bytes,
            shared_item_table_current_bytes=0,shared_item_table_peak_bytes=0,
            shared_item_table_resize_peak_bytes=0,item_projection_batch_peak_bytes=0,
            item_projection_gather_peak_bytes=0,item_cache_entries=0,
            cpu_metadata_bytes_estimate=0)
        if self.capacity:
            self._ensure_table(self.capacity)
        if self.static_mode and len(self.static_ids):
            wanted = torch.as_tensor(self.static_ids,device=self.device,dtype=torch.long)[None]
            features = self.model.lookup_item_embeddings(wanted)
            terms = features[0].to(self.dtype) @ self.item_weight + self.bias
            self.table.copy_(terms)
            self.slots.update((int(item),slot) for slot,item in enumerate(self.static_ids))
            self._next_slot = len(self.static_ids)
            self.counters["item_projection_flops"] += len(self.static_ids)*(2*self.item_width+1)*self.latent_width
            self.counters["item_lookup_bytes"] += features.numel()*features.element_size()
            self.counters["static_precomputed_items"] += len(self.static_ids)
            self.counters["item_table_write_bytes"] += terms.numel()*terms.element_size()
            self.memory["item_projection_batch_peak_bytes"] = terms.numel()*terms.element_size()
            self._record_metadata()

    def _ensure_table(self, required):
        if required <= self.table.shape[0]:
            return
        rows = self.capacity if self.capacity is not None else max(64,2*self.table.shape[0],required)
        new_table = torch.empty((rows,self.latent_width),device=self.device,dtype=self.dtype)
        old_bytes = self.table.numel()*self.table.element_size()
        new_bytes = new_table.numel()*new_table.element_size()
        self.memory["shared_item_table_resize_peak_bytes"] = max(
            self.memory["shared_item_table_resize_peak_bytes"],old_bytes+new_bytes)
        if self._next_slot:
            new_table[:self._next_slot].copy_(self.table[:self._next_slot])
            self.counters["item_table_growth_copy_bytes"] += self._next_slot*self.latent_width*self.table.element_size()
        self.table = new_table
        self.memory["shared_item_table_current_bytes"] = new_bytes
        self.memory["shared_item_table_peak_bytes"] = max(self.memory["shared_item_table_peak_bytes"],new_bytes)
        self.counters["cache_peak_rows"] = max(self.counters["cache_peak_rows"],rows)
        self.counters["cache_peak_bytes"] = self.memory["shared_item_table_peak_bytes"]

    def _update_cache(self, unique_ids, unique_terms, inverse):
        if self.static_mode or self.capacity == 0:
            return
        if self.capacity is None:
            new = sum(int(item) not in self.slots for item in unique_ids)
            self._ensure_table(len(self.slots)+new)
        # Processing last occurrences gives the final batch access order. All
        # call-start hits were gathered before eviction, so their batch-local
        # terms remain usable even if this metadata update evicts their slot.
        last = np.full(len(unique_ids),-1,dtype=np.int64)
        np.maximum.at(last,inverse,np.arange(len(inverse)))
        writes = {}
        for index in np.argsort(last):
            item = int(unique_ids[index])
            if item in self.slots:
                self.slots.move_to_end(item)
                continue
            if self.capacity is not None and len(self.slots) >= self.capacity:
                _,slot = self.slots.popitem(last=False)
                self.counters["item_cache_evictions"] += 1
            else:
                slot = self._next_slot
                self._next_slot += 1
            self.slots[item] = slot
            writes[slot] = int(index)
        if writes:
            slots = torch.tensor(list(writes),device=self.device,dtype=torch.long)
            indices = torch.tensor(list(writes.values()),device=self.device,dtype=torch.long)
            self.table.index_copy_(0,slots,unique_terms.index_select(0,indices))
            self.counters["item_table_write_bytes"] += len(writes)*self.latent_width*self.table.element_size()
        self._record_metadata()

    def _record_metadata(self):
        self.memory["item_cache_entries"] = len(self.slots)
        # OrderedDict nodes/arrays are included by getsizeof; key/value Python
        # integers are added separately. This excludes allocator fragmentation.
        self.memory["cpu_metadata_bytes_estimate"] = sys.getsizeof(self.slots) + 2*len(self.slots)*sys.getsizeof(0)

    @torch.no_grad()
    def encode(self, cache, item_ids):
        self.counters["parameter_preparation_flops"] = self.parameter_preparation_flops
        if cache.k.shape[1] != 1:
            raise ValueError("the experiment's factored encoder serves one owner per call")
        ids = (item_ids.detach().cpu().numpy() if isinstance(item_ids,torch.Tensor)
               else np.asarray(item_ids))
        ids = np.asarray(ids,dtype=np.int64).reshape(-1)
        if len(ids) != cache.seq_len:
            raise ValueError("item IDs and native K/V rows must align")
        unique,inverse = np.unique(ids,return_inverse=True)
        terms = torch.empty((len(unique),self.latent_width),device=self.device,dtype=self.dtype)
        hits = [i for i,item in enumerate(unique) if int(item) in self.slots]
        misses = [i for i,item in enumerate(unique) if int(item) not in self.slots]
        if hits:
            slots = torch.tensor([self.slots[int(unique[i])] for i in hits],device=self.device,dtype=torch.long)
            targets = torch.tensor(hits,device=self.device,dtype=torch.long)
            terms.index_copy_(0,targets,self.table.index_select(0,slots))
            self.counters["item_table_hit_read_bytes"] += len(hits)*self.latent_width*terms.element_size()
        if misses:
            targets = torch.tensor(misses,device=self.device,dtype=torch.long)
            wanted = torch.as_tensor(unique[misses],device=self.device,dtype=torch.long)[None]
            item_features = self.model.lookup_item_embeddings(wanted)
            projected = item_features[0].to(self.dtype) @ self.item_weight + self.bias
            terms.index_copy_(0,targets,projected)
            self.counters["item_projection_flops"] += len(misses)*(2*self.item_width+1)*self.latent_width
            self.counters["item_lookup_bytes"] += item_features.numel()*item_features.element_size()
        self.counters["unique_item_misses"] += len(misses)
        self.counters["unique_item_hits"] += len(hits)
        self.counters["misses"] += len(misses)
        self.counters["hits"] += len(hits)
        self.counters["unique_items_in_calls"] += len(unique)
        self._update_cache(unique,terms,inverse)
        self.memory["item_projection_batch_peak_bytes"] = max(
            self.memory["item_projection_batch_peak_bytes"],terms.numel()*terms.element_size())
        gather = terms.index_select(0,torch.as_tensor(inverse,device=self.device,dtype=torch.long))
        gather = gather.reshape(cache.seq_len,self.layers,self.heads,self.rank).permute(1,0,2,3)
        self.memory["item_projection_gather_peak_bytes"] = max(
            self.memory["item_projection_gather_peak_bytes"],gather.numel()*gather.element_size())
        codes = []
        for layer in range(self.layers):
            shape = (1,cache.seq_len,self.heads,self.dim)
            key,value = cache.k[layer].reshape(shape),cache.v[layer].reshape(shape)
            kv = torch.cat((key,value),dim=-1).to(self.dtype).transpose(1,2)
            kv_term = (kv @ self.kv_weight[layer][None]).transpose(1,2)
            codes.append(F.silu(kv_term + gather[layer][None]).to(cache.k.dtype))
        self.counters["encoded_rows"] += cache.seq_len
        self.counters["row_kv_flops"] += cache.seq_len*self.layers*self.heads*(
            2*(2*self.dim)*self.rank+2*self.rank)
        # Peak fields are monotone counters: deltas telescope to each release's
        # maximum. Absolute and current footprints are also in `memory`.
        self.counters["folded_parameter_bytes"] = self.memory["folded_parameter_bytes"]
        changes = {key:int(value-self._reported.get(key,0)) for key,value in self.counters.items()
                   if value != self._reported.get(key,0)}
        self._reported = Counter(self.counters)
        self.last_delta = changes
        return LightweightValueState(torch.stack(codes),cache.seq_len)
