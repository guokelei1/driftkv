"""Original lazy native lifecycle with structured codes and static item reuse."""
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from design_system.lightweight_lazy import LazyLightweightSession
from hstu_kvcache.design_one.lightweight import LightweightValueState

ROOT = Path(__file__).resolve().parents[2]
ITEM_TABLES = ROOT / 'results/partial_refresh_2026_10/round2/item_tables/item_ids.npz'


class StructuredEncoder:
    @torch.no_grad()
    def __init__(self, adapter, model, version):
        self.adapter, self.model = adapter, model
        weights = adapter.effective_weights()
        self.branches, self.kv_weights = weights['branches'], weights['kv_weights']
        self.item_weight = weights['item_weight'].permute(1, 0, 2).reshape(192, -1).contiguous()
        self.bias = weights['bias'].reshape(-1)
        with np.load(ITEM_TABLES) as tables:
            self.static_ids = np.unique(tables[f'v{version}_4096'])
        self.slots = {int(item): i for i, item in enumerate(self.static_ids)}
        device = self.item_weight.device
        item = model.lookup_item_embeddings(torch.as_tensor(self.static_ids, device=device)[None])[0]
        self.table = item @ self.item_weight + self.bias
        self.counters = Counter(static_table_preparation_flops=self.table.numel() * 385,
            static_table_bytes=self.table.numel() * self.table.element_size())

    @torch.no_grad()
    def encode(self, cache, items):
        unique, inverse = np.unique(np.asarray(items, dtype=np.int64), return_inverse=True)
        terms = torch.empty((len(unique), self.item_weight.shape[1]), device=cache.k.device)
        hits = [i for i, item in enumerate(unique) if int(item) in self.slots]
        misses = [i for i, item in enumerate(unique) if int(item) not in self.slots]
        if hits:
            slots = torch.tensor([self.slots[int(unique[i])] for i in hits], device=cache.k.device)
            terms[torch.tensor(hits, device=cache.k.device)] = self.table.index_select(0, slots)
        if misses:
            ids = torch.tensor(unique[misses], device=cache.k.device)[None]
            item = self.model.lookup_item_embeddings(ids)[0]
            terms[torch.tensor(misses, device=cache.k.device)] = item @ self.item_weight + self.bias
        gather = terms[torch.tensor(inverse, device=cache.k.device)].reshape(cache.seq_len,
            self.adapter.num_layers, 48).permute(1, 0, 2)
        encoded = []
        for layer in range(self.adapter.num_layers):
            k = cache.k[layer].reshape(1, cache.seq_len, 6, 32)
            v = cache.v[layer].reshape(1, cache.seq_len, 6, 32)
            branch_codes = []
            for branch, weight in zip(self.branches, self.kv_weights, strict=True):
                selected_k = k[:, :, branch['heads']].flatten(-2)
                selected_v = v[:, :, branch['heads']].flatten(-2)
                branch_codes.append(torch.cat((selected_k, selected_v), -1) @ weight[layer])
            code = F.silu(torch.cat(branch_codes, -1) + gather[layer][None])
            encoded.append(code.reshape(1, cache.seq_len, 6, 8))
        self.counters['row_kv_flops'] += cache.seq_len * self.adapter.num_layers * (
            sum(4 * len(b['heads']) * 32 * b['code_width'] for b in self.branches) + 96)
        self.counters['item_projection_flops'] += len(misses) * 385 * self.item_weight.shape[1]
        self.counters['hits'] += len(hits)
        self.counters['misses'] += len(misses)
        return LightweightValueState(torch.stack(encoded), cache.seq_len)


class StructuredSession(LazyLightweightSession):
    shared_encoders = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, kernel='torch', **kwargs)
        self.verified = None
        self.max_code_error = 0.

    @torch.no_grad()
    def _encode(self, cache, items, cost_key):
        key = id(self.adapter), id(self.trajectory.model)
        if key not in self.shared_encoders:
            encoder = StructuredEncoder(self.adapter, self.trajectory.model, self.trajectory.version)
            self.shared_encoders[key] = encoder
            self.costs['static_table_preparation_flops'] += encoder.counters['static_table_preparation_flops']
        encoder = self.shared_encoders[key]
        before = Counter(encoder.counters)
        state = encoder.encode(cache, items)
        self.costs[cost_key] += encoder.counters['row_kv_flops'] - before['row_kv_flops']
        self.costs['item_projection_flops'] += encoder.counters['item_projection_flops'] - before['item_projection_flops']
        self.costs['unique_item_hits'] += encoder.counters['hits'] - before['hits']
        self.costs['unique_item_misses'] += encoder.counters['misses'] - before['misses']
        self.costs['first_read_encode_rows' if cost_key == 'initial_encode_flops' else 'append_encode_rows'] += cache.seq_len
        self.costs['encode_calls'] += 1
        flags = self.support.flags[:, -cache.seq_len:]
        state.codes.mul_(torch.as_tensor(flags.copy(), device=cache.k.device)[:, None, :, None, None])
        self.costs['support_mask_flops'] += state.codes.numel()
        return state

    def read_callback(self, candidates, *, force_torch=False):
        return None if self.state is None else self.adapter.make_history_read(self.state)

    @torch.no_grad()
    def score(self, at, candidates):
        actual = super().score(at, candidates)
        if self.state is not None:
            self.costs['corrected_token_query_pairs'] += self.cache.seq_len * candidates.numel()
        return actual

    @torch.no_grad()
    def verify(self):
        stamp = self.trajectory.version, self.trajectory.cursor
        if self.state is None or stamp == self.verified:
            return
        ids = torch.as_tensor(self.trajectory.retained_events[1], device=self.cache.k.device)[None]
        wanted = self.adapter.encode_cache(self.cache, self.trajectory.model.lookup_item_embeddings(ids))
        wanted.codes.mul_(torch.as_tensor(self.support.flags.copy(), device=self.cache.k.device)[:, None, :, None, None])
        self.max_code_error = max(self.max_code_error, float((wanted.codes - self.state.codes).abs().max()))
        if not torch.allclose(wanted.codes, self.state.codes, atol=2e-5, rtol=2e-5):
            raise RuntimeError('static factorization or incremental maintenance changed codes')
        self.verified = stamp
