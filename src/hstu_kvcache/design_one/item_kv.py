"""K/V read view with the current item embedding as a local input.

The caller supplies frozen item vectors aligned with the native cache rows.
This module contains no foundation embedding table and retains no item
metadata after generating the separate mapped read view.
"""

import torch

from hstu_kvcache.design_one.nonlinear import (
    _NormalizedMLP, KVViewLayer, KVReadViewAdapter,
)
from hstu_kvcache.models import HSTUKVCache


class ItemKVViewLayer(KVViewLayer):
    """Cross-head [K,V,E_current(item)] -> [delta K,delta V] SiLU residual."""

    def __init__(self, heads, head_dim, hidden_width=64):
        width = heads * head_dim
        _NormalizedMLP.__init__(self, heads, head_dim, hidden_width,
                                2 * width, extra_inputs=width)

    def map_tokens(self, k, v, item_features):
        source = torch.cat((k, v), dim=-1)
        features = torch.cat((source, item_features.to(source)), dim=-1)
        mapped = source + self._delta(features).to(source.dtype)
        return mapped.split(self.width, dim=-1)


class ItemKVReadViewAdapter(KVReadViewAdapter):
    """Generate a reusable view from native K/V and aligned current item vectors.

    The normalization, zero-output identity initialization, serialization,
    actual-query override and FLOPs helpers are shared with the existing MLP.
    Its input width is 3W, output width 2W. Relative to the same hidden width
    without item vectors, forward costs an extra 2W*(hidden_width+1) per row
    per layer; backward adds 2W*hidden_width with frozen feature inputs.
    Embedding lookup bytes belong to the caller's lifecycle ledger.
    """

    kind = "design_one_item_kv_view_v1"
    layer_class = ItemKVViewLayer

    def __init__(self, num_layers, heads, head_dim, hidden_width=64, max_length=1024):
        super().__init__(num_layers, heads, head_dim, hidden_width, max_length)

    def map_cache(self, cache, item_features, fitted_layers=None):
        prefix = len(self.layers) if fitted_layers is None else fitted_layers
        if prefix == 0:
            return HSTUKVCache(cache.k[:0], cache.v[:0], cache.seq_len)
        if item_features.shape != (cache.k.shape[1], cache.seq_len, self.heads * self.head_dim):
            raise ValueError("current item features must align with the supplied native K/V rows")
        return HSTUKVCache.from_layer_list([
            self.layers[index].map_tokens(cache.k[index], cache.v[index], item_features)
            for index in range(prefix)
        ], cache.seq_len)
