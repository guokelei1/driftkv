"""Frozen Item-KV maps with one learned native-producer scale per layer.

Parent rows retain the original Item-KV correction. Current-produced rows
multiply that correction by a free scalar, initialized to one. The separate
mapped view never changes persistent native K/V or stores producer metadata.
"""

import torch
from torch import nn

from hstu_kvcache.design_one.item_kv import ItemKVReadViewAdapter, ItemKVViewLayer
from hstu_kvcache.models import HSTUKVCache


class ProducerScaledItemKVLayer(ItemKVViewLayer):
    def __init__(self, heads, head_dim, hidden_width=64):
        super().__init__(heads, head_dim, hidden_width)
        self.native_scale = nn.Parameter(torch.ones(()))
        self.freeze_base()

    def freeze_base(self):
        self.requires_grad_(False)
        self.native_scale.requires_grad_(True)
        return self

    def map_tokens(self, k, v, item_features, producer_mask=None):
        """Map rows; a boolean [B,N] mask marks Current-produced entries."""
        source = torch.cat((k, v), dim=-1)
        features = torch.cat((source, item_features.to(source)), dim=-1)
        delta = self._delta(features).to(source.dtype)
        if producer_mask is not None:
            if producer_mask.shape != source.shape[:-1] or producer_mask.dtype != torch.bool:
                raise ValueError("producer_mask must be boolean [B,N], True for Current rows")
            mask = producer_mask.to(device=source.device)
            # Indexing limits scale arithmetic to Current rows and keeps the
            # Parent expression bitwise identical to the original Item map.
            delta[mask] = delta[mask] * self.native_scale
        return (source + delta).split(self.width, dim=-1)

    def fit_flops(self, *, batch=1, length=1, current_rows=0):
        """Frozen inputs/base network; backward is only the scalar gradient.

        current_rows counts Current entries across the whole batch. Gradient
        multiplication and reduction each charge 2W operations per row; the
        same rows-vs-rows-minus-one convention is used by the shared ledger.
        """
        scale = int(current_rows) * 2 * self.width
        return dict(forward=super().fit_flops(batch=batch, length=length)["forward"] + scale,
                    backward=2 * scale)

    def token_forward_flops(self, *, batch=1, length=1, current_rows=0):
        return self.fit_flops(batch=batch, length=length, current_rows=current_rows)["forward"]


class ProducerScaledItemKVAdapter(ItemKVReadViewAdapter):
    kind = "design_one_producer_scaled_item_kv_view_v1"
    layer_class = ProducerScaledItemKVLayer

    @classmethod
    def from_item_adapter(cls, adapter):
        """Copy an existing Item-KV model exactly; initialize every scale to 1."""
        first = next(adapter.parameters())
        result = cls(**adapter.get_config()).to(device=first.device, dtype=first.dtype)
        for source, target in zip(adapter.layers, result.layers, strict=True):
            target.load_state_dict({**source.state_dict(),
                                    "native_scale": target.native_scale.detach()})
        return result.freeze_base()

    def freeze_base(self):
        for layer in self.layers:
            layer.freeze_base()
        return self

    def map_cache(self, cache, item_features, producer_mask=None, fitted_layers=None):
        prefix = len(self.layers) if fitted_layers is None else fitted_layers
        if prefix == 0:
            return HSTUKVCache(cache.k[:0], cache.v[:0], cache.seq_len)
        if item_features.shape != (cache.k.shape[1], cache.seq_len, self.heads * self.head_dim):
            raise ValueError("current item features must align with the supplied native K/V rows")
        return HSTUKVCache.from_layer_list([
            self.layers[index].map_tokens(cache.k[index], cache.v[index], item_features, producer_mask)
            for index in range(prefix)
        ], cache.seq_len)

    def estimate_flops(self, *, batch=1, tokens=1, candidates=1, current_rows=0):
        result = super().estimate_flops(batch=batch, tokens=tokens, candidates=candidates)
        scale = int(current_rows) * len(self.layers) * 2 * self.heads * self.head_dim
        result["token_transform"] += scale
        # Informational subitem of token_transform; callers must not add twice.
        result["native_scale_flops"] = scale
        return result
