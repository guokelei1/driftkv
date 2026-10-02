"""A frozen Item-KV read view followed by a query-dependent Response map.

The caller maintains the Item view through native appends and evictions.
Each callback reads that view once and corrects its response at the actual
query. The caller's preceding native history read remains part of the cost.
"""

import torch
from torch import nn

from hstu_kvcache.design_one.item_kv import ItemKVReadViewAdapter
from hstu_kvcache.design_one.nonlinear import NonlinearResponseAdapter


class FrozenItemResponseAdapter(nn.Module):
    kind = "design_one_item_response_v1"

    def __init__(self, item_adapter, response_adapter):
        super().__init__()
        for name in ("num_layers", "heads", "head_dim", "max_length"):
            if getattr(item_adapter, name) != getattr(response_adapter, name):
                raise ValueError(f"Item and Response must share {name}")
        self.item = item_adapter
        self.response = response_adapter
        self.num_layers = item_adapter.num_layers
        self.heads, self.head_dim = item_adapter.heads, item_adapter.head_dim
        self.max_length = item_adapter.max_length
        self.freeze_item()

    @property
    def layers(self):
        """Expose Item layers for the existing view lifecycle's layer count."""
        return self.item.layers

    def freeze_item(self):
        self.item.eval().requires_grad_(False)
        return self

    @classmethod
    def from_item_adapter(cls, item_adapter, response_hidden_width=256):
        first = next(item_adapter.parameters())
        response = NonlinearResponseAdapter(
            item_adapter.num_layers, item_adapter.heads, item_adapter.head_dim,
            hidden_width=response_hidden_width, max_length=item_adapter.max_length,
        ).to(device=first.device, dtype=first.dtype)
        return cls(item_adapter, response)

    def map_cache(self, cache, item_features, fitted_layers=None):
        return self.item.map_cache(cache, item_features, fitted_layers=fitted_layers)

    def make_history_override(self, model, mapped_cache, counts=None, fitted_layers=None):
        """Mapped read then Response; fitted_layers limits Response only.

        A zero Response prefix still reads the frozen Item view at every layer.
        Calibration may equivalently score the mapped cache directly with the
        ordinary Response callback. No callback starts a new score pass.
        """
        item_read = self.item.make_history_override(model, mapped_cache)
        if counts is None:
            counts = torch.full((mapped_cache.k.shape[1],), mapped_cache.seq_len,
                                device=mapped_cache.k.device)
        prefix = self.num_layers if fitted_layers is None else fitted_layers

        def override(layer, query, native):
            if layer >= mapped_cache.k.shape[0]:
                return native
            mapped = item_read(layer, query, native)
            return self.response.layers[layer](query, mapped, counts) if layer < prefix else mapped

        return override

    def estimate_flops(self, *, batch=1, tokens=1, candidates=1):
        # Three disjoint components: stored map, extra mapped read, Response.
        return {**self.item.estimate_flops(batch=batch, tokens=tokens, candidates=candidates),
                **self.response.estimate_flops(batch=batch, candidates=candidates)}

    def get_config(self):
        return dict(item=self.item.get_config(), response=self.response.get_config())

    def export_state(self):
        return dict(config=self.get_config(), item=self.item.export_state(),
                    response=self.response.export_state())

    @classmethod
    def from_state_dict(cls, state):
        result = cls(ItemKVReadViewAdapter.from_state_dict(state["item"]),
                     NonlinearResponseAdapter.from_state_dict(state["response"]))
        if result.get_config() != state["config"]:
            raise ValueError("composite configuration differs from its two saved adapters")
        return result
