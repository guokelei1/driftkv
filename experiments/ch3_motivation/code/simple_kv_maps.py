"""Cheap native-KV controls: one constant shift and one head-local affine map.

These maps are fitted to rebuilt K/V directly, unlike the frozen F whose
objective aligns reads and logits.  They never use candidate IDs or labels.
"""
from dataclasses import dataclass
from pathlib import Path

import torch

from hstu_kvcache.models import HSTUKVCache


def head_rows(cache, num_heads):
    """Return [layer*head, batch*row, concat(Khead,Vhead)]."""
    layers, batch, rows, width = cache.k.shape
    if cache.v.shape != cache.k.shape or width % num_heads:
        raise ValueError("unaligned native K/V head geometry")
    dim = width // num_heads
    def split(value):
        return value.reshape(layers, batch, rows, num_heads, dim).permute(0, 3, 1, 2, 4).reshape(
            layers * num_heads, batch * rows, dim)
    return torch.cat((split(cache.k), split(cache.v)), dim=-1)


def rows_to_cache(value, template, num_heads):
    layers, batch, rows, width = template.k.shape
    dim = width // num_heads
    value = value.reshape(layers, num_heads, batch, rows, 2*dim)
    def join(part):
        return part.permute(0, 2, 3, 1, 4).reshape(layers, batch, rows, width).contiguous()
    return HSTUKVCache(join(value[..., :dim]), join(value[..., dim:]), template.seq_len)


@dataclass
class SimpleKVMap:
    kind: str
    num_heads: int
    shift: torch.Tensor | None = None
    xmean: torch.Tensor | None = None
    xscale: torch.Tensor | None = None
    weight: torch.Tensor | None = None
    ymean: torch.Tensor | None = None
    metadata: dict | None = None

    @torch.no_grad()
    def __call__(self, cache):
        x = head_rows(cache, self.num_heads)
        if self.kind == "kv_shift":
            value = x + self.shift[:, None]
        elif self.kind == "kv_linear":
            value = torch.bmm((x-self.xmean[:, None])/self.xscale[:, None], self.weight) + self.ymean[:, None]
        else:
            raise ValueError("unknown simple KV map")
        return rows_to_cache(value, cache, self.num_heads)

    def materialization_flops(self, rows, batch=1):
        reference = self.shift if self.kind == "kv_shift" else self.xmean
        pairs, width = reference.shape
        per_row = width if self.kind == "kv_shift" else 2*width*width + 3*width
        return int(batch * rows * pairs * per_row)


def maps_from_tensors(tensors, num_heads, metadata=None):
    return dict(kv_shift=SimpleKVMap("kv_shift", num_heads, shift=tensors["shift"], metadata=metadata),
        kv_linear=SimpleKVMap("kv_linear", num_heads, xmean=tensors["xmean"],
            xscale=tensors["xscale"], weight=tensors["linear_weight"],
            ymean=tensors["ymean"], metadata=metadata))


def load_maps(path, device="cpu"):
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if payload["format"] != "simple_kv_maps_v1":
        raise ValueError("not a simple-KV-map checkpoint")
    return maps_from_tensors(payload["tensors"], payload["num_heads"], payload["metadata"])
