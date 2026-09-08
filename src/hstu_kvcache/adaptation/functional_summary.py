"""Additive observations of the actual ELU+1 read function at fixed probes.

Probes are shared and frozen when this writer is enabled, never reselected at
a release. Changing them requires charged backfill of the retained source KV.
"""

import torch
import torch.nn.functional as F


class FunctionalSummary:
    def __init__(self, probes, attention_scale, producers=6):
        # [layers, heads, probes, head_dim]; zero probe is a sum-V reference.
        self.probes = probes
        self.attention_scale = attention_scale
        self.responses = probes.new_zeros(producers, *probes.shape)
        self.counts = torch.zeros(producers, device=probes.device, dtype=torch.long)

    @torch.no_grad()
    def update(self, key, value, producer, sign=1):
        """Add/subtract already stored [layers, events, width] ordinary KV."""
        layers, events, _ = key.shape
        heads, _, dimension = self.probes.shape[1:]
        k = key.float().reshape(layers, events, heads, dimension)
        v = value.float().reshape(layers, events, heads, dimension)
        weights = F.elu(torch.einsum("lhmd,lnhd->lnhm", self.probes, k) * self.attention_scale) + 1
        # Reduce by producer without retaining an expanded per-event sketch.
        for p in producer.unique().tolist():
            mask = producer == p
            self.responses[p].add_(torch.einsum("lnhm,lnhd->lhmd", weights[:, mask], v[:, mask]), alpha=sign)
            self.counts[p] += sign * int(mask.sum())
            if self.counts[p] == 0:
                self.responses[p].zero_()

    @property
    def state_bytes(self):
        # Shared probe bytes are charged separately, once per writer schema.
        return self.responses.numel()*self.responses.element_size() + self.counts.numel()*self.counts.element_size()


def source_producers(state):
    return torch.tensor([state.writer.segments[event[0]]["producer"] for event in state.writer.events],
                        device=state.cache.k.device, dtype=torch.long)
