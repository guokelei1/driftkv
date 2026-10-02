"""Producer aggregates of the actual retained cache, without learned sketches.

An append/eviction touches only entering/departing K/V. Feature preparation
reads the maintained aggregates, never the complete history. Batch members
share retained lengths, while their producer lineages may differ.
"""

import torch

from hstu_kvcache.models import HSTUKVCache


class ProducerSummary:
    def __init__(self, sums, counts, producers, *, producer_ids, max_length):
        # sums: [batch, producers, layers, K/V, width]; counts: [batch, producers].
        self.sums = sums
        self.counts = counts
        self.producers = producers  # [batch, retained events]
        self.producer_ids = tuple(producer_ids)
        self.max_length = max_length
        self.native_writes = sums.new_zeros(sums.shape[0])
        self.revision = 0

    @classmethod
    @torch.no_grad()
    def from_cache(cls, cache: HSTUKVCache, producer, *, producer_ids=(4, 5),
                   max_length=1024):
        """Scan once; ``producer`` is an id or chronological producer-id list."""
        layers, batch, _, width = cache.k.shape
        if isinstance(producer, int):
            producers = torch.full((batch, cache.seq_len), producer,
                                   device=cache.k.device, dtype=torch.long)
        else:
            producers = torch.as_tensor(producer, device=cache.k.device, dtype=torch.long).clone()
            if producers.ndim == 1:
                producers = producers[None].expand(batch, -1).clone()
        if (producers.shape != (batch, cache.seq_len)
                or not torch.isin(producers, producers.new_tensor(producer_ids)).all()):
            raise ValueError("summary producer membership must match the retained cache")
        sums = torch.zeros(batch, len(producer_ids), layers, 2, width,
                           dtype=torch.float32, device=cache.k.device)
        counts = sums.new_zeros((batch, len(producer_ids)))
        for group, identity in enumerate(producer_ids):
            mask = (producers == identity).float()
            counts[:, group] = mask.sum(1)
            sums[:, group, :, 0] = torch.einsum("lbnw,bn->blw", cache.k.float(), mask)
            sums[:, group, :, 1] = torch.einsum("lbnw,bn->blw", cache.v.float(), mask)
        return cls(sums, counts, producers, producer_ids=producer_ids, max_length=max_length)

    @property
    def total_count(self):
        return self.producers.shape[1]

    def batch_counts(self):
        return self.sums.new_full((self.sums.shape[0],), self.total_count)

    @torch.no_grad()
    def features(self, mode="producer_mean"):
        """[producer K/V features(L,2,W), count/context, fraction], writes/context.

        Counts are structural metadata; absent producers contribute exact zero
        features. ``producer_mass`` divides each producer's sums by the total
        retained count, so its contribution scales with its actual fraction.
        ``producer_mean`` divides by that producer's count and preserves the
        initial development implementation. Writes count all real appends since
        initialization, including events that have left the retained state.
        """
        count = self.counts
        if mode == "producer_mean":
            values = self.sums / count.clamp_min(1)[:, :, None, None, None]
        elif mode == "producer_mass":
            values = self.sums / max(1, self.total_count)
        else:
            raise ValueError(f"unknown producer summary mode: {mode}")
        metadata = torch.stack((count / self.max_length,
                                count / max(1, self.total_count)), dim=-1)
        grouped = torch.cat((values.flatten(2), metadata), dim=-1)
        writes = self.native_writes[:, None] / self.max_length
        return torch.cat((grouped.flatten(1), writes), dim=-1)

    @torch.no_grad()
    def add(self, new: HSTUKVCache, producer: int):
        """Add already-computed native writes, retaining no duplicate K/V."""
        group = self.producer_ids.index(producer)
        if not new.seq_len:
            return
        self.sums[:, group, :, 0] += new.k.float().sum(2).transpose(0, 1)
        self.sums[:, group, :, 1] += new.v.float().sum(2).transpose(0, 1)
        self.counts[:, group] += new.seq_len
        self.producers = torch.cat((self.producers, self.producers.new_full(
            (self.sums.shape[0], new.seq_len), producer)), dim=1)
        self.native_writes += new.seq_len
        self.revision += 1

    @torch.no_grad()
    def evict_prefix(self, cache: HSTUKVCache, count: int):
        """Subtract departing contributions before the caller crops its cache."""
        if not 0 <= count <= self.total_count:
            raise ValueError("eviction must refer to the actual retained prefix")
        if not count:
            return
        removed = self.producers[:, :count]
        for group, identity in enumerate(self.producer_ids):
            mask = (removed == identity).float()
            self.counts[:, group] -= mask.sum(1)
            self.sums[:, group, :, 0] -= torch.einsum("lbnw,bn->blw", cache.k[:, :, :count].float(), mask)
            self.sums[:, group, :, 1] -= torch.einsum("lbnw,bn->blw", cache.v[:, :, :count].float(), mask)
            # Avoid a floating subtraction residual for an absent producer.
            self.sums[:, group] *= (self.counts[:, group] > 0)[:, None, None, None]
        self.producers = self.producers[:, count:]
        self.revision += 1

    def select(self, indices):
        """Copy owners for an independent equal-length native append batch."""
        selected = ProducerSummary(self.sums[indices].clone(), self.counts[indices].clone(),
                                   self.producers[indices].clone(),
                                   producer_ids=self.producer_ids, max_length=self.max_length)
        selected.native_writes = self.native_writes[indices].clone()
        selected.revision = self.revision
        return selected

    def put(self, indices, updated):
        """Return changed owners to a same-length batch; no K/V scan occurs."""
        if updated.total_count != self.total_count or updated.producer_ids != self.producer_ids:
            raise ValueError("put requires equal retained lengths and producer coordinates")
        self.sums[indices] = updated.sums
        self.counts[indices] = updated.counts
        self.producers[indices] = updated.producers
        self.native_writes[indices] = updated.native_writes
        self.revision += 1

    def storage_bytes(self):
        """Logical tensor and integer bytes, excluding Python container overhead."""
        return sum(value.numel() * value.element_size() for value in
                   (self.sums, self.producers, self.counts, self.native_writes))

    def estimate_flops(self, operation, *, length=None):
        """Dense arithmetic for this implementation; copies/indices are excluded.

        ``scan`` is the initial producer-masked reduction, ``add``/``evict``
        refer only to changed events, and ``features`` uses aggregate dimensions.
        Reductions conservatively count N terms and multiply-adds as two FLOPs.
        """
        batch, producers, layers, _, width = self.sums.shape
        length = self.total_count if length is None else length
        payload = 2 * batch * producers * layers * width
        if operation in ("add", "evict") and length == 0:
            return 0
        if operation == "scan":
            return 2 * payload * length + batch * producers * length
        if operation == "add":
            return 2 * batch * layers * width * (length + 1) + 2 * batch
        if operation == "evict":
            return (2 * payload * length + 2 * payload
                    + batch * producers * (length + 1))
        if operation == "features":
            return payload + 2 * batch * producers + batch
        raise ValueError(f"unknown summary operation: {operation}")
