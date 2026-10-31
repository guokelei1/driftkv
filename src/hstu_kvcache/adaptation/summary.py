"""Fixed event membership, FP32 sums, and exact subtraction on eviction."""

from collections import deque
from dataclasses import dataclass, replace
from itertools import repeat

import torch

from hstu_kvcache.models import HSTUKVCache


@dataclass
class Summary:
    # [segments, slots, layers, K/V, width]; metadata never comes from a translator.
    payload: torch.Tensor
    count: torch.Tensor
    ordinal: torch.Tensor
    timestamp: torch.Tensor
    producer: torch.Tensor
    segment_ids: tuple[int, ...]
    read_mode: str = "kv"
    release_age: int = 0
    temporal_coefficients: torch.Tensor | None = None  # [layers, native time features, width]
    second_moment: torch.Tensor | None = None  # Same membership as payload; raw E[K²], E[V²].
    query_coefficients: torch.Tensor | None = None  # [layers, heads, query dim, response dim], per event.

    def with_payload(self, payload: torch.Tensor) -> "Summary":
        return replace(self, payload=payload)

    def select(self, indices) -> "Summary":
        return Summary(self.payload[indices], self.count[indices], self.ordinal[indices],
                       self.timestamp[indices], self.producer[indices],
                       tuple(self.segment_ids[i] for i in indices), self.read_mode, self.release_age,
                       self.temporal_coefficients,
                       None if self.second_moment is None else self.second_moment[indices], self.query_coefficients)


class SummaryWriter:
    def __init__(self, layers: int, width: int, device, segment_size=64, slots=2, second_moments=False):
        if segment_size % slots:
            raise ValueError("slots must evenly partition a segment")
        self.layers, self.width, self.device = layers, width, device
        self.segment_size, self.slots = segment_size, slots
        self.second_moments = second_moments
        self.slot_size = segment_size // slots
        self.segments = {}
        self.events = deque()
        self.next_ordinal = 0
        self.next_segment = 0
        self.open_segment = None

    @classmethod
    @torch.no_grad()
    def from_cache(cls, cache, timestamps, producer, *, segment_size=64, slots=2, second_moments=False):
        writer = cls(cache.k.shape[0], cache.k.shape[-1], cache.k.device,
                     segment_size, slots, second_moments)
        length = cache.seq_len
        if not length:
            return writer
        stamps = timestamps.tolist() if hasattr(timestamps, "tolist") else list(timestamps)
        segments = (length + segment_size - 1) // segment_size
        values = torch.stack((cache.k[:, 0], cache.v[:, 0]), 1).float()
        padding = segments * segment_size - length
        if padding:
            values = torch.nn.functional.pad(values, (0, 0, 0, padding))
        sums = values.reshape(writer.layers, 2, segments, slots, writer.slot_size, writer.width)
        sums = sums.sum(4).permute(2, 3, 0, 1, 4).contiguous()
        counts, ordinals, times = [], [], []
        for sid in range(segments):
            bounds = [(min(length, sid * segment_size + slot * writer.slot_size),
                       min(length, sid * segment_size + (slot + 1) * writer.slot_size))
                      for slot in range(slots)]
            counts.append([stop - start for start, stop in bounds])
            ordinals.append([(start + stop - 1) * (stop - start) // 2 for start, stop in bounds])
            times.append([sum(stamps[start:stop]) for start, stop in bounds])
        # Structural counters live on CPU; no GPU kernel is needed for an
        # integer increment. Materialize their tensors only when a view reads them.
        writer.segments = {sid: dict(producer=producer,
            written=min(segment_size, length-sid*segment_size), sums=sums[sid],
            count=counts[sid], ordinal=ordinals[sid], timestamp=times[sid]) for sid in range(segments)}
        if second_moments:
            squared = values.square().reshape(writer.layers, 2, segments, slots, writer.slot_size, writer.width)
            squared = squared.sum(4).permute(2, 3, 0, 1, 4).contiguous()
            for sid in range(segments):
                writer.segments[sid]["squared_sums"] = squared[sid]
        if segments == slots == 1:
            writer.events = deque(zip(repeat(0, length), repeat(0, length), range(length), stamps, strict=True))
        else:
            writer.events = deque((index // segment_size, index % segment_size // writer.slot_size,
                                   index, timestamp) for index, timestamp in enumerate(stamps))
        writer.next_ordinal = length
        writer.next_segment = segments
        writer.open_segment = segments - 1 if length % segment_size else None
        return writer

    def close_segment(self):
        self.open_segment = None

    @torch.no_grad()
    def add(self, cache: HSTUKVCache, timestamps, producer: int):
        """Add only real, already-computed event K/V; candidates never call this."""
        position = 0
        while position < cache.seq_len:
            if self.open_segment is None:
                sid = self.next_segment
                self.next_segment += 1
                self.open_segment = sid
                self.segments[sid] = {
                    "producer": producer, "written": 0,
                    "sums": torch.zeros(self.slots, self.layers, 2, self.width,
                                        device=self.device, dtype=torch.float32),
                    "count": [0] * self.slots,
                    "ordinal": [0] * self.slots,
                    "timestamp": [0] * self.slots,
                }
                if self.second_moments:
                    self.segments[sid]["squared_sums"] = torch.zeros_like(self.segments[sid]["sums"])
            sid = self.open_segment
            seg = self.segments[sid]
            if seg["producer"] != producer:
                raise ValueError("release must close the previous producer's open segment")
            slot = seg["written"] // self.slot_size
            count = min(cache.seq_len - position, self.slot_size - seg["written"] % self.slot_size)
            end = position + count
            seg["sums"][slot, :, 0] += cache.k[:, 0, position:end].float().sum(1)
            seg["sums"][slot, :, 1] += cache.v[:, 0, position:end].float().sum(1)
            if self.second_moments:
                seg["squared_sums"][slot, :, 0] += cache.k[:, 0, position:end].float().square().sum(1)
                seg["squared_sums"][slot, :, 1] += cache.v[:, 0, position:end].float().square().sum(1)
            seg["count"][slot] += count
            ordinals = range(self.next_ordinal, self.next_ordinal + count)
            times = [int(t) for t in timestamps[position:end]]
            seg["ordinal"][slot] += sum(ordinals)
            seg["timestamp"][slot] += sum(times)
            self.events.extend((sid, slot, ordinal, timestamp) for ordinal, timestamp in zip(ordinals, times, strict=True))
            self.next_ordinal += count
            seg["written"] += count
            position = end
            if seg["written"] == self.segment_size:
                self.close_segment()

    @torch.no_grad()
    def evict_first(self, cache: HSTUKVCache) -> int:
        """Subtract the exact stored contribution, before cropping the cache."""
        sid, slot, ordinal, timestamp = self.events.popleft()
        seg = self.segments[sid]
        producer = seg["producer"]
        seg["sums"][slot, :, 0] -= cache.k[:, 0, 0].float()
        seg["sums"][slot, :, 1] -= cache.v[:, 0, 0].float()
        if self.second_moments:
            seg["squared_sums"][slot, :, 0] -= cache.k[:, 0, 0].float().square()
            seg["squared_sums"][slot, :, 1] -= cache.v[:, 0, 0].float().square()
        seg["count"][slot] -= 1
        seg["ordinal"][slot] -= ordinal
        seg["timestamp"][slot] -= timestamp
        if not self.events or self.events[0][0] != sid:
            del self.segments[sid]
            if self.open_segment == sid:
                self.close_segment()
        return producer

    def pack(self, target: int, *, include_current=False) -> Summary | None:
        selected = [(sid, seg) for sid, seg in self.segments.items() if include_current or seg["producer"] != target]
        if not selected:
            return None
        counts = torch.tensor([seg["count"] for _, seg in selected], device=self.device, dtype=torch.float32)
        divisor = counts.clamp_min(1)
        payload = torch.stack([seg["sums"] for _, seg in selected]) / divisor[..., None, None, None]
        payload = payload * (counts > 0)[..., None, None, None]
        return Summary(
            payload=payload,
            count=counts,
            ordinal=torch.tensor([seg["ordinal"] for _, seg in selected], device=self.device, dtype=torch.float64) / divisor,
            timestamp=torch.tensor([seg["timestamp"] for _, seg in selected], device=self.device, dtype=torch.float64) / divisor,
            producer=torch.tensor([seg["producer"] for _, seg in selected], device=self.device),
            segment_ids=tuple(sid for sid, _ in selected),
            second_moment=(torch.stack([seg["squared_sums"] for _, seg in selected])/divisor[..., None, None, None]
                           * (counts > 0)[..., None, None, None]
                           if self.second_moments else None),
        )

    @torch.no_grad()
    def evict_prefix(self, cache: HSTUKVCache, count: int) -> dict[int, int]:
        """Subtract a replay prefix once per slot, before bulk-adding real writes.

        An emptied open segment keeps its write coordinate: during sequential
        replay the new events would have filled that segment before it expired.
        """
        groups = {}
        producers = {}
        for position in range(count):
            sid, slot, ordinal, timestamp = self.events.popleft()
            groups.setdefault((sid, slot), []).append((position, ordinal, timestamp))
            producer = self.segments[sid]["producer"]
            producers[producer] = producers.get(producer, 0) + 1
        retained_segments = {event[0] for event in self.events}
        for (sid, slot), entries in groups.items():
            segment = self.segments[sid]
            # Fixed chronological slot membership makes each group contiguous.
            start, stop = entries[0][0], entries[-1][0] + 1
            segment["sums"][slot, :, 0] -= cache.k[:, 0, start:stop].float().sum(1)
            segment["sums"][slot, :, 1] -= cache.v[:, 0, start:stop].float().sum(1)
            if self.second_moments:
                segment["squared_sums"][slot, :, 0] -= cache.k[:, 0, start:stop].float().square().sum(1)
                segment["squared_sums"][slot, :, 1] -= cache.v[:, 0, start:stop].float().square().sum(1)
            segment["count"][slot] -= len(entries)
            segment["ordinal"][slot] -= sum(entry[1] for entry in entries)
            segment["timestamp"][slot] -= sum(entry[2] for entry in entries)
        for sid in {sid for sid, _ in groups} - retained_segments:
            if self.open_segment == sid:
                self.segments[sid]["sums"].zero_()
                if self.second_moments:
                    self.segments[sid]["squared_sums"].zero_()
                for key in ("count", "ordinal", "timestamp"):
                    self.segments[sid][key] = [0] * self.slots
            else:
                del self.segments[sid]
        return producers

    def reference_summary(self, target_cache: HSTUKVCache, target: int) -> Summary | None:
        """Diagnostic/teacher only: target means with the source's event membership."""
        source = self.pack(target)
        if source is None:
            return None
        payload = torch.zeros_like(source.payload)
        second = torch.zeros_like(payload) if self.second_moments else None
        rows = {sid: i for i, sid in enumerate(source.segment_ids)}
        groups = {}
        for index, (sid, slot, _, _) in enumerate(self.events):
            if sid in rows:
                groups.setdefault((rows[sid], slot), []).append(index)
        for (row, slot), indices in groups.items():
            payload[row, slot, :, 0] = target_cache.k[:, 0, indices].float().mean(1)
            payload[row, slot, :, 1] = target_cache.v[:, 0, indices].float().mean(1)
            if second is not None:
                second[row, slot, :, 0] = target_cache.k[:, 0, indices].float().square().mean(1)
                second[row, slot, :, 1] = target_cache.v[:, 0, indices].float().square().mean(1)
        return replace(source, payload=payload, second_moment=second)

    def old_mask(self, target: int):
        return torch.tensor([self.segments[sid]["producer"] != target for sid, _, _, _ in self.events],
                            device=self.device, dtype=torch.bool)

    def storage_bytes(self):
        # Views of a bulk backfill keep their underlying allocation alive until
        # its last segment is evicted. Count that allocation once, in full.
        storages = {value.untyped_storage().data_ptr(): value.untyped_storage().nbytes()
                    for seg in self.segments.values() for value in seg.values()
                    if isinstance(value, torch.Tensor)}
        tensor_bytes = sum(storages.values())
        # Logical metadata bytes, excluding Python-container overhead (reported separately).
        return tensor_bytes + len(self.events) * 4 * 8 + len(self.segments) * (3 + 3*self.slots) * 8
