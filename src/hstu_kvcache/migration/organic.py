from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

import torch

from ..models import HSTU, HSTUKVCache
from .cohort_jagged import JaggedMigratedKVBatch


def _offsets(lengths: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (
            torch.zeros(
                1,
                dtype=torch.long,
                device=lengths.device,
            ),
            lengths.long().cumsum(0),
        )
    )


@dataclass(frozen=True)
class JaggedTokenSlice:
    record_ids: tuple[int, ...]
    migration_anchor_version: str
    served_kv_target: str
    starts: tuple[int, ...]
    stops: tuple[int, ...]
    retained_rows: tuple[int, ...]
    empty_rows: tuple[int, ...]
    num_layers: int
    kv_width: int
    dtype: torch.dtype
    device: torch.device
    cache: JaggedMigratedKVBatch | None

    def __post_init__(self) -> None:
        batch_size = len(self.record_ids)
        if (
            batch_size < 1
            or len(self.starts) != batch_size
            or len(self.stops) != batch_size
            or self.num_layers < 1
            or self.kv_width < 1
            or not self.dtype.is_floating_point
            or not self.migration_anchor_version
            or not self.served_kv_target
        ):
            raise ValueError("jagged token slice metadata is invalid")
        if any(
            start < 0 or stop < start
            for start, stop in zip(self.starts, self.stops, strict=True)
        ):
            raise ValueError("jagged token slice bounds are invalid")
        expected_retained = tuple(
            row
            for row, (start, stop) in enumerate(
                zip(self.starts, self.stops, strict=True)
            )
            if stop > start
        )
        expected_empty = tuple(
            row
            for row, (start, stop) in enumerate(
                zip(self.starts, self.stops, strict=True)
            )
            if stop == start
        )
        if (
            self.retained_rows != expected_retained
            or self.empty_rows != expected_empty
        ):
            raise ValueError("jagged token slice row partition is invalid")
        if self.cache is None:
            if self.retained_rows:
                raise ValueError("nonempty slice rows require a cache")
            return
        expected_ids = tuple(self.record_ids[row] for row in self.retained_rows)
        expected_lengths = torch.tensor(
            [self.stops[row] - self.starts[row] for row in self.retained_rows],
            dtype=torch.long,
            device=self.device,
        )
        if (
            self.cache.record_ids != expected_ids
            or self.cache.migration_anchor_version
            != self.migration_anchor_version
            or self.cache.served_kv_target != self.served_kv_target
            or self.cache.k.shape[0] != self.num_layers
            or self.cache.k.shape[2] != self.kv_width
            or self.cache.k.dtype != self.dtype
            or self.cache.k.device != self.device
            or not torch.equal(self.cache.lengths.long(), expected_lengths)
        ):
            raise ValueError("jagged token slice cache differs from metadata")

    @property
    def lengths(self) -> tuple[int, ...]:
        return tuple(
            stop - start
            for start, stop in zip(self.starts, self.stops, strict=True)
        )


@dataclass(frozen=True)
class OrganicAppendResult:
    cache: JaggedMigratedKVBatch
    last_appended_hidden: torch.Tensor | None
    appended_mask: torch.Tensor

    def __post_init__(self) -> None:
        if (
            self.appended_mask.shape != (self.cache.batch_size,)
            or self.appended_mask.dtype != torch.bool
            or self.appended_mask.device != self.cache.k.device
        ):
            raise ValueError("organic append mask differs from output cache")
        if self.last_appended_hidden is None:
            if bool(torch.any(self.appended_mask)):
                raise ValueError("appended rows require last hidden states")
        elif (
            self.last_appended_hidden.ndim != 2
            or self.last_appended_hidden.shape[0] != self.cache.batch_size
            or self.last_appended_hidden.device != self.cache.k.device
        ):
            raise ValueError("organic append hidden states differ from output cache")


@dataclass(frozen=True)
class HistoryOverlapPlan:
    old_length: int
    new_prefix_length: int
    overlap_length: int
    evicted_tokens: int
    appended_tokens: int
    retained_old_start: int
    retained_old_stop: int
    appended_new_start: int
    appended_new_stop: int

    def __post_init__(self) -> None:
        if (
            self.old_length < 0
            or self.new_prefix_length < 0
            or not 0
            <= self.overlap_length
            <= min(self.old_length, self.new_prefix_length)
            or self.evicted_tokens != self.old_length - self.overlap_length
            or self.appended_tokens
            != self.new_prefix_length - self.overlap_length
            or self.retained_old_start != self.evicted_tokens
            or self.retained_old_stop != self.old_length
            or self.appended_new_start != self.overlap_length
            or self.appended_new_stop != self.new_prefix_length
        ):
            raise ValueError("history overlap plan is inconsistent")


def _range_values(
    values: Sequence[int] | torch.Tensor,
    name: str,
) -> tuple[int, ...]:
    if isinstance(values, torch.Tensor):
        if values.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        prepared = tuple(values.detach().cpu().tolist())
    else:
        prepared = tuple(values)
    if any(
        isinstance(value, bool) or not isinstance(value, Integral)
        for value in prepared
    ):
        raise ValueError(f"{name} must contain integers")
    return tuple(int(value) for value in prepared)


def slice_jagged_token_ranges(
    cache: JaggedMigratedKVBatch,
    starts: Sequence[int] | torch.Tensor,
    stops: Sequence[int] | torch.Tensor,
) -> JaggedTokenSlice:
    prepared_starts = _range_values(starts, "jagged token starts")
    prepared_stops = _range_values(stops, "jagged token stops")
    if (
        len(prepared_starts) != cache.batch_size
        or len(prepared_stops) != cache.batch_size
    ):
        raise ValueError("jagged token range count differs from batch size")
    source_lengths = tuple(int(value) for value in cache.lengths.tolist())
    if any(
        start < 0 or stop < start or stop > length
        for start, stop, length in zip(
            prepared_starts,
            prepared_stops,
            source_lengths,
            strict=True,
        )
    ):
        raise ValueError("jagged token range is outside its record")
    retained_rows = tuple(
        row
        for row, (start, stop) in enumerate(
            zip(prepared_starts, prepared_stops, strict=True)
        )
        if stop > start
    )
    empty_rows = tuple(
        row
        for row, (start, stop) in enumerate(
            zip(prepared_starts, prepared_stops, strict=True)
        )
        if stop == start
    )
    selected = None
    if retained_rows:
        k = []
        v = []
        lengths = []
        for row in retained_rows:
            source_start = int(cache.offsets[row]) + prepared_starts[row]
            source_stop = int(cache.offsets[row]) + prepared_stops[row]
            k.append(cache.k[:, source_start:source_stop])
            v.append(cache.v[:, source_start:source_stop])
            lengths.append(source_stop - source_start)
        length_tensor = torch.tensor(
            lengths,
            dtype=torch.long,
            device=cache.k.device,
        )
        selected = JaggedMigratedKVBatch(
            record_ids=tuple(cache.record_ids[row] for row in retained_rows),
            migration_anchor_version=cache.migration_anchor_version,
            served_kv_target=cache.served_kv_target,
            k=torch.cat(k, dim=1).contiguous(),
            v=torch.cat(v, dim=1).contiguous(),
            lengths=length_tensor,
            offsets=_offsets(length_tensor),
        )
    return JaggedTokenSlice(
        record_ids=cache.record_ids,
        migration_anchor_version=cache.migration_anchor_version,
        served_kv_target=cache.served_kv_target,
        starts=prepared_starts,
        stops=prepared_stops,
        retained_rows=retained_rows,
        empty_rows=empty_rows,
        num_layers=cache.k.shape[0],
        kv_width=cache.k.shape[2],
        dtype=cache.k.dtype,
        device=cache.k.device,
        cache=selected,
    )


def tail_slice_jagged_cache(
    cache: JaggedMigratedKVBatch,
    keep_lengths: Sequence[int] | torch.Tensor,
) -> JaggedTokenSlice:
    prepared = _range_values(keep_lengths, "jagged tail lengths")
    if len(prepared) != cache.batch_size:
        raise ValueError("jagged tail count differs from batch size")
    source_lengths = tuple(int(value) for value in cache.lengths.tolist())
    if any(
        keep < 0 or keep > length
        for keep, length in zip(prepared, source_lengths, strict=True)
    ):
        raise ValueError("jagged tail length is outside its record")
    return slice_jagged_token_ranges(
        cache,
        tuple(
            length - keep
            for length, keep in zip(
                source_lengths,
                prepared,
                strict=True,
            )
        ),
        source_lengths,
    )


def _selected_cache(
    sliced: JaggedTokenSlice,
    full_rows: tuple[int, ...],
) -> JaggedMigratedKVBatch:
    if sliced.cache is None:
        raise ValueError("selected retained cache is unavailable")
    local_by_full = {
        full_row: local_row
        for local_row, full_row in enumerate(sliced.retained_rows)
    }
    local_rows = tuple(local_by_full[row] for row in full_rows)
    k = []
    v = []
    lengths = []
    for local_row in local_rows:
        start = int(sliced.cache.offsets[local_row])
        stop = int(sliced.cache.offsets[local_row + 1])
        k.append(sliced.cache.k[:, start:stop])
        v.append(sliced.cache.v[:, start:stop])
        lengths.append(stop - start)
    length_tensor = torch.tensor(
        lengths,
        dtype=torch.long,
        device=sliced.device,
    )
    return JaggedMigratedKVBatch(
        record_ids=tuple(sliced.record_ids[row] for row in full_rows),
        migration_anchor_version=sliced.migration_anchor_version,
        served_kv_target=sliced.served_kv_target,
        k=torch.cat(k, dim=1).contiguous(),
        v=torch.cat(v, dim=1).contiguous(),
        lengths=length_tensor,
        offsets=_offsets(length_tensor),
    )


def _unpack_jagged_cache(
    cache: JaggedMigratedKVBatch,
) -> HSTUKVCache:
    width = int(cache.lengths.max())
    shape = (
        cache.k.shape[0],
        cache.batch_size,
        width,
        cache.k.shape[2],
    )
    k = torch.zeros(shape, dtype=torch.float32, device=cache.k.device)
    v = torch.zeros_like(k)
    for row in range(cache.batch_size):
        start = int(cache.offsets[row])
        stop = int(cache.offsets[row + 1])
        length = stop - start
        k[:, row, :length].copy_(cache.k[:, start:stop])
        v[:, row, :length].copy_(cache.v[:, start:stop])
    return HSTUKVCache(k=k, v=v, seq_len=width)


def _validate_suffix(
    sliced: JaggedTokenSlice,
    item_ids: torch.Tensor,
    behaviors: torch.Tensor,
    time_deltas: torch.Tensor,
    lengths: torch.Tensor,
) -> tuple[int, ...]:
    if (
        item_ids.ndim != 2
        or item_ids.shape != behaviors.shape
        or item_ids.shape != time_deltas.shape
        or item_ids.shape[0] != len(sliced.record_ids)
        or lengths.shape != (len(sliced.record_ids),)
        or item_ids.device != sliced.device
        or behaviors.device != sliced.device
        or time_deltas.device != sliced.device
        or lengths.device != sliced.device
        or item_ids.dtype not in {torch.int32, torch.int64}
        or behaviors.dtype not in {torch.int32, torch.int64}
        or not time_deltas.is_floating_point()
        or lengths.dtype
        not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }
    ):
        raise ValueError("organic suffix tensors are invalid")
    prepared = tuple(int(value) for value in lengths.tolist())
    if any(
        length < 0 or length > item_ids.shape[1]
        for length in prepared
    ):
        raise ValueError("organic suffix length is outside its padded width")
    if any(
        retained + appended < 1
        for retained, appended in zip(
            sliced.lengths,
            prepared,
            strict=True,
        )
    ):
        raise ValueError("organic append cannot produce an empty record")
    return prepared


@torch.inference_mode()
def append_jagged_suffix(
    model: HSTU,
    sliced: JaggedTokenSlice,
    item_ids: torch.Tensor,
    behaviors: torch.Tensor,
    time_deltas: torch.Tensor,
    lengths: torch.Tensor,
    dtype: torch.dtype = torch.float16,
) -> OrganicAppendResult:
    if not dtype.is_floating_point:
        raise ValueError("organic output dtype must be floating point")
    try:
        model_device = next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("organic append model has no parameters") from exc
    if model_device != sliced.device:
        raise ValueError("organic append model and cache devices differ")
    if model.training:
        raise ValueError("organic append requires an evaluation-mode model")
    if (
        model.cfg.num_layers != sliced.num_layers
        or model.blocks[0].attn.inner != sliced.kv_width
    ):
        raise ValueError("organic append model and cache shapes differ")
    appended_lengths = _validate_suffix(
        sliced,
        item_ids,
        behaviors,
        time_deltas,
        lengths,
    )
    retained_lengths = sliced.lengths
    both_rows = tuple(
        row
        for row, (retained, appended) in enumerate(
            zip(retained_lengths, appended_lengths, strict=True)
        )
        if retained > 0 and appended > 0
    )
    retained_only_rows = tuple(
        row
        for row, (retained, appended) in enumerate(
            zip(retained_lengths, appended_lengths, strict=True)
        )
        if retained > 0 and appended == 0
    )
    fresh_rows = tuple(
        row
        for row, (retained, appended) in enumerate(
            zip(retained_lengths, appended_lengths, strict=True)
        )
        if retained == 0 and appended > 0
    )
    row_k: list[torch.Tensor | None] = [None] * len(sliced.record_ids)
    row_v: list[torch.Tensor | None] = [None] * len(sliced.record_ids)
    row_hidden: list[torch.Tensor | None] = [None] * len(sliced.record_ids)
    if retained_only_rows:
        selected = _selected_cache(sliced, retained_only_rows)
        for local_row, full_row in enumerate(retained_only_rows):
            start = int(selected.offsets[local_row])
            stop = int(selected.offsets[local_row + 1])
            row_k[full_row] = selected.k[:, start:stop]
            row_v[full_row] = selected.v[:, start:stop]
    if both_rows:
        selected = _selected_cache(sliced, both_rows)
        padded = _unpack_jagged_cache(selected)
        selected_index = torch.tensor(
            both_rows,
            dtype=torch.long,
            device=sliced.device,
        )
        appended_width = max(appended_lengths[row] for row in both_rows)
        selected_items = item_ids.index_select(0, selected_index)[
            :, :appended_width
        ]
        selected_behaviors = behaviors.index_select(0, selected_index)[
            :, :appended_width
        ]
        selected_deltas = time_deltas.index_select(0, selected_index)[
            :, :appended_width
        ]
        hidden, updated = model.forward_with_cache(
            padded,
            selected_items,
            selected_behaviors,
            selected_deltas,
        )
        retained_width = padded.seq_len
        for local_row, full_row in enumerate(both_rows):
            retained = retained_lengths[full_row]
            appended = appended_lengths[full_row]
            row_k[full_row] = torch.cat(
                (
                    updated.k[:, local_row, :retained],
                    updated.k[
                        :,
                        local_row,
                        retained_width : retained_width + appended,
                    ],
                ),
                dim=1,
            )
            row_v[full_row] = torch.cat(
                (
                    updated.v[:, local_row, :retained],
                    updated.v[
                        :,
                        local_row,
                        retained_width : retained_width + appended,
                    ],
                ),
                dim=1,
            )
            row_hidden[full_row] = hidden[local_row, appended - 1]
    if fresh_rows:
        selected_index = torch.tensor(
            fresh_rows,
            dtype=torch.long,
            device=sliced.device,
        )
        fresh_width = max(appended_lengths[row] for row in fresh_rows)
        fresh_lengths = torch.tensor(
            [appended_lengths[row] for row in fresh_rows],
            dtype=torch.long,
            device=sliced.device,
        )
        fresh_hidden, fresh_cache = model(
            item_ids.index_select(0, selected_index)[:, :fresh_width],
            behaviors.index_select(0, selected_index)[:, :fresh_width],
            time_deltas.index_select(0, selected_index)[:, :fresh_width],
            return_kv=True,
            lengths=fresh_lengths,
        )
        if fresh_cache is None:
            raise RuntimeError("organic fresh append did not return K/V")
        for local_row, full_row in enumerate(fresh_rows):
            appended = appended_lengths[full_row]
            row_k[full_row] = fresh_cache.k[:, local_row, :appended]
            row_v[full_row] = fresh_cache.v[:, local_row, :appended]
            row_hidden[full_row] = fresh_hidden[local_row, appended - 1]
    if any(value is None for value in row_k) or any(
        value is None for value in row_v
    ):
        raise RuntimeError("organic append did not produce every cache row")
    output_lengths = torch.tensor(
        [
            retained + appended
            for retained, appended in zip(
                retained_lengths,
                appended_lengths,
                strict=True,
            )
        ],
        dtype=torch.long,
        device=sliced.device,
    )
    output = JaggedMigratedKVBatch(
        record_ids=sliced.record_ids,
        migration_anchor_version=sliced.migration_anchor_version,
        served_kv_target=sliced.served_kv_target,
        k=torch.cat(
            [value.to(dtype=dtype) for value in row_k if value is not None],
            dim=1,
        ).contiguous(),
        v=torch.cat(
            [value.to(dtype=dtype) for value in row_v if value is not None],
            dim=1,
        ).contiguous(),
        lengths=output_lengths,
        offsets=_offsets(output_lengths),
    )
    appended_mask = lengths > 0
    hidden_values = [value for value in row_hidden if value is not None]
    last_hidden = None
    if hidden_values:
        exemplar = hidden_values[0]
        last_hidden = torch.stack(
            [
                value
                if value is not None
                else torch.zeros_like(exemplar)
                for value in row_hidden
            ]
        )
    return OrganicAppendResult(
        cache=output,
        last_appended_hidden=last_hidden,
        appended_mask=appended_mask,
    )


def drop_last_jagged_token(
    cache: JaggedMigratedKVBatch,
) -> JaggedMigratedKVBatch:
    lengths = tuple(int(value) for value in cache.lengths.tolist())
    if any(length < 2 for length in lengths):
        raise ValueError("dropping the last token would create an empty prefix")
    sliced = slice_jagged_token_ranges(
        cache,
        (0,) * cache.batch_size,
        tuple(length - 1 for length in lengths),
    )
    if sliced.cache is None or sliced.empty_rows:
        raise RuntimeError("nonempty prefix slicing failed")
    return sliced.cache


def _integer_sequence(
    values: Sequence[int] | torch.Tensor,
    name: str,
) -> tuple[int, ...]:
    if isinstance(values, torch.Tensor):
        if values.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        raw = values.detach().cpu().tolist()
    else:
        raw = list(values)
    if any(
        isinstance(value, bool) or not isinstance(value, Integral)
        for value in raw
    ):
        raise ValueError(f"{name} must contain integers")
    return tuple(int(value) for value in raw)


def plan_history_overlap(
    old_timestamps: Sequence[int] | torch.Tensor,
    old_item_ids: Sequence[int] | torch.Tensor,
    old_behaviors: Sequence[int] | torch.Tensor,
    new_prefix_timestamps: Sequence[int] | torch.Tensor,
    new_prefix_item_ids: Sequence[int] | torch.Tensor,
    new_prefix_behaviors: Sequence[int] | torch.Tensor,
) -> HistoryOverlapPlan:
    old_timestamp_values = _integer_sequence(
        old_timestamps,
        "old timestamps",
    )
    old_item_values = _integer_sequence(old_item_ids, "old item IDs")
    old_behavior_values = _integer_sequence(
        old_behaviors,
        "old behaviors",
    )
    new_timestamp_values = _integer_sequence(
        new_prefix_timestamps,
        "new prefix timestamps",
    )
    new_item_values = _integer_sequence(
        new_prefix_item_ids,
        "new prefix item IDs",
    )
    new_behavior_values = _integer_sequence(
        new_prefix_behaviors,
        "new prefix behaviors",
    )
    if (
        len(
            {
                len(old_timestamp_values),
                len(old_item_values),
                len(old_behavior_values),
            }
        )
        != 1
        or len(
            {
                len(new_timestamp_values),
                len(new_item_values),
                len(new_behavior_values),
            }
        )
        != 1
    ):
        raise ValueError("history event fields have different lengths")
    if any(
        right < left
        for left, right in zip(
            old_timestamp_values,
            old_timestamp_values[1:],
            strict=False,
        )
    ) or any(
        right < left
        for left, right in zip(
            new_timestamp_values,
            new_timestamp_values[1:],
            strict=False,
        )
    ):
        raise ValueError("history timestamps must be nondecreasing")
    old_events = list(
        zip(
            old_timestamp_values,
            old_item_values,
            old_behavior_values,
            strict=True,
        )
    )
    new_events = list(
        zip(
            new_timestamp_values,
            new_item_values,
            new_behavior_values,
            strict=True,
        )
    )
    sentinel = object()
    combined: list[tuple[int, int, int] | object] = [
        *new_events,
        sentinel,
        *old_events,
    ]
    prefix = [0] * len(combined)
    for index in range(1, len(combined)):
        candidate = prefix[index - 1]
        while candidate > 0 and combined[index] != combined[candidate]:
            candidate = prefix[candidate - 1]
        if combined[index] == combined[candidate]:
            candidate += 1
        prefix[index] = candidate
    overlap = prefix[-1] if combined else 0
    overlap = min(overlap, len(old_events), len(new_events))
    return HistoryOverlapPlan(
        old_length=len(old_events),
        new_prefix_length=len(new_events),
        overlap_length=overlap,
        evicted_tokens=len(old_events) - overlap,
        appended_tokens=len(new_events) - overlap,
        retained_old_start=len(old_events) - overlap,
        retained_old_stop=len(old_events),
        appended_new_start=overlap,
        appended_new_stop=len(new_events),
    )
