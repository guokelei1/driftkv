from __future__ import annotations

import argparse
import gc
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist

from hstu_kvcache.data import load_prepared_kuairand_plan
from hstu_kvcache.migration import (
    D2ActionPlan,
    D2ActionRecord,
    D3GroupPlan,
    D3WorkManifest,
    audit_d3_group_plan,
    audit_d3_work_manifest,
    build_d2_record_owner_map,
    canonical_json_bytes,
)
from hstu_kvcache.migration.cohort_jagged import JaggedMigratedKVBatch
from hstu_kvcache.migration.design2_distributed import (
    close_d2_distributed_runtime,
    init_d2_distributed_runtime,
)
from hstu_kvcache.migration.design2_embedding import (
    build_modulo_sharded_hstu_from_cpu,
)
from hstu_kvcache.migration.design2_integrated import (
    IntegratedAppendOnlyKVBatch,
    integrated_sharded_append_only,
    integrated_sharded_exact,
    slice_integrated_jagged_ranges,
)
from hstu_kvcache.migration.recompute import RawHistoryBatch
from hstu_kvcache.migration.stage45_oldkv import (
    DirectOldKVFusedOperator,
    load_direct_oldkv_program,
)
from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.streaming import (
    reconstruct_organic_windows,
    validate_long_context_plan,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "evokv_design3_m0_pageable_s0_development_v0"
DEFAULT_ACTION_PLAN = (
    "configs/cohortkv_d2/"
    "action_plan_theta1_theta2_staggered_renewal_h12.json"
)
DEFAULT_STAGE_A_SUMMARY = "configs/cohortkv_d2/stage_a_summary.json"
DEFAULT_TRAINING = (
    "results/motivation_scale/"
    "long_context_4plus12_training_exploration_seed0.json"
)
DEFAULT_CHECKPOINT_DIR = (
    "checkpoints/kuairand_long_context_4plus12_exploration/seed0"
)
DEFAULT_WORK_MANIFEST = (
    "configs/evokv_d3/development/h12_w2_m0_work_manifest.json"
)
DEFAULT_GROUP_PLAN = (
    "configs/evokv_d3/development/h12_w2_m0_group_plan.json"
)
DEFAULT_OUTPUT = (
    "configs/evokv_d3/development/h12_w2_m0_s0_canary.json"
)


@dataclass(frozen=True)
class SelectedGroup:
    source_ordinal: int
    pool: str
    record_ids_by_rank: tuple[tuple[int, ...], ...]


@dataclass
class PinnedSlot:
    source_k: torch.Tensor
    source_v: torch.Tensor
    source_lengths: torch.Tensor
    source_offsets: torch.Tensor
    history_item_ids: torch.Tensor
    history_behaviors: torch.Tensor
    history_time_deltas: torch.Tensor
    history_lengths: torch.Tensor
    output_k: torch.Tensor
    output_v: torch.Tensor
    output_lengths: torch.Tensor
    output_offsets: torch.Tensor
    output_hidden: torch.Tensor

    @property
    def nbytes(self) -> int:
        storages = {}
        for value in self.__dict__.values():
            storage = value.untyped_storage()
            storages[storage.data_ptr()] = storage.nbytes()
        return sum(storages.values())


@dataclass(frozen=True)
class PublishedOutput:
    segments: tuple[JaggedMigratedKVBatch, ...]
    last_hidden: torch.Tensor
    final_lengths: torch.Tensor
    final_offsets: torch.Tensor

    def __post_init__(self) -> None:
        if (
            not self.segments
            or any(
                value.record_ids != self.segments[0].record_ids
                for value in self.segments
            )
            or self.final_lengths.shape
            != self.segments[0].lengths.shape
            or self.final_offsets.shape
            != self.segments[0].offsets.shape
            or not torch.equal(
                self.final_lengths,
                sum(
                    (
                        value.lengths
                        for value in self.segments
                    ),
                    torch.zeros_like(self.segments[0].lengths),
                ),
            )
            or int(self.final_offsets[0]) != 0
            or not torch.equal(
                self.final_offsets[1:] - self.final_offsets[:-1],
                self.final_lengths,
            )
        ):
            raise ValueError("D3 published output metadata differs")

    @property
    def record_ids(self) -> tuple[int, ...]:
        return self.segments[0].record_ids

    @property
    def nbytes(self) -> int:
        return (
            sum(value.nbytes for value in self.segments)
            + self.last_hidden.numel() * self.last_hidden.element_size()
            + self.final_lengths.numel()
            * self.final_lengths.element_size()
            + self.final_offsets.numel()
            * self.final_offsets.element_size()
        )


@dataclass(frozen=True)
class PinnedKVTransfer:
    record_ids: tuple[int, ...]
    migration_anchor_version: str
    served_kv_target: str
    k: torch.Tensor
    v: torch.Tensor
    lengths: torch.Tensor
    offsets: torch.Tensor

    def finish(self) -> JaggedMigratedKVBatch:
        return JaggedMigratedKVBatch(
            record_ids=self.record_ids,
            migration_anchor_version=self.migration_anchor_version,
            served_kv_target=self.served_kv_target,
            k=self.k,
            v=self.v,
            lengths=self.lengths,
            offsets=self.offsets,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action-plan", default=DEFAULT_ACTION_PLAN)
    parser.add_argument("--stage-a-summary", default=DEFAULT_STAGE_A_SUMMARY)
    parser.add_argument("--training-result", default=DEFAULT_TRAINING)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--work-manifest", default=DEFAULT_WORK_MANIFEST)
    parser.add_argument("--group-plan", default=DEFAULT_GROUP_PLAN)
    parser.add_argument(
        "--scope",
        choices=("canary", "full"),
        default="canary",
    )
    parser.add_argument("--canary-records-per-rank", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--expected-visible-devices", default="0,1")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _checkpoint_path(checkpoint_dir: Path, version: str) -> Path:
    return checkpoint_dir / f"theta_{int(version.removeprefix('theta'))}.pt"


def _load_cpu_model(cfg: HSTUConfig, checkpoint: Path) -> HSTU:
    model = HSTU(cfg)
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True)
    )
    model.eval()
    return model


def _program_descriptor(
    stage_a: dict[str, object],
) -> dict[str, object]:
    capacity = json.loads(
        _path(stage_a["artifacts"]["capacity"]["path"]).read_text()
    )
    return capacity["program"]


def _select_groups(
    plan: D3GroupPlan,
    scope: str,
    canary_records_per_rank: int,
) -> tuple[SelectedGroup, ...]:
    if scope == "full":
        return tuple(
            SelectedGroup(
                source_ordinal=value.ordinal,
                pool=value.pool,
                record_ids_by_rank=value.record_ids_by_rank,
            )
            for value in plan.groups
        )
    if canary_records_per_rank < 1:
        raise ValueError("D3 canary record count must be positive")
    selected = []
    for pool in ("compiled", "exact"):
        source = next(value for value in plan.groups if value.pool == pool)
        selected.append(
            SelectedGroup(
                source_ordinal=source.ordinal,
                pool=source.pool,
                record_ids_by_rank=tuple(
                    tuple(record_ids[:canary_records_per_rank])
                    for record_ids in source.record_ids_by_rank
                ),
            )
        )
    return tuple(selected)


def _history_batch(
    actions: tuple[D2ActionRecord, ...],
    window,
    starts: tuple[int, ...],
    stops: tuple[int, ...],
    version: str,
) -> RawHistoryBatch:
    lengths = tuple(
        stop - start
        for start, stop in zip(starts, stops, strict=True)
    )
    if (
        len(actions) != len(starts)
        or len(actions) != len(stops)
        or any(value < 0 for value in lengths)
    ):
        raise ValueError("D3 history ranges differ")
    width = max(max(lengths, default=0), 1)
    item_ids = torch.zeros((len(actions), width), dtype=torch.long)
    behaviors = torch.zeros_like(item_ids)
    time_deltas = torch.zeros(
        (len(actions), width),
        dtype=torch.float32,
    )
    for row, (action, start, stop) in enumerate(
        zip(actions, starts, stops, strict=True)
    ):
        history = window.records[action.prepared_user_id].history
        if history is None or not 0 <= start <= stop <= len(history):
            raise ValueError("D3 history range exceeds its record")
        length = stop - start
        if length:
            item_ids[row, :length].copy_(
                torch.from_numpy(history.item_ids[start:stop].copy())
            )
            behaviors[row, :length].copy_(
                torch.from_numpy(history.behaviors[start:stop].copy())
            )
            time_deltas[row, :length].copy_(
                torch.from_numpy(history.time_deltas[start:stop].copy())
            )
    return RawHistoryBatch(
        record_ids=tuple(value.record_id for value in actions),
        migration_anchor_version=version,
        item_ids=item_ids,
        behaviors=behaviors,
        time_deltas=time_deltas,
        lengths=torch.tensor(lengths, dtype=torch.long),
    )


def _actions_for_group(
    group: SelectedGroup,
    rank: int,
    actions_by_id: dict[int, D2ActionRecord],
) -> tuple[D2ActionRecord, ...]:
    return tuple(
        actions_by_id[record_id]
        for record_id in group.record_ids_by_rank[rank]
    )


def _assert_pageable(batch: JaggedMigratedKVBatch) -> None:
    tensors = (batch.k, batch.v, batch.lengths, batch.offsets)
    if any(value.device.type != "cpu" for value in tensors) or any(
        value.is_pinned() for value in tensors
    ):
        raise ValueError("D3 DRAM K/V is not pageable CPU memory")


def _pageable_tensor(value: torch.Tensor) -> torch.Tensor:
    output = torch.empty(
        value.shape,
        dtype=value.dtype,
        device="cpu",
    )
    output.copy_(value)
    if output.is_pinned():
        raise RuntimeError("D3 publication unexpectedly remained pinned")
    return output


def _pageable_kv(batch: JaggedMigratedKVBatch) -> JaggedMigratedKVBatch:
    output = JaggedMigratedKVBatch(
        record_ids=batch.record_ids,
        migration_anchor_version=batch.migration_anchor_version,
        served_kv_target=batch.served_kv_target,
        k=_pageable_tensor(batch.k),
        v=_pageable_tensor(batch.v),
        lengths=_pageable_tensor(batch.lengths),
        offsets=_pageable_tensor(batch.offsets),
    )
    _assert_pageable(output)
    return output


def _direct_destination(
    source: JaggedMigratedKVBatch,
    target_version: str,
) -> JaggedMigratedKVBatch:
    return JaggedMigratedKVBatch(
        record_ids=source.record_ids,
        migration_anchor_version=source.migration_anchor_version,
        served_kv_target=target_version,
        k=torch.empty_like(source.k),
        v=torch.empty_like(source.v),
        lengths=source.lengths.clone(),
        offsets=source.offsets.clone(),
    )


def _slot_requirements(
    groups: tuple[SelectedGroup, ...],
    rank: int,
    actions_by_id: dict[int, D2ActionRecord],
) -> dict[str, int]:
    source_tokens = []
    output_tokens = []
    history_elements = []
    rows = []
    for group in groups:
        actions = _actions_for_group(group, rank, actions_by_id)
        rows.append(len(actions))
        source_tokens.append(
            sum(value.retained_tokens for value in actions)
            if group.pool == "compiled"
            else 0
        )
        output_tokens.append(sum(value.final_tokens for value in actions))
        history_lengths = [
            (
                value.final_tokens - value.delta_start
                if group.pool == "compiled"
                else value.final_tokens
            )
            for value in actions
        ]
        history_elements.append(
            len(actions) * max(max(history_lengths, default=0), 1)
        )
    return {
        "source_tokens": max(source_tokens, default=0),
        "output_tokens": max(output_tokens, default=0),
        "history_elements": max(history_elements, default=0),
        "rows": max(rows, default=0),
    }


def _allocate_slot(
    cfg: HSTUConfig,
    requirements: dict[str, int],
) -> PinnedSlot:
    layers = cfg.num_layers
    width = cfg.hidden_size
    source_values = max(
        layers * requirements["source_tokens"] * width,
        1,
    )
    output_values = max(
        layers * requirements["output_tokens"] * width,
        1,
    )
    history_values = max(requirements["history_elements"], 1)
    rows = max(requirements["rows"], 1)
    return PinnedSlot(
        source_k=torch.empty(
            source_values,
            dtype=torch.float16,
            pin_memory=True,
        ),
        source_v=torch.empty(
            source_values,
            dtype=torch.float16,
            pin_memory=True,
        ),
        source_lengths=torch.empty(
            rows,
            dtype=torch.long,
            pin_memory=True,
        ),
        source_offsets=torch.empty(
            rows + 1,
            dtype=torch.long,
            pin_memory=True,
        ),
        history_item_ids=torch.empty(
            history_values,
            dtype=torch.long,
            pin_memory=True,
        ),
        history_behaviors=torch.empty(
            history_values,
            dtype=torch.long,
            pin_memory=True,
        ),
        history_time_deltas=torch.empty(
            history_values,
            dtype=torch.float32,
            pin_memory=True,
        ),
        history_lengths=torch.empty(
            rows,
            dtype=torch.long,
            pin_memory=True,
        ),
        output_k=torch.empty(
            output_values,
            dtype=torch.float16,
            pin_memory=True,
        ),
        output_v=torch.empty(
            output_values,
            dtype=torch.float16,
            pin_memory=True,
        ),
        output_lengths=torch.empty(
            2 * rows,
            dtype=torch.long,
            pin_memory=True,
        ),
        output_offsets=torch.empty(
            2 * (rows + 1),
            dtype=torch.long,
            pin_memory=True,
        ),
        output_hidden=torch.empty(
            rows * cfg.hidden_size,
            dtype=torch.float32,
            pin_memory=True,
        ),
    )


def _stage_source(
    slot: PinnedSlot,
    batch: JaggedMigratedKVBatch | None,
) -> JaggedMigratedKVBatch | None:
    if batch is None:
        return None
    _assert_pageable(batch)
    k = slot.source_k[: batch.k.numel()].view(batch.k.shape)
    v = slot.source_v[: batch.v.numel()].view(batch.v.shape)
    lengths = slot.source_lengths[: batch.batch_size]
    offsets = slot.source_offsets[: batch.batch_size + 1]
    k.copy_(batch.k)
    v.copy_(batch.v)
    lengths.copy_(batch.lengths)
    offsets.copy_(batch.offsets)
    return JaggedMigratedKVBatch(
        record_ids=batch.record_ids,
        migration_anchor_version=batch.migration_anchor_version,
        served_kv_target=batch.served_kv_target,
        k=k,
        v=v,
        lengths=lengths,
        offsets=offsets,
    )


def _stage_history(
    slot: PinnedSlot,
    batch: RawHistoryBatch,
) -> RawHistoryBatch:
    elements = batch.item_ids.numel()
    item_ids = slot.history_item_ids[:elements].view(batch.item_ids.shape)
    behaviors = slot.history_behaviors[:elements].view(
        batch.behaviors.shape
    )
    time_deltas = slot.history_time_deltas[:elements].view(
        batch.time_deltas.shape
    )
    lengths = slot.history_lengths[: batch.batch_size]
    item_ids.copy_(batch.item_ids)
    behaviors.copy_(batch.behaviors)
    time_deltas.copy_(batch.time_deltas)
    lengths.copy_(batch.lengths)
    return RawHistoryBatch(
        record_ids=batch.record_ids,
        migration_anchor_version=batch.migration_anchor_version,
        item_ids=item_ids,
        behaviors=behaviors,
        time_deltas=time_deltas,
        lengths=lengths,
    )


def _pinned_output_kv(
    slot: PinnedSlot,
    batch: JaggedMigratedKVBatch,
    value_offset: int,
    length_offset: int,
    offsets_offset: int,
) -> tuple[PinnedKVTransfer, int, int, int]:
    count = batch.k.numel()
    k = slot.output_k[value_offset : value_offset + count].view(
        batch.k.shape
    )
    v = slot.output_v[value_offset : value_offset + count].view(
        batch.v.shape
    )
    k.copy_(batch.k, non_blocking=True)
    v.copy_(batch.v, non_blocking=True)
    lengths = slot.output_lengths[
        length_offset : length_offset + batch.batch_size
    ]
    length_offset += batch.batch_size
    offsets = slot.output_offsets[
        offsets_offset : offsets_offset + batch.batch_size + 1
    ]
    offsets_offset += batch.batch_size + 1
    lengths.copy_(batch.lengths, non_blocking=True)
    offsets.copy_(batch.offsets, non_blocking=True)
    return (
        PinnedKVTransfer(
            record_ids=batch.record_ids,
            migration_anchor_version=batch.migration_anchor_version,
            served_kv_target=batch.served_kv_target,
            k=k,
            v=v,
            lengths=lengths,
            offsets=offsets,
        ),
        value_offset + count,
        length_offset,
        offsets_offset,
    )


def _copy_output_to_slot(
    slot: PinnedSlot,
    fragment: JaggedMigratedKVBatch | IntegratedAppendOnlyKVBatch,
    last_hidden: torch.Tensor,
) -> tuple[tuple[PinnedKVTransfer, ...], torch.Tensor]:
    value_offset = 0
    length_offset = 0
    offsets_offset = 0
    if isinstance(fragment, JaggedMigratedKVBatch):
        segment, value_offset, length_offset, offsets_offset = (
            _pinned_output_kv(
                slot,
                fragment,
                value_offset,
                length_offset,
                offsets_offset,
            )
        )
        segments = (segment,)
    else:
        retained, value_offset, length_offset, offsets_offset = (
            _pinned_output_kv(
                slot,
                fragment.retained,
                value_offset,
                length_offset,
                offsets_offset,
            )
        )
        suffix, value_offset, length_offset, offsets_offset = (
            _pinned_output_kv(
                slot,
                fragment.suffix,
                value_offset,
                length_offset,
                offsets_offset,
            )
        )
        segments = (retained, suffix)
    hidden = slot.output_hidden[: last_hidden.numel()].view(
        last_hidden.shape
    )
    hidden.copy_(last_hidden, non_blocking=True)
    return segments, hidden


def _publish_output(
    segments: tuple[PinnedKVTransfer, ...],
    hidden: torch.Tensor,
) -> PublishedOutput:
    pageable_segments = tuple(
        _pageable_kv(value.finish()) for value in segments
    )
    final_lengths = sum(
        (value.lengths for value in pageable_segments),
        torch.zeros_like(pageable_segments[0].lengths),
    )
    final_offsets = torch.cat(
        (
            torch.zeros(1, dtype=torch.long),
            final_lengths.cumsum(0),
        )
    )
    output = PublishedOutput(
        segments=pageable_segments,
        last_hidden=_pageable_tensor(hidden),
        final_lengths=final_lengths,
        final_offsets=final_offsets,
    )
    if (
        not output.record_ids
        or output.last_hidden.is_pinned()
        or output.final_lengths.is_pinned()
        or output.final_offsets.is_pinned()
    ):
        raise RuntimeError("D3 pageable publication differs")
    return output


def _validate_published_action_lengths(
    output: PublishedOutput,
    actions: tuple[D2ActionRecord, ...],
    pool: str,
) -> None:
    expected_final = torch.tensor(
        [value.final_tokens for value in actions],
        dtype=torch.long,
    )
    if (
        output.record_ids != tuple(value.record_id for value in actions)
        or not torch.equal(output.final_lengths, expected_final)
    ):
        raise RuntimeError("D3 published final lengths differ")
    if pool == "compiled":
        expected_retained = torch.tensor(
            [value.retained_tokens for value in actions],
            dtype=torch.long,
        )
        if (
            len(output.segments) != 2
            or not torch.equal(
                output.segments[0].lengths,
                expected_retained,
            )
            or not torch.equal(
                output.segments[1].lengths,
                expected_final - expected_retained,
            )
        ):
            raise RuntimeError("D3 compiled segment lengths differ")
    elif len(output.segments) != 1 or not torch.equal(
        output.segments[0].lengths,
        expected_final,
    ):
        raise RuntimeError("D3 exact segment lengths differ")


def _sample_is_finite(output: PublishedOutput) -> bool:
    for segment in output.segments:
        for value in (segment.k, segment.v):
            flattened = value.flatten()
            sample = torch.cat(
                (
                    flattened[: min(flattened.numel(), 512)],
                    flattened[max(flattened.numel() - 512, 0) :],
                )
            )
            if not bool(torch.isfinite(sample).all()):
                return False
    return bool(torch.isfinite(output.last_hidden).all())


def _materialize_sources(
    groups: tuple[SelectedGroup, ...],
    rank: int,
    actions_by_id: dict[int, D2ActionRecord],
    source_window,
    action_plan: D2ActionPlan,
    source_model,
    device: torch.device,
) -> tuple[
    dict[int, JaggedMigratedKVBatch],
    list[dict[str, object]],
]:
    sources = {}
    lookup_reports = []
    for group in groups:
        if group.pool != "compiled":
            continue
        actions = _actions_for_group(group, rank, actions_by_id)
        history = _history_batch(
            actions,
            source_window,
            (0,) * len(actions),
            tuple(value.old_tokens for value in actions),
            action_plan.source_version,
        ).to(device)
        exact = integrated_sharded_exact(
            source_model,
            history,
            action_plan.source_version,
        )
        lookup_reports.append(
            {
                "source_ordinal": group.source_ordinal,
                **exact.lookup_metrics.to_dict(),
            }
        )
        if exact.fragment is not None:
            retained = slice_integrated_jagged_ranges(
                exact.fragment,
                tuple(value.retained_start for value in actions),
                tuple(value.old_tokens for value in actions),
            )
            source = retained.to("cpu")
            _assert_pageable(source)
            sources[group.source_ordinal] = source
        del history
        del exact
    torch.cuda.synchronize(device)
    return sources, lookup_reports


def _run_s0(
    groups: tuple[SelectedGroup, ...],
    rank: int,
    actions_by_id: dict[int, D2ActionRecord],
    target_histories: dict[int, RawHistoryBatch],
    sources: dict[int, JaggedMigratedKVBatch],
    target_model,
    operator: DirectOldKVFusedOperator,
    program,
    target_version: str,
    slot: PinnedSlot,
    device: torch.device,
) -> tuple[dict[str, object], dict[int, PublishedOutput]]:
    phase_seconds = {
        "pageable_to_pinned": 0.0,
        "h2d": 0.0,
        "d2_compute_collective_and_wait": 0.0,
        "d2h": 0.0,
        "pinned_to_pageable": 0.0,
    }
    lookup_collective_seconds = 0.0
    lookup_reports = []
    outputs = {}
    group_reports = []
    h2d_bytes = 0
    d2h_bytes = 0
    h2d_stream = torch.cuda.Stream(device=device)
    d2h_stream = torch.cuda.Stream(device=device)
    torch.cuda.reset_peak_memory_stats(device)
    baseline_allocated = torch.cuda.memory_allocated(device)
    wall_started = time.perf_counter()
    for group in groups:
        actions = _actions_for_group(group, rank, actions_by_id)
        source_pageable = sources.get(group.source_ordinal)
        history_pageable = target_histories[group.source_ordinal]
        group_h2d_bytes = history_pageable.nbytes + (
            0 if source_pageable is None else source_pageable.nbytes
        )
        h2d_bytes += group_h2d_bytes
        stage_started = time.perf_counter()
        source_pinned = _stage_source(slot, source_pageable)
        history_pinned = _stage_history(slot, history_pageable)
        stage_seconds = time.perf_counter() - stage_started
        phase_seconds["pageable_to_pinned"] += stage_seconds
        h2d_start = torch.cuda.Event(enable_timing=True)
        h2d_end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(h2d_stream):
            h2d_start.record(h2d_stream)
            source_device = (
                None
                if source_pinned is None
                else source_pinned.to(device, non_blocking=True)
            )
            history_device = history_pinned.to(
                device,
                non_blocking=True,
            )
            h2d_end.record(h2d_stream)
        h2d_end.synchronize()
        h2d_seconds = h2d_start.elapsed_time(h2d_end) / 1000.0
        phase_seconds["h2d"] += h2d_seconds
        compute_start = torch.cuda.Event(enable_timing=True)
        compute_end = torch.cuda.Event(enable_timing=True)
        compute_start.record()
        retained = None
        if group.pool == "compiled":
            retained = (
                None
                if source_device is None
                else operator.execute_into(
                    program,
                    source_device,
                    _direct_destination(source_device, target_version),
                )
            )
            result = integrated_sharded_append_only(
                target_model,
                retained,
                history_device,
                target_version,
            )
        else:
            result = integrated_sharded_exact(
                target_model,
                history_device,
                target_version,
            )
        compute_end.record()
        compute_end.synchronize()
        compute_seconds = (
            compute_start.elapsed_time(compute_end) / 1000.0
        )
        phase_seconds["d2_compute_collective_and_wait"] += (
            compute_seconds
        )
        group_collective_seconds = (
            result.lookup_metrics.collective_seconds
        )
        lookup_collective_seconds += group_collective_seconds
        lookup_reports.append(
            {
                "source_ordinal": group.source_ordinal,
                "pool": group.pool,
                **result.lookup_metrics.to_dict(),
            }
        )
        if result.fragment is None:
            if actions:
                raise RuntimeError("D3 local work produced no output")
            group_reports.append(
                {
                    "source_ordinal": group.source_ordinal,
                    "pool": group.pool,
                    "records": 0,
                    "pageable_to_pinned_seconds": stage_seconds,
                    "h2d_seconds": h2d_seconds,
                    "d2_compute_collective_and_wait_seconds": (
                        compute_seconds
                    ),
                    "lookup_collective_seconds": (
                        group_collective_seconds
                    ),
                    "d2h_seconds": 0.0,
                    "pinned_to_pageable_seconds": 0.0,
                    "h2d_bytes": group_h2d_bytes,
                    "d2h_bytes": 0,
                }
            )
            del source_device
            del history_device
            del result
            del retained
            continue
        d2h_start = torch.cuda.Event(enable_timing=True)
        d2h_end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(d2h_stream):
            d2h_stream.wait_event(compute_end)
            d2h_start.record(d2h_stream)
            pinned_segments, pinned_hidden = _copy_output_to_slot(
                slot,
                result.fragment,
                result.last_hidden,
            )
            d2h_end.record(d2h_stream)
        d2h_end.synchronize()
        d2h_seconds = d2h_start.elapsed_time(d2h_end) / 1000.0
        phase_seconds["d2h"] += d2h_seconds
        publish_started = time.perf_counter()
        published = _publish_output(pinned_segments, pinned_hidden)
        publish_seconds = time.perf_counter() - publish_started
        phase_seconds["pinned_to_pageable"] += publish_seconds
        d2h_bytes += published.nbytes
        _validate_published_action_lengths(
            published,
            actions,
            group.pool,
        )
        if not _sample_is_finite(published):
            raise RuntimeError("D3 published output differs")
        outputs[group.source_ordinal] = published
        group_reports.append(
            {
                "source_ordinal": group.source_ordinal,
                "pool": group.pool,
                "records": len(actions),
                "pageable_to_pinned_seconds": stage_seconds,
                "h2d_seconds": h2d_seconds,
                "d2_compute_collective_and_wait_seconds": (
                    compute_seconds
                ),
                "lookup_collective_seconds": (
                    group_collective_seconds
                ),
                "d2h_seconds": d2h_seconds,
                "pinned_to_pageable_seconds": publish_seconds,
                "h2d_bytes": group_h2d_bytes,
                "d2h_bytes": published.nbytes,
                "published_bytes": published.nbytes,
            }
        )
        del source_device
        del history_device
        del result
        del retained
        del pinned_segments
        del pinned_hidden
    torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - wall_started
    report = {
        "phase_seconds": phase_seconds,
        "lookup_collective_seconds": lookup_collective_seconds,
        "d2_compute_excluding_lookup_collective_estimate_seconds": (
            max(
                phase_seconds["d2_compute_collective_and_wait"]
                - lookup_collective_seconds,
                0.0,
            )
        ),
        "wall_seconds": wall_seconds,
        "baseline_hbm_allocated_bytes": baseline_allocated,
        "peak_hbm_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_hbm_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "pinned_slot_bytes": slot.nbytes,
        "h2d_bytes": h2d_bytes,
        "d2h_bytes": d2h_bytes,
        "published_bytes": sum(value.nbytes for value in outputs.values()),
        "group_reports": group_reports,
        "lookup_reports": lookup_reports,
    }
    return report, outputs


def run(args: argparse.Namespace) -> dict[str, object] | None:
    runtime = init_d2_distributed_runtime(
        timeout_seconds=args.timeout_seconds,
    )
    try:
        if (
            runtime.world_size != 2
            or runtime.backend != "nccl"
            or runtime.device.type != "cuda"
        ):
            raise RuntimeError(
                "D3 M0 requires two torchrun NCCL ranks"
            )
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        visible_tokens = tuple(
            value.strip()
            for value in visible_devices.split(",")
            if value.strip()
        )
        expected_tokens = tuple(
            value.strip()
            for value in args.expected_visible_devices.split(",")
            if value.strip()
        )
        if visible_tokens != expected_tokens or len(visible_tokens) != 2:
            raise RuntimeError(
                "D3 M0 physical GPU pair differs from the requested pair"
            )
        action_path = _path(args.action_plan)
        stage_a_path = _path(args.stage_a_summary)
        training_path = _path(args.training_result)
        checkpoint_dir = _path(args.checkpoint_dir)
        manifest_path = _path(args.work_manifest)
        group_path = _path(args.group_plan)
        action_plan = D2ActionPlan.load(action_path)
        manifest = D3WorkManifest.load(manifest_path)
        group_plan = D3GroupPlan.load(group_path)
        owner_map = build_d2_record_owner_map(
            action_plan,
            runtime.world_size,
            "strict_cow_lpt",
        )
        audit_d3_work_manifest(manifest, action_plan, owner_map)
        audit_d3_group_plan(manifest, group_plan)
        groups = _select_groups(
            group_plan,
            args.scope,
            args.canary_records_per_rank,
        )
        selected_ids = {
            record_id
            for group in groups
            for record_ids in group.record_ids_by_rank
            for record_id in record_ids
        }
        action_lookup = {
            value.record_id: value for value in action_plan.records
        }
        actions_by_id = {
            record_id: action_lookup[record_id]
            for record_id in selected_ids
        }
        local_actions = tuple(
            actions_by_id[record_id]
            for group in groups
            for record_id in group.record_ids_by_rank[runtime.rank]
        )
        stage_a = json.loads(stage_a_path.read_text())
        training = json.loads(training_path.read_text())
        if (
            stage_a["status"] != "complete"
            or stage_a["action_plan"]["content_sha256"]
            != action_plan.content_sha256
        ):
            raise ValueError("D3 Stage A binding differs")
        data_plan, prepared_metadata = load_prepared_kuairand_plan(
            _path(action_plan.provenance.prepared_data)
        )
        validate_long_context_plan(data_plan, prepared_metadata, 4)
        windows = reconstruct_organic_windows(
            data_plan,
            (value.prepared_user_id for value in local_actions),
        )
        source_window = windows[
            int(action_plan.source_version.removeprefix("theta"))
        ]
        target_window = windows[
            int(action_plan.target_version.removeprefix("theta"))
        ]
        for action in local_actions:
            source_record = source_window.records[action.prepared_user_id]
            target_record = target_window.records[action.prepared_user_id]
            if (
                source_record.history is None
                or source_record.history_sha256
                != action.old_history_sha256
                or len(source_record.history) != action.old_tokens
                or target_record.history is None
                or target_record.history_sha256
                != action.target_history_sha256
                or len(target_record.history) != action.final_tokens
            ):
                raise ValueError("D3 reconstructed history differs")
        cfg = HSTUConfig(**training["model"])
        target_histories = {}
        for group in groups:
            actions = _actions_for_group(
                group,
                runtime.rank,
                actions_by_id,
            )
            starts = tuple(
                value.delta_start if group.pool == "compiled" else 0
                for value in actions
            )
            target_histories[group.source_ordinal] = _history_batch(
                actions,
                target_window,
                starts,
                tuple(value.final_tokens for value in actions),
                action_plan.target_version,
            )
        source_setup_started = time.perf_counter()
        source_cpu = _load_cpu_model(
            cfg,
            _checkpoint_path(
                checkpoint_dir,
                action_plan.source_version,
            ),
        )
        source_model = build_modulo_sharded_hstu_from_cpu(
            source_cpu,
            runtime.rank,
            runtime.world_size,
            runtime.device,
        )
        del source_cpu
        dist.barrier()
        sources, source_lookups = _materialize_sources(
            groups,
            runtime.rank,
            actions_by_id,
            source_window,
            action_plan,
            source_model,
            runtime.device,
        )
        source_setup_seconds = time.perf_counter() - source_setup_started
        del source_model
        gc.collect()
        torch.cuda.empty_cache()
        target_setup_started = time.perf_counter()
        target_cpu = _load_cpu_model(
            cfg,
            _checkpoint_path(
                checkpoint_dir,
                action_plan.target_version,
            ),
        )
        target_model = build_modulo_sharded_hstu_from_cpu(
            target_cpu,
            runtime.rank,
            runtime.world_size,
            runtime.device,
        )
        del target_cpu
        descriptor = _program_descriptor(stage_a)
        program_cpu, loaded_program = load_direct_oldkv_program(
            _path(descriptor["path"]),
            expected_sha256=descriptor["sha256"],
            expected_source_version=action_plan.source_version,
            expected_target_version=action_plan.target_version,
            expected_num_layers=cfg.num_layers,
            expected_kv_width=cfg.hidden_size,
        )
        operator = DirectOldKVFusedOperator()
        program = operator.prepare_program(
            program_cpu,
            runtime.device,
        )
        del program_cpu
        requirements = _slot_requirements(
            groups,
            runtime.rank,
            actions_by_id,
        )
        slot = _allocate_slot(cfg, requirements)
        target_setup_seconds = time.perf_counter() - target_setup_started
        dist.barrier()
        local_s0, outputs = _run_s0(
            groups,
            runtime.rank,
            actions_by_id,
            target_histories,
            sources,
            target_model,
            operator,
            program,
            action_plan.target_version,
            slot,
            runtime.device,
        )
        expected_local_ids = {
            value.record_id for value in local_actions
        }
        observed_local_ids = {
            record_id
            for output in outputs.values()
            for record_id in output.record_ids
        }
        local_report = {
            "rank": runtime.rank,
            "local_rank": runtime.local_rank,
            "physical_visible_token": visible_tokens[runtime.local_rank],
            "device_name": torch.cuda.get_device_name(runtime.device),
            "device_uuid": (
                f"GPU-{torch.cuda.get_device_properties(runtime.device).uuid}"
            ),
            "records": len(local_actions),
            "record_ids": sorted(expected_local_ids),
            "pools": {
                pool: sum(
                    group.pool == pool
                    for group in groups
                    for _ in group.record_ids_by_rank[runtime.rank]
                )
                for pool in ("compiled", "exact")
            },
            "source_materialized_bytes": sum(
                value.nbytes for value in sources.values()
            ),
            "source_setup_seconds": source_setup_seconds,
            "target_setup_seconds": target_setup_seconds,
            "slot_requirements": requirements,
            "source_lookup_reports": source_lookups,
            "s0": local_s0,
            "exactly_once_pass": (
                expected_local_ids == observed_local_ids
                and sum(
                    len(output.record_ids)
                    for output in outputs.values()
                )
                == len(expected_local_ids)
            ),
        }
        gathered: list[object] = [None] * runtime.world_size
        dist.all_gather_object(gathered, local_report)
        if not runtime.is_primary:
            return None
        rank_reports = [dict(value) for value in gathered]
        complete = all(
            value["exactly_once_pass"] for value in rank_reports
        )
        report = {
            "protocol": PROTOCOL,
            "status": "complete" if complete else "failed",
            "scientific_result": False,
            "formal_design3": False,
            "capacity_emulation": True,
            "physical_oversubscription": False,
            "group_budget_kind": "logical_payload_estimate",
            "scope": args.scope,
            "stack_revision": manifest.stack_revision,
            "benchmark_role": (
                "two_gpu_pageable_dram_sequential_foundation"
            ),
            "timing_role": (
                "single_pass_development_profile_not_a_comparison"
            ),
            "execution": {
                "world_size": runtime.world_size,
                "logical_gpu_pair": [0, 1],
                "physical_visible_devices": list(visible_tokens),
                "transport": (
                    "pageable_dram_to_reusable_pinned_slot_to_hbm_"
                    "compute_to_pinned_slot_to_pageable_dram"
                ),
                "buffer_depth": 1,
                "group_order": "sequential_route_pure",
                "source_store_scope": (
                    "action_required_compiled_retained_oldkv"
                ),
                "capacity_accounting": (
                    "full_committed_oldkv_from_manifest"
                ),
                "primary_timer": (
                    "pageable_to_pinned_plus_h2d_plus_d2_compute_plus_"
                    "d2h_plus_pinned_to_pageable"
                ),
                "excluded_setup": (
                    "history_reconstruction_source_fixture_"
                    "materialization_model_program_and_slot_setup"
                ),
                "warmup": 0,
                "repeats": 1,
            },
            "bindings": {
                "action_plan": str(action_path),
                "action_plan_sha256": action_plan.content_sha256,
                "work_manifest": str(manifest_path),
                "work_manifest_sha256": manifest.dev_sha256,
                "group_plan": str(group_path),
                "group_plan_sha256": group_plan.dev_sha256,
                "loaded_program": loaded_program,
            },
            "selected_groups": [
                {
                    "source_ordinal": value.source_ordinal,
                    "pool": value.pool,
                    "record_ids_by_rank": [
                        list(record_ids)
                        for record_ids in value.record_ids_by_rank
                    ],
                }
                for value in groups
            ],
            "rank_reports": rank_reports,
            "makespan_seconds": max(
                value["s0"]["wall_seconds"]
                for value in rank_reports
            ),
            "aggregate_published_bytes": sum(
                value["s0"]["published_bytes"]
                for value in rank_reports
            ),
            "exactly_once_pass": complete,
            "next": (
                (
                    "run the full S0 boundary, then add S1"
                    if args.scope == "canary"
                    else "add a same-revision S1 double buffer"
                )
                + " without freezing the final D3 planner"
            ),
        }
        output = _path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(canonical_json_bytes(report))
        return report
    finally:
        close_d2_distributed_runtime(runtime)


def main(argv: list[str] | None = None) -> None:
    report = run(parse_args(argv))
    if report is not None:
        print(
            f"status={report['status']} scope={report['scope']} "
            f"makespan={report['makespan_seconds']:.6f}s"
        )


if __name__ == "__main__":
    main()
