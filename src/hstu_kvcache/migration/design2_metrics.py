from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass

import torch
from torch import nn

from .design2_plan import (
    D2ActionPlan,
    build_d2_record_owner_map,
    d2_record_owner_map_sha256,
)

D2_PHASE_LEDGER_PROTOCOL = "cohortkv_d2_phase_ledger_v1"
D2_REQUEST_CHARACTERIZATION_PROTOCOL = (
    "cohortkv_d2_stage_a_request_characterization_v1"
)
D2_CAPACITY_CHARACTERIZATION_PROTOCOL = (
    "cohortkv_d2_stage_a_capacity_characterization_v1"
)


@dataclass(frozen=True)
class D2LookupObservation:
    phase: str
    lookup_calls: int
    padded_lookup_elements: int
    nonpadding_lookup_elements: int
    logical_lookup_tokens: int

    def __post_init__(self) -> None:
        if (
            not self.phase
            or min(
                self.lookup_calls,
                self.padded_lookup_elements,
                self.nonpadding_lookup_elements,
                self.logical_lookup_tokens,
            )
            < 0
            or self.nonpadding_lookup_elements
            > self.padded_lookup_elements
            or (
                self.lookup_calls == 0
                and (
                    self.padded_lookup_elements != 0
                    or self.nonpadding_lookup_elements != 0
                )
            )
        ):
            raise ValueError("D2 lookup observation is invalid")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class D2EmbeddingLookupCounter:
    def __init__(self, embedding: nn.Module) -> None:
        self._embedding = embedding
        self._active_phase: str | None = None
        self._closed = False
        self._values: dict[str, list[int]] = {}
        self._handle = embedding.register_forward_pre_hook(
            self._observe
        )

    def _observe(
        self,
        module: nn.Module,
        inputs: tuple[object, ...],
    ) -> None:
        if module is not self._embedding or self._active_phase is None:
            raise RuntimeError("embedding lookup occurred outside a D2 phase")
        if len(inputs) != 1 or not isinstance(inputs[0], torch.Tensor):
            raise RuntimeError("D2 embedding lookup input is invalid")
        item_ids = inputs[0]
        values = self._values.setdefault(
            self._active_phase,
            [0, 0, 0, 0],
        )
        values[0] += 1
        values[1] += int(item_ids.numel())
        values[2] += int(torch.count_nonzero(item_ids).item())

    @contextmanager
    def phase(
        self,
        phase: str,
        logical_lookup_tokens: int,
    ) -> Iterator[None]:
        if (
            self._closed
            or self._active_phase is not None
            or not phase
            or logical_lookup_tokens < 0
        ):
            raise RuntimeError("D2 lookup phase transition is invalid")
        self._active_phase = phase
        values = self._values.setdefault(phase, [0, 0, 0, 0])
        try:
            yield
        finally:
            values[3] += logical_lookup_tokens
            self._active_phase = None

    def observations(self) -> tuple[D2LookupObservation, ...]:
        if self._active_phase is not None:
            raise RuntimeError("D2 lookup phase is still active")
        return tuple(
            D2LookupObservation(
                phase=phase,
                lookup_calls=values[0],
                padded_lookup_elements=values[1],
                nonpadding_lookup_elements=values[2],
                logical_lookup_tokens=values[3],
            )
            for phase, values in self._values.items()
        )

    def close(self) -> None:
        if self._active_phase is not None:
            raise RuntimeError("cannot close an active D2 lookup phase")
        if not self._closed:
            self._handle.remove()
            self._closed = True

    def __enter__(self) -> D2EmbeddingLookupCounter:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


@dataclass(frozen=True)
class D2PhaseLedgerEntry:
    phase: str
    records: int
    compute_tokens: int
    lookup_tokens: int
    embedding_id_bytes: int
    embedding_vector_bytes: Mapping[str, int]
    padded_lookup_elements: int | None = None
    unique_ids: int | None = None
    local_ids: int | None = None
    remote_ids: int | None = None
    physical_collective_bytes: int | None = None
    lookup_calls: int | None = None
    collective_calls: int | None = None

    def __post_init__(self) -> None:
        optional_counts = (
            self.padded_lookup_elements,
            self.unique_ids,
            self.local_ids,
            self.remote_ids,
            self.physical_collective_bytes,
            self.lookup_calls,
            self.collective_calls,
        )
        if (
            not self.phase
            or min(
                self.records,
                self.compute_tokens,
                self.lookup_tokens,
                self.embedding_id_bytes,
            )
            < 0
            or self.lookup_tokens > self.compute_tokens
            or any(value < 0 for value in self.embedding_vector_bytes.values())
            or any(
                value is not None and value < 0
                for value in optional_counts
            )
        ):
            raise ValueError("D2 phase ledger entry is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "embedding_vector_bytes": dict(self.embedding_vector_bytes),
        }


@dataclass(frozen=True)
class D2PhaseLedger:
    action_plan_sha256: str
    embedding_dim: int
    embedding_id_bytes: int
    transport_element_bytes: Mapping[str, int]
    mixed: tuple[D2PhaseLedgerEntry, ...]
    all_exact: tuple[D2PhaseLedgerEntry, ...]
    boundaries: Mapping[str, object]
    checks: Mapping[str, bool]
    protocol: str = D2_PHASE_LEDGER_PROTOCOL

    def __post_init__(self) -> None:
        if (
            self.protocol != D2_PHASE_LEDGER_PROTOCOL
            or len(self.action_plan_sha256) != 64
            or self.embedding_dim < 1
            or self.embedding_id_bytes < 1
            or not self.transport_element_bytes
            or any(value < 1 for value in self.transport_element_bytes.values())
            or not self.mixed
            or not self.all_exact
            or not self.boundaries
            or not self.checks
            or not all(self.checks.values())
        ):
            raise ValueError("D2 phase ledger is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol": self.protocol,
            "action_plan_sha256": self.action_plan_sha256,
            "embedding_dim": self.embedding_dim,
            "embedding_id_bytes": self.embedding_id_bytes,
            "transport_element_bytes": dict(
                self.transport_element_bytes
            ),
            "mixed": [value.to_dict() for value in self.mixed],
            "all_exact": [
                value.to_dict() for value in self.all_exact
            ],
            "boundaries": dict(self.boundaries),
            "checks": dict(self.checks),
        }


def _entry(
    phase: str,
    records: int,
    compute_tokens: int,
    lookup_tokens: int,
    embedding_dim: int,
    embedding_id_bytes: int,
    transport_element_bytes: Mapping[str, int],
) -> D2PhaseLedgerEntry:
    return D2PhaseLedgerEntry(
        phase=phase,
        records=records,
        compute_tokens=compute_tokens,
        lookup_tokens=lookup_tokens,
        embedding_id_bytes=lookup_tokens * embedding_id_bytes,
        embedding_vector_bytes={
            name: lookup_tokens * embedding_dim * element_bytes
            for name, element_bytes in transport_element_bytes.items()
        },
    )


def build_d2_phase_ledger(
    plan: D2ActionPlan,
    embedding_dim: int,
    embedding_id_bytes: int = 8,
    transport_element_bytes: Mapping[str, int] | None = None,
) -> D2PhaseLedger:
    if transport_element_bytes is None:
        transport_element_bytes = {
            "fp32_current_exact": 4,
            "bf16_candidate": 2,
            "fp16_candidate": 2,
        }
    compiled = tuple(
        value
        for value in plan.records
        if value.requested_reason == "migrate"
    )
    scheduled = tuple(
        value
        for value in plan.records
        if value.requested_reason == "scheduled_exact"
    )
    natural = tuple(
        value
        for value in plan.records
        if value.requested_reason == "natural_exact"
    )
    reusable = compiled + scheduled

    def total(records, field: str) -> int:
        return sum(int(getattr(value, field)) for value in records)

    compiled_retained = total(compiled, "retained_tokens")
    scheduled_retained = total(scheduled, "retained_tokens")
    reusable_retained = total(reusable, "retained_tokens")
    natural_prefix = total(natural, "target_prefix_tokens")
    reusable_delta = total(reusable, "delta_tokens")
    latest = total(plan.records, "latest_tokens")
    mixed = (
        _entry(
            "compiled_retained",
            len(compiled),
            compiled_retained,
            0,
            embedding_dim,
            embedding_id_bytes,
            transport_element_bytes,
        ),
        _entry(
            "scheduled_exact_retained",
            len(scheduled),
            scheduled_retained,
            scheduled_retained,
            embedding_dim,
            embedding_id_bytes,
            transport_element_bytes,
        ),
        _entry(
            "natural_exact_target_prefix",
            len(natural),
            natural_prefix,
            natural_prefix,
            embedding_dim,
            embedding_id_bytes,
            transport_element_bytes,
        ),
        _entry(
            "delta_append",
            len(reusable),
            reusable_delta,
            reusable_delta,
            embedding_dim,
            embedding_id_bytes,
            transport_element_bytes,
        ),
        _entry(
            "latest_append",
            len(plan.records),
            latest,
            latest,
            embedding_dim,
            embedding_id_bytes,
            transport_element_bytes,
        ),
    )
    all_exact = (
        _entry(
            "all_exact_retained",
            len(reusable),
            reusable_retained,
            reusable_retained,
            embedding_dim,
            embedding_id_bytes,
            transport_element_bytes,
        ),
        _entry(
            "natural_exact_target_prefix",
            len(natural),
            natural_prefix,
            natural_prefix,
            embedding_dim,
            embedding_id_bytes,
            transport_element_bytes,
        ),
        _entry(
            "delta_append",
            len(reusable),
            reusable_delta,
            reusable_delta,
            embedding_dim,
            embedding_id_bytes,
            transport_element_bytes,
        ),
        _entry(
            "latest_append",
            len(plan.records),
            latest,
            latest,
            embedding_dim,
            embedding_id_bytes,
            transport_element_bytes,
        ),
    )
    mixed_lookup = sum(value.lookup_tokens for value in mixed)
    exact_lookup = sum(value.lookup_tokens for value in all_exact)
    append_lookup = reusable_delta + latest
    boundaries = {
        "retained_prefix": {
            "mixed_lookup_tokens": scheduled_retained,
            "all_exact_lookup_tokens": reusable_retained,
            "lookup_reduction": (
                float(reusable_retained / scheduled_retained)
                if scheduled_retained
                else None
            ),
        },
        "integrated_post_append": {
            "mixed_lookup_tokens": mixed_lookup,
            "all_exact_lookup_tokens": exact_lookup,
            "lookup_reduction": (
                float(exact_lookup / mixed_lookup)
                if mixed_lookup
                else None
            ),
        },
        "method_independent_append": {
            "delta_lookup_tokens": reusable_delta,
            "latest_lookup_tokens": latest,
            "lookup_tokens": append_lookup,
        },
    }
    checks = {
        "record_counts_match": (
            len(compiled) == plan.counts.compiled
            and len(scheduled) == plan.counts.scheduled_exact
            and len(natural) == plan.counts.natural_exact
        ),
        "compiled_retained_lookup_zero": mixed[0].lookup_tokens == 0,
        "scheduled_retained_only_lookup": (
            boundaries["retained_prefix"]["mixed_lookup_tokens"]
            == scheduled_retained
        ),
        "append_is_method_independent": (
            sum(
                value.lookup_tokens
                for value in mixed
                if value.phase in {"delta_append", "latest_append"}
            )
            == sum(
                value.lookup_tokens
                for value in all_exact
                if value.phase in {"delta_append", "latest_append"}
            )
        ),
        "mixed_full_boundary_closes": (
            mixed_lookup
            == scheduled_retained
            + natural_prefix
            + reusable_delta
            + latest
        ),
        "all_exact_full_boundary_closes": (
            exact_lookup
            == reusable_retained
            + natural_prefix
            + reusable_delta
            + latest
        ),
    }
    return D2PhaseLedger(
        action_plan_sha256=plan.content_sha256,
        embedding_dim=embedding_dim,
        embedding_id_bytes=embedding_id_bytes,
        transport_element_bytes=transport_element_bytes,
        mixed=mixed,
        all_exact=all_exact,
        boundaries=boundaries,
        checks=checks,
    )


def _request_scope(
    item_ids: Sequence[int],
    embedding_dim: int,
    embedding_id_bytes: int,
    transport_element_bytes: Mapping[str, int],
) -> dict[str, object]:
    requested = len(item_ids)
    unique = len(set(item_ids))
    return {
        "requested_ids": requested,
        "unique_ids": unique,
        "unique_over_requested": (
            float(unique / requested) if requested else None
        ),
        "maximum_dedup_fraction": (
            float(1.0 - unique / requested) if requested else None
        ),
        "request_id_bytes": requested * embedding_id_bytes,
        "return_vector_bytes_without_dedup": {
            name: requested * embedding_dim * element_bytes
            for name, element_bytes in transport_element_bytes.items()
        },
        "return_vector_bytes_at_global_unique_floor": {
            name: unique * embedding_dim * element_bytes
            for name, element_bytes in transport_element_bytes.items()
        },
    }


def characterize_d2_requests(
    plan: D2ActionPlan,
    target_item_ids: Mapping[int, Sequence[int]],
    embedding_dim: int,
    embedding_id_bytes: int = 8,
    transport_element_bytes: Mapping[str, int] | None = None,
) -> dict[str, object]:
    if transport_element_bytes is None:
        transport_element_bytes = {
            "fp32_current_exact": 4,
            "bf16_candidate": 2,
            "fp16_candidate": 2,
        }
    expected_ids = {value.record_id for value in plan.records}
    if set(target_item_ids) != expected_ids:
        raise ValueError("D2 request histories do not cover the action plan")
    mixed: dict[str, list[int]] = {
        "compiled_retained": [],
        "scheduled_exact_retained": [],
        "natural_exact_target_prefix": [],
        "delta_append": [],
        "latest_append": [],
    }
    all_exact: dict[str, list[int]] = {
        "all_exact_retained": [],
        "natural_exact_target_prefix": [],
        "delta_append": [],
        "latest_append": [],
    }
    for record in plan.records:
        item_ids = tuple(
            int(value) for value in target_item_ids[record.record_id]
        )
        if (
            len(item_ids) != record.final_tokens
            or any(value < 0 for value in item_ids)
        ):
            raise ValueError("D2 request history differs from action plan")
        retained = item_ids[: record.retained_tokens]
        delta = item_ids[
            record.delta_start : record.target_prefix_tokens
        ]
        latest = item_ids[
            record.target_prefix_tokens : record.final_tokens
        ]
        if record.requested_reason == "migrate":
            mixed["compiled_retained"].extend(())
            all_exact["all_exact_retained"].extend(retained)
            mixed["delta_append"].extend(delta)
            all_exact["delta_append"].extend(delta)
        elif record.requested_reason == "scheduled_exact":
            mixed["scheduled_exact_retained"].extend(retained)
            all_exact["all_exact_retained"].extend(retained)
            mixed["delta_append"].extend(delta)
            all_exact["delta_append"].extend(delta)
        else:
            prefix = item_ids[: record.target_prefix_tokens]
            mixed["natural_exact_target_prefix"].extend(prefix)
            all_exact["natural_exact_target_prefix"].extend(prefix)
        mixed["latest_append"].extend(latest)
        all_exact["latest_append"].extend(latest)

    def characterize_branch(
        phases: Mapping[str, Sequence[int]],
    ) -> dict[str, object]:
        phase_values = {
            name: _request_scope(
                values,
                embedding_dim,
                embedding_id_bytes,
                transport_element_bytes,
            )
            for name, values in phases.items()
        }
        concatenated = [
            item_id for values in phases.values() for item_id in values
        ]
        phase_local_unique = sum(
            int(value["unique_ids"]) for value in phase_values.values()
        )
        return {
            "phases": phase_values,
            "full_wave": {
                **_request_scope(
                    concatenated,
                    embedding_dim,
                    embedding_id_bytes,
                    transport_element_bytes,
                ),
                "phase_local_unique_sum": phase_local_unique,
                "cross_phase_reuse_ids": (
                    phase_local_unique - len(set(concatenated))
                ),
                "global_unique_is_ceiling_only": True,
            },
        }

    exact_prefix_ids = (
        mixed["scheduled_exact_retained"]
        + mixed["natural_exact_target_prefix"]
    )
    append_ids = mixed["delta_append"] + mixed["latest_append"]
    return {
        "protocol": D2_REQUEST_CHARACTERIZATION_PROTOCOL,
        "scientific_result": False,
        "action_plan_sha256": plan.content_sha256,
        "embedding_dim": embedding_dim,
        "embedding_id_bytes": embedding_id_bytes,
        "transport_element_bytes": dict(transport_element_bytes),
        "mixed": characterize_branch(mixed),
        "all_exact": characterize_branch(all_exact),
        "coalescing_ceilings": {
            "exact_retained_or_natural": {
                **_request_scope(
                    exact_prefix_ids,
                    embedding_dim,
                    embedding_id_bytes,
                    transport_element_bytes,
                ),
                "logical_subphases": [
                    "scheduled_exact_retained",
                    "natural_exact_target_prefix",
                ],
            },
            "delta_plus_latest_append_candidate": {
                **_request_scope(
                    append_ids,
                    embedding_dim,
                    embedding_id_bytes,
                    transport_element_bytes,
                ),
                "logical_subphases": [
                    "delta_append",
                    "latest_append",
                ],
            },
        },
        "scope": {
            "request_multiset_known_before_wave": True,
            "unique_counts_are_global_static_ceilings": True,
            "actual_remote_fraction_measured": False,
            "actual_collective_bytes_measured": False,
            "transport_dtype_selected": False,
        },
    }


def characterize_d2_capacity(
    plan: D2ActionPlan,
    model_bytes: int,
    item_embedding_bytes: int,
    program_bytes: int,
    capacity_bytes: int,
    world_sizes: Sequence[int] = (1, 2, 4),
    owner_strategies: Sequence[str] = (
        "modulo",
        "old_kv_lpt",
        "strict_cow_lpt",
    ),
    num_layers: int = 16,
    kv_width: int = 512,
    kv_element_bytes: int = 2,
    transient_extent_multiplier: int = 8,
    allocator_context_margin_bytes: int = 2 * 1024**3,
) -> dict[str, object]:
    if (
        min(
            model_bytes,
            item_embedding_bytes,
            program_bytes,
            capacity_bytes,
            num_layers,
            kv_width,
            kv_element_bytes,
            transient_extent_multiplier,
            allocator_context_margin_bytes,
        )
        < 0
        or item_embedding_bytes > model_bytes
        or capacity_bytes < 1
        or not world_sizes
        or any(value < 1 for value in world_sizes)
        or not owner_strategies
    ):
        raise ValueError("D2 capacity characterization input is invalid")
    bytes_per_token = (
        num_layers * kv_width * 2 * kv_element_bytes
    )
    old_by_record = {
        value.record_id: value.old_tokens * bytes_per_token
        for value in plan.records
    }
    new_by_record = {
        value.record_id: value.final_tokens * bytes_per_token
        for value in plan.records
    }
    old_total = sum(old_by_record.values())
    new_total = sum(new_by_record.values())
    maximum_extent = max(
        max(old_by_record.values()),
        max(new_by_record.values()),
    )
    transient_bytes = maximum_extent * transient_extent_multiplier
    dense_model_bytes = model_bytes - item_embedding_bytes
    layouts = []
    owner_maps = {}
    for world_size in world_sizes:
        for strategy in owner_strategies:
            owner_map = build_d2_record_owner_map(
                plan,
                world_size,
                strategy,
            )
            map_key = f"{strategy}_w{world_size}"
            owner_maps[map_key] = {
                "sha256": d2_record_owner_map_sha256(owner_map),
                "record_owner_map": {
                    str(record_id): owner
                    for record_id, owner in owner_map.items()
                },
            }
            rank_values = []
            for rank in range(world_size):
                record_ids = tuple(
                    record_id
                    for record_id, owner in owner_map.items()
                    if owner == rank
                )
                old_bytes = sum(
                    old_by_record[value] for value in record_ids
                )
                new_bytes = sum(
                    new_by_record[value] for value in record_ids
                )
                full_model_required = (
                    old_bytes
                    + new_bytes
                    + model_bytes
                    + program_bytes
                    + transient_bytes
                    + allocator_context_margin_bytes
                )
                estimated_sharded_model_bytes = (
                    dense_model_bytes
                    + (
                        item_embedding_bytes
                        + world_size
                        - 1
                    )
                    // world_size
                )
                sharded_required = (
                    old_bytes
                    + new_bytes
                    + estimated_sharded_model_bytes
                    + program_bytes
                    + transient_bytes
                    + allocator_context_margin_bytes
                )
                rank_values.append(
                    {
                        "rank": rank,
                        "records": len(record_ids),
                        "old_kv_bytes": old_bytes,
                        "complete_new_kv_bytes": new_bytes,
                        "strict_cow_kv_bytes": old_bytes
                        + new_bytes,
                        "measured_full_model_replica_bytes": (
                            model_bytes
                        ),
                        "estimated_sharded_embedding_dense_bytes": (
                            estimated_sharded_model_bytes
                        ),
                        "program_bytes": program_bytes,
                        "transient_bytes": transient_bytes,
                        "allocator_context_margin_bytes": (
                            allocator_context_margin_bytes
                        ),
                        "full_model_required_bytes": (
                            full_model_required
                        ),
                        "full_model_total_capacity_admitted": (
                            full_model_required <= capacity_bytes
                        ),
                        "estimated_sharded_required_bytes": (
                            sharded_required
                        ),
                        "estimated_sharded_total_capacity_admitted": (
                            sharded_required <= capacity_bytes
                        ),
                    }
                )
            layouts.append(
                {
                    "world_size": world_size,
                    "owner_strategy": strategy,
                    "owner_map_sha256": owner_maps[map_key][
                        "sha256"
                    ],
                    "rank_values": rank_values,
                    "maximum_strict_cow_kv_bytes": max(
                        value["strict_cow_kv_bytes"]
                        for value in rank_values
                    ),
                    "all_full_model_total_capacity_admitted": all(
                        value[
                            "full_model_total_capacity_admitted"
                        ]
                        for value in rank_values
                    ),
                    "all_estimated_sharded_total_capacity_admitted": all(
                        value[
                            "estimated_sharded_total_capacity_admitted"
                        ]
                        for value in rank_values
                    ),
                }
            )
    return {
        "protocol": D2_CAPACITY_CHARACTERIZATION_PROTOCOL,
        "scientific_result": False,
        "action_plan_sha256": plan.content_sha256,
        "kv_layout": {
            "num_layers": num_layers,
            "kv_width_each": kv_width,
            "kv_tensors": 2,
            "element_bytes": kv_element_bytes,
            "bytes_per_token": bytes_per_token,
        },
        "standing_components": {
            "measured_full_model_replica_bytes": model_bytes,
            "measured_item_embedding_replica_bytes": (
                item_embedding_bytes
            ),
            "measured_dense_and_other_replica_bytes": (
                dense_model_bytes
            ),
            "measured_program_bytes": program_bytes,
            "transient_extent_multiplier": (
                transient_extent_multiplier
            ),
            "transient_bytes": transient_bytes,
            "allocator_context_margin_bytes": (
                allocator_context_margin_bytes
            ),
            "device_total_capacity_bytes": capacity_bytes,
        },
        "cohort": {
            "records": len(plan.records),
            "old_kv_bytes": old_total,
            "complete_new_kv_bytes": new_total,
            "strict_cow_kv_bytes": old_total + new_total,
            "maximum_record_extent_bytes": maximum_extent,
        },
        "layouts": layouts,
        "owner_maps": owner_maps,
        "scope": {
            "strict_cow": True,
            "actual_hbm_source_manifest_available": False,
            "owner_maps_are_deterministic_stage_a_candidates": True,
            "full_model_replica_is_measured_tensor_layout": True,
            "sharded_embedding_dense_is_estimated": True,
            "driver_context_bytes_in_margin_not_measured": True,
            "current_free_memory_used_for_admission": False,
        },
    }


def characterize_d2_scoped_dedup(
    plan: D2ActionPlan,
    target_item_ids: Mapping[int, Sequence[int]],
    num_embedding_rows: int,
    world_sizes: Sequence[int] = (1, 2, 4),
    batch_sizes: Sequence[int] = (
        1,
        4,
        8,
        16,
        32,
        64,
        128,
        682,
    ),
    owner_strategies: Sequence[str] = (
        "modulo",
        "old_kv_lpt",
    ),
    embedding_owner_rules: Sequence[str] = (
        "modulo",
        "contiguous",
    ),
    embedding_dim: int = 512,
    transport_element_bytes: int = 4,
) -> dict[str, object]:
    if (
        num_embedding_rows < 1
        or embedding_dim < 1
        or transport_element_bytes < 1
        or not world_sizes
        or not batch_sizes
        or any(value < 1 for value in world_sizes)
        or any(value < 1 for value in batch_sizes)
        or set(embedding_owner_rules)
        - {"modulo", "contiguous"}
    ):
        raise ValueError("D2 scoped dedup input is invalid")
    phases: dict[str, list[tuple[int, tuple[int, ...]]]] = {
        "scheduled_exact_retained": [],
        "natural_exact_target_prefix": [],
        "delta_append": [],
        "latest_append": [],
    }
    exact_prefix_records = []
    append_records = []
    for record in plan.records:
        item_ids = tuple(
            int(value) for value in target_item_ids[record.record_id]
        )
        if len(item_ids) != record.final_tokens:
            raise ValueError("D2 scoped dedup history differs")
        if record.requested_reason == "scheduled_exact":
            exact_values = item_ids[: record.retained_tokens]
            phases["scheduled_exact_retained"].append(
                (record.record_id, exact_values)
            )
            exact_prefix_records.append(
                (record.record_id, exact_values)
            )
        elif record.requested_reason == "natural_exact":
            exact_values = item_ids[: record.target_prefix_tokens]
            phases["natural_exact_target_prefix"].append(
                (record.record_id, exact_values)
            )
            exact_prefix_records.append(
                (record.record_id, exact_values)
            )
        append_values = ()
        if record.requested_reason != "natural_exact":
            delta_values = item_ids[
                record.delta_start : record.target_prefix_tokens
            ]
            phases["delta_append"].append(
                (record.record_id, delta_values)
            )
            append_values = delta_values
        latest_values = item_ids[
            record.target_prefix_tokens : record.final_tokens
        ]
        phases["latest_append"].append(
            (record.record_id, latest_values)
        )
        append_records.append(
            (
                record.record_id,
                append_values + latest_values,
            )
        )

    def embedding_owner(
        item_id: int,
        world_size: int,
        rule: str,
    ) -> int:
        if rule == "modulo":
            return item_id % world_size
        return min(
            item_id * world_size // num_embedding_rows,
            world_size - 1,
        )

    def scope_values(
        records: Sequence[tuple[int, tuple[int, ...]]],
        owner_map: Mapping[int, int],
        world_size: int,
        owner_rule: str,
        batch_size: int,
    ) -> dict[str, object]:
        requested = 0
        unique = 0
        remote_requested = 0
        remote_unique = 0
        requester_ranks: dict[int, set[int]] = {}
        for rank in range(world_size):
            rank_records = sorted(
                (
                    value
                    for value in records
                    if owner_map[value[0]] == rank
                ),
                key=lambda value: value[0],
            )
            for start in range(0, len(rank_records), batch_size):
                batch = rank_records[start : start + batch_size]
                batch_ids = [
                    item_id
                    for _, values in batch
                    for item_id in values
                ]
                remote = [
                    item_id
                    for item_id in batch_ids
                    if embedding_owner(
                        item_id,
                        world_size,
                        owner_rule,
                    )
                    != rank
                ]
                for item_id in set(batch_ids):
                    requester_ranks.setdefault(item_id, set()).add(rank)
                requested += len(batch_ids)
                unique += len(set(batch_ids))
                remote_requested += len(remote)
                remote_unique += len(set(remote))
        fanout = {}
        for ranks in requester_ranks.values():
            count = len(ranks)
            fanout[str(count)] = fanout.get(str(count), 0) + 1
        return {
            "requested_ids": requested,
            "unique_ids": unique,
            "dedup_reduction": (
                float(1.0 - unique / requested)
                if requested
                else None
            ),
            "remote_requested_ids": remote_requested,
            "remote_unique_ids": remote_unique,
            "remote_requested_fraction": (
                float(remote_requested / requested)
                if requested
                else None
            ),
            "remote_return_reduction": (
                float(1.0 - remote_unique / remote_requested)
                if remote_requested
                else None
            ),
            "remote_vector_bytes_without_dedup": (
                remote_requested
                * embedding_dim
                * transport_element_bytes
            ),
            "remote_vector_bytes_at_scope_unique": (
                remote_unique
                * embedding_dim
                * transport_element_bytes
            ),
            "requester_rank_fanout_distribution": fanout,
            "maximum_requester_rank_fanout": (
                max(
                    (len(value) for value in requester_ranks.values()),
                    default=0,
                )
            ),
        }

    points = []
    for world_size in world_sizes:
        for strategy in owner_strategies:
            owner_map = build_d2_record_owner_map(
                plan,
                world_size,
                strategy,
            )
            for owner_rule in embedding_owner_rules:
                for batch_size in batch_sizes:
                    phase_values = {
                        phase: scope_values(
                            records,
                            owner_map,
                            world_size,
                            owner_rule,
                            batch_size,
                        )
                        for phase, records in phases.items()
                    }
                    exact_coalesced = scope_values(
                        exact_prefix_records,
                        owner_map,
                        world_size,
                        owner_rule,
                        batch_size,
                    )
                    append_coalesced = scope_values(
                        append_records,
                        owner_map,
                        world_size,
                        owner_rule,
                        batch_size,
                    )
                    requested = sum(
                        value["requested_ids"]
                        for value in phase_values.values()
                    )
                    unique = sum(
                        value["unique_ids"]
                        for value in phase_values.values()
                    )
                    remote_requested = sum(
                        value["remote_requested_ids"]
                        for value in phase_values.values()
                    )
                    remote_unique = sum(
                        value["remote_unique_ids"]
                        for value in phase_values.values()
                    )
                    exact_prefix = {
                        key: sum(
                            phase_values[phase][key]
                            for phase in (
                                "scheduled_exact_retained",
                                "natural_exact_target_prefix",
                            )
                        )
                        for key in (
                            "requested_ids",
                            "unique_ids",
                            "remote_requested_ids",
                            "remote_unique_ids",
                        )
                    }
                    exact_prefix.update(
                        {
                            "dedup_reduction": (
                                float(
                                    1.0
                                    - exact_prefix["unique_ids"]
                                    / exact_prefix["requested_ids"]
                                )
                                if exact_prefix["requested_ids"]
                                else None
                            ),
                            "remote_return_reduction": (
                                float(
                                    1.0
                                    - exact_prefix["remote_unique_ids"]
                                    / exact_prefix[
                                        "remote_requested_ids"
                                    ]
                                )
                                if exact_prefix[
                                    "remote_requested_ids"
                                ]
                                else None
                            ),
                        }
                    )
                    points.append(
                        {
                            "world_size": world_size,
                            "record_owner_strategy": strategy,
                            "record_owner_map_sha256": (
                                d2_record_owner_map_sha256(owner_map)
                            ),
                            "embedding_owner_rule": owner_rule,
                            "batch_records": batch_size,
                            "phases": phase_values,
                            "exact_prefix_phase_separated": (
                                exact_prefix
                            ),
                            "exact_prefix_coalesced": exact_coalesced,
                            "append_coalesced_candidate": (
                                append_coalesced
                            ),
                            "full_wave_phase_separated": {
                                "requested_ids": requested,
                                "unique_ids": unique,
                                "dedup_reduction": (
                                    float(
                                        1.0 - unique / requested
                                    )
                                    if requested
                                    else None
                                ),
                                "remote_requested_ids": (
                                    remote_requested
                                ),
                                "remote_unique_ids": remote_unique,
                                "remote_return_reduction": (
                                    float(
                                        1.0
                                        - remote_unique
                                        / remote_requested
                                    )
                                    if remote_requested
                                    else None
                                ),
                            },
                            "full_wave_coalesced_by_primitive": {
                                "requested_ids": (
                                    exact_coalesced["requested_ids"]
                                    + append_coalesced["requested_ids"]
                                ),
                                "unique_ids": (
                                    exact_coalesced["unique_ids"]
                                    + append_coalesced["unique_ids"]
                                ),
                                "remote_requested_ids": (
                                    exact_coalesced[
                                        "remote_requested_ids"
                                    ]
                                    + append_coalesced[
                                        "remote_requested_ids"
                                    ]
                                ),
                                "remote_unique_ids": (
                                    exact_coalesced[
                                        "remote_unique_ids"
                                    ]
                                    + append_coalesced[
                                        "remote_unique_ids"
                                    ]
                                ),
                            },
                        }
                    )
                    coalesced = points[-1][
                        "full_wave_coalesced_by_primitive"
                    ]
                    coalesced["dedup_reduction"] = (
                        float(
                            1.0
                            - coalesced["unique_ids"]
                            / coalesced["requested_ids"]
                        )
                        if coalesced["requested_ids"]
                        else None
                    )
                    coalesced["remote_return_reduction"] = (
                        float(
                            1.0
                            - coalesced["remote_unique_ids"]
                            / coalesced["remote_requested_ids"]
                        )
                        if coalesced["remote_requested_ids"]
                        else None
                    )
    return {
        "num_embedding_rows": num_embedding_rows,
        "embedding_dim": embedding_dim,
        "transport_element_bytes": transport_element_bytes,
        "phase_boundary": (
            "dedup never crosses exact-prefix and append primitives; "
            "logical-subphase-separated and coalesced candidates are both reported"
        ),
        "points": points,
    }
