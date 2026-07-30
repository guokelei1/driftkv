from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from .design2_plan import canonical_json_bytes, canonical_sha256

D3_RESIDENCY_PROFILE_PROTOCOL = "evokv_d3_residency_profile_v0"
D3_RESIDENCY_PLAN_PROTOCOL = "evokv_d3_residency_plan_v1"


def d3_stack_revision_sha256(
    *,
    bindings: Mapping[str, object],
    world_size: int,
    group_records_per_rank: int,
    records: int,
    counts: Mapping[str, object],
    embedding_scale_role: str,
    device_names: Sequence[str],
    source_checkpoint_sha256: Sequence[str],
    target_checkpoint_sha256: Sequence[str],
    capacity_groups: Mapping[str, object],
    execution_identity: Mapping[str, object],
) -> str:
    if (
        world_size < 1
        or group_records_per_rank < 1
        or records < 1
        or len(device_names) != world_size
        or len(source_checkpoint_sha256) != world_size
        or len(target_checkpoint_sha256) != world_size
        or not embedding_scale_role
        or not execution_identity
    ):
        raise ValueError("D3 stack revision inputs are invalid")
    return canonical_sha256(
        {
            "bindings": dict(bindings),
            "world_size": world_size,
            "group_records_per_rank": group_records_per_rank,
            "records": records,
            "counts": dict(counts),
            "embedding_scale_role": embedding_scale_role,
            "device_names": list(device_names),
            "source_checkpoint_sha256": list(
                source_checkpoint_sha256
            ),
            "target_checkpoint_sha256": list(
                target_checkpoint_sha256
            ),
            "capacity_groups": dict(capacity_groups),
            "execution_identity": dict(execution_identity),
        }
    )


@dataclass(frozen=True)
class D3ResidencyGroup:
    ordinal: int
    route: str
    records_by_rank: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            self.ordinal < 0
            or self.route not in {"compiled", "exact"}
            or not self.records_by_rank
            or min(self.records_by_rank) < 0
            or max(self.records_by_rank) < 1
        ):
            raise ValueError("D3 residency group is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "route": self.route,
            "records_by_rank": list(self.records_by_rank),
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
    ) -> D3ResidencyGroup:
        return cls(
            ordinal=int(value["ordinal"]),
            route=str(value["route"]),
            records_by_rank=tuple(
                int(item) for item in value["records_by_rank"]
            ),
        )


@dataclass(frozen=True)
class D3RouteGranularity:
    input_segment_records: int
    compute_batch_records: int
    output_segment_records: int

    def __post_init__(self) -> None:
        if min(
            self.input_segment_records,
            self.compute_batch_records,
            self.output_segment_records,
        ) < 1:
            raise ValueError("D3 route granularity is invalid")

    def to_dict(self) -> dict[str, int]:
        return {
            "input_segment_records": self.input_segment_records,
            "compute_batch_records": self.compute_batch_records,
            "output_segment_records": self.output_segment_records,
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
    ) -> D3RouteGranularity:
        return cls(
            input_segment_records=int(
                value["input_segment_records"]
            ),
            compute_batch_records=int(
                value["compute_batch_records"]
            ),
            output_segment_records=int(
                value["output_segment_records"]
            ),
        )


@dataclass(frozen=True)
class D3RouteStageProfile:
    route: str
    granularity: D3RouteGranularity
    reference_records: int
    input_seconds_by_rank: tuple[float, ...]
    compute_seconds_by_rank: tuple[float, ...]
    output_seconds_by_rank: tuple[float, ...]
    peak_hbm_reserved_bytes_by_rank: tuple[int, ...]
    pinned_bytes_by_rank: tuple[int, ...]
    sample_groups: int
    source_sha256: str
    protocol: str = D3_RESIDENCY_PROFILE_PROTOCOL

    def __post_init__(self) -> None:
        world_size = len(self.input_seconds_by_rank)
        stage_values = (
            *self.input_seconds_by_rank,
            *self.compute_seconds_by_rank,
            *self.output_seconds_by_rank,
        )
        if (
            self.protocol != D3_RESIDENCY_PROFILE_PROTOCOL
            or self.route not in {"compiled", "exact"}
            or self.reference_records < 1
            or world_size < 1
            or len(self.compute_seconds_by_rank) != world_size
            or len(self.output_seconds_by_rank) != world_size
            or len(self.peak_hbm_reserved_bytes_by_rank) != world_size
            or len(self.pinned_bytes_by_rank) != world_size
            or any(
                not math.isfinite(value) or value < 0
                for value in stage_values
            )
            or max(self.compute_seconds_by_rank) <= 0
            or min(self.peak_hbm_reserved_bytes_by_rank) < 0
            or min(self.pinned_bytes_by_rank) < 0
            or self.sample_groups < 1
            or len(self.source_sha256) != 64
            or any(
                value not in "0123456789abcdef"
                for value in self.source_sha256
            )
        ):
            raise ValueError("D3 route stage profile is invalid")

    @property
    def content_sha256(self) -> str:
        return canonical_sha256(self.payload_dict())

    def payload_dict(self) -> dict[str, object]:
        return {
            "protocol": self.protocol,
            "route": self.route,
            "granularity": self.granularity.to_dict(),
            "reference_records": self.reference_records,
            "input_seconds_by_rank": list(
                self.input_seconds_by_rank
            ),
            "compute_seconds_by_rank": list(
                self.compute_seconds_by_rank
            ),
            "output_seconds_by_rank": list(
                self.output_seconds_by_rank
            ),
            "peak_hbm_reserved_bytes_by_rank": list(
                self.peak_hbm_reserved_bytes_by_rank
            ),
            "pinned_bytes_by_rank": list(
                self.pinned_bytes_by_rank
            ),
            "sample_groups": self.sample_groups,
            "source_sha256": self.source_sha256,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self.payload_dict(),
            "content_sha256": self.content_sha256,
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
    ) -> D3RouteStageProfile:
        profile = cls(
            protocol=str(value["protocol"]),
            route=str(value["route"]),
            granularity=D3RouteGranularity.from_dict(
                value["granularity"]
            ),
            reference_records=int(value["reference_records"]),
            input_seconds_by_rank=tuple(
                float(item)
                for item in value["input_seconds_by_rank"]
            ),
            compute_seconds_by_rank=tuple(
                float(item)
                for item in value["compute_seconds_by_rank"]
            ),
            output_seconds_by_rank=tuple(
                float(item)
                for item in value["output_seconds_by_rank"]
            ),
            peak_hbm_reserved_bytes_by_rank=tuple(
                int(item)
                for item in value[
                    "peak_hbm_reserved_bytes_by_rank"
                ]
            ),
            pinned_bytes_by_rank=tuple(
                int(item) for item in value["pinned_bytes_by_rank"]
            ),
            sample_groups=int(value["sample_groups"]),
            source_sha256=str(value["source_sha256"]),
        )
        if value.get("content_sha256") != profile.content_sha256:
            raise ValueError("D3 route profile hash differs")
        return profile


@dataclass(frozen=True)
class D3ResidencyPlan:
    group_plan_sha256: str
    stack_revision_sha256: str
    compiled: D3RouteGranularity
    exact: D3RouteGranularity
    compiled_profile_sha256: str
    exact_profile_sha256: str
    compiled_profile: D3RouteStageProfile
    exact_profile: D3RouteStageProfile
    original_group_order: tuple[int, ...]
    launch_order: tuple[int, ...]
    predicted_makespan_seconds: float
    projected_peak_hbm_reserved_bytes_by_rank: tuple[int, ...]
    projected_pinned_bytes_by_rank: tuple[int, ...]
    hbm_total_bytes_by_rank: tuple[int, ...]
    pinned_limit_bytes_by_rank: tuple[int, ...]
    hbm_margin_bytes: int
    selector_tie_fraction: float
    protocol: str = D3_RESIDENCY_PLAN_PROTOCOL

    def __post_init__(self) -> None:
        world_size = len(
            self.projected_peak_hbm_reserved_bytes_by_rank
        )
        hashes = (
            self.group_plan_sha256,
            self.stack_revision_sha256,
            self.compiled_profile_sha256,
            self.exact_profile_sha256,
        )
        if (
            self.protocol != D3_RESIDENCY_PLAN_PROTOCOL
            or any(
                len(value) != 64
                or any(
                    token not in "0123456789abcdef"
                    for token in value
                )
                for value in hashes
            )
            or not self.original_group_order
            or len(set(self.original_group_order))
            != len(self.original_group_order)
            or set(self.launch_order) != set(self.original_group_order)
            or len(self.launch_order) != len(self.original_group_order)
            or not math.isfinite(self.predicted_makespan_seconds)
            or self.predicted_makespan_seconds <= 0
            or world_size < 1
            or len(self.projected_pinned_bytes_by_rank)
            != world_size
            or len(self.hbm_total_bytes_by_rank) != world_size
            or len(self.pinned_limit_bytes_by_rank) != world_size
            or min(
                self.projected_peak_hbm_reserved_bytes_by_rank
            )
            < 0
            or min(self.projected_pinned_bytes_by_rank) < 0
            or min(self.hbm_total_bytes_by_rank) < 1
            or min(self.pinned_limit_bytes_by_rank) < 1
            or self.compiled_profile.route != "compiled"
            or self.exact_profile.route != "exact"
            or self.compiled_profile.content_sha256
            != self.compiled_profile_sha256
            or self.exact_profile.content_sha256
            != self.exact_profile_sha256
            or self.compiled_profile.source_sha256
            != self.exact_profile.source_sha256
            or self.compiled_profile.granularity != self.compiled
            or self.exact_profile.granularity != self.exact
            or len(self.compiled_profile.input_seconds_by_rank)
            != world_size
            or len(self.exact_profile.input_seconds_by_rank)
            != world_size
            or any(
                observed > total - self.hbm_margin_bytes
                for observed, total in zip(
                    self.projected_peak_hbm_reserved_bytes_by_rank,
                    self.hbm_total_bytes_by_rank,
                    strict=True,
                )
            )
            or any(
                observed > limit
                for observed, limit in zip(
                    self.projected_pinned_bytes_by_rank,
                    self.pinned_limit_bytes_by_rank,
                    strict=True,
                )
            )
            or self.hbm_margin_bytes < 0
            or not 0 <= self.selector_tie_fraction < 1
        ):
            raise ValueError("D3 residency plan is invalid")

    @property
    def content_sha256(self) -> str:
        return canonical_sha256(self.payload_dict())

    def payload_dict(self) -> dict[str, object]:
        return {
            "protocol": self.protocol,
            "group_plan_sha256": self.group_plan_sha256,
            "stack_revision_sha256": self.stack_revision_sha256,
            "compiled": self.compiled.to_dict(),
            "exact": self.exact.to_dict(),
            "compiled_profile_sha256": (
                self.compiled_profile_sha256
            ),
            "exact_profile_sha256": self.exact_profile_sha256,
            "compiled_profile": self.compiled_profile.to_dict(),
            "exact_profile": self.exact_profile.to_dict(),
            "original_group_order": list(self.original_group_order),
            "launch_order": list(self.launch_order),
            "predicted_makespan_seconds": (
                self.predicted_makespan_seconds
            ),
            "projected_peak_hbm_reserved_bytes_by_rank": list(
                self.projected_peak_hbm_reserved_bytes_by_rank
            ),
            "projected_pinned_bytes_by_rank": list(
                self.projected_pinned_bytes_by_rank
            ),
            "hbm_total_bytes_by_rank": list(
                self.hbm_total_bytes_by_rank
            ),
            "pinned_limit_bytes_by_rank": list(
                self.pinned_limit_bytes_by_rank
            ),
            "hbm_margin_bytes": self.hbm_margin_bytes,
            "selector_tie_fraction": self.selector_tie_fraction,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self.payload_dict(),
            "content_sha256": self.content_sha256,
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
    ) -> D3ResidencyPlan:
        plan = cls(
            protocol=str(value["protocol"]),
            group_plan_sha256=str(value["group_plan_sha256"]),
            stack_revision_sha256=str(
                value["stack_revision_sha256"]
            ),
            compiled=D3RouteGranularity.from_dict(
                value["compiled"]
            ),
            exact=D3RouteGranularity.from_dict(value["exact"]),
            compiled_profile_sha256=str(
                value["compiled_profile_sha256"]
            ),
            exact_profile_sha256=str(
                value["exact_profile_sha256"]
            ),
            compiled_profile=D3RouteStageProfile.from_dict(
                value["compiled_profile"]
            ),
            exact_profile=D3RouteStageProfile.from_dict(
                value["exact_profile"]
            ),
            original_group_order=tuple(
                int(item) for item in value["original_group_order"]
            ),
            launch_order=tuple(
                int(item) for item in value["launch_order"]
            ),
            predicted_makespan_seconds=float(
                value["predicted_makespan_seconds"]
            ),
            projected_peak_hbm_reserved_bytes_by_rank=tuple(
                int(item)
                for item in value[
                    "projected_peak_hbm_reserved_bytes_by_rank"
                ]
            ),
            projected_pinned_bytes_by_rank=tuple(
                int(item)
                for item in value["projected_pinned_bytes_by_rank"]
            ),
            hbm_total_bytes_by_rank=tuple(
                int(item) for item in value["hbm_total_bytes_by_rank"]
            ),
            pinned_limit_bytes_by_rank=tuple(
                int(item)
                for item in value["pinned_limit_bytes_by_rank"]
            ),
            hbm_margin_bytes=int(value["hbm_margin_bytes"]),
            selector_tie_fraction=float(
                value["selector_tie_fraction"]
            ),
        )
        if value.get("content_sha256") != plan.content_sha256:
            raise ValueError("D3 residency plan hash differs")
        return plan

    @classmethod
    def load(cls, path: str | Path) -> D3ResidencyPlan:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def write(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(canonical_json_bytes(self.to_dict()))


def bounded_flowshop_makespan(
    ordered_stages: Sequence[tuple[float, float, float]],
) -> float:
    if not ordered_stages:
        raise ValueError("D3 flowshop stages are empty")
    input_done: list[float] = []
    compute_started: list[float] = []
    compute_done: list[float] = []
    output_done: list[float] = []
    for index, (input_seconds, compute_seconds, output_seconds) in enumerate(
        ordered_stages
    ):
        if (
            min(input_seconds, compute_seconds, output_seconds) < 0
            or not all(
                math.isfinite(value)
                for value in (
                    input_seconds,
                    compute_seconds,
                    output_seconds,
                )
            )
        ):
            raise ValueError("D3 flowshop service time is invalid")
        input_started = (
            0.0 if index == 0 else compute_started[index - 1]
        )
        input_done.append(input_started + input_seconds)
        prior_output_credit = (
            output_done[index - 2] if index >= 2 else 0.0
        )
        compute_started.append(
            max(
                compute_done[index - 1] if index else 0.0,
                input_done[index],
                prior_output_credit,
            )
        )
        compute_done.append(
            compute_started[index] + compute_seconds
        )
        output_started = max(
            compute_done[index],
            output_done[index - 1] if index else 0.0,
        )
        output_done.append(output_started + output_seconds)
    return output_done[-1]


def deterministic_route_interleavings(
    groups: Sequence[D3ResidencyGroup],
    max_interleavings: int = 100_000,
) -> tuple[tuple[int, ...], ...]:
    if max_interleavings < 1 or not groups:
        raise ValueError("D3 interleaving request is invalid")
    compiled = tuple(
        value.ordinal for value in groups if value.route == "compiled"
    )
    exact = tuple(
        value.ordinal for value in groups if value.route == "exact"
    )
    if not compiled or not exact:
        return (tuple(value.ordinal for value in groups),)
    count = math.comb(len(groups), len(compiled))
    if count <= max_interleavings:
        output = []
        positions = range(len(groups))
        for compiled_positions in combinations(
            positions,
            len(compiled),
        ):
            selected = set(compiled_positions)
            compiled_index = 0
            exact_index = 0
            order = []
            for position in positions:
                if position in selected:
                    order.append(compiled[compiled_index])
                    compiled_index += 1
                else:
                    order.append(exact[exact_index])
                    exact_index += 1
            output.append(tuple(order))
        return tuple(output)
    unit_stages = {
        value.ordinal: (
            (1.0, 1.0, 1.0)
            if value.route == "compiled"
            else (1.0, 1.0, 1.0)
        )
        for value in groups
    }
    return (
        _best_route_interleaving_dp(
            groups,
            unit_stages,
            beam_width=min(max_interleavings, 512),
        )[0],
    )


@dataclass(frozen=True)
class _FlowState:
    compute_started: float
    compute_done: float
    prior_output_done: float
    output_done: float
    order: tuple[int, ...]


def _advance_flow_state(
    state: _FlowState,
    ordinal: int,
    stages: tuple[float, float, float],
) -> _FlowState:
    input_seconds, compute_seconds, output_seconds = stages
    input_done = state.compute_started + input_seconds
    compute_started = max(
        state.compute_done,
        input_done,
        state.prior_output_done,
    )
    compute_done = compute_started + compute_seconds
    output_done = (
        max(compute_done, state.output_done) + output_seconds
    )
    return _FlowState(
        compute_started=compute_started,
        compute_done=compute_done,
        prior_output_done=state.output_done,
        output_done=output_done,
        order=(*state.order, ordinal),
    )


def _dominates(left: _FlowState, right: _FlowState) -> bool:
    left_values = (
        left.compute_started,
        left.compute_done,
        left.prior_output_done,
        left.output_done,
    )
    right_values = (
        right.compute_started,
        right.compute_done,
        right.prior_output_done,
        right.output_done,
    )
    return all(
        left_value <= right_value
        for left_value, right_value in zip(
            left_values,
            right_values,
            strict=True,
        )
    ) and (
        left_values != right_values or left.order < right.order
    )


def _pareto_beam(
    states: Sequence[_FlowState],
    beam_width: int,
) -> tuple[_FlowState, ...]:
    unique: dict[tuple[float, float, float, float], _FlowState] = {}
    for state in states:
        key = (
            state.compute_started,
            state.compute_done,
            state.prior_output_done,
            state.output_done,
        )
        incumbent = unique.get(key)
        if incumbent is None or state.order < incumbent.order:
            unique[key] = state
    ordered = sorted(
        unique.values(),
        key=lambda value: (
            value.output_done,
            value.compute_done,
            value.compute_started,
            value.prior_output_done,
            value.order,
        ),
    )
    frontier: list[_FlowState] = []
    for state in ordered:
        if any(_dominates(value, state) for value in frontier):
            continue
        frontier = [
            value
            for value in frontier
            if not _dominates(state, value)
        ]
        frontier.append(state)
    if len(frontier) > beam_width:
        frontier = sorted(
            frontier,
            key=lambda value: (
                value.output_done,
                value.compute_done,
                value.compute_started,
                value.prior_output_done,
                value.order,
            ),
        )[:beam_width]
    return tuple(frontier)


def _best_route_interleaving_dp(
    groups: Sequence[D3ResidencyGroup],
    stages_by_ordinal: Mapping[int, tuple[float, float, float]],
    *,
    beam_width: int,
) -> tuple[tuple[int, ...], float]:
    if beam_width < 1:
        raise ValueError("D3 interleaving beam is invalid")
    compiled = tuple(
        value.ordinal for value in groups if value.route == "compiled"
    )
    exact = tuple(
        value.ordinal for value in groups if value.route == "exact"
    )
    initial = _FlowState(0.0, 0.0, 0.0, 0.0, ())
    cells: dict[tuple[int, int], tuple[_FlowState, ...]] = {
        (0, 0): (initial,)
    }
    for total in range(1, len(groups) + 1):
        lower = max(0, total - len(exact))
        upper = min(len(compiled), total)
        for compiled_count in range(lower, upper + 1):
            exact_count = total - compiled_count
            candidates: list[_FlowState] = []
            if compiled_count:
                ordinal = compiled[compiled_count - 1]
                candidates.extend(
                    _advance_flow_state(
                        state,
                        ordinal,
                        stages_by_ordinal[ordinal],
                    )
                    for state in cells[
                        (compiled_count - 1, exact_count)
                    ]
                )
            if exact_count:
                ordinal = exact[exact_count - 1]
                candidates.extend(
                    _advance_flow_state(
                        state,
                        ordinal,
                        stages_by_ordinal[ordinal],
                    )
                    for state in cells[
                        (compiled_count, exact_count - 1)
                    ]
                )
            cells[(compiled_count, exact_count)] = _pareto_beam(
                candidates,
                beam_width,
            )
    best = min(
        cells[(len(compiled), len(exact))],
        key=lambda value: (value.output_done, value.order),
    )
    return best.order, best.output_done


def _group_stages(
    group: D3ResidencyGroup,
    profile: D3RouteStageProfile,
) -> tuple[float, float, float]:
    records = max(group.records_by_rank)
    granularity = profile.granularity

    def scale(value: int) -> float:
        return math.ceil(records / value) / math.ceil(
            profile.reference_records / value
        )

    return (
        max(profile.input_seconds_by_rank)
        * scale(granularity.input_segment_records),
        max(profile.compute_seconds_by_rank)
        * scale(granularity.compute_batch_records),
        max(profile.output_seconds_by_rank)
        * scale(granularity.output_segment_records),
    )


def _segment_count(
    groups: Sequence[D3ResidencyGroup],
    compiled: D3RouteGranularity,
    exact: D3RouteGranularity,
) -> int:
    total = 0
    for group in groups:
        records = max(group.records_by_rank)
        granularity = compiled if group.route == "compiled" else exact
        total += sum(
            math.ceil(records / value)
            for value in (
                granularity.input_segment_records,
                granularity.compute_batch_records,
                granularity.output_segment_records,
            )
        )
    return total


def select_residency_plan(
    groups: Sequence[D3ResidencyGroup],
    profiles_by_route: Mapping[
        str,
        Sequence[D3RouteStageProfile],
    ],
    hbm_total_bytes_by_rank: Sequence[int],
    *,
    group_plan_sha256: str,
    stack_revision_sha256: str,
    hbm_margin_bytes: int = 2 * 1024**3,
    pinned_limit_bytes_by_rank: Sequence[int],
    selector_tie_fraction: float = 0.03,
    max_interleavings: int = 100_000,
) -> D3ResidencyPlan:
    if (
        not groups
        or set(profiles_by_route) != {"compiled", "exact"}
        or not profiles_by_route["compiled"]
        or not profiles_by_route["exact"]
        or hbm_margin_bytes < 0
        or not 0 <= selector_tie_fraction < 1
        or any(
            len(value) != 64
            or any(
                token not in "0123456789abcdef"
                for token in value
            )
            for value in (
                group_plan_sha256,
                stack_revision_sha256,
            )
        )
    ):
        raise ValueError("D3 residency selection inputs are invalid")
    original_order = tuple(value.ordinal for value in groups)
    if (
        len(set(original_order)) != len(original_order)
        or set(value.route for value in groups)
        != {"compiled", "exact"}
    ):
        raise ValueError("D3 residency groups differ")
    hbm_totals = tuple(int(value) for value in hbm_total_bytes_by_rank)
    world_size = len(hbm_totals)
    if world_size < 1 or min(hbm_totals) <= hbm_margin_bytes:
        raise ValueError("D3 HBM capacity is invalid")
    pinned_limits = tuple(
        int(value) for value in pinned_limit_bytes_by_rank
    )
    if (
        len(pinned_limits) != world_size
        or min(pinned_limits) < 1
    ):
        raise ValueError("D3 pinned capacity is invalid")
    for route, profiles in profiles_by_route.items():
        for profile in profiles:
            if (
                profile.route != route
                or len(profile.input_seconds_by_rank) != world_size
            ):
                raise ValueError("D3 route profile rank differs")
    candidates: list[
        tuple[
            float,
            tuple[int, int, int, str, str, tuple[int, ...]],
            D3ResidencyPlan,
        ]
    ] = []
    compiled_profiles = sorted(
        profiles_by_route["compiled"],
        key=lambda value: value.content_sha256,
    )
    exact_profiles_by_source: dict[
        str,
        list[D3RouteStageProfile],
    ] = {}
    for profile in sorted(
        profiles_by_route["exact"],
        key=lambda value: value.content_sha256,
    ):
        exact_profiles_by_source.setdefault(
            profile.source_sha256,
            [],
        ).append(profile)
    for compiled_profile in compiled_profiles:
        for exact_profile in exact_profiles_by_source.get(
            compiled_profile.source_sha256,
            (),
        ):
            projected_hbm = tuple(
                max(
                    compiled_profile.peak_hbm_reserved_bytes_by_rank[
                        rank
                    ],
                    exact_profile.peak_hbm_reserved_bytes_by_rank[
                        rank
                    ],
                )
                for rank in range(world_size)
            )
            projected_pinned = tuple(
                max(
                    compiled_profile.pinned_bytes_by_rank[rank],
                    exact_profile.pinned_bytes_by_rank[rank],
                )
                for rank in range(world_size)
            )
            if any(
                observed > total - hbm_margin_bytes
                for observed, total in zip(
                    projected_hbm,
                    hbm_totals,
                    strict=True,
                )
            ) or any(
                observed > limit
                for observed, limit in zip(
                    projected_pinned,
                    pinned_limits,
                    strict=True,
                )
            ):
                continue
            profile_for_route = {
                "compiled": compiled_profile,
                "exact": exact_profile,
            }
            stages_by_ordinal = {
                group.ordinal: _group_stages(
                    group,
                    profile_for_route[group.route],
                )
                for group in groups
            }
            interleaving_count = math.comb(
                len(groups),
                sum(
                    value.route == "compiled" for value in groups
                ),
            )
            if interleaving_count <= max_interleavings:
                interleavings = deterministic_route_interleavings(
                    groups,
                    max_interleavings,
                )
                best_order = min(
                    interleavings,
                    key=lambda order: (
                        bounded_flowshop_makespan(
                            tuple(
                                stages_by_ordinal[ordinal]
                                for ordinal in order
                            )
                        ),
                        order,
                    ),
                )
                makespan = bounded_flowshop_makespan(
                    tuple(
                        stages_by_ordinal[ordinal]
                        for ordinal in best_order
                    )
                )
            else:
                best_order, makespan = (
                    _best_route_interleaving_dp(
                        groups,
                        stages_by_ordinal,
                        beam_width=min(max_interleavings, 2048),
                    )
                )
            segments = _segment_count(
                groups,
                compiled_profile.granularity,
                exact_profile.granularity,
            )
            tie_key = (
                max(projected_hbm),
                sum(projected_pinned),
                segments,
                compiled_profile.content_sha256,
                exact_profile.content_sha256,
                best_order,
            )
            plan = D3ResidencyPlan(
                group_plan_sha256=group_plan_sha256,
                stack_revision_sha256=stack_revision_sha256,
                compiled=compiled_profile.granularity,
                exact=exact_profile.granularity,
                compiled_profile_sha256=(
                    compiled_profile.content_sha256
                ),
                exact_profile_sha256=(
                    exact_profile.content_sha256
                ),
                compiled_profile=compiled_profile,
                exact_profile=exact_profile,
                original_group_order=original_order,
                launch_order=best_order,
                predicted_makespan_seconds=makespan,
                projected_peak_hbm_reserved_bytes_by_rank=(
                    projected_hbm
                ),
                projected_pinned_bytes_by_rank=projected_pinned,
                hbm_total_bytes_by_rank=hbm_totals,
                pinned_limit_bytes_by_rank=pinned_limits,
                hbm_margin_bytes=hbm_margin_bytes,
                selector_tie_fraction=selector_tie_fraction,
            )
            candidates.append((makespan, tie_key, plan))
    if not candidates:
        common_sources = {
            value.source_sha256
            for value in profiles_by_route["compiled"]
        } & {
            value.source_sha256
            for value in profiles_by_route["exact"]
        }
        if not common_sources:
            raise ValueError(
                "D3 route profiles do not share a joint source"
            )
        raise ValueError("D3 residency candidates exceed capacity")
    global_min = min(value[0] for value in candidates)
    eligible = [
        value
        for value in candidates
        if value[0]
        <= global_min * (1 + selector_tie_fraction)
    ]
    return min(
        eligible,
        key=lambda value: (value[1], value[0]),
    )[2]
