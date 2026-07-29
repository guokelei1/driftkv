from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .design2_plan import (
    D2ActionPlan,
    canonical_json_bytes,
    canonical_sha256,
    d2_record_owner_map_sha256,
)

D3_DEV_WORK_MANIFEST_PROTOCOL = "evokv_d3_dev_work_manifest_v0"
D3_DEV_GROUP_PLAN_PROTOCOL = "evokv_d3_dev_group_plan_v0"


@dataclass(frozen=True)
class D3WorkBytes:
    old_kv_allocated: int
    old_kv_read: int
    history_read: int
    target_write: int
    grouping_estimate: int
    extra: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            min(
                self.old_kv_allocated,
                self.old_kv_read,
                self.history_read,
                self.target_write,
                self.grouping_estimate,
            )
            < 0
            or self.old_kv_allocated < self.old_kv_read
            or self.grouping_estimate
            < self.old_kv_read + self.history_read + self.target_write
            or any(value < 0 for value in self.extra.values())
        ):
            raise ValueError("D3 work bytes are invalid")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> D3WorkBytes:
        return cls(
            old_kv_allocated=int(value["old_kv_allocated"]),
            old_kv_read=int(value["old_kv_read"]),
            history_read=int(value["history_read"]),
            target_write=int(value["target_write"]),
            grouping_estimate=int(value["grouping_estimate"]),
            extra={
                str(key): int(item)
                for key, item in dict(value.get("extra", {})).items()
            },
        )

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "extra": dict(self.extra),
        }


@dataclass(frozen=True)
class D3WorkItem:
    record_id: int
    prepared_user_id: int
    owner_rank: int
    pool: str
    reason: str
    old_kv_ref: str | None
    history_ref: str
    target_ref: str
    retained_start: int
    retained_tokens: int
    history_start: int
    history_tokens: int
    final_tokens: int
    bytes: D3WorkBytes
    extra: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            self.record_id < 0
            or self.prepared_user_id < 1
            or self.owner_rank < 0
            or not self.pool
            or not self.reason
            or not self.history_ref
            or not self.target_ref
            or min(
                self.retained_start,
                self.retained_tokens,
                self.history_start,
                self.history_tokens,
            )
            < 0
            or self.final_tokens < 1
            or self.old_kv_ref is None
            and self.bytes.old_kv_allocated != 0
        ):
            raise ValueError("D3 work item is invalid")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> D3WorkItem:
        return cls(
            record_id=int(value["record_id"]),
            prepared_user_id=int(value["prepared_user_id"]),
            owner_rank=int(value["owner_rank"]),
            pool=str(value["pool"]),
            reason=str(value["reason"]),
            old_kv_ref=(
                None
                if value.get("old_kv_ref") is None
                else str(value["old_kv_ref"])
            ),
            history_ref=str(value["history_ref"]),
            target_ref=str(value["target_ref"]),
            retained_start=int(value["retained_start"]),
            retained_tokens=int(value["retained_tokens"]),
            history_start=int(value["history_start"]),
            history_tokens=int(value["history_tokens"]),
            final_tokens=int(value["final_tokens"]),
            bytes=D3WorkBytes.from_dict(value["bytes"]),
            extra=dict(value.get("extra", {})),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            "prepared_user_id": self.prepared_user_id,
            "owner_rank": self.owner_rank,
            "pool": self.pool,
            "reason": self.reason,
            "old_kv_ref": self.old_kv_ref,
            "history_ref": self.history_ref,
            "target_ref": self.target_ref,
            "retained_start": self.retained_start,
            "retained_tokens": self.retained_tokens,
            "history_start": self.history_start,
            "history_tokens": self.history_tokens,
            "final_tokens": self.final_tokens,
            "bytes": self.bytes.to_dict(),
            "extra": dict(self.extra),
        }


@dataclass(frozen=True)
class D3WorkManifest:
    stack_revision: str
    world_size: int
    action_plan_ref: str
    action_plan_sha256: str
    owner_strategy: str
    owner_map_sha256: str
    source_version: str
    target_version: str
    program_ref: str | None
    records: tuple[D3WorkItem, ...]
    scientific_result: bool = False
    extra: Mapping[str, object] = field(default_factory=dict)
    protocol: str = D3_DEV_WORK_MANIFEST_PROTOCOL

    def __post_init__(self) -> None:
        record_ids = tuple(value.record_id for value in self.records)
        if (
            not self.protocol
            or not self.stack_revision
            or self.world_size < 1
            or not self.action_plan_ref
            or not self.action_plan_sha256
            or not self.owner_strategy
            or not self.owner_map_sha256
            or not self.source_version
            or not self.target_version
            or not self.records
            or record_ids != tuple(sorted(record_ids))
            or len(set(record_ids)) != len(record_ids)
            or any(
                value.owner_rank >= self.world_size
                for value in self.records
            )
            or self.scientific_result
        ):
            raise ValueError("D3 work manifest is invalid")

    def payload_dict(self) -> dict[str, object]:
        return {
            "protocol": self.protocol,
            "stack_revision": self.stack_revision,
            "world_size": self.world_size,
            "action_plan_ref": self.action_plan_ref,
            "action_plan_sha256": self.action_plan_sha256,
            "owner_strategy": self.owner_strategy,
            "owner_map_sha256": self.owner_map_sha256,
            "source_version": self.source_version,
            "target_version": self.target_version,
            "program_ref": self.program_ref,
            "records": [value.to_dict() for value in self.records],
            "scientific_result": self.scientific_result,
            "extra": dict(self.extra),
        }

    @property
    def dev_sha256(self) -> str:
        return canonical_sha256(self.payload_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            **self.payload_dict(),
            "dev_sha256": self.dev_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> D3WorkManifest:
        manifest = cls(
            protocol=str(value["protocol"]),
            stack_revision=str(value["stack_revision"]),
            world_size=int(value["world_size"]),
            action_plan_ref=str(value["action_plan_ref"]),
            action_plan_sha256=str(value["action_plan_sha256"]),
            owner_strategy=str(value["owner_strategy"]),
            owner_map_sha256=str(value["owner_map_sha256"]),
            source_version=str(value["source_version"]),
            target_version=str(value["target_version"]),
            program_ref=(
                None
                if value.get("program_ref") is None
                else str(value["program_ref"])
            ),
            records=tuple(
                D3WorkItem.from_dict(record)
                for record in value["records"]
            ),
            scientific_result=bool(value["scientific_result"]),
            extra=dict(value.get("extra", {})),
        )
        if value.get("dev_sha256") != manifest.dev_sha256:
            raise ValueError("D3 work manifest development hash differs")
        return manifest

    @classmethod
    def load(cls, path: str | Path) -> D3WorkManifest:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def write(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(canonical_json_bytes(self.to_dict()))


@dataclass(frozen=True)
class D3WorkGroup:
    ordinal: int
    pool: str
    record_ids_by_rank: tuple[tuple[int, ...], ...]
    source_read_bytes_by_rank: tuple[int, ...]
    target_write_bytes_by_rank: tuple[int, ...]
    estimated_bytes_by_rank: tuple[int, ...]

    def __post_init__(self) -> None:
        world_size = len(self.record_ids_by_rank)
        if (
            self.ordinal < 0
            or not self.pool
            or world_size < 1
            or len(self.source_read_bytes_by_rank) != world_size
            or len(self.target_write_bytes_by_rank) != world_size
            or len(self.estimated_bytes_by_rank) != world_size
            or all(not value for value in self.record_ids_by_rank)
            or any(
                len(set(record_ids)) != len(record_ids)
                for record_ids in self.record_ids_by_rank
            )
            or any(
                value < 0
                for values in (
                    self.source_read_bytes_by_rank,
                    self.target_write_bytes_by_rank,
                    self.estimated_bytes_by_rank,
                )
                for value in values
            )
        ):
            raise ValueError("D3 work group is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "pool": self.pool,
            "record_ids_by_rank": [
                list(value) for value in self.record_ids_by_rank
            ],
            "source_read_bytes_by_rank": list(
                self.source_read_bytes_by_rank
            ),
            "target_write_bytes_by_rank": list(
                self.target_write_bytes_by_rank
            ),
            "estimated_bytes_by_rank": list(
                self.estimated_bytes_by_rank
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> D3WorkGroup:
        return cls(
            ordinal=int(value["ordinal"]),
            pool=str(value["pool"]),
            record_ids_by_rank=tuple(
                tuple(int(record_id) for record_id in record_ids)
                for record_ids in value["record_ids_by_rank"]
            ),
            source_read_bytes_by_rank=tuple(
                int(item) for item in value["source_read_bytes_by_rank"]
            ),
            target_write_bytes_by_rank=tuple(
                int(item) for item in value["target_write_bytes_by_rank"]
            ),
            estimated_bytes_by_rank=tuple(
                int(item) for item in value["estimated_bytes_by_rank"]
            ),
        )


@dataclass(frozen=True)
class D3GroupPlan:
    stack_revision: str
    work_manifest_sha256: str
    group_budget_bytes_by_rank: tuple[int, ...]
    groups: tuple[D3WorkGroup, ...]
    extra: Mapping[str, object] = field(default_factory=dict)
    protocol: str = D3_DEV_GROUP_PLAN_PROTOCOL

    def __post_init__(self) -> None:
        if (
            not self.protocol
            or not self.stack_revision
            or not self.work_manifest_sha256
            or not self.group_budget_bytes_by_rank
            or any(value < 1 for value in self.group_budget_bytes_by_rank)
            or not self.groups
            or tuple(group.ordinal for group in self.groups)
            != tuple(range(len(self.groups)))
        ):
            raise ValueError("D3 group plan is invalid")

    def payload_dict(self) -> dict[str, object]:
        return {
            "protocol": self.protocol,
            "stack_revision": self.stack_revision,
            "work_manifest_sha256": self.work_manifest_sha256,
            "group_budget_bytes_by_rank": list(
                self.group_budget_bytes_by_rank
            ),
            "groups": [value.to_dict() for value in self.groups],
            "extra": dict(self.extra),
        }

    @property
    def dev_sha256(self) -> str:
        return canonical_sha256(self.payload_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            **self.payload_dict(),
            "dev_sha256": self.dev_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> D3GroupPlan:
        plan = cls(
            protocol=str(value["protocol"]),
            stack_revision=str(value["stack_revision"]),
            work_manifest_sha256=str(value["work_manifest_sha256"]),
            group_budget_bytes_by_rank=tuple(
                int(item)
                for item in value["group_budget_bytes_by_rank"]
            ),
            groups=tuple(
                D3WorkGroup.from_dict(group)
                for group in value["groups"]
            ),
            extra=dict(value.get("extra", {})),
        )
        if value.get("dev_sha256") != plan.dev_sha256:
            raise ValueError("D3 group plan development hash differs")
        return plan

    @classmethod
    def load(cls, path: str | Path) -> D3GroupPlan:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def write(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(canonical_json_bytes(self.to_dict()))


def build_d3_work_manifest(
    action_plan: D2ActionPlan,
    owner_map: Mapping[int, int],
    *,
    world_size: int,
    stack_revision: str,
    action_plan_ref: str,
    owner_strategy: str = "strict_cow_lpt",
    program_ref: str | None = None,
    kv_bytes_per_token: int = 32768,
    raw_history_bytes_per_token: int = 20,
    per_record_grouping_overhead_bytes: int = 4096,
    extra: Mapping[str, object] | None = None,
) -> D3WorkManifest:
    if (
        world_size < 1
        or not stack_revision
        or not action_plan_ref
        or set(owner_map)
        != {value.record_id for value in action_plan.records}
        or any(
            owner < 0 or owner >= world_size
            for owner in owner_map.values()
        )
        or min(
            kv_bytes_per_token,
            raw_history_bytes_per_token,
        )
        < 1
        or per_record_grouping_overhead_bytes < 0
    ):
        raise ValueError("D3 work manifest inputs are invalid")
    records = []
    for record in action_plan.records:
        pool = record.requested_action
        old_kv_available = record.previous_cache_present
        history_start = (
            record.delta_start if pool == "compiled" else 0
        )
        history_tokens = record.final_tokens - history_start
        old_kv_read = (
            record.retained_tokens * kv_bytes_per_token
            if pool == "compiled"
            else 0
        )
        history_read = history_tokens * raw_history_bytes_per_token
        target_write = record.final_tokens * kv_bytes_per_token
        records.append(
            D3WorkItem(
                record_id=record.record_id,
                prepared_user_id=record.prepared_user_id,
                owner_rank=int(owner_map[record.record_id]),
                pool=pool,
                reason=record.requested_reason,
                old_kv_ref=(
                    f"oldkv:record:{record.record_id}"
                    if old_kv_available
                    else None
                ),
                history_ref=(
                    f"history:user:{record.prepared_user_id}:"
                    f"{record.target_history_sha256}"
                ),
                target_ref=f"target:record:{record.record_id}",
                retained_start=record.retained_start,
                retained_tokens=record.retained_tokens,
                history_start=history_start,
                history_tokens=history_tokens,
                final_tokens=record.final_tokens,
                bytes=D3WorkBytes(
                    old_kv_allocated=(
                        record.old_tokens * kv_bytes_per_token
                        if old_kv_available
                        else 0
                    ),
                    old_kv_read=old_kv_read,
                    history_read=history_read,
                    target_write=target_write,
                    grouping_estimate=(
                        old_kv_read
                        + history_read
                        + target_write
                        + per_record_grouping_overhead_bytes
                    ),
                ),
                extra={
                    "old_tokens": record.old_tokens,
                    "target_history_sha256": (
                        record.target_history_sha256
                    ),
                },
            )
        )
    return D3WorkManifest(
        stack_revision=stack_revision,
        world_size=world_size,
        action_plan_ref=action_plan_ref,
        action_plan_sha256=action_plan.content_sha256,
        owner_strategy=owner_strategy,
        owner_map_sha256=d2_record_owner_map_sha256(owner_map),
        source_version=action_plan.source_version,
        target_version=action_plan.target_version,
        program_ref=program_ref,
        records=tuple(records),
        extra={
            "byte_model": {
                "kv_bytes_per_token": kv_bytes_per_token,
                "raw_history_bytes_per_token": (
                    raw_history_bytes_per_token
                ),
                "per_record_grouping_overhead_bytes": (
                    per_record_grouping_overhead_bytes
                ),
            },
            **({} if extra is None else dict(extra)),
        },
    )


def audit_d3_work_manifest(
    manifest: D3WorkManifest,
    action_plan: D2ActionPlan,
    owner_map: Mapping[int, int],
) -> None:
    actions = {value.record_id: value for value in action_plan.records}
    if (
        manifest.action_plan_sha256 != action_plan.content_sha256
        or manifest.source_version != action_plan.source_version
        or manifest.target_version != action_plan.target_version
        or any(
            owner < 0 or owner >= manifest.world_size
            for owner in owner_map.values()
        )
        or manifest.owner_map_sha256
        != d2_record_owner_map_sha256(owner_map)
        or set(owner_map) != set(actions)
        or {value.record_id for value in manifest.records} != set(actions)
    ):
        raise ValueError("D3 work manifest differs from its action plan")
    for item in manifest.records:
        action = actions[item.record_id]
        expected_pool = action.requested_action
        expected_history_start = (
            action.delta_start if expected_pool == "compiled" else 0
        )
        if (
            item.owner_rank != owner_map[item.record_id]
            or item.pool != expected_pool
            or item.reason != action.requested_reason
            or item.retained_start != action.retained_start
            or item.retained_tokens != action.retained_tokens
            or item.history_start != expected_history_start
            or item.history_tokens
            != action.final_tokens - expected_history_start
            or item.final_tokens != action.final_tokens
        ):
            raise ValueError("D3 work record differs from its action")


def audit_d3_group_plan(
    manifest: D3WorkManifest,
    group_plan: D3GroupPlan,
) -> None:
    if (
        group_plan.work_manifest_sha256 != manifest.dev_sha256
        or group_plan.stack_revision != manifest.stack_revision
        or len(group_plan.group_budget_bytes_by_rank)
        != manifest.world_size
    ):
        raise ValueError("D3 group plan differs from its work manifest")
    items = {value.record_id: value for value in manifest.records}
    observed = []
    for group in group_plan.groups:
        if (
            len(group.record_ids_by_rank) != manifest.world_size
            or len(group.source_read_bytes_by_rank)
            != manifest.world_size
            or len(group.target_write_bytes_by_rank)
            != manifest.world_size
            or len(group.estimated_bytes_by_rank)
            != manifest.world_size
        ):
            raise ValueError("D3 group rank arrays differ")
        for rank, record_ids in enumerate(group.record_ids_by_rank):
            for record_id in record_ids:
                item = items.get(record_id)
                if (
                    item is None
                    or item.owner_rank != rank
                    or item.pool != group.pool
                ):
                    raise ValueError(
                        "D3 grouped work differs from its manifest"
                    )
                observed.append(record_id)
    expected = [value.record_id for value in manifest.records]
    if len(observed) != len(set(observed)) or set(observed) != set(expected):
        raise ValueError("D3 grouped work is not exactly once")


GroupEstimator = Callable[[int, tuple[D3WorkItem, ...]], int]
GroupOrderKey = Callable[[D3WorkItem], object]


def _default_estimator(
    rank: int,
    records: tuple[D3WorkItem, ...],
) -> int:
    del rank
    return sum(value.bytes.grouping_estimate for value in records)


def _default_order_key(item: D3WorkItem) -> object:
    if item.pool == "compiled":
        return (
            item.history_tokens,
            item.retained_tokens,
            item.final_tokens,
            item.record_id,
        )
    return (item.final_tokens, item.record_id)


def _partition_rank_records(
    rank: int,
    records: tuple[D3WorkItem, ...],
    budget_bytes: int,
    order_key: GroupOrderKey,
    estimator: GroupEstimator,
) -> tuple[tuple[D3WorkItem, ...], ...]:
    chunks = []
    current: tuple[D3WorkItem, ...] = ()
    for record in sorted(records, key=order_key):
        candidate = (*current, record)
        required = estimator(rank, candidate)
        if required <= budget_bytes:
            current = candidate
            continue
        if not current:
            raise ValueError(
                "D3 record exceeds group budget: "
                f"record={record.record_id} rank={rank} "
                f"required={required} cap={budget_bytes}"
            )
        chunks.append(current)
        current = (record,)
        required = estimator(rank, current)
        if required > budget_bytes:
            raise ValueError(
                "D3 record exceeds group budget: "
                f"record={record.record_id} rank={rank} "
                f"required={required} cap={budget_bytes}"
            )
    if current:
        chunks.append(current)
    return tuple(chunks)


def build_d3_byte_bounded_groups(
    manifest: D3WorkManifest,
    group_budget_bytes_by_rank: Sequence[int],
    *,
    pool_order: Sequence[str] | None = None,
    order_key: GroupOrderKey | None = None,
    estimator: GroupEstimator | None = None,
    extra: Mapping[str, object] | None = None,
) -> D3GroupPlan:
    budgets = tuple(int(value) for value in group_budget_bytes_by_rank)
    if (
        len(budgets) != manifest.world_size
        or any(value < 1 for value in budgets)
    ):
        raise ValueError("D3 group budget differs from its manifest")
    observed_pools = tuple(
        dict.fromkeys(value.pool for value in manifest.records)
    )
    if pool_order is None:
        requested_pools = tuple(
            value
            for value in ("compiled", "exact")
            if value in observed_pools
        ) + tuple(
            sorted(
                value
                for value in observed_pools
                if value not in {"compiled", "exact"}
            )
        )
    else:
        requested_pools = tuple(pool_order)
    if (
        len(set(requested_pools)) != len(requested_pools)
        or set(requested_pools) != set(observed_pools)
    ):
        raise ValueError("D3 pool order differs from its manifest")
    resolved_order_key = _default_order_key if order_key is None else order_key
    resolved_estimator = (
        _default_estimator if estimator is None else estimator
    )
    groups = []
    ordinal = 0
    for pool in requested_pools:
        chunks_by_rank = tuple(
            _partition_rank_records(
                rank,
                tuple(
                    record
                    for record in manifest.records
                    if record.pool == pool
                    and record.owner_rank == rank
                ),
                budgets[rank],
                resolved_order_key,
                resolved_estimator,
            )
            for rank in range(manifest.world_size)
        )
        steps = max((len(value) for value in chunks_by_rank), default=0)
        for step in range(steps):
            rank_chunks = tuple(
                chunks[step] if step < len(chunks) else ()
                for chunks in chunks_by_rank
            )
            groups.append(
                D3WorkGroup(
                    ordinal=ordinal,
                    pool=pool,
                    record_ids_by_rank=tuple(
                        tuple(record.record_id for record in chunk)
                        for chunk in rank_chunks
                    ),
                    source_read_bytes_by_rank=tuple(
                        sum(
                            record.bytes.old_kv_read
                            + record.bytes.history_read
                            for record in chunk
                        )
                        for chunk in rank_chunks
                    ),
                    target_write_bytes_by_rank=tuple(
                        sum(
                            record.bytes.target_write
                            for record in chunk
                        )
                        for chunk in rank_chunks
                    ),
                    estimated_bytes_by_rank=tuple(
                        resolved_estimator(rank, chunk)
                        for rank, chunk in enumerate(rank_chunks)
                    ),
                )
            )
            ordinal += 1
    observed_ids = [
        record_id
        for group in groups
        for record_ids in group.record_ids_by_rank
        for record_id in record_ids
    ]
    expected_ids = [value.record_id for value in manifest.records]
    if (
        len(observed_ids) != len(expected_ids)
        or set(observed_ids) != set(expected_ids)
        or any(
            observed > budget
            for group in groups
            for observed, budget in zip(
                group.estimated_bytes_by_rank,
                budgets,
                strict=True,
            )
        )
    ):
        raise RuntimeError("D3 group coverage or capacity differs")
    return D3GroupPlan(
        stack_revision=manifest.stack_revision,
        work_manifest_sha256=manifest.dev_sha256,
        group_budget_bytes_by_rank=budgets,
        groups=tuple(groups),
        extra={} if extra is None else dict(extra),
    )
