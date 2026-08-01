from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace

FOUNDATION_LIFECYCLE_PROTOCOL = "evokv_foundation_rolling_lifecycle_canary_v0"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: Mapping[str, object]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return _sha256_bytes(payload)


def deterministic_extent_payload(
    record_id: int,
    version: str,
    allocated_nbytes: int,
) -> bytes:
    if record_id < 0 or not version or allocated_nbytes < 1:
        raise ValueError("deterministic extent request is invalid")
    block = hashlib.sha256(f"{record_id}:{version}".encode()).digest()
    repeats, remainder = divmod(allocated_nbytes, len(block))
    return block * repeats + block[:remainder]


@dataclass(frozen=True)
class FoundationRecordSpec:
    record_id: int
    route: str
    source_version: str
    target_version: str
    old_valid_tokens: int
    old_allocated_tokens: int
    target_valid_tokens: int
    target_allocated_tokens: int

    def __post_init__(self) -> None:
        if (
            self.record_id < 0
            or not self.route
            or not self.source_version
            or not self.target_version
            or self.source_version == self.target_version
            or min(
                self.old_valid_tokens,
                self.old_allocated_tokens,
                self.target_valid_tokens,
                self.target_allocated_tokens,
            )
            < 1
            or self.old_valid_tokens > self.old_allocated_tokens
            or self.target_valid_tokens > self.target_allocated_tokens
        ):
            raise ValueError("foundation record specification is invalid")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FoundationGroupSpec:
    group_id: str
    records: tuple[FoundationRecordSpec, ...]
    staging_bytes: int
    workspace_bytes: int

    def __post_init__(self) -> None:
        record_ids = tuple(value.record_id for value in self.records)
        if (
            not self.group_id
            or not self.records
            or len(record_ids) != len(set(record_ids))
            or self.staging_bytes < 0
            or self.workspace_bytes < 0
        ):
            raise ValueError("foundation group specification is invalid")

    @property
    def identity_sha256(self) -> str:
        return _canonical_sha256(
            {
                "group_id": self.group_id,
                "records": [value.to_dict() for value in self.records],
                "staging_bytes": self.staging_bytes,
                "workspace_bytes": self.workspace_bytes,
            }
        )


@dataclass(frozen=True)
class FoundationTargetExtent:
    record_id: int
    source_version: str
    source_lineage_sha256: str
    target_version: str
    target_lineage_sha256: str
    valid_tokens: int
    allocated_tokens: int
    payload: bytes
    payload_sha256: str

    def __post_init__(self) -> None:
        if (
            self.record_id < 0
            or not self.source_version
            or not self.source_lineage_sha256
            or not self.target_version
            or not self.target_lineage_sha256
            or self.valid_tokens < 1
            or self.allocated_tokens < self.valid_tokens
            or not self.payload
            or not self.payload_sha256
        ):
            raise ValueError("foundation target extent is invalid")

    def descriptor(self) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            "source_version": self.source_version,
            "source_lineage_sha256": self.source_lineage_sha256,
            "target_version": self.target_version,
            "target_lineage_sha256": self.target_lineage_sha256,
            "valid_tokens": self.valid_tokens,
            "allocated_tokens": self.allocated_tokens,
            "payload_nbytes": len(self.payload),
            "payload_sha256": self.payload_sha256,
        }


@dataclass(frozen=True)
class FoundationGroupReceipt:
    group_id: str
    group_sha256: str
    execution_sha256: str
    status: str
    commit_ordinal: int
    record_ids: tuple[int, ...]
    committed_valid_bytes: int
    committed_allocated_bytes: int
    reclaimed_valid_bytes: int
    reclaimed_allocated_bytes: int

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["record_ids"] = list(self.record_ids)
        return value


@dataclass(frozen=True)
class _LiveExtent:
    record_id: int
    version: str
    lineage_sha256: str
    valid_tokens: int
    allocated_tokens: int
    payload: bytes
    payload_sha256: str


def _source_lineage(
    record: FoundationRecordSpec,
    payload_sha256: str,
) -> str:
    return _canonical_sha256(
        {
            "kind": "foundation_source_extent",
            "record_id": record.record_id,
            "version": record.source_version,
            "route": record.route,
            "valid_tokens": record.old_valid_tokens,
            "allocated_tokens": record.old_allocated_tokens,
            "payload_sha256": payload_sha256,
        }
    )


def _target_lineage(
    record: FoundationRecordSpec,
    source_lineage_sha256: str,
    payload_sha256: str,
) -> str:
    return _canonical_sha256(
        {
            "kind": "foundation_target_extent",
            "record_id": record.record_id,
            "route": record.route,
            "source_version": record.source_version,
            "source_lineage_sha256": source_lineage_sha256,
            "target_version": record.target_version,
            "valid_tokens": record.target_valid_tokens,
            "allocated_tokens": record.target_allocated_tokens,
            "payload_sha256": payload_sha256,
        }
    )


class FoundationRollingLifecycle:
    def __init__(
        self,
        records: Sequence[FoundationRecordSpec],
        source_payloads: Mapping[int, bytes],
        *,
        bytes_per_token: int,
        shadow_capacity_bytes: int,
        staging_capacity_bytes: int,
        workspace_capacity_bytes: int,
    ) -> None:
        resolved = tuple(records)
        record_ids = tuple(value.record_id for value in resolved)
        if (
            not resolved
            or len(record_ids) != len(set(record_ids))
            or set(record_ids) != set(source_payloads)
            or bytes_per_token < 1
            or shadow_capacity_bytes < 1
            or staging_capacity_bytes < 0
            or workspace_capacity_bytes < 0
        ):
            raise ValueError("foundation lifecycle configuration is invalid")
        self.records = {value.record_id: value for value in resolved}
        self.bytes_per_token = int(bytes_per_token)
        self.shadow_capacity_bytes = int(shadow_capacity_bytes)
        self.staging_capacity_bytes = int(staging_capacity_bytes)
        self.workspace_capacity_bytes = int(workspace_capacity_bytes)
        self._live: dict[int, _LiveExtent] = {}
        for record in resolved:
            payload = bytes(source_payloads[record.record_id])
            expected_nbytes = record.old_allocated_tokens * self.bytes_per_token
            if len(payload) != expected_nbytes:
                raise ValueError("source extent allocated length differs")
            payload_sha256 = _sha256_bytes(payload)
            self._live[record.record_id] = _LiveExtent(
                record_id=record.record_id,
                version=record.source_version,
                lineage_sha256=_source_lineage(record, payload_sha256),
                valid_tokens=record.old_valid_tokens,
                allocated_tokens=record.old_allocated_tokens,
                payload=payload,
                payload_sha256=payload_sha256,
            )
        self._shadow: dict[int, FoundationTargetExtent] = {}
        self._receipts: dict[str, FoundationGroupReceipt] = {}
        self._execution_hashes: dict[str, str] = {}
        self._commit_counts: Counter[int] = Counter()
        self._attempted_groups = 0
        self._committed_groups = 0
        self._failed_groups = 0
        self._idempotent_groups = 0
        self._validated_records = 0
        self._discarded_shadow_bytes = 0
        self._reclaimed_valid_bytes = 0
        self._reclaimed_allocated_bytes = 0
        self._sum_shadow_allocated_bytes = 0
        initial_valid, initial_allocated = self._live_bytes()
        self._initial_live_valid_bytes = initial_valid
        self._initial_live_allocated_bytes = initial_allocated
        self._peak_live_valid_bytes = initial_valid
        self._peak_live_allocated_bytes = initial_allocated
        self._peak_shadow_valid_bytes = 0
        self._peak_shadow_allocated_bytes = 0
        self._peak_staging_bytes = 0
        self._peak_workspace_bytes = 0
        self._peak_total_allocated_bytes = initial_allocated

    def _live_bytes(self) -> tuple[int, int]:
        valid = sum(
            value.valid_tokens * self.bytes_per_token
            for value in self._live.values()
        )
        allocated = sum(
            value.allocated_tokens * self.bytes_per_token
            for value in self._live.values()
        )
        return valid, allocated

    def state(self, record_id: int) -> dict[str, object]:
        value = self._live.get(int(record_id))
        if value is None:
            raise KeyError(f"unknown foundation record {record_id}")
        return {
            "record_id": value.record_id,
            "version": value.version,
            "lineage_sha256": value.lineage_sha256,
            "valid_tokens": value.valid_tokens,
            "allocated_tokens": value.allocated_tokens,
            "payload_nbytes": len(value.payload),
            "payload_sha256": value.payload_sha256,
        }

    def prepare_target(
        self,
        record_id: int,
        payload: bytes,
    ) -> FoundationTargetExtent:
        record = self.records.get(int(record_id))
        source = self._live.get(int(record_id))
        if record is None or source is None:
            raise KeyError(f"unknown foundation record {record_id}")
        payload_value = bytes(payload)
        payload_sha256 = _sha256_bytes(payload_value)
        return FoundationTargetExtent(
            record_id=record.record_id,
            source_version=source.version,
            source_lineage_sha256=source.lineage_sha256,
            target_version=record.target_version,
            target_lineage_sha256=_target_lineage(
                record,
                source.lineage_sha256,
                payload_sha256,
            ),
            valid_tokens=record.target_valid_tokens,
            allocated_tokens=record.target_allocated_tokens,
            payload=payload_value,
            payload_sha256=payload_sha256,
        )

    def _execution_sha256(
        self,
        group: FoundationGroupSpec,
        outputs: Sequence[FoundationTargetExtent],
    ) -> str:
        return _canonical_sha256(
            {
                "group_sha256": group.identity_sha256,
                "outputs": sorted(
                    (value.descriptor() for value in outputs),
                    key=lambda value: int(value["record_id"]),
                ),
            }
        )

    def _abort(self) -> None:
        self._failed_groups += 1
        self._discarded_shadow_bytes += sum(
            value.allocated_tokens * self.bytes_per_token
            for value in self._shadow.values()
        )
        self._shadow.clear()

    def execute_group(
        self,
        group: FoundationGroupSpec,
        outputs: Sequence[FoundationTargetExtent],
    ) -> FoundationGroupReceipt:
        self._attempted_groups += 1
        execution_sha256 = self._execution_sha256(group, outputs)
        prior = self._receipts.get(group.group_id)
        if prior is not None:
            if self._execution_hashes[group.group_id] != execution_sha256:
                self._failed_groups += 1
                raise ValueError("committed group identity differs on replay")
            self._idempotent_groups += 1
            return replace(prior, status="already_committed")
        try:
            expected_records = {
                value.record_id: value for value in group.records
            }
            if any(
                self.records.get(record_id) != record
                for record_id, record in expected_records.items()
            ):
                raise ValueError("group record specification differs")
            output_ids = tuple(value.record_id for value in outputs)
            if (
                len(output_ids) != len(set(output_ids))
                or set(output_ids) != set(expected_records)
            ):
                raise ValueError("group output coverage differs")
            self._shadow = {value.record_id: value for value in outputs}
            shadow_valid = sum(
                value.valid_tokens * self.bytes_per_token
                for value in outputs
            )
            shadow_allocated = sum(
                value.allocated_tokens * self.bytes_per_token
                for value in outputs
            )
            if shadow_allocated > self.shadow_capacity_bytes:
                raise MemoryError("group exceeds bounded shadow capacity")
            if group.staging_bytes > self.staging_capacity_bytes:
                raise MemoryError("group exceeds bounded staging capacity")
            if group.workspace_bytes > self.workspace_capacity_bytes:
                raise MemoryError("group exceeds bounded workspace capacity")
            live_valid, live_allocated = self._live_bytes()
            self._peak_shadow_valid_bytes = max(
                self._peak_shadow_valid_bytes,
                shadow_valid,
            )
            self._peak_shadow_allocated_bytes = max(
                self._peak_shadow_allocated_bytes,
                shadow_allocated,
            )
            self._peak_staging_bytes = max(
                self._peak_staging_bytes,
                group.staging_bytes,
            )
            self._peak_workspace_bytes = max(
                self._peak_workspace_bytes,
                group.workspace_bytes,
            )
            self._peak_total_allocated_bytes = max(
                self._peak_total_allocated_bytes,
                live_allocated
                + shadow_allocated
                + group.staging_bytes
                + group.workspace_bytes,
            )
            old_extents: dict[int, _LiveExtent] = {}
            for record_id, record in expected_records.items():
                source = self._live[record_id]
                target = self._shadow[record_id]
                expected_payload_nbytes = (
                    record.target_allocated_tokens * self.bytes_per_token
                )
                if (
                    source.version != record.source_version
                    or target.source_version != source.version
                    or target.source_lineage_sha256
                    != source.lineage_sha256
                    or target.target_version != record.target_version
                    or target.valid_tokens != record.target_valid_tokens
                    or target.allocated_tokens
                    != record.target_allocated_tokens
                    or len(target.payload) != expected_payload_nbytes
                    or _sha256_bytes(target.payload)
                    != target.payload_sha256
                    or target.target_lineage_sha256
                    != _target_lineage(
                        record,
                        source.lineage_sha256,
                        target.payload_sha256,
                    )
                ):
                    raise ValueError(
                        "group target payload, length, version, or lineage differs"
                    )
                old_extents[record_id] = source
            next_live = dict(self._live)
            for record_id, target in self._shadow.items():
                next_live[record_id] = _LiveExtent(
                    record_id=record_id,
                    version=target.target_version,
                    lineage_sha256=target.target_lineage_sha256,
                    valid_tokens=target.valid_tokens,
                    allocated_tokens=target.allocated_tokens,
                    payload=target.payload,
                    payload_sha256=target.payload_sha256,
                )
            reclaimed_valid = sum(
                value.valid_tokens * self.bytes_per_token
                for value in old_extents.values()
            )
            reclaimed_allocated = sum(
                value.allocated_tokens * self.bytes_per_token
                for value in old_extents.values()
            )
            self._live = next_live
            self._shadow.clear()
            self._committed_groups += 1
            self._validated_records += len(expected_records)
            self._sum_shadow_allocated_bytes += shadow_allocated
            self._reclaimed_valid_bytes += reclaimed_valid
            self._reclaimed_allocated_bytes += reclaimed_allocated
            self._commit_counts.update(expected_records.keys())
            final_valid, final_allocated = self._live_bytes()
            self._peak_live_valid_bytes = max(
                self._peak_live_valid_bytes,
                final_valid,
            )
            self._peak_live_allocated_bytes = max(
                self._peak_live_allocated_bytes,
                final_allocated,
            )
            receipt = FoundationGroupReceipt(
                group_id=group.group_id,
                group_sha256=group.identity_sha256,
                execution_sha256=execution_sha256,
                status="committed",
                commit_ordinal=self._committed_groups,
                record_ids=tuple(sorted(expected_records)),
                committed_valid_bytes=shadow_valid,
                committed_allocated_bytes=shadow_allocated,
                reclaimed_valid_bytes=reclaimed_valid,
                reclaimed_allocated_bytes=reclaimed_allocated,
            )
            self._receipts[group.group_id] = receipt
            self._execution_hashes[group.group_id] = execution_sha256
            return receipt
        except (KeyError, MemoryError, ValueError):
            self._abort()
            raise

    def ledger(self) -> dict[str, object]:
        final_valid, final_allocated = self._live_bytes()
        expected_ids = tuple(sorted(self.records))
        committed_ids = tuple(
            record_id
            for record_id in expected_ids
            if self._commit_counts[record_id] > 0
        )
        pending_ids = tuple(
            record_id
            for record_id in expected_ids
            if self._commit_counts[record_id] == 0
        )
        duplicate_ids = tuple(
            record_id
            for record_id in expected_ids
            if self._commit_counts[record_id] > 1
        )
        states = [
            self.state(record_id)
            for record_id in expected_ids
        ]
        return {
            "protocol": FOUNDATION_LIFECYCLE_PROTOCOL,
            "scientific_result": False,
            "formal_design3": False,
            "executes_d1_d2_numeric": False,
            "backend": "in_memory_full_payload_transaction_canary",
            "bytes_per_token": self.bytes_per_token,
            "records": len(expected_ids),
            "groups_attempted": self._attempted_groups,
            "groups_committed": self._committed_groups,
            "groups_failed": self._failed_groups,
            "idempotent_group_replays": self._idempotent_groups,
            "full_payload_length_hash_validated_records": (
                self._validated_records
            ),
            "capacity": {
                "shadow_capacity_bytes": self.shadow_capacity_bytes,
                "staging_capacity_bytes": self.staging_capacity_bytes,
                "workspace_capacity_bytes": self.workspace_capacity_bytes,
                "initial_live_valid_bytes": self._initial_live_valid_bytes,
                "initial_live_allocated_bytes": (
                    self._initial_live_allocated_bytes
                ),
                "final_live_valid_bytes": final_valid,
                "final_live_allocated_bytes": final_allocated,
                "peak_live_valid_bytes": self._peak_live_valid_bytes,
                "peak_live_allocated_bytes": (
                    self._peak_live_allocated_bytes
                ),
                "peak_shadow_valid_bytes": self._peak_shadow_valid_bytes,
                "peak_shadow_allocated_bytes": (
                    self._peak_shadow_allocated_bytes
                ),
                "peak_staging_bytes": self._peak_staging_bytes,
                "peak_workspace_bytes": self._peak_workspace_bytes,
                "peak_total_allocated_bytes": (
                    self._peak_total_allocated_bytes
                ),
                "sum_group_shadow_allocated_bytes": (
                    self._sum_shadow_allocated_bytes
                ),
                "shadow_reuse_groups": max(
                    self._committed_groups - 1,
                    0,
                ),
                "shadow_capacity_bound_respected": (
                    self._peak_shadow_allocated_bytes
                    <= self.shadow_capacity_bytes
                ),
            },
            "transaction": {
                "validate_before_commit": True,
                "old_reclaim_after_commit": True,
                "failed_group_publishes": False,
                "discarded_failed_shadow_bytes": (
                    self._discarded_shadow_bytes
                ),
                "reclaimed_old_valid_bytes": (
                    self._reclaimed_valid_bytes
                ),
                "reclaimed_old_allocated_bytes": (
                    self._reclaimed_allocated_bytes
                ),
                "receipts": [
                    value.to_dict()
                    for value in sorted(
                        self._receipts.values(),
                        key=lambda item: item.commit_ordinal,
                    )
                ],
            },
            "coverage": {
                "expected_record_ids": list(expected_ids),
                "committed_record_ids": list(committed_ids),
                "pending_record_ids": list(pending_ids),
                "duplicate_commit_record_ids": list(duplicate_ids),
                "complete": not pending_ids,
                "exactly_once": not pending_ids and not duplicate_ids,
            },
            "live_states": states,
        }
