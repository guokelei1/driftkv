from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from .design2_plan import canonical_sha256

D2_DEV_EPOCH_SCOPE = "development_only_pure_logic_not_formal_stage_c"
D2_DEV_PREPARE_PROTOCOL = "cohortkv_d2_dev_epoch_private_prepare_v1"
D2_DEV_COMMIT_PROTOCOL = "cohortkv_d2_dev_epoch_commit_certificate_v1"
D2_DEV_READBACK_PROTOCOL = "cohortkv_d2_dev_epoch_readback_ack_v1"
D2_DEV_PUBLICATION_PROTOCOL = "cohortkv_d2_dev_epoch_publication_certificate_v1"
D2_DEV_POINTER_PROTOCOL = "cohortkv_d2_dev_epoch_visible_pointer_v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ACTIONS = frozenset(
    {"compiled", "scheduled_exact", "natural_exact"}
)


def _is_sha256(value: str) -> bool:
    return _SHA256.fullmatch(value) is not None


@dataclass(frozen=True)
class D2DevEpochRecord:
    record_id: int
    owner_rank: int
    token_length: int
    action: str
    lineage_sha256: str
    payload_sha256: str

    def __post_init__(self) -> None:
        if (
            self.record_id < 0
            or self.owner_rank < 0
            or self.token_length < 1
            or self.action not in _ACTIONS
            or not _is_sha256(self.lineage_sha256)
            or not _is_sha256(self.payload_sha256)
        ):
            raise ValueError("D2 development epoch record is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            "owner_rank": self.owner_rank,
            "token_length": self.token_length,
            "action": self.action,
            "lineage_sha256": self.lineage_sha256,
            "payload_sha256": self.payload_sha256,
        }


def _validate_record_sequence(
    records: Sequence[D2DevEpochRecord],
) -> None:
    record_ids = tuple(value.record_id for value in records)
    if (
        record_ids != tuple(sorted(record_ids))
        or len(set(record_ids)) != len(record_ids)
    ):
        raise ValueError("D2 development epoch records are not canonical")


@dataclass(frozen=True)
class D2DevEpochSpec:
    action_plan_sha256: str
    source_version: str
    target_version: str
    source_epoch: int
    target_epoch: int
    world_size: int
    records: tuple[D2DevEpochRecord, ...]

    def __post_init__(self) -> None:
        _validate_record_sequence(self.records)
        if (
            not _is_sha256(self.action_plan_sha256)
            or not self.source_version
            or not self.target_version
            or self.source_version == self.target_version
            or self.source_epoch < 0
            or self.target_epoch <= self.source_epoch
            or self.world_size < 1
            or not self.records
            or any(
                value.owner_rank >= self.world_size
                for value in self.records
            )
        ):
            raise ValueError("D2 development epoch specification is invalid")

    @property
    def record_manifest_sha256(self) -> str:
        return canonical_sha256(
            {
                "records": [value.to_dict() for value in self.records],
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "scope": D2_DEV_EPOCH_SCOPE,
            "scientific_result": False,
            "formal_stage_c": False,
            "timing_included": False,
            "action_plan_sha256": self.action_plan_sha256,
            "source_version": self.source_version,
            "target_version": self.target_version,
            "source_epoch": self.source_epoch,
            "target_epoch": self.target_epoch,
            "world_size": self.world_size,
            "record_manifest_sha256": self.record_manifest_sha256,
            "records": [value.to_dict() for value in self.records],
        }


def _prepare_body(
    *,
    action_plan_sha256: str,
    source_version: str,
    target_version: str,
    source_epoch: int,
    target_epoch: int,
    rank: int,
    world_size: int,
    records: Sequence[D2DevEpochRecord],
    failure_reason: str | None,
) -> dict[str, object]:
    return {
        "protocol": D2_DEV_PREPARE_PROTOCOL,
        "action_plan_sha256": action_plan_sha256,
        "source_version": source_version,
        "target_version": target_version,
        "source_epoch": source_epoch,
        "target_epoch": target_epoch,
        "rank": rank,
        "world_size": world_size,
        "records": [value.to_dict() for value in records],
        "failure_reason": failure_reason,
    }


@dataclass(frozen=True)
class D2DevPrivatePrepare:
    action_plan_sha256: str
    source_version: str
    target_version: str
    source_epoch: int
    target_epoch: int
    rank: int
    world_size: int
    records: tuple[D2DevEpochRecord, ...]
    fragment_sha256: str
    failure_reason: str | None = None
    protocol: str = D2_DEV_PREPARE_PROTOCOL

    def __post_init__(self) -> None:
        _validate_record_sequence(self.records)
        if (
            self.protocol != D2_DEV_PREPARE_PROTOCOL
            or not _is_sha256(self.action_plan_sha256)
            or not self.source_version
            or not self.target_version
            or self.source_epoch < 0
            or self.target_epoch < 0
            or self.rank < 0
            or self.world_size < 1
            or self.rank >= self.world_size
            or any(value.owner_rank >= self.world_size for value in self.records)
            or not _is_sha256(self.fragment_sha256)
            or self.failure_reason == ""
        ):
            raise ValueError("D2 development private prepare is invalid")

    @classmethod
    def create(
        cls,
        *,
        spec: D2DevEpochSpec,
        rank: int,
        records: Sequence[D2DevEpochRecord],
        failure_reason: str | None = None,
    ) -> D2DevPrivatePrepare:
        prepared = tuple(sorted(records, key=lambda value: value.record_id))
        body = _prepare_body(
            action_plan_sha256=spec.action_plan_sha256,
            source_version=spec.source_version,
            target_version=spec.target_version,
            source_epoch=spec.source_epoch,
            target_epoch=spec.target_epoch,
            rank=rank,
            world_size=spec.world_size,
            records=prepared,
            failure_reason=failure_reason,
        )
        return cls(
            action_plan_sha256=spec.action_plan_sha256,
            source_version=spec.source_version,
            target_version=spec.target_version,
            source_epoch=spec.source_epoch,
            target_epoch=spec.target_epoch,
            rank=rank,
            world_size=spec.world_size,
            records=prepared,
            fragment_sha256=canonical_sha256(body),
            failure_reason=failure_reason,
        )

    @property
    def checksum_valid(self) -> bool:
        return self.fragment_sha256 == canonical_sha256(
            _prepare_body(
                action_plan_sha256=self.action_plan_sha256,
                source_version=self.source_version,
                target_version=self.target_version,
                source_epoch=self.source_epoch,
                target_epoch=self.target_epoch,
                rank=self.rank,
                world_size=self.world_size,
                records=self.records,
                failure_reason=self.failure_reason,
            )
        )

    def to_dict(self) -> dict[str, object]:
        value = _prepare_body(
            action_plan_sha256=self.action_plan_sha256,
            source_version=self.source_version,
            target_version=self.target_version,
            source_epoch=self.source_epoch,
            target_epoch=self.target_epoch,
            rank=self.rank,
            world_size=self.world_size,
            records=self.records,
            failure_reason=self.failure_reason,
        )
        value["fragment_sha256"] = self.fragment_sha256
        value["checksum_valid"] = self.checksum_valid
        return value


def _prepare_set_sha256(
    prepares: Sequence[D2DevPrivatePrepare],
) -> str:
    return canonical_sha256(
        {
            "private_prepares": [
                value.to_dict()
                for value in sorted(
                    prepares,
                    key=lambda item: (
                        item.rank,
                        item.fragment_sha256,
                        canonical_sha256(item.to_dict()),
                    ),
                )
            ]
        }
    )


@dataclass(frozen=True)
class D2DevCommitCertificate:
    status: str
    action_plan_sha256: str
    target_version: str
    target_epoch: int
    world_size: int
    record_manifest_sha256: str
    expected_record_ids: tuple[int, ...]
    covered_record_ids: tuple[int, ...]
    prepared_ranks: tuple[int, ...]
    prepare_set_sha256: str
    failure_reasons: tuple[str, ...]
    protocol: str = D2_DEV_COMMIT_PROTOCOL
    scope: str = D2_DEV_EPOCH_SCOPE
    scientific_result: bool = False
    formal_stage_c: bool = False
    timing_included: bool = False
    publishes_target_epoch: bool = False

    def __post_init__(self) -> None:
        if (
            self.protocol != D2_DEV_COMMIT_PROTOCOL
            or self.scope != D2_DEV_EPOCH_SCOPE
            or self.status not in {"commit", "abort"}
            or not _is_sha256(self.action_plan_sha256)
            or not self.target_version
            or self.target_epoch < 1
            or self.world_size < 1
            or not _is_sha256(self.record_manifest_sha256)
            or self.expected_record_ids
            != tuple(sorted(self.expected_record_ids))
            or self.covered_record_ids
            != tuple(sorted(self.covered_record_ids))
            or self.prepared_ranks != tuple(sorted(self.prepared_ranks))
            or len(set(self.prepared_ranks)) != len(self.prepared_ranks)
            or not _is_sha256(self.prepare_set_sha256)
            or self.failure_reasons
            != tuple(sorted(set(self.failure_reasons)))
            or self.scientific_result
            or self.formal_stage_c
            or self.timing_included
            or self.publishes_target_epoch
            or (
                self.status == "commit"
                and (
                    self.failure_reasons
                    or self.covered_record_ids != self.expected_record_ids
                    or self.prepared_ranks != tuple(range(self.world_size))
                )
            )
            or (self.status == "abort" and not self.failure_reasons)
        ):
            raise ValueError("D2 development commit certificate is invalid")

    def _body(self) -> dict[str, object]:
        return {
            "protocol": self.protocol,
            "scope": self.scope,
            "scientific_result": self.scientific_result,
            "formal_stage_c": self.formal_stage_c,
            "timing_included": self.timing_included,
            "status": self.status,
            "action_plan_sha256": self.action_plan_sha256,
            "target_version": self.target_version,
            "target_epoch": self.target_epoch,
            "world_size": self.world_size,
            "record_manifest_sha256": self.record_manifest_sha256,
            "expected_record_ids": list(self.expected_record_ids),
            "covered_record_ids": list(self.covered_record_ids),
            "prepared_ranks": list(self.prepared_ranks),
            "prepare_set_sha256": self.prepare_set_sha256,
            "failure_reasons": list(self.failure_reasons),
            "publishes_target_epoch": self.publishes_target_epoch,
        }

    @property
    def certificate_sha256(self) -> str:
        return canonical_sha256(self._body())

    @property
    def committed(self) -> bool:
        return self.status == "commit"

    def to_dict(self) -> dict[str, object]:
        value = self._body()
        value["certificate_sha256"] = self.certificate_sha256
        return value


def validate_d2_dev_prepares(
    spec: D2DevEpochSpec,
    prepares: Sequence[D2DevPrivatePrepare],
) -> D2DevCommitCertificate:
    failures: list[str] = []
    ordered = tuple(
        sorted(
            prepares,
            key=lambda value: (
                value.rank,
                value.fragment_sha256,
                canonical_sha256(value.to_dict()),
            ),
        )
    )
    rank_counts = Counter(value.rank for value in ordered)
    expected_ranks = set(range(spec.world_size))
    observed_ranks = set(rank_counts)
    missing_ranks = tuple(sorted(expected_ranks - observed_ranks))
    unexpected_ranks = tuple(sorted(observed_ranks - expected_ranks))
    duplicate_ranks = tuple(
        sorted(rank for rank, count in rank_counts.items() if count > 1)
    )
    if missing_ranks:
        failures.append(f"missing rank prepares: {missing_ranks}")
    if unexpected_ranks:
        failures.append(f"unexpected rank prepares: {unexpected_ranks}")
    if duplicate_ranks:
        failures.append(f"duplicate rank prepares: {duplicate_ranks}")
    for value in ordered:
        if value.world_size != spec.world_size:
            failures.append(f"rank {value.rank} world size mismatch")
        if value.action_plan_sha256 != spec.action_plan_sha256:
            failures.append(f"rank {value.rank} action plan mismatch")
        if value.source_version != spec.source_version:
            failures.append(f"rank {value.rank} source version mismatch")
        if value.target_version != spec.target_version:
            failures.append(f"rank {value.rank} target version mismatch")
        if value.source_epoch != spec.source_epoch:
            failures.append(f"rank {value.rank} source epoch mismatch")
        if value.target_epoch != spec.target_epoch:
            failures.append(f"rank {value.rank} target epoch mismatch")
        if not value.checksum_valid:
            failures.append(f"rank {value.rank} prepare checksum mismatch")
        if value.failure_reason is not None:
            failures.append(
                f"rank {value.rank} prepare fault: {value.failure_reason}"
            )
    observed_records = tuple(
        record for value in ordered for record in value.records
    )
    record_counts = Counter(value.record_id for value in observed_records)
    expected_by_id = {value.record_id: value for value in spec.records}
    expected_ids = set(expected_by_id)
    observed_ids = set(record_counts)
    duplicate_records = tuple(
        sorted(
            record_id
            for record_id, count in record_counts.items()
            if count > 1
        )
    )
    missing_records = tuple(sorted(expected_ids - observed_ids))
    unexpected_records = tuple(sorted(observed_ids - expected_ids))
    if duplicate_records:
        failures.append(f"duplicate record prepares: {duplicate_records}")
    if missing_records:
        failures.append(f"missing record prepares: {missing_records}")
    if unexpected_records:
        failures.append(f"unexpected record prepares: {unexpected_records}")
    for prepare in ordered:
        for record in prepare.records:
            expected = expected_by_id.get(record.record_id)
            if expected is None:
                continue
            if (
                record.owner_rank != expected.owner_rank
                or prepare.rank != expected.owner_rank
            ):
                failures.append(
                    f"record {record.record_id} owner mismatch"
                )
            if record.token_length != expected.token_length:
                failures.append(
                    f"record {record.record_id} length mismatch"
                )
            if record.action != expected.action:
                failures.append(
                    f"record {record.record_id} action mismatch"
                )
            if record.lineage_sha256 != expected.lineage_sha256:
                failures.append(
                    f"record {record.record_id} lineage mismatch"
                )
            if record.payload_sha256 != expected.payload_sha256:
                failures.append(
                    f"record {record.record_id} payload checksum mismatch"
                )
    failure_reasons = tuple(sorted(set(failures)))
    return D2DevCommitCertificate(
        status="abort" if failure_reasons else "commit",
        action_plan_sha256=spec.action_plan_sha256,
        target_version=spec.target_version,
        target_epoch=spec.target_epoch,
        world_size=spec.world_size,
        record_manifest_sha256=spec.record_manifest_sha256,
        expected_record_ids=tuple(sorted(expected_ids)),
        covered_record_ids=tuple(sorted(observed_ids)),
        prepared_ranks=tuple(sorted(observed_ranks)),
        prepare_set_sha256=_prepare_set_sha256(ordered),
        failure_reasons=failure_reasons,
    )


def _readback_body(
    *,
    action_plan_sha256: str,
    commit_certificate_sha256: str,
    target_version: str,
    target_epoch: int,
    rank: int,
    world_size: int,
    records: Sequence[D2DevEpochRecord],
    failure_reason: str | None,
) -> dict[str, object]:
    return {
        "protocol": D2_DEV_READBACK_PROTOCOL,
        "action_plan_sha256": action_plan_sha256,
        "commit_certificate_sha256": commit_certificate_sha256,
        "target_version": target_version,
        "target_epoch": target_epoch,
        "rank": rank,
        "world_size": world_size,
        "records": [value.to_dict() for value in records],
        "failure_reason": failure_reason,
    }


@dataclass(frozen=True)
class D2DevReadbackAck:
    action_plan_sha256: str
    commit_certificate_sha256: str
    target_version: str
    target_epoch: int
    rank: int
    world_size: int
    records: tuple[D2DevEpochRecord, ...]
    ack_sha256: str
    failure_reason: str | None = None
    protocol: str = D2_DEV_READBACK_PROTOCOL

    def __post_init__(self) -> None:
        _validate_record_sequence(self.records)
        if (
            self.protocol != D2_DEV_READBACK_PROTOCOL
            or not _is_sha256(self.action_plan_sha256)
            or not _is_sha256(self.commit_certificate_sha256)
            or not self.target_version
            or self.target_epoch < 0
            or self.rank < 0
            or self.world_size < 1
            or self.rank >= self.world_size
            or any(value.owner_rank >= self.world_size for value in self.records)
            or not _is_sha256(self.ack_sha256)
            or self.failure_reason == ""
        ):
            raise ValueError("D2 development readback ACK is invalid")

    @classmethod
    def create(
        cls,
        *,
        spec: D2DevEpochSpec,
        commit: D2DevCommitCertificate,
        rank: int,
        records: Sequence[D2DevEpochRecord],
        failure_reason: str | None = None,
    ) -> D2DevReadbackAck:
        if not commit.committed:
            raise ValueError("D2 development readback requires commit")
        prepared = tuple(sorted(records, key=lambda value: value.record_id))
        body = _readback_body(
            action_plan_sha256=spec.action_plan_sha256,
            commit_certificate_sha256=commit.certificate_sha256,
            target_version=spec.target_version,
            target_epoch=spec.target_epoch,
            rank=rank,
            world_size=spec.world_size,
            records=prepared,
            failure_reason=failure_reason,
        )
        return cls(
            action_plan_sha256=spec.action_plan_sha256,
            commit_certificate_sha256=commit.certificate_sha256,
            target_version=spec.target_version,
            target_epoch=spec.target_epoch,
            rank=rank,
            world_size=spec.world_size,
            records=prepared,
            ack_sha256=canonical_sha256(body),
            failure_reason=failure_reason,
        )

    @property
    def checksum_valid(self) -> bool:
        return self.ack_sha256 == canonical_sha256(
            _readback_body(
                action_plan_sha256=self.action_plan_sha256,
                commit_certificate_sha256=self.commit_certificate_sha256,
                target_version=self.target_version,
                target_epoch=self.target_epoch,
                rank=self.rank,
                world_size=self.world_size,
                records=self.records,
                failure_reason=self.failure_reason,
            )
        )

    def to_dict(self) -> dict[str, object]:
        value = _readback_body(
            action_plan_sha256=self.action_plan_sha256,
            commit_certificate_sha256=self.commit_certificate_sha256,
            target_version=self.target_version,
            target_epoch=self.target_epoch,
            rank=self.rank,
            world_size=self.world_size,
            records=self.records,
            failure_reason=self.failure_reason,
        )
        value["ack_sha256"] = self.ack_sha256
        value["checksum_valid"] = self.checksum_valid
        return value


def _readback_set_sha256(
    readbacks: Sequence[D2DevReadbackAck],
) -> str:
    return canonical_sha256(
        {
            "readback_acks": [
                value.to_dict()
                for value in sorted(
                    readbacks,
                    key=lambda item: (
                        item.rank,
                        item.ack_sha256,
                        canonical_sha256(item.to_dict()),
                    ),
                )
            ]
        }
    )


@dataclass(frozen=True)
class D2DevEpochPointer:
    version: str
    epoch: int
    certificate_sha256: str
    authoritative: bool = True
    protocol: str = D2_DEV_POINTER_PROTOCOL

    def __post_init__(self) -> None:
        if (
            self.protocol != D2_DEV_POINTER_PROTOCOL
            or not self.version
            or self.epoch < 0
            or not _is_sha256(self.certificate_sha256)
            or not self.authoritative
        ):
            raise ValueError("D2 development visible pointer is invalid")

    @property
    def pointer_sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol": self.protocol,
            "version": self.version,
            "epoch": self.epoch,
            "certificate_sha256": self.certificate_sha256,
            "authoritative": self.authoritative,
        }


@dataclass(frozen=True)
class D2DevPublicationCertificate:
    status: str
    action_plan_sha256: str
    target_version: str
    target_epoch: int
    world_size: int
    commit_certificate_sha256: str
    source_pointer_sha256: str
    readback_set_sha256: str
    acknowledged_ranks: tuple[int, ...]
    failure_reasons: tuple[str, ...]
    protocol: str = D2_DEV_PUBLICATION_PROTOCOL
    scope: str = D2_DEV_EPOCH_SCOPE
    scientific_result: bool = False
    formal_stage_c: bool = False
    timing_included: bool = False

    def __post_init__(self) -> None:
        if (
            self.protocol != D2_DEV_PUBLICATION_PROTOCOL
            or self.scope != D2_DEV_EPOCH_SCOPE
            or self.status not in {"published", "abort"}
            or not _is_sha256(self.action_plan_sha256)
            or not self.target_version
            or self.target_epoch < 1
            or self.world_size < 1
            or not _is_sha256(self.commit_certificate_sha256)
            or not _is_sha256(self.source_pointer_sha256)
            or not _is_sha256(self.readback_set_sha256)
            or self.acknowledged_ranks
            != tuple(sorted(self.acknowledged_ranks))
            or len(set(self.acknowledged_ranks))
            != len(self.acknowledged_ranks)
            or self.failure_reasons
            != tuple(sorted(set(self.failure_reasons)))
            or self.scientific_result
            or self.formal_stage_c
            or self.timing_included
            or (
                self.status == "published"
                and (
                    self.failure_reasons
                    or self.acknowledged_ranks
                    != tuple(range(self.world_size))
                )
            )
            or (self.status == "abort" and not self.failure_reasons)
        ):
            raise ValueError(
                "D2 development publication certificate is invalid"
            )

    @property
    def publishes_target_epoch(self) -> bool:
        return self.status == "published"

    def _body(self) -> dict[str, object]:
        return {
            "protocol": self.protocol,
            "scope": self.scope,
            "scientific_result": self.scientific_result,
            "formal_stage_c": self.formal_stage_c,
            "timing_included": self.timing_included,
            "status": self.status,
            "action_plan_sha256": self.action_plan_sha256,
            "target_version": self.target_version,
            "target_epoch": self.target_epoch,
            "world_size": self.world_size,
            "commit_certificate_sha256": (
                self.commit_certificate_sha256
            ),
            "source_pointer_sha256": self.source_pointer_sha256,
            "readback_set_sha256": self.readback_set_sha256,
            "acknowledged_ranks": list(self.acknowledged_ranks),
            "failure_reasons": list(self.failure_reasons),
            "publishes_target_epoch": self.publishes_target_epoch,
        }

    @property
    def certificate_sha256(self) -> str:
        return canonical_sha256(self._body())

    def to_dict(self) -> dict[str, object]:
        value = self._body()
        value["certificate_sha256"] = self.certificate_sha256
        return value


def validate_d2_dev_readbacks(
    spec: D2DevEpochSpec,
    commit: D2DevCommitCertificate,
    source_pointer: D2DevEpochPointer,
    readbacks: Sequence[D2DevReadbackAck],
) -> D2DevPublicationCertificate:
    if (
        not commit.committed
        or commit.action_plan_sha256 != spec.action_plan_sha256
        or commit.target_version != spec.target_version
        or commit.target_epoch != spec.target_epoch
        or commit.world_size != spec.world_size
        or commit.record_manifest_sha256 != spec.record_manifest_sha256
        or source_pointer.version != spec.source_version
        or source_pointer.epoch != spec.source_epoch
    ):
        raise ValueError("D2 development readback commit is invalid")
    failures: list[str] = []
    ordered = tuple(
        sorted(
            readbacks,
            key=lambda value: (
                value.rank,
                value.ack_sha256,
                canonical_sha256(value.to_dict()),
            ),
        )
    )
    rank_counts = Counter(value.rank for value in ordered)
    expected_ranks = set(range(spec.world_size))
    observed_ranks = set(rank_counts)
    missing_ranks = tuple(sorted(expected_ranks - observed_ranks))
    unexpected_ranks = tuple(sorted(observed_ranks - expected_ranks))
    duplicate_ranks = tuple(
        sorted(rank for rank, count in rank_counts.items() if count > 1)
    )
    if missing_ranks:
        failures.append(f"missing readback ACK ranks: {missing_ranks}")
    if unexpected_ranks:
        failures.append(
            f"unexpected readback ACK ranks: {unexpected_ranks}"
        )
    if duplicate_ranks:
        failures.append(f"duplicate readback ACK ranks: {duplicate_ranks}")
    for value in ordered:
        if value.world_size != spec.world_size:
            failures.append(f"rank {value.rank} readback world size mismatch")
        if value.action_plan_sha256 != spec.action_plan_sha256:
            failures.append(f"rank {value.rank} readback action plan mismatch")
        if (
            value.commit_certificate_sha256
            != commit.certificate_sha256
        ):
            failures.append(
                f"rank {value.rank} readback commit mismatch"
            )
        if value.target_version != spec.target_version:
            failures.append(
                f"rank {value.rank} readback target version mismatch"
            )
        if value.target_epoch != spec.target_epoch:
            failures.append(
                f"rank {value.rank} readback target epoch mismatch"
            )
        if not value.checksum_valid:
            failures.append(
                f"rank {value.rank} readback ACK checksum mismatch"
            )
        if value.failure_reason is not None:
            failures.append(
                f"rank {value.rank} readback fault: {value.failure_reason}"
            )
    observed_records = tuple(
        record for value in ordered for record in value.records
    )
    record_counts = Counter(value.record_id for value in observed_records)
    expected_by_id = {value.record_id: value for value in spec.records}
    expected_ids = set(expected_by_id)
    observed_ids = set(record_counts)
    duplicate_records = tuple(
        sorted(
            record_id
            for record_id, count in record_counts.items()
            if count > 1
        )
    )
    missing_records = tuple(sorted(expected_ids - observed_ids))
    unexpected_records = tuple(sorted(observed_ids - expected_ids))
    if duplicate_records:
        failures.append(f"duplicate readback records: {duplicate_records}")
    if missing_records:
        failures.append(f"missing readback records: {missing_records}")
    if unexpected_records:
        failures.append(f"unexpected readback records: {unexpected_records}")
    for readback in ordered:
        for record in readback.records:
            expected = expected_by_id.get(record.record_id)
            if expected is None:
                continue
            if (
                record.owner_rank != expected.owner_rank
                or readback.rank != expected.owner_rank
            ):
                failures.append(
                    f"record {record.record_id} readback owner mismatch"
                )
            if record.token_length != expected.token_length:
                failures.append(
                    f"record {record.record_id} readback length mismatch"
                )
            if record.action != expected.action:
                failures.append(
                    f"record {record.record_id} readback action mismatch"
                )
            if record.lineage_sha256 != expected.lineage_sha256:
                failures.append(
                    f"record {record.record_id} readback lineage mismatch"
                )
            if record.payload_sha256 != expected.payload_sha256:
                failures.append(
                    f"record {record.record_id} readback payload checksum mismatch"
                )
    failure_reasons = tuple(sorted(set(failures)))
    return D2DevPublicationCertificate(
        status="abort" if failure_reasons else "published",
        action_plan_sha256=spec.action_plan_sha256,
        target_version=spec.target_version,
        target_epoch=spec.target_epoch,
        world_size=spec.world_size,
        commit_certificate_sha256=commit.certificate_sha256,
        source_pointer_sha256=source_pointer.pointer_sha256,
        readback_set_sha256=_readback_set_sha256(ordered),
        acknowledged_ranks=tuple(sorted(observed_ranks)),
        failure_reasons=failure_reasons,
    )


class D2DevEpochStateMachine:
    def __init__(
        self,
        spec: D2DevEpochSpec,
        source_pointer: D2DevEpochPointer,
    ) -> None:
        if (
            source_pointer.version != spec.source_version
            or source_pointer.epoch != spec.source_epoch
        ):
            raise ValueError(
                "D2 development source pointer differs from specification"
            )
        self._spec = spec
        self._source_pointer = source_pointer
        self._visible_pointer = source_pointer
        self._commit: D2DevCommitCertificate | None = None
        self._publication: D2DevPublicationCertificate | None = None

    @property
    def state(self) -> str:
        if self._publication is not None:
            return (
                "published"
                if self._publication.publishes_target_epoch
                else "aborted"
            )
        if self._commit is None:
            return "private_prepare"
        return "awaiting_readback" if self._commit.committed else "aborted"

    @property
    def visible_pointer(self) -> D2DevEpochPointer:
        return self._visible_pointer

    @property
    def target_visible(self) -> bool:
        return (
            self._visible_pointer.version == self._spec.target_version
            and self._visible_pointer.epoch == self._spec.target_epoch
        )

    @property
    def commit_certificate(self) -> D2DevCommitCertificate | None:
        return self._commit

    @property
    def publication_certificate(
        self,
    ) -> D2DevPublicationCertificate | None:
        return self._publication

    def resolve_visible(self, version: str) -> D2DevEpochPointer:
        if version != self._visible_pointer.version:
            raise KeyError(f"D2 epoch {version} is not globally visible")
        return self._visible_pointer

    def decide_prepares(
        self,
        prepares: Sequence[D2DevPrivatePrepare],
    ) -> D2DevCommitCertificate:
        if self._commit is not None:
            raise RuntimeError("D2 development prepare decision already exists")
        self._commit = validate_d2_dev_prepares(self._spec, prepares)
        return self._commit

    def validate_readbacks(
        self,
        readbacks: Sequence[D2DevReadbackAck],
    ) -> D2DevPublicationCertificate:
        if self._commit is None:
            raise RuntimeError(
                "D2 development readback precedes prepare decision"
            )
        if not self._commit.committed:
            raise RuntimeError(
                "D2 development aborted prepare cannot publish"
            )
        if self._publication is not None:
            raise RuntimeError(
                "D2 development publication decision already exists"
            )
        self._publication = validate_d2_dev_readbacks(
            self._spec,
            self._commit,
            self._source_pointer,
            readbacks,
        )
        if self._publication.publishes_target_epoch:
            self._visible_pointer = D2DevEpochPointer(
                version=self._spec.target_version,
                epoch=self._spec.target_epoch,
                certificate_sha256=self._publication.certificate_sha256,
            )
        return self._publication
