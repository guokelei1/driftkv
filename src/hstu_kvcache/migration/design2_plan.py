from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

D2_ACTION_PLAN_PROTOCOL = "cohortkv_d2_action_plan_v1"
_STAGE49_PROTOCOL = "cohortkv_single_config_stage4_9_same_device_confirmation_v2"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_VERSION = re.compile(r"^theta([0-9]+)$")


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _version_index(value: str) -> int:
    match = _VERSION.fullmatch(value)
    if match is None:
        raise ValueError("D2 version must have thetaN form")
    return int(match.group(1))


def build_d2_record_owner_map(
    plan: D2ActionPlan,
    world_size: int,
    strategy: str,
) -> dict[int, int]:
    if world_size < 1:
        raise ValueError("D2 owner world size must be positive")
    if strategy == "modulo":
        return {
            value.record_id: value.record_id % world_size
            for value in plan.records
        }
    if strategy not in {"old_kv_lpt", "strict_cow_lpt"}:
        raise ValueError("D2 owner strategy is unsupported")
    loads = [0] * world_size
    if strategy == "old_kv_lpt":
        weight = lambda value: value.old_tokens
    else:
        weight = lambda value: value.old_tokens + value.final_tokens
    ordered = sorted(
        plan.records,
        key=lambda value: (-weight(value), value.record_id),
    )
    owner_map = {}
    for record in ordered:
        owner = min(range(world_size), key=lambda rank: (loads[rank], rank))
        owner_map[record.record_id] = owner
        loads[owner] += weight(record)
    return dict(sorted(owner_map.items()))


def d2_record_owner_map_sha256(
    owner_map: Mapping[int, int],
) -> str:
    return canonical_sha256(
        {
            "record_owner_map": [
                {
                    "record_id": int(record_id),
                    "owner_rank": int(owner),
                }
                for record_id, owner in sorted(owner_map.items())
            ]
        }
    )


@dataclass(frozen=True)
class D2ActionProvenance:
    artifact: str
    artifact_sha256: str
    artifact_protocol: str
    step_index: int
    action_partition_sha256: str
    lineage_sha256: str
    prepared_data: str
    prepared_data_sha256: str
    manifest_content_sha256: str
    target_window_content_sha256: str

    def __post_init__(self) -> None:
        if (
            not self.artifact
            or not _SHA256.fullmatch(self.artifact_sha256)
            or self.artifact_protocol != _STAGE49_PROTOCOL
            or self.step_index < 0
            or not _SHA256.fullmatch(self.action_partition_sha256)
            or not _SHA256.fullmatch(self.lineage_sha256)
            or not self.prepared_data
            or not _SHA256.fullmatch(self.prepared_data_sha256)
            or not _SHA256.fullmatch(self.manifest_content_sha256)
            or not _SHA256.fullmatch(
                self.target_window_content_sha256
            )
        ):
            raise ValueError("D2 action provenance is invalid")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> D2ActionProvenance:
        return cls(
            artifact=str(value["artifact"]),
            artifact_sha256=str(value["artifact_sha256"]),
            artifact_protocol=str(value["artifact_protocol"]),
            step_index=int(value["step_index"]),
            action_partition_sha256=str(value["action_partition_sha256"]),
            lineage_sha256=str(value["lineage_sha256"]),
            prepared_data=str(value["prepared_data"]),
            prepared_data_sha256=str(value["prepared_data_sha256"]),
            manifest_content_sha256=str(
                value["manifest_content_sha256"]
            ),
            target_window_content_sha256=str(
                value["target_window_content_sha256"]
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class D2ActionRecord:
    record_id: int
    prepared_user_id: int
    requested_action: str
    requested_reason: str
    old_tokens: int
    retained_start: int
    retained_tokens: int
    delta_start: int
    delta_tokens: int
    target_prefix_tokens: int
    latest_tokens: int
    final_tokens: int
    last_exact_version: str | None
    migration_depth: int
    previous_cache_expected: bool
    previous_cache_present: bool
    old_history_sha256: str | None
    target_history_sha256: str
    retained_identity_sha256: str
    delta_identity_sha256: str
    target_prefix_identity_sha256: str

    def __post_init__(self) -> None:
        hashes = (
            self.target_history_sha256,
            self.retained_identity_sha256,
            self.delta_identity_sha256,
            self.target_prefix_identity_sha256,
        )
        if (
            self.record_id < 0
            or self.prepared_user_id < 1
            or self.requested_action not in {"compiled", "exact"}
            or self.requested_reason
            not in {"migrate", "scheduled_exact", "natural_exact"}
            or (
                self.requested_action == "compiled"
                and self.requested_reason != "migrate"
            )
            or (
                self.requested_action == "exact"
                and self.requested_reason
                not in {"scheduled_exact", "natural_exact"}
            )
            or any(
                value < 0
                for value in (
                    self.old_tokens,
                    self.retained_start,
                    self.retained_tokens,
                    self.delta_start,
                    self.delta_tokens,
                    self.target_prefix_tokens,
                    self.latest_tokens,
                    self.migration_depth,
                )
            )
            or self.final_tokens < 1
            or self.retained_start + self.retained_tokens > self.old_tokens
            or self.delta_start != self.retained_tokens
            or self.retained_tokens + self.delta_tokens
            != self.target_prefix_tokens
            or self.target_prefix_tokens + self.latest_tokens
            != self.final_tokens
            or self.latest_tokens != 1
            or self.previous_cache_present and not self.previous_cache_expected
            or (
                self.previous_cache_present
                and (
                    self.old_tokens < 1
                    or self.old_history_sha256 is None
                    or self.retained_start + self.retained_tokens
                    != self.old_tokens
                )
            )
            or (
                self.requested_action == "compiled"
                and (
                    self.retained_tokens < 1
                    or not self.previous_cache_present
                    or self.last_exact_version is None
                )
            )
            or (
                self.requested_reason == "scheduled_exact"
                and (
                    self.retained_tokens < 1
                    or not self.previous_cache_present
                    or self.last_exact_version is None
                )
            )
            or (
                self.last_exact_version is not None
                and _VERSION.fullmatch(self.last_exact_version) is None
            )
            or any(not _SHA256.fullmatch(value) for value in hashes)
            or (
                self.old_history_sha256 is not None
                and not _SHA256.fullmatch(self.old_history_sha256)
            )
        ):
            raise ValueError("D2 action record is invalid")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> D2ActionRecord:
        return cls(
            record_id=int(value["record_id"]),
            prepared_user_id=int(value["prepared_user_id"]),
            requested_action=str(value["requested_action"]),
            requested_reason=str(value["requested_reason"]),
            old_tokens=int(value["old_tokens"]),
            retained_start=int(value["retained_start"]),
            retained_tokens=int(value["retained_tokens"]),
            delta_start=int(value["delta_start"]),
            delta_tokens=int(value["delta_tokens"]),
            target_prefix_tokens=int(value["target_prefix_tokens"]),
            latest_tokens=int(value["latest_tokens"]),
            final_tokens=int(value["final_tokens"]),
            last_exact_version=(
                None
                if value.get("last_exact_version") is None
                else str(value["last_exact_version"])
            ),
            migration_depth=int(value["migration_depth"]),
            previous_cache_expected=bool(value["previous_cache_expected"]),
            previous_cache_present=bool(value["previous_cache_present"]),
            old_history_sha256=(
                None
                if value.get("old_history_sha256") is None
                else str(value["old_history_sha256"])
            ),
            target_history_sha256=str(value["target_history_sha256"]),
            retained_identity_sha256=str(value["retained_identity_sha256"]),
            delta_identity_sha256=str(value["delta_identity_sha256"]),
            target_prefix_identity_sha256=str(
                value["target_prefix_identity_sha256"]
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class D2ActionCounts:
    compiled: int
    scheduled_exact: int
    natural_exact: int
    records: int

    def __post_init__(self) -> None:
        if (
            min(
                self.compiled,
                self.scheduled_exact,
                self.natural_exact,
                self.records,
            )
            < 0
            or self.compiled + self.scheduled_exact + self.natural_exact
            != self.records
        ):
            raise ValueError("D2 action counts are invalid")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> D2ActionCounts:
        return cls(
            compiled=int(value["compiled"]),
            scheduled_exact=int(value["scheduled_exact"]),
            natural_exact=int(value["natural_exact"]),
            records=int(value["records"]),
        )

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class D2ActionPlan:
    source_version: str
    target_version: str
    producer: str
    policy: str
    provenance: D2ActionProvenance
    records: tuple[D2ActionRecord, ...]
    counts: D2ActionCounts
    protocol: str = D2_ACTION_PLAN_PROTOCOL

    def __post_init__(self) -> None:
        source_index = _version_index(self.source_version)
        target_index = _version_index(self.target_version)
        record_ids = tuple(value.record_id for value in self.records)
        observed = Counter(
            (
                "compiled"
                if value.requested_action == "compiled"
                else value.requested_reason
            )
            for value in self.records
        )
        expected_partition = canonical_sha256(
            {
                "migrate_ids": [
                    value.record_id
                    for value in self.records
                    if value.requested_reason == "migrate"
                ],
                "scheduled_exact_ids": [
                    value.record_id
                    for value in self.records
                    if value.requested_reason == "scheduled_exact"
                ],
                "natural_exact_ids": [
                    value.record_id
                    for value in self.records
                    if value.requested_reason == "natural_exact"
                ],
            }
        )
        if (
            self.protocol != D2_ACTION_PLAN_PROTOCOL
            or target_index <= source_index
            or not self.producer
            or not self.policy
            or not self.records
            or record_ids != tuple(sorted(record_ids))
            or len(set(record_ids)) != len(record_ids)
            or self.counts.records != len(self.records)
            or observed["compiled"] != self.counts.compiled
            or observed["scheduled_exact"] != self.counts.scheduled_exact
            or observed["natural_exact"] != self.counts.natural_exact
            or expected_partition
            != self.provenance.action_partition_sha256
            or any(
                value.last_exact_version is not None
                and _version_index(value.last_exact_version) > source_index
                for value in self.records
            )
            or any(
                value.migration_depth
                != (
                    0
                    if value.last_exact_version is None
                    else source_index
                    - _version_index(value.last_exact_version)
                )
                for value in self.records
            )
        ):
            raise ValueError("D2 action plan is invalid")

    def payload_dict(self) -> dict[str, object]:
        return {
            "protocol": self.protocol,
            "source_version": self.source_version,
            "target_version": self.target_version,
            "producer": self.producer,
            "policy": self.policy,
            "provenance": self.provenance.to_dict(),
            "records": [value.to_dict() for value in self.records],
            "counts": self.counts.to_dict(),
        }

    @property
    def content_sha256(self) -> str:
        return canonical_sha256(self.payload_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            **self.payload_dict(),
            "content_sha256": self.content_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> D2ActionPlan:
        plan = cls(
            protocol=str(value["protocol"]),
            source_version=str(value["source_version"]),
            target_version=str(value["target_version"]),
            producer=str(value["producer"]),
            policy=str(value["policy"]),
            provenance=D2ActionProvenance.from_dict(value["provenance"]),
            records=tuple(
                D2ActionRecord.from_dict(record)
                for record in value["records"]
            ),
            counts=D2ActionCounts.from_dict(value["counts"]),
        )
        if value.get("content_sha256") != plan.content_sha256:
            raise ValueError("D2 action plan content hash differs")
        return plan

    @classmethod
    def load(cls, path: str | Path) -> D2ActionPlan:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def write(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(canonical_json_bytes(self.to_dict()))


def export_stage49_h12_action_plan(
    artifact_path: str | Path,
    step_index: int = 1,
) -> D2ActionPlan:
    artifact_path = Path(artifact_path)
    artifact = json.loads(artifact_path.read_text())
    if (
        artifact.get("protocol") != _STAGE49_PROTOCOL
        or artifact.get("candidate_name") != "staggered_renewal_h12"
        or artifact.get("status") != "complete"
        or not bool(artifact.get("scientific_result"))
    ):
        raise ValueError("D2 upstream H12 artifact differs")
    steps = artifact.get("steps")
    if not isinstance(steps, Sequence) or not 0 <= step_index < len(steps):
        raise ValueError("D2 upstream step index is invalid")
    step = steps[step_index]
    source_index = int(step["source_version"])
    target_index = int(step["target_version"])
    lineage = step["lineage"]
    if (
        not isinstance(lineage, list)
        or canonical_sha256(lineage) != step["lineage_sha256"]
    ):
        raise ValueError("D2 upstream lineage differs")
    records = []
    for value in sorted(lineage, key=lambda item: int(item["record_id"])):
        upstream_action = str(value["action"])
        if upstream_action not in {
            "migrate",
            "scheduled_exact",
            "natural_exact",
        }:
            raise ValueError("D2 upstream action differs")
        last_exact_index = value.get("last_exact_version_before")
        migration_depth = (
            0
            if last_exact_index is None
            else source_index - int(last_exact_index)
        )
        if migration_depth < 0:
            raise ValueError("D2 upstream migration depth differs")
        expected_after_depth = (
            migration_depth + 1
            if upstream_action == "migrate"
            else 0
        )
        if int(value["migration_depth_after"]) != expected_after_depth:
            raise ValueError("D2 upstream post-action depth differs")
        records.append(
            D2ActionRecord(
                record_id=int(value["record_id"]),
                prepared_user_id=int(value["user_id"]),
                requested_action=(
                    "compiled" if upstream_action == "migrate" else "exact"
                ),
                requested_reason=upstream_action,
                old_tokens=int(value["old_tokens"]),
                retained_start=int(value["retained_start"]),
                retained_tokens=int(value["retained_tokens"]),
                delta_start=int(value["delta_start"]),
                delta_tokens=int(value["delta_tokens"]),
                target_prefix_tokens=int(value["target_prefix_tokens"]),
                latest_tokens=int(value["latest_tokens"]),
                final_tokens=int(value["final_tokens"]),
                last_exact_version=(
                    None
                    if last_exact_index is None
                    else f"theta{int(last_exact_index)}"
                ),
                migration_depth=migration_depth,
                previous_cache_expected=bool(
                    value["previous_cache_expected"]
                ),
                previous_cache_present=bool(
                    value["previous_cache_present"]
                ),
                old_history_sha256=value.get("old_history_sha256"),
                target_history_sha256=str(
                    value["target_history_sha256"]
                ),
                retained_identity_sha256=str(
                    value["retained_identity_sha256"]
                ),
                delta_identity_sha256=str(
                    value["delta_identity_sha256"]
                ),
                target_prefix_identity_sha256=str(
                    value["target_prefix_identity_sha256"]
                ),
            )
        )
    counts = Counter(value.requested_reason for value in records)
    action_counts = step["actions"]
    if (
        counts["migrate"] != int(action_counts["migrate"])
        or counts["scheduled_exact"]
        != int(action_counts["scheduled_exact"])
        or counts["natural_exact"]
        != int(action_counts["natural_exact"])
    ):
        raise ValueError("D2 upstream action counts differ")
    input_provenance = artifact["input_provenance"]
    prepared_data = input_provenance["prepared_data"]
    manifest = input_provenance["manifest"]
    target_windows = [
        value
        for value in input_provenance["windows"]
        if int(value["version"]) == target_index
    ]
    if (
        len(target_windows) != 1
        or manifest["records"] != len(records)
    ):
        raise ValueError("D2 upstream input provenance differs")
    return D2ActionPlan(
        source_version=f"theta{source_index}",
        target_version=f"theta{target_index}",
        producer="h12_frozen_lineage",
        policy="staggered_renewal_h12",
        provenance=D2ActionProvenance(
            artifact=str(artifact_path),
            artifact_sha256=file_sha256(artifact_path),
            artifact_protocol=str(artifact["protocol"]),
            step_index=step_index,
            action_partition_sha256=str(
                step["scheduler"]["action_partition_sha256"]
            ),
            lineage_sha256=str(step["lineage_sha256"]),
            prepared_data=str(prepared_data["path"]),
            prepared_data_sha256=str(prepared_data["sha256"]),
            manifest_content_sha256=str(
                manifest["content_sha256"]
            ),
            target_window_content_sha256=str(
                target_windows[0]["content_sha256"]
            ),
        ),
        records=tuple(records),
        counts=D2ActionCounts(
            compiled=counts["migrate"],
            scheduled_exact=counts["scheduled_exact"],
            natural_exact=counts["natural_exact"],
            records=len(records),
        ),
    )
