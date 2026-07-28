from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
import pandas as pd

from ..data import StreamingDataPlan

ORGANIC_STREAM_PROTOCOL = "kuairand_long_context_4plus12_organic_stream_v1"


def _array(values, dtype) -> np.ndarray:
    output = np.array(values, dtype=dtype, copy=True, order="C")
    output.setflags(write=False)
    return output


def _update_array_hash(
    digest: hashlib._Hash,
    name: str,
    values: np.ndarray,
) -> None:
    digest.update(name.encode("utf-8"))
    digest.update(struct.pack("<Q", len(values)))
    digest.update(values.tobytes(order="C"))


def stable_event_identity(
    user_id: int,
    item_id: int,
    behavior: int,
    label: int,
    timestamp_ms: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"cohortkv-organic-event-v1")
    digest.update(
        struct.pack(
            "<qqqqq",
            int(user_id),
            int(item_id),
            int(behavior),
            int(label),
            int(timestamp_ms),
        )
    )
    return digest.hexdigest()


@dataclass(frozen=True)
class CanonicalEvents:
    item_ids: np.ndarray
    behaviors: np.ndarray
    time_deltas: np.ndarray
    labels: np.ndarray
    timestamps: np.ndarray

    def __post_init__(self) -> None:
        arrays = {
            "item_ids": _array(self.item_ids, np.int64),
            "behaviors": _array(self.behaviors, np.int64),
            "time_deltas": _array(self.time_deltas, np.float32),
            "labels": _array(self.labels, np.int64),
            "timestamps": _array(self.timestamps, np.int64),
        }
        lengths = {len(values) for values in arrays.values()}
        if len(lengths) != 1:
            raise ValueError("canonical event arrays must have equal lengths")
        for name, values in arrays.items():
            object.__setattr__(self, name, values)

    @classmethod
    def empty(cls) -> CanonicalEvents:
        return cls(
            item_ids=np.empty(0, dtype=np.int64),
            behaviors=np.empty(0, dtype=np.int64),
            time_deltas=np.empty(0, dtype=np.float32),
            labels=np.empty(0, dtype=np.int64),
            timestamps=np.empty(0, dtype=np.int64),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> CanonicalEvents:
        return cls(
            item_ids=value["item_ids"],
            behaviors=value["behaviors"],
            time_deltas=value["time_deltas"],
            labels=value["labels"],
            timestamps=value["timestamps"],
        )

    def __len__(self) -> int:
        return len(self.item_ids)

    def identities(self, user_id: int) -> tuple[str, ...]:
        return tuple(
            stable_event_identity(
                user_id,
                item_id,
                behavior,
                label,
                timestamp,
            )
            for item_id, behavior, label, timestamp in zip(
                self.item_ids,
                self.behaviors,
                self.labels,
                self.timestamps,
                strict=True,
            )
        )

    def to_dict(self) -> dict[str, list[int] | list[float]]:
        return {
            "item_ids": self.item_ids.tolist(),
            "behaviors": self.behaviors.tolist(),
            "time_deltas": self.time_deltas.tolist(),
            "labels": self.labels.tolist(),
            "timestamps": self.timestamps.tolist(),
        }


@dataclass(frozen=True)
class CanonicalHistory:
    events: CanonicalEvents
    available_length_before_token_cap: int
    token_truncated: bool

    def __post_init__(self) -> None:
        available = int(self.available_length_before_token_cap)
        if available < len(self.events):
            raise ValueError("available history length is smaller than resident history")
        if bool(self.token_truncated) != (available > len(self.events)):
            raise ValueError("history truncation metadata differs from resident history")
        object.__setattr__(self, "available_length_before_token_cap", available)
        object.__setattr__(self, "token_truncated", bool(self.token_truncated))

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> CanonicalHistory:
        return cls(
            events=CanonicalEvents.from_dict(value["events"]),
            available_length_before_token_cap=int(
                value["available_length_before_token_cap"]
            ),
            token_truncated=bool(value["token_truncated"]),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "events": self.events.to_dict(),
            "available_length_before_token_cap": (
                self.available_length_before_token_cap
            ),
            "token_truncated": self.token_truncated,
        }

    @property
    def item_ids(self) -> np.ndarray:
        return self.events.item_ids

    @property
    def behaviors(self) -> np.ndarray:
        return self.events.behaviors

    @property
    def time_deltas(self) -> np.ndarray:
        return self.events.time_deltas

    @property
    def labels(self) -> np.ndarray:
        return self.events.labels

    @property
    def timestamps(self) -> np.ndarray:
        return self.events.timestamps

    def __len__(self) -> int:
        return len(self.events)


def stable_history_hash(
    user_id: int,
    history: CanonicalHistory | None,
) -> str | None:
    if history is None:
        return None
    digest = hashlib.sha256()
    digest.update(b"cohortkv-organic-history-v1")
    digest.update(struct.pack("<q", int(user_id)))
    digest.update(
        struct.pack(
            "<Q?",
            history.available_length_before_token_cap,
            history.token_truncated,
        )
    )
    _update_array_hash(digest, "item_ids", history.events.item_ids)
    _update_array_hash(digest, "behaviors", history.events.behaviors)
    _update_array_hash(digest, "time_deltas", history.events.time_deltas)
    _update_array_hash(digest, "labels", history.events.labels)
    _update_array_hash(digest, "timestamps", history.events.timestamps)
    return digest.hexdigest()


@dataclass(frozen=True)
class OrganicStreamRecord:
    user_id: int
    as_of_timestamp_ms: int
    history: CanonicalHistory | None
    engaged_positive_item_ids: tuple[int, ...]
    new_events: CanonicalEvents
    history_sha256: str | None = field(init=False)
    history_event_identities: tuple[str, ...] = field(init=False)
    new_event_identities: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        user_id = int(self.user_id)
        positives = tuple(int(value) for value in self.engaged_positive_item_ids)
        if user_id < 1:
            raise ValueError("organic record user id must be positive")
        if len(set(positives)) != len(positives):
            raise ValueError("engaged positives must be unique")
        object.__setattr__(self, "user_id", user_id)
        object.__setattr__(self, "as_of_timestamp_ms", int(self.as_of_timestamp_ms))
        object.__setattr__(self, "engaged_positive_item_ids", positives)
        object.__setattr__(
            self,
            "history_sha256",
            stable_history_hash(user_id, self.history),
        )
        object.__setattr__(
            self,
            "history_event_identities",
            (
                ()
                if self.history is None
                else self.history.events.identities(user_id)
            ),
        )
        object.__setattr__(
            self,
            "new_event_identities",
            self.new_events.identities(user_id),
        )

    @property
    def active(self) -> bool:
        return len(self.new_events) > 0

    @property
    def positives(self) -> tuple[int, ...]:
        return self.engaged_positive_item_ids

    @property
    def history_hash(self) -> str | None:
        return self.history_sha256

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> OrganicStreamRecord:
        history_value = value.get("history")
        record = cls(
            user_id=int(value["user_id"]),
            as_of_timestamp_ms=int(value["as_of_timestamp_ms"]),
            history=(
                None
                if history_value is None
                else CanonicalHistory.from_dict(history_value)
            ),
            engaged_positive_item_ids=tuple(
                int(item) for item in value["engaged_positive_item_ids"]
            ),
            new_events=CanonicalEvents.from_dict(value["new_events"]),
        )
        expected = {
            "history_sha256": record.history_sha256,
            "history_event_identities": list(record.history_event_identities),
            "new_event_identities": list(record.new_event_identities),
        }
        for name, actual in expected.items():
            if name in value and value[name] != actual:
                raise ValueError(f"serialized organic record {name} differs")
        return record

    def to_dict(self) -> dict[str, object]:
        return {
            "user_id": self.user_id,
            "as_of_timestamp_ms": self.as_of_timestamp_ms,
            "active": self.active,
            "history": None if self.history is None else self.history.to_dict(),
            "history_sha256": self.history_sha256,
            "history_event_identities": list(self.history_event_identities),
            "engaged_positive_item_ids": list(self.engaged_positive_item_ids),
            "new_events": self.new_events.to_dict(),
            "new_event_identities": list(self.new_event_identities),
        }


@dataclass(frozen=True)
class OrganicStreamSnapshot:
    version: int
    target_date: str
    wall_clock_as_of_timestamp_ms: int
    records: Mapping[int, OrganicStreamRecord]

    def __post_init__(self) -> None:
        version = int(self.version)
        if isinstance(self.records, Mapping):
            supplied = tuple(self.records.values())
            if any(
                int(key) != record.user_id
                for key, record in self.records.items()
            ):
                raise ValueError("organic snapshot record key differs from user id")
        else:
            supplied = tuple(self.records)
        records = tuple(sorted(supplied, key=lambda record: record.user_id))
        user_ids = tuple(record.user_id for record in records)
        if not 0 <= version <= 11:
            raise ValueError("organic snapshot version must be in [0, 11]")
        if not self.target_date:
            raise ValueError("organic snapshot target date is empty")
        if user_ids != tuple(sorted(user_ids)) or len(set(user_ids)) != len(user_ids):
            raise ValueError("organic snapshot users must be unique and sorted")
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "target_date", str(self.target_date))
        object.__setattr__(
            self,
            "wall_clock_as_of_timestamp_ms",
            int(self.wall_clock_as_of_timestamp_ms),
        )
        object.__setattr__(
            self,
            "records",
            MappingProxyType(
                {record.user_id: record for record in records}
            ),
        )

    def _payload(self) -> dict[str, object]:
        return {
            "protocol": ORGANIC_STREAM_PROTOCOL,
            "version": self.version,
            "target_date": self.target_date,
            "wall_clock_as_of_timestamp_ms": self.wall_clock_as_of_timestamp_ms,
            "records": [
                record.to_dict() for record in self.records.values()
            ],
        }

    @property
    def content_sha256(self) -> str:
        encoded = json.dumps(
            self._payload(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> OrganicStreamSnapshot:
        if value.get("protocol") != ORGANIC_STREAM_PROTOCOL:
            raise ValueError("organic snapshot protocol differs")
        snapshot = cls(
            version=int(value["version"]),
            target_date=str(value["target_date"]),
            wall_clock_as_of_timestamp_ms=int(
                value["wall_clock_as_of_timestamp_ms"]
            ),
            records={
                record.user_id: record
                for record in (
                    OrganicStreamRecord.from_dict(serialized)
                    for serialized in value["records"]
                )
            },
        )
        if (
            "content_sha256" in value
            and value["content_sha256"] != snapshot.content_sha256
        ):
            raise ValueError("serialized organic snapshot hash differs")
        return snapshot

    def to_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "content_sha256": self.content_sha256,
        }


def _history(sequence: dict | None) -> CanonicalHistory | None:
    if sequence is None:
        return None
    return CanonicalHistory(
        events=CanonicalEvents(
            item_ids=sequence["item_ids"],
            behaviors=sequence["behaviors"],
            time_deltas=sequence["time_deltas"],
            labels=sequence["labels"],
            timestamps=sequence["timestamps"],
        ),
        available_length_before_token_cap=int(
            sequence["available_length_before_token_cap"]
        ),
        token_truncated=bool(sequence["token_truncated"]),
    )


def _new_events(
    plan: StreamingDataPlan,
    user_id: int,
    frame: pd.DataFrame | None,
) -> CanonicalEvents:
    if frame is None or frame.empty:
        return CanonicalEvents.empty()
    ordered = frame.sort_values("time_ms")
    timestamps = ordered["time_ms"].to_numpy(dtype=np.int64)
    time_deltas = np.zeros(len(ordered), dtype=np.float32)
    previous = plan.user_histories[user_id]["timestamps"]
    if len(previous):
        time_deltas[0] = max(
            0.0,
            (int(timestamps[0]) - int(previous[-1])) / 1000.0,
        )
    if len(timestamps) > 1:
        time_deltas[1:] = (
            np.maximum(np.diff(timestamps), 0).astype(np.float64) / 1000.0
        )
    return CanonicalEvents(
        item_ids=ordered["item_idx"].to_numpy(dtype=np.int64),
        behaviors=ordered["behavior"].to_numpy(dtype=np.int64),
        time_deltas=time_deltas,
        labels=ordered["label"].to_numpy(dtype=np.int64),
        timestamps=timestamps,
    )


def _validate_plan(
    plan: StreamingDataPlan,
    user_ids: tuple[int, ...],
) -> None:
    if len(plan.base_dates) != 4 or len(plan.stream_dates) != 12:
        raise ValueError("organic stream requires a 4+12 plan")
    if not user_ids or len(set(user_ids)) != len(user_ids):
        raise ValueError("base-cohort user ids must be nonempty and unique")
    missing = set(user_ids) - set(plan.user_histories)
    if missing:
        raise ValueError(f"base-cohort users are absent from the plan: {sorted(missing)}")
    if any(
        len(history["item_ids"]) > 0
        for history in plan.user_histories.values()
    ):
        raise ValueError("organic stream replay requires an uninitialized plan")


def iter_4plus12_causal_snapshots(
    plan: StreamingDataPlan,
    user_ids: Iterable[int],
) -> Iterator[OrganicStreamSnapshot]:
    ordered_user_ids = tuple(sorted(int(value) for value in user_ids))
    _validate_plan(plan, ordered_user_ids)
    plan.init_base()
    empty_base = [
        user_id
        for user_id in ordered_user_ids
        if len(plan.user_histories[user_id]["item_ids"]) == 0
    ]
    if empty_base:
        raise ValueError(f"users lack base-period history: {empty_base}")
    for version, target_date in enumerate(plan.stream_dates):
        target = plan.daily_segments.get(target_date)
        if target is None or target.empty:
            raise ValueError(f"target date has no events: {target_date}")
        wall_clock = int(target["time_ms"].min())
        by_user = {
            int(user_id): frame
            for user_id, frame in target.groupby("user_idx", sort=False)
        }
        records = {}
        for user_id in ordered_user_ids:
            frame = by_user.get(user_id)
            as_of = (
                wall_clock
                if frame is None
                else int(frame["time_ms"].min())
            )
            positives = (
                ()
                if frame is None
                else tuple(
                    int(value)
                    for value in frame.loc[
                        frame["label"] > 0,
                        "item_idx",
                    ].unique()
                )
            )
            record = OrganicStreamRecord(
                user_id=user_id,
                as_of_timestamp_ms=as_of,
                history=_history(
                    plan._build_seq(
                        user_id,
                        as_of_timestamp=as_of,
                    )
                ),
                engaged_positive_item_ids=positives,
                new_events=_new_events(plan, user_id, frame),
            )
            records[user_id] = record
        yield OrganicStreamSnapshot(
            version=version,
            target_date=target_date,
            wall_clock_as_of_timestamp_ms=wall_clock,
            records=records,
        )
        if version < len(plan.stream_dates) - 1:
            plan.ingest_day(target_date)


def build_4plus12_causal_snapshots(
    plan: StreamingDataPlan,
    user_ids: Iterable[int],
) -> tuple[OrganicStreamSnapshot, ...]:
    return tuple(iter_4plus12_causal_snapshots(plan, user_ids))


OrganicWindowRecord = OrganicStreamRecord
OrganicWindow = OrganicStreamSnapshot


def reconstruct_organic_windows(
    plan: StreamingDataPlan,
    user_ids: Iterable[int],
) -> tuple[OrganicWindow, ...]:
    return build_4plus12_causal_snapshots(plan, user_ids)
