from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import torch

from .cohort_jagged import JaggedMigratedKVBatch
from .design2_plan import D2ActionRecord, canonical_sha256
from .stage5_closure import jagged_kv_sha256

D2_DEV_WAVE_PROTOCOL = "cohortkv_d2_dev_c0_wave_v1"
D2_DEV_LINEAGE_PROTOCOL = "cohortkv_d2_dev_c0_lineage_v1"
D2_DEV_RECORD_PAYLOAD_PROTOCOL = (
    "cohortkv_d2_dev_c0_record_payload_v1"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ROUTES = {"compiled", "scheduled_exact", "natural_exact"}


def d2_dev_route(action: D2ActionRecord) -> str:
    if action.requested_action == "compiled":
        return "compiled"
    if action.requested_reason == "scheduled_exact":
        return "scheduled_exact"
    if action.requested_reason == "natural_exact":
        return "natural_exact"
    raise ValueError("D2 dev action has no integrated-wave route")


@dataclass(frozen=True)
class D2DevWaveLineage:
    record_id: int
    owner_rank: int
    world_size: int
    route: str
    source_version: str
    target_version: str
    old_tokens: int
    retained_start: int
    retained_tokens: int
    delta_start: int
    delta_tokens: int
    target_prefix_tokens: int
    latest_tokens: int
    final_tokens: int
    source_history_sha256: str | None
    target_history_sha256: str
    protocol: str = D2_DEV_LINEAGE_PROTOCOL

    def __post_init__(self) -> None:
        if (
            self.protocol != D2_DEV_LINEAGE_PROTOCOL
            or self.record_id < 0
            or self.world_size < 1
            or not 0 <= self.owner_rank < self.world_size
            or self.route not in _ROUTES
            or not self.source_version
            or not self.target_version
            or self.source_version == self.target_version
            or min(
                self.old_tokens,
                self.retained_start,
                self.retained_tokens,
                self.delta_start,
                self.delta_tokens,
                self.target_prefix_tokens,
                self.latest_tokens,
            )
            < 0
            or self.final_tokens < 1
            or self.retained_start + self.retained_tokens
            > self.old_tokens
            or self.delta_start != self.retained_tokens
            or self.retained_tokens + self.delta_tokens
            != self.target_prefix_tokens
            or self.target_prefix_tokens + self.latest_tokens
            != self.final_tokens
            or self.latest_tokens < 1
            or not _SHA256.fullmatch(self.target_history_sha256)
            or (
                self.source_history_sha256 is not None
                and not _SHA256.fullmatch(self.source_history_sha256)
            )
            or (
                self.route in {"compiled", "scheduled_exact"}
                and self.retained_tokens < 1
            )
            or (
                self.route == "compiled"
                and self.source_history_sha256 is None
            )
            or (
                self.route == "natural_exact"
                and (
                    self.retained_tokens != 0
                    or self.delta_start != 0
                    or self.delta_tokens != self.target_prefix_tokens
                )
            )
        ):
            raise ValueError("D2 dev integrated-wave lineage is invalid")

    @classmethod
    def from_action(
        cls,
        action: D2ActionRecord,
        owner_rank: int,
        world_size: int,
        source_version: str,
        target_version: str,
    ) -> D2DevWaveLineage:
        return cls(
            record_id=action.record_id,
            owner_rank=owner_rank,
            world_size=world_size,
            route=d2_dev_route(action),
            source_version=source_version,
            target_version=target_version,
            old_tokens=action.old_tokens,
            retained_start=action.retained_start,
            retained_tokens=action.retained_tokens,
            delta_start=action.delta_start,
            delta_tokens=action.delta_tokens,
            target_prefix_tokens=action.target_prefix_tokens,
            latest_tokens=action.latest_tokens,
            final_tokens=action.final_tokens,
            source_history_sha256=action.old_history_sha256,
            target_history_sha256=action.target_history_sha256,
        )

    @property
    def phase_tokens(self) -> dict[str, int]:
        return {
            "source_old_kv_fixture": (
                self.old_tokens if self.route == "compiled" else 0
            ),
            "compiled_retained": (
                self.retained_tokens if self.route == "compiled" else 0
            ),
            "scheduled_exact_retained": (
                self.retained_tokens
                if self.route == "scheduled_exact"
                else 0
            ),
            "natural_exact_prefix": (
                self.target_prefix_tokens
                if self.route == "natural_exact"
                else 0
            ),
            "delta_append": (
                self.delta_tokens
                if self.route in {"compiled", "scheduled_exact"}
                else 0
            ),
            "latest_append": self.latest_tokens,
            "final": self.final_tokens,
        }

    @property
    def lineage_sha256(self) -> str:
        return canonical_sha256(self.to_dict(include_hash=False))

    def to_dict(self, include_hash: bool = True) -> dict[str, object]:
        value = asdict(self)
        value["phase_tokens"] = self.phase_tokens
        if include_hash:
            value["lineage_sha256"] = self.lineage_sha256
        return value


def build_d2_dev_lineages(
    actions: Sequence[D2ActionRecord],
    owner_map: Mapping[int, int],
    world_size: int,
    source_version: str,
    target_version: str,
) -> tuple[D2DevWaveLineage, ...]:
    record_ids = tuple(value.record_id for value in actions)
    if (
        not record_ids
        or len(set(record_ids)) != len(record_ids)
        or set(record_ids) - set(owner_map)
    ):
        raise ValueError("D2 dev actions and owner map differ")
    return tuple(
        D2DevWaveLineage.from_action(
            action,
            owner_map[action.record_id],
            world_size,
            source_version,
            target_version,
        )
        for action in sorted(actions, key=lambda value: value.record_id)
    )


def assemble_d2_dev_jagged(
    record_ids: Sequence[int],
    fragments: Sequence[JaggedMigratedKVBatch | None],
    source_version: str,
    target_version: str,
) -> JaggedMigratedKVBatch | None:
    prepared_ids = tuple(int(value) for value in record_ids)
    if len(set(prepared_ids)) != len(prepared_ids):
        raise ValueError("D2 dev assembled record IDs must be unique")
    rows: dict[int, tuple[JaggedMigratedKVBatch, int]] = {}
    signature = None
    for fragment in fragments:
        if fragment is None:
            continue
        if (
            fragment.migration_anchor_version != source_version
            or fragment.served_kv_target != target_version
        ):
            raise ValueError("D2 dev assembled fragment versions differ")
        current_signature = (
            fragment.k.shape[0],
            fragment.k.shape[2],
            fragment.k.dtype,
            fragment.k.device,
        )
        if signature is None:
            signature = current_signature
        elif signature != current_signature:
            raise ValueError("D2 dev assembled fragment layouts differ")
        for row, record_id in enumerate(fragment.record_ids):
            if record_id in rows:
                raise ValueError("D2 dev assembled record appears twice")
            rows[record_id] = (fragment, row)
    if set(rows) != set(prepared_ids):
        raise ValueError("D2 dev assembled fragments do not close coverage")
    if not prepared_ids:
        return None
    k_rows = []
    v_rows = []
    lengths = []
    for record_id in prepared_ids:
        fragment, row = rows[record_id]
        start = int(fragment.offsets[row])
        stop = int(fragment.offsets[row + 1])
        k_rows.append(fragment.k[:, start:stop])
        v_rows.append(fragment.v[:, start:stop])
        lengths.append(stop - start)
    length_tensor = torch.tensor(
        lengths,
        dtype=torch.long,
        device=k_rows[0].device,
    )
    offsets = torch.cat(
        (
            torch.zeros(
                1,
                dtype=torch.long,
                device=length_tensor.device,
            ),
            length_tensor.cumsum(0),
        )
    )
    return JaggedMigratedKVBatch(
        record_ids=prepared_ids,
        migration_anchor_version=source_version,
        served_kv_target=target_version,
        k=torch.cat(k_rows, dim=1).contiguous(),
        v=torch.cat(v_rows, dim=1).contiguous(),
        lengths=length_tensor,
        offsets=offsets,
    )


def d2_dev_record_payload_sha256(
    fragment: JaggedMigratedKVBatch,
    record_id: int,
) -> str:
    row = fragment.record_index(record_id)
    start = int(fragment.offsets[row])
    stop = int(fragment.offsets[row + 1])
    digest = hashlib.sha256()
    digest.update(D2_DEV_RECORD_PAYLOAD_PROTOCOL.encode("utf-8"))
    digest.update(str(record_id).encode("utf-8"))
    digest.update(fragment.migration_anchor_version.encode("utf-8"))
    digest.update(fragment.served_kv_target.encode("utf-8"))
    for value in (
        fragment.k[:, start:stop],
        fragment.v[:, start:stop],
        fragment.lengths[row : row + 1],
    ):
        tensor = value.detach().contiguous()
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class D2DevWaveClosure:
    rank: int
    world_size: int
    source_version: str
    target_version: str
    record_ids: tuple[int, ...]
    record_payload_sha256: tuple[str, ...]
    record_lineage_sha256: tuple[str, ...]
    token_count: int
    payload_bytes: int
    fragment_sha256: str
    lineage_set_sha256: str
    fp16: bool
    finite: bool
    owner_closed: bool
    length_closed: bool
    lineage_closed: bool
    scientific_result: bool = False
    formal_stage_c: bool = False
    protocol: str = D2_DEV_WAVE_PROTOCOL

    def __post_init__(self) -> None:
        if (
            self.protocol != D2_DEV_WAVE_PROTOCOL
            or self.rank < 0
            or self.world_size < 1
            or self.rank >= self.world_size
            or not self.source_version
            or not self.target_version
            or len(self.record_ids) != len(self.record_payload_sha256)
            or len(self.record_ids) != len(self.record_lineage_sha256)
            or tuple(sorted(self.record_ids)) != self.record_ids
            or len(set(self.record_ids)) != len(self.record_ids)
            or self.token_count < 0
            or self.payload_bytes < 0
            or any(
                not _SHA256.fullmatch(value)
                for value in (
                    *self.record_payload_sha256,
                    *self.record_lineage_sha256,
                    self.fragment_sha256,
                    self.lineage_set_sha256,
                )
            )
            or self.scientific_result
            or self.formal_stage_c
        ):
            raise ValueError("D2 dev integrated-wave closure is invalid")

    @property
    def passed(self) -> bool:
        return all(
            (
                self.fp16,
                self.finite,
                self.owner_closed,
                self.length_closed,
                self.lineage_closed,
            )
        )

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["passed"] = self.passed
        value["records"] = [
            {
                "record_id": record_id,
                "payload_sha256": payload,
                "lineage_sha256": lineage,
            }
            for record_id, payload, lineage in zip(
                self.record_ids,
                self.record_payload_sha256,
                self.record_lineage_sha256,
                strict=True,
            )
        ]
        return value


def close_d2_dev_wave(
    fragment: JaggedMigratedKVBatch | None,
    lineages: Sequence[D2DevWaveLineage],
    rank: int,
    world_size: int,
    source_version: str,
    target_version: str,
) -> D2DevWaveClosure:
    prepared = tuple(sorted(lineages, key=lambda value: value.record_id))
    record_ids = tuple(value.record_id for value in prepared)
    owner_closed = all(
        value.owner_rank == rank and value.world_size == world_size
        for value in prepared
    )
    lineage_closed = (
        len(set(record_ids)) == len(record_ids)
        and all(
            value.source_version == source_version
            and value.target_version == target_version
            for value in prepared
        )
    )
    if fragment is None:
        payload_hashes: tuple[str, ...] = ()
        token_count = 0
        payload_bytes = 0
        fragment_hash = canonical_sha256(
            {
                "protocol": D2_DEV_WAVE_PROTOCOL,
                "rank": rank,
                "empty": True,
            }
        )
        fp16 = not prepared
        finite = not prepared
        length_closed = not prepared
    else:
        if fragment.record_ids != record_ids:
            raise ValueError("D2 dev final fragment order differs from lineage")
        payload_hashes = tuple(
            d2_dev_record_payload_sha256(fragment, record_id)
            for record_id in record_ids
        )
        lengths = tuple(
            int(value) for value in fragment.lengths.detach().cpu()
        )
        token_count = fragment.token_count
        payload_bytes = fragment.nbytes
        fragment_hash = jagged_kv_sha256(fragment)
        fp16 = fragment.k.dtype == torch.float16
        finite = bool(
            torch.isfinite(fragment.k).all()
            and torch.isfinite(fragment.v).all()
        )
        length_closed = lengths == tuple(
            value.final_tokens for value in prepared
        )
        lineage_closed = (
            lineage_closed
            and fragment.migration_anchor_version == target_version
            and fragment.served_kv_target == target_version
        )
    lineage_hashes = tuple(value.lineage_sha256 for value in prepared)
    return D2DevWaveClosure(
        rank=rank,
        world_size=world_size,
        source_version=source_version,
        target_version=target_version,
        record_ids=record_ids,
        record_payload_sha256=payload_hashes,
        record_lineage_sha256=lineage_hashes,
        token_count=token_count,
        payload_bytes=payload_bytes,
        fragment_sha256=fragment_hash,
        lineage_set_sha256=canonical_sha256(
            {
                "protocol": D2_DEV_LINEAGE_PROTOCOL,
                "lineages": [
                    value.to_dict() for value in prepared
                ],
            }
        ),
        fp16=fp16,
        finite=finite,
        owner_closed=owner_closed,
        length_closed=length_closed,
        lineage_closed=lineage_closed,
    )
