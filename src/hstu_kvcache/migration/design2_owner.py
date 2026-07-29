from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Protocol

import torch

from .cohort_jagged import JaggedMigratedKVBatch
from .stage45_oldkv import (
    DirectOldKVProgram,
    execute_direct_oldkv_reference,
)

D2_OWNER_COMPUTE_PROTOCOL = "cohortkv_d2_owner_compute_v1"
D2_OWNER_FRAGMENT_PROTOCOL = "cohortkv_d2_owner_fragment_v1"
D2_PRELIMINARY_PLACEMENT_PROTOCOL = (
    "cohortkv_d2_preliminary_placement_ledger_v1"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class D2DirectOldKVOperator(Protocol):
    def execute_into(
        self,
        program: DirectOldKVProgram,
        source: JaggedMigratedKVBatch,
        destination: JaggedMigratedKVBatch,
    ) -> JaggedMigratedKVBatch: ...


class _Digest(Protocol):
    def update(self, value: bytes) -> None: ...


D2DirectOldKVExecutor = Callable[
    [
        DirectOldKVProgram,
        JaggedMigratedKVBatch,
        JaggedMigratedKVBatch,
    ],
    JaggedMigratedKVBatch,
]


@dataclass(frozen=True)
class D2CompiledRetainedPhaseCounters:
    item_lookup_calls: int = 0
    embedding_collective_count: int = 0
    embedding_collective_bytes: int = 0
    old_kv_p2p_bytes: int = 0

    def __post_init__(self) -> None:
        if min(
            self.item_lookup_calls,
            self.embedding_collective_count,
            self.embedding_collective_bytes,
            self.old_kv_p2p_bytes,
        ) < 0:
            raise ValueError("compiled-retained phase counters must be nonnegative")

    def assert_normal_path(self) -> None:
        if (
            self.item_lookup_calls != 0
            or self.embedding_collective_count != 0
            or self.embedding_collective_bytes != 0
            or self.old_kv_p2p_bytes != 0
        ):
            raise RuntimeError(
                "compiled-retained owner-compute communication invariant failed"
            )

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class D2OwnerFragmentMetadata:
    owner_rank: int
    source_version: str
    target_version: str
    record_ids: tuple[int, ...]
    lengths: tuple[int, ...]
    token_count: int
    num_layers: int
    kv_width: int
    dtype: str
    device: str
    kv_payload_bytes: int
    extent_bytes: int
    checksum_sha256: str
    ready: bool = True
    protocol: str = D2_OWNER_FRAGMENT_PROTOCOL

    def __post_init__(self) -> None:
        if (
            self.protocol != D2_OWNER_FRAGMENT_PROTOCOL
            or self.owner_rank < 0
            or not self.source_version
            or not self.target_version
            or len(self.record_ids) != len(self.lengths)
            or len(set(self.record_ids)) != len(self.record_ids)
            or any(record_id < 0 for record_id in self.record_ids)
            or any(length < 1 for length in self.lengths)
            or self.token_count != sum(self.lengths)
            or self.num_layers < 1
            or self.kv_width < 1
            or not self.dtype
            or not self.device
            or self.kv_payload_bytes < 0
            or self.extent_bytes < self.kv_payload_bytes
            or not _SHA256.fullmatch(self.checksum_sha256)
            or not self.ready
            or (
                not self.record_ids
                and (
                    self.lengths
                    or self.token_count != 0
                    or self.kv_payload_bytes != 0
                    or self.extent_bytes != 0
                )
            )
        ):
            raise ValueError("D2 owner fragment metadata is invalid")

    @property
    def empty(self) -> bool:
        return not self.record_ids

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class D2OwnerComputeMetrics:
    owner_rank: int
    operator: str
    record_count: int
    token_count: int
    source_extent_bytes: int
    output_extent_bytes: int
    program_bytes: int
    elapsed_seconds: float
    phase_counters: D2CompiledRetainedPhaseCounters
    private_output: bool
    scientific_result: bool = False
    protocol: str = D2_OWNER_COMPUTE_PROTOCOL

    def __post_init__(self) -> None:
        if (
            self.protocol != D2_OWNER_COMPUTE_PROTOCOL
            or self.owner_rank < 0
            or not self.operator
            or min(
                self.record_count,
                self.token_count,
                self.source_extent_bytes,
                self.output_extent_bytes,
                self.program_bytes,
            )
            < 0
            or self.elapsed_seconds < 0
            or not self.private_output
            or self.scientific_result
        ):
            raise ValueError("D2 owner-compute metrics are invalid")
        self.phase_counters.assert_normal_path()

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "phase_counters": self.phase_counters.to_dict(),
        }


@dataclass(frozen=True)
class D2OwnerComputeFragment:
    metadata: D2OwnerFragmentMetadata
    output: JaggedMigratedKVBatch | None
    metrics: D2OwnerComputeMetrics

    def __post_init__(self) -> None:
        if (
            self.metadata.owner_rank != self.metrics.owner_rank
            or len(self.metadata.record_ids) != self.metrics.record_count
            or self.metadata.token_count != self.metrics.token_count
            or self.metadata.extent_bytes != self.metrics.output_extent_bytes
            or self.metadata.empty != (self.output is None)
        ):
            raise ValueError("D2 owner-compute fragment is inconsistent")
        if self.output is not None and (
            self.output.record_ids != self.metadata.record_ids
            or self.output.migration_anchor_version
            != self.metadata.source_version
            or self.output.served_kv_target != self.metadata.target_version
            or tuple(int(value) for value in self.output.lengths.detach().cpu())
            != self.metadata.lengths
            or self.output.token_count != self.metadata.token_count
            or self.output.nbytes != self.metadata.extent_bytes
            or d2_owner_fragment_sha256(
                self.metadata.source_version,
                self.metadata.target_version,
                self.output,
            )
            != self.metadata.checksum_sha256
        ):
            raise ValueError("D2 owner-compute output differs from metadata")

    @property
    def ready(self) -> bool:
        return self.metadata.ready

    @property
    def empty(self) -> bool:
        return self.metadata.empty

    def to_dict(self) -> dict[str, object]:
        return {
            "metadata": self.metadata.to_dict(),
            "metrics": self.metrics.to_dict(),
        }


@dataclass(frozen=True)
class D2PlacementRankBytes:
    rank: int
    old_kv_send_bytes: int
    old_kv_receive_bytes: int
    target_kv_send_bytes: int
    target_kv_receive_bytes: int

    def __post_init__(self) -> None:
        if self.rank < 0 or min(
            self.old_kv_send_bytes,
            self.old_kv_receive_bytes,
            self.target_kv_send_bytes,
            self.target_kv_receive_bytes,
        ) < 0:
            raise ValueError("D2 placement rank bytes are invalid")

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class D2PreliminaryPlacementLedger:
    record_ids: tuple[int, ...]
    owner_local_records: int
    p2p_steal_records: int
    nonowner_output_return_records: int
    old_kv_p2p_bytes: int
    target_kv_return_bytes: int
    total_p2p_bytes: int
    per_rank: tuple[D2PlacementRankBytes, ...]
    preliminary_baseline: bool = True
    measured_transport: bool = False
    scientific_result: bool = False
    protocol: str = D2_PRELIMINARY_PLACEMENT_PROTOCOL

    def __post_init__(self) -> None:
        ranks = tuple(value.rank for value in self.per_rank)
        if (
            self.protocol != D2_PRELIMINARY_PLACEMENT_PROTOCOL
            or not self.record_ids
            or self.record_ids != tuple(sorted(self.record_ids))
            or len(set(self.record_ids)) != len(self.record_ids)
            or min(
                self.owner_local_records,
                self.p2p_steal_records,
                self.nonowner_output_return_records,
                self.old_kv_p2p_bytes,
                self.target_kv_return_bytes,
                self.total_p2p_bytes,
            )
            < 0
            or self.owner_local_records + self.p2p_steal_records
            != len(self.record_ids)
            or self.total_p2p_bytes
            != self.old_kv_p2p_bytes + self.target_kv_return_bytes
            or ranks != tuple(sorted(ranks))
            or len(set(ranks)) != len(ranks)
            or sum(value.old_kv_send_bytes for value in self.per_rank)
            != self.old_kv_p2p_bytes
            or sum(value.old_kv_receive_bytes for value in self.per_rank)
            != self.old_kv_p2p_bytes
            or sum(value.target_kv_send_bytes for value in self.per_rank)
            != self.target_kv_return_bytes
            or sum(value.target_kv_receive_bytes for value in self.per_rank)
            != self.target_kv_return_bytes
            or not self.preliminary_baseline
            or self.measured_transport
            or self.scientific_result
        ):
            raise ValueError("D2 preliminary placement ledger is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "per_rank": [value.to_dict() for value in self.per_rank],
        }


def _update_tensor_digest(
    digest: _Digest,
    value: torch.Tensor,
) -> None:
    tensor = value.detach().contiguous()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(tensor.view(torch.uint8).cpu().numpy().tobytes())


def d2_owner_fragment_sha256(
    source_version: str,
    target_version: str,
    output: JaggedMigratedKVBatch | None,
) -> str:
    if not source_version or not target_version:
        raise ValueError("D2 fragment versions must be nonempty")
    digest = hashlib.sha256()
    digest.update(D2_OWNER_FRAGMENT_PROTOCOL.encode("utf-8"))
    digest.update(source_version.encode("utf-8"))
    digest.update(target_version.encode("utf-8"))
    if output is None:
        digest.update(b"empty")
        return digest.hexdigest()
    if (
        output.migration_anchor_version != source_version
        or output.served_kv_target != target_version
    ):
        raise ValueError("D2 fragment versions differ from output")
    digest.update(str(output.record_ids).encode("utf-8"))
    for value in (output.k, output.v, output.lengths, output.offsets):
        _update_tensor_digest(digest, value)
    return digest.hexdigest()


def _operator_name(
    operator: D2DirectOldKVOperator | D2DirectOldKVExecutor,
) -> str:
    name = getattr(operator, "name", None)
    if isinstance(name, str) and name:
        return name
    function_name = getattr(operator, "__name__", None)
    if isinstance(function_name, str) and function_name:
        return function_name
    return type(operator).__name__


def _prepare_program(
    operator: D2DirectOldKVOperator | D2DirectOldKVExecutor,
    program: DirectOldKVProgram,
    device: torch.device,
) -> DirectOldKVProgram:
    prepare = getattr(operator, "prepare_program", None)
    if prepare is None:
        return program
    prepared = prepare(program, device)
    if not isinstance(prepared, DirectOldKVProgram):
        raise TypeError("direct old-K/V operator returned an invalid program")
    if (
        prepared.source_version != program.source_version
        or prepared.target_version != program.target_version
        or prepared.num_layers != program.num_layers
        or prepared.kv_width != program.kv_width
    ):
        raise ValueError("prepared direct old-K/V program changed its signature")
    return prepared


def _execute_operator(
    operator: D2DirectOldKVOperator | D2DirectOldKVExecutor,
    program: DirectOldKVProgram,
    source: JaggedMigratedKVBatch,
    destination: JaggedMigratedKVBatch,
) -> JaggedMigratedKVBatch:
    execute_into = getattr(operator, "execute_into", None)
    if execute_into is not None:
        result = execute_into(program, source, destination)
    elif callable(operator):
        result = operator(program, source, destination)
    else:
        raise TypeError("direct old-K/V operator is not executable")
    if result is not destination:
        raise RuntimeError("direct old-K/V operator did not return private destination")
    return result


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _validate_owner_map(
    record_owner_by_id: Mapping[int, int],
) -> None:
    if any(
        not isinstance(record_id, int)
        or isinstance(record_id, bool)
        or record_id < 0
        or not isinstance(owner, int)
        or isinstance(owner, bool)
        or owner < 0
        for record_id, owner in record_owner_by_id.items()
    ):
        raise ValueError("D2 record owner map is invalid")


@torch.no_grad()
def execute_compiled_retained_owner_compute(
    program: DirectOldKVProgram,
    source: JaggedMigratedKVBatch | None,
    record_owner_by_id: Mapping[int, int],
    rank: int,
    operator: D2DirectOldKVOperator | D2DirectOldKVExecutor = (
        execute_direct_oldkv_reference
    ),
    phase_counters: D2CompiledRetainedPhaseCounters | None = None,
) -> D2OwnerComputeFragment:
    if not isinstance(rank, int) or isinstance(rank, bool) or rank < 0:
        raise ValueError("D2 owner-compute rank is invalid")
    _validate_owner_map(record_owner_by_id)
    if phase_counters is None:
        phase_counters = D2CompiledRetainedPhaseCounters()
    phase_counters.assert_normal_path()
    operator_name = _operator_name(operator)
    if source is None:
        checksum = d2_owner_fragment_sha256(
            program.source_version,
            program.target_version,
            None,
        )
        metadata = D2OwnerFragmentMetadata(
            owner_rank=rank,
            source_version=program.source_version,
            target_version=program.target_version,
            record_ids=(),
            lengths=(),
            token_count=0,
            num_layers=program.num_layers,
            kv_width=program.kv_width,
            dtype=str(program.weights.dtype),
            device=str(program.device),
            kv_payload_bytes=0,
            extent_bytes=0,
            checksum_sha256=checksum,
        )
        metrics = D2OwnerComputeMetrics(
            owner_rank=rank,
            operator=operator_name,
            record_count=0,
            token_count=0,
            source_extent_bytes=0,
            output_extent_bytes=0,
            program_bytes=program.nbytes,
            elapsed_seconds=0.0,
            phase_counters=phase_counters,
            private_output=True,
        )
        return D2OwnerComputeFragment(
            metadata=metadata,
            output=None,
            metrics=metrics,
        )
    missing = tuple(
        record_id
        for record_id in source.record_ids
        if record_id not in record_owner_by_id
    )
    nonowners = tuple(
        record_id
        for record_id in source.record_ids
        if record_owner_by_id.get(record_id) != rank
    )
    if missing:
        raise ValueError("source records are missing owner assignments")
    if nonowners:
        raise ValueError("rank-local source contains non-owner records")
    if (
        source.migration_anchor_version != program.source_version
        or source.served_kv_target != program.source_version
    ):
        raise ValueError("rank-local source is not exact source-version K/V")
    prepared = _prepare_program(operator, program, source.k.device)
    output = JaggedMigratedKVBatch(
        record_ids=source.record_ids,
        migration_anchor_version=program.source_version,
        served_kv_target=program.target_version,
        k=torch.empty_like(source.k),
        v=torch.empty_like(source.v),
        lengths=source.lengths.clone(),
        offsets=source.offsets.clone(),
    )
    _synchronize(source.k.device)
    started = time.perf_counter()
    result = _execute_operator(operator, prepared, source, output)
    _synchronize(source.k.device)
    elapsed = time.perf_counter() - started
    if (
        result.k.untyped_storage().data_ptr()
        in {
            source.k.untyped_storage().data_ptr(),
            source.v.untyped_storage().data_ptr(),
            prepared.weights.untyped_storage().data_ptr(),
            prepared.biases.untyped_storage().data_ptr(),
        }
        or result.v.untyped_storage().data_ptr()
        in {
            source.k.untyped_storage().data_ptr(),
            source.v.untyped_storage().data_ptr(),
            prepared.weights.untyped_storage().data_ptr(),
            prepared.biases.untyped_storage().data_ptr(),
        }
    ):
        raise RuntimeError("compiled-retained output aliases a source or program")
    lengths = tuple(int(value) for value in result.lengths.detach().cpu())
    kv_payload_bytes = sum(
        value.numel() * value.element_size() for value in (result.k, result.v)
    )
    checksum = d2_owner_fragment_sha256(
        program.source_version,
        program.target_version,
        result,
    )
    metadata = D2OwnerFragmentMetadata(
        owner_rank=rank,
        source_version=program.source_version,
        target_version=program.target_version,
        record_ids=result.record_ids,
        lengths=lengths,
        token_count=result.token_count,
        num_layers=result.k.shape[0],
        kv_width=result.k.shape[2],
        dtype=str(result.k.dtype),
        device=str(result.k.device),
        kv_payload_bytes=kv_payload_bytes,
        extent_bytes=result.nbytes,
        checksum_sha256=checksum,
    )
    metrics = D2OwnerComputeMetrics(
        owner_rank=rank,
        operator=operator_name,
        record_count=result.batch_size,
        token_count=result.token_count,
        source_extent_bytes=source.nbytes,
        output_extent_bytes=result.nbytes,
        program_bytes=prepared.nbytes,
        elapsed_seconds=elapsed,
        phase_counters=phase_counters,
        private_output=True,
    )
    return D2OwnerComputeFragment(
        metadata=metadata,
        output=result,
        metrics=metrics,
    )


def jagged_kv_payload_bytes_by_record(
    batch: JaggedMigratedKVBatch,
) -> dict[int, int]:
    bytes_per_token = (
        2
        * batch.k.shape[0]
        * batch.k.shape[2]
        * batch.k.element_size()
    )
    values = {
        record_id: int(batch.lengths[index]) * bytes_per_token
        for index, record_id in enumerate(batch.record_ids)
    }
    if sum(values.values()) != sum(
        tensor.numel() * tensor.element_size()
        for tensor in (batch.k, batch.v)
    ):
        raise RuntimeError("jagged K/V record byte accounting is inconsistent")
    return values


def characterize_p2p_steal_and_return(
    old_kv_bytes_by_record: Mapping[int, int],
    target_kv_bytes_by_record: Mapping[int, int],
    source_owner_by_record: Mapping[int, int],
    compute_rank_by_record: Mapping[int, int],
    target_owner_by_record: Mapping[int, int] | None = None,
) -> D2PreliminaryPlacementLedger:
    target_owners = (
        source_owner_by_record
        if target_owner_by_record is None
        else target_owner_by_record
    )
    record_ids = tuple(sorted(old_kv_bytes_by_record))
    key_sets = (
        set(target_kv_bytes_by_record),
        set(source_owner_by_record),
        set(compute_rank_by_record),
        set(target_owners),
    )
    if (
        not record_ids
        or any(keys != set(record_ids) for keys in key_sets)
        or any(
            not isinstance(record_id, int)
            or isinstance(record_id, bool)
            or record_id < 0
            for record_id in record_ids
        )
        or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            for mapping in (
                old_kv_bytes_by_record,
                target_kv_bytes_by_record,
                source_owner_by_record,
                compute_rank_by_record,
                target_owners,
            )
            for value in mapping.values()
        )
    ):
        raise ValueError("D2 preliminary placement inputs are invalid")
    ranks = sorted(
        set(source_owner_by_record.values())
        | set(compute_rank_by_record.values())
        | set(target_owners.values())
    )
    rank_values = {
        rank: {
            "old_send": 0,
            "old_receive": 0,
            "target_send": 0,
            "target_receive": 0,
        }
        for rank in ranks
    }
    p2p_steal_records = 0
    nonowner_output_return_records = 0
    old_kv_p2p_bytes = 0
    target_kv_return_bytes = 0
    for record_id in record_ids:
        source_owner = source_owner_by_record[record_id]
        compute_rank = compute_rank_by_record[record_id]
        target_owner = target_owners[record_id]
        if compute_rank != source_owner:
            value = old_kv_bytes_by_record[record_id]
            p2p_steal_records += 1
            old_kv_p2p_bytes += value
            rank_values[source_owner]["old_send"] += value
            rank_values[compute_rank]["old_receive"] += value
        if compute_rank != target_owner:
            value = target_kv_bytes_by_record[record_id]
            nonowner_output_return_records += 1
            target_kv_return_bytes += value
            rank_values[compute_rank]["target_send"] += value
            rank_values[target_owner]["target_receive"] += value
    per_rank = tuple(
        D2PlacementRankBytes(
            rank=rank,
            old_kv_send_bytes=rank_values[rank]["old_send"],
            old_kv_receive_bytes=rank_values[rank]["old_receive"],
            target_kv_send_bytes=rank_values[rank]["target_send"],
            target_kv_receive_bytes=rank_values[rank]["target_receive"],
        )
        for rank in ranks
    )
    return D2PreliminaryPlacementLedger(
        record_ids=record_ids,
        owner_local_records=len(record_ids) - p2p_steal_records,
        p2p_steal_records=p2p_steal_records,
        nonowner_output_return_records=(
            nonowner_output_return_records
        ),
        old_kv_p2p_bytes=old_kv_p2p_bytes,
        target_kv_return_bytes=target_kv_return_bytes,
        total_p2p_bytes=old_kv_p2p_bytes + target_kv_return_bytes,
        per_rank=per_rank,
    )
