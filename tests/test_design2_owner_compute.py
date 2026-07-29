import pytest
import torch

from hstu_kvcache.migration.cohort_jagged import JaggedMigratedKVBatch
from hstu_kvcache.migration.design2_owner import (
    D2CompiledRetainedPhaseCounters,
    characterize_p2p_steal_and_return,
    d2_owner_fragment_sha256,
    execute_compiled_retained_owner_compute,
    jagged_kv_payload_bytes_by_record,
)
from hstu_kvcache.migration.stage45_oldkv import DirectOldKVProgram


def _program() -> DirectOldKVProgram:
    weights = torch.eye(4, dtype=torch.float16).reshape(1, 4, 4)
    biases = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0]],
        dtype=torch.float16,
    )
    return DirectOldKVProgram(
        source_version="theta0",
        target_version="theta1",
        weights=weights,
        biases=biases,
    )


def _source() -> JaggedMigratedKVBatch:
    return JaggedMigratedKVBatch(
        record_ids=(10, 20),
        migration_anchor_version="theta0",
        served_kv_target="theta0",
        k=torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]],
            dtype=torch.float16,
        ),
        v=torch.tensor(
            [[[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]],
            dtype=torch.float16,
        ),
        lengths=torch.tensor([2, 1], dtype=torch.long),
        offsets=torch.tensor([0, 2, 3], dtype=torch.long),
    )


def test_owner_compute_cpu_reference_private_fragment_and_checksum() -> None:
    source = _source()
    fragment = execute_compiled_retained_owner_compute(
        _program(),
        source,
        {10: 1, 20: 1},
        rank=1,
    )

    assert fragment.ready
    assert not fragment.empty
    assert fragment.output is not None
    assert fragment.metadata.record_ids == (10, 20)
    assert fragment.metadata.lengths == (2, 1)
    assert fragment.metadata.token_count == 3
    assert fragment.metadata.kv_payload_bytes == 24
    assert fragment.metadata.extent_bytes == 64
    assert fragment.metrics.source_extent_bytes == source.nbytes
    assert fragment.metrics.output_extent_bytes == fragment.output.nbytes
    assert fragment.metrics.phase_counters.to_dict() == {
        "item_lookup_calls": 0,
        "embedding_collective_count": 0,
        "embedding_collective_bytes": 0,
        "old_kv_p2p_bytes": 0,
    }
    assert torch.equal(
        fragment.output.k,
        source.k + torch.tensor([1.0, 2.0], dtype=torch.float16),
    )
    assert torch.equal(
        fragment.output.v,
        source.v + torch.tensor([3.0, 4.0], dtype=torch.float16),
    )
    assert fragment.output.k.untyped_storage().data_ptr() != (
        source.k.untyped_storage().data_ptr()
    )
    assert fragment.output.v.untyped_storage().data_ptr() != (
        source.v.untyped_storage().data_ptr()
    )
    assert fragment.metadata.checksum_sha256 == d2_owner_fragment_sha256(
        "theta0",
        "theta1",
        fragment.output,
    )


def test_owner_compute_rejects_nonowner_before_operator_execution() -> None:
    calls = 0

    def operator(program, source, destination):
        nonlocal calls
        calls += 1
        return destination

    with pytest.raises(ValueError, match="non-owner"):
        execute_compiled_retained_owner_compute(
            _program(),
            _source(),
            {10: 1, 20: 0},
            rank=1,
            operator=operator,
        )

    assert calls == 0


def test_owner_compute_empty_rank_returns_ready_empty_fragment() -> None:
    fragment = execute_compiled_retained_owner_compute(
        _program(),
        None,
        {10: 0, 20: 0},
        rank=1,
    )

    assert fragment.ready
    assert fragment.empty
    assert fragment.output is None
    assert fragment.metadata.record_ids == ()
    assert fragment.metadata.token_count == 0
    assert fragment.metadata.kv_payload_bytes == 0
    assert fragment.metadata.extent_bytes == 0
    assert fragment.metrics.record_count == 0
    assert fragment.metrics.elapsed_seconds == 0.0
    assert fragment.metadata.checksum_sha256 == d2_owner_fragment_sha256(
        "theta0",
        "theta1",
        None,
    )

    bucket_empty = execute_compiled_retained_owner_compute(
        _program(),
        None,
        {10: 1},
        rank=1,
    )
    assert bucket_empty.ready
    assert bucket_empty.empty


def test_owner_compute_hard_rejects_nonzero_normal_path_counters() -> None:
    with pytest.raises(RuntimeError, match="communication invariant"):
        execute_compiled_retained_owner_compute(
            _program(),
            _source(),
            {10: 1, 20: 1},
            rank=1,
            phase_counters=D2CompiledRetainedPhaseCounters(
                embedding_collective_count=1,
            ),
        )


def test_preliminary_p2p_steal_and_nonowner_return_byte_ledger() -> None:
    source = _source()
    payload = jagged_kv_payload_bytes_by_record(source)
    assert payload == {10: 16, 20: 8}

    local = characterize_p2p_steal_and_return(
        payload,
        payload,
        {10: 0, 20: 1},
        {10: 0, 20: 1},
    )
    assert local.owner_local_records == 2
    assert local.old_kv_p2p_bytes == 0
    assert local.target_kv_return_bytes == 0
    assert local.preliminary_baseline
    assert not local.measured_transport
    assert not local.scientific_result

    stolen = characterize_p2p_steal_and_return(
        {10: 100, 20: 200},
        {10: 110, 20: 220},
        {10: 0, 20: 1},
        {10: 1, 20: 0},
    )
    assert stolen.owner_local_records == 0
    assert stolen.p2p_steal_records == 2
    assert stolen.nonowner_output_return_records == 2
    assert stolen.old_kv_p2p_bytes == 300
    assert stolen.target_kv_return_bytes == 330
    assert stolen.total_p2p_bytes == 630
    assert [value.to_dict() for value in stolen.per_rank] == [
        {
            "rank": 0,
            "old_kv_send_bytes": 100,
            "old_kv_receive_bytes": 200,
            "target_kv_send_bytes": 220,
            "target_kv_receive_bytes": 110,
        },
        {
            "rank": 1,
            "old_kv_send_bytes": 200,
            "old_kv_receive_bytes": 100,
            "target_kv_send_bytes": 110,
            "target_kv_receive_bytes": 220,
        },
    ]
