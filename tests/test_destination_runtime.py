from pathlib import Path

import pytest
import torch

from hstu_kvcache.migration import (
    DRAMKVUpdateDestination,
    FilesystemKVUpdateDestination,
    HBMKVUpdateDestination,
    InMemoryRemoteObjectStore,
    JaggedMigrationCapsuleBatch,
    MigrationCapsuleBatch,
    OutOfCoreKVUpdateEngine,
    PackedJaggedMigrationOperator,
    PublicationMode,
    RemoteKVUpdateDestination,
    capture_layerwise_state,
    compile_migration_program,
)
from hstu_kvcache.models import HSTU, HSTUConfig


def make_destination_inputs(dtype=torch.float32):
    torch.manual_seed(71)
    model = HSTU(
        HSTUConfig(
            num_items=100,
            num_behaviors=8,
            hidden_size=16,
            num_layers=2,
            num_heads=2,
            head_dim=8,
            max_seq_len=6,
            input_dropout=0.0,
        )
    )
    model.eval()
    lengths = torch.tensor([6, 5, 4, 3])
    valid = torch.arange(6).unsqueeze(0) < lengths.unsqueeze(1)
    state = capture_layerwise_state(
        model,
        torch.randint(1, 101, (4, 6)) * valid,
        torch.randint(1, 9, (4, 6)) * valid,
        torch.rand(4, 6) * 100 * valid,
        lengths,
    )
    dense = MigrationCapsuleBatch.from_layerwise_state(
        state,
        migration_anchor_version="theta-0",
        record_ids=(200, 201, 202, 203),
    )
    dense = MigrationCapsuleBatch(
        record_ids=dense.record_ids,
        migration_anchor_version=dense.migration_anchor_version,
        normed=dense.normed.to(dtype),
        lengths=dense.lengths,
    )
    program = compile_migration_program(
        model,
        source_version="theta-0",
        target_version="theta-1",
    )
    batches = tuple(
        make_jagged_batch(batch)
        for batch in dense.split(2)
    )
    return program, batches


def make_jagged_batch(capsule):
    lengths = capsule.lengths.clone()
    offsets = torch.cat(
        (
            torch.zeros(1, dtype=torch.long),
            lengths.cumsum(dim=0),
        )
    )
    normed = torch.cat(
        [
            capsule.normed[:, row, : int(length)]
            for row, length in enumerate(lengths)
        ],
        dim=1,
    ).contiguous()
    return JaggedMigrationCapsuleBatch(
        record_ids=capsule.record_ids,
        migration_anchor_version=capsule.migration_anchor_version,
        normed=normed,
        lengths=lengths,
        offsets=offsets,
    )


def expected_by_record(program, batches):
    operator = PackedJaggedMigrationOperator(torch.float32)
    prepared = operator.prepare_program(program, "cpu")
    return {
        record_id: result.record_kv(record_id)
        for batch in batches
        for result in (operator.execute(prepared, batch),)
        for record_id in batch.record_ids
    }


def assert_destination_matches(
    destination,
    manifest,
    expected,
    atol=1e-5,
    rtol=1e-5,
):
    actual = {
        record_id: batch.record_kv(record_id)
        for extent in manifest.extents
        for batch in (
            destination.load_extent(
                manifest.target_version,
                extent.extent_id,
            ),
        )
        for record_id in batch.record_ids
    }
    assert set(actual) == set(expected)
    for record_id in expected:
        actual_k, actual_v = actual[record_id]
        expected_k, expected_v = expected[record_id]
        assert torch.allclose(
            actual_k.float().cpu(),
            expected_k.float().cpu(),
            atol=atol,
            rtol=rtol,
        )
        assert torch.allclose(
            actual_v.float().cpu(),
            expected_v.float().cpu(),
            atol=atol,
            rtol=rtol,
        )


@pytest.mark.parametrize("backend", ["dram", "filesystem", "remote"])
def test_out_of_core_engine_publishes_complete_atomic_version(
    backend,
    tmp_path: Path,
):
    program, batches = make_destination_inputs()
    if backend == "dram":
        destination = DRAMKVUpdateDestination()
    elif backend == "filesystem":
        destination = FilesystemKVUpdateDestination(tmp_path / "ssd")
    else:
        destination = RemoteKVUpdateDestination(InMemoryRemoteObjectStore())
    engine = OutOfCoreKVUpdateEngine(
        program,
        devices=("cpu",),
        destination=destination,
        wave_batch_limit=1,
        operator=PackedJaggedMigrationOperator(torch.float32),
    )

    report = engine.run("job-001", batches)

    assert report.manifest.record_ids == (200, 201, 202, 203)
    assert report.manifest.record_count == 4
    assert report.manifest.token_count == 18
    assert report.metrics.wave_count == 2
    assert report.metrics.batch_count == 2
    assert report.metrics.publication_mode == PublicationMode.HOST_STAGED
    assert destination.manifest("theta-1") == report.manifest
    assert_destination_matches(
        destination,
        report.manifest,
        expected_by_record(program, batches),
    )


def test_incomplete_filesystem_publication_is_not_visible(tmp_path: Path):
    program, batches = make_destination_inputs()
    operator = PackedJaggedMigrationOperator(torch.float32)
    prepared = operator.prepare_program(program, "cpu")
    first = operator.execute(prepared, batches[0])
    destination = FilesystemKVUpdateDestination(tmp_path / "ssd")
    transaction = destination.begin(
        job_id="job-incomplete",
        target_version="theta-1",
        expected_record_ids=(200, 201, 202, 203),
    )
    transaction.stage("extent-00000000", first)

    with pytest.raises(ValueError, match="missing"):
        transaction.commit()
    transaction.abort()

    with pytest.raises(KeyError, match="not published"):
        destination.manifest("theta-1")
    assert not any((tmp_path / "ssd" / "versions").iterdir())


def test_remote_manifest_is_the_commit_marker():
    program, batches = make_destination_inputs()
    store = InMemoryRemoteObjectStore()
    destination = RemoteKVUpdateDestination(store)
    operator = PackedJaggedMigrationOperator(torch.float32)
    prepared = operator.prepare_program(program, "cpu")
    transaction = destination.begin(
        job_id="job-remote",
        target_version="theta-1",
        expected_record_ids=(200, 201, 202, 203),
    )
    transaction.stage(
        "extent-00000000",
        operator.execute(prepared, batches[0]),
    )

    with pytest.raises(KeyError, match="not published"):
        destination.manifest("theta-1")
    transaction.abort()

    with pytest.raises(KeyError, match="not published"):
        destination.manifest("theta-1")


def test_invalid_transaction_input_does_not_create_staging():
    destination = DRAMKVUpdateDestination()

    with pytest.raises(ValueError, match="job_id"):
        destination.begin("", "theta-1", (200,))

    assert destination._staging == {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_out_of_core_engine_directly_publishes_hbm_extents():
    program, batches = make_destination_inputs(torch.float16)
    destination = HBMKVUpdateDestination(("cuda:0",))
    engine = OutOfCoreKVUpdateEngine(
        program,
        devices=("cuda:0",),
        destination=destination,
        wave_batch_limit=1,
    )

    report = engine.run("job-hbm", batches)

    assert report.metrics.publication_mode == PublicationMode.DIRECT_DEVICE
    assert report.metrics.wave_count == 1
    assert all(extent.device == "cuda:0" for extent in report.manifest.extents)
    assert all(
        destination.load_extent("theta-1", extent.extent_id).k.device
        == torch.device("cuda:0")
        for extent in report.manifest.extents
    )
    assert_destination_matches(
        destination,
        report.manifest,
        expected_by_record(program, batches),
        atol=2e-2,
        rtol=2e-2,
    )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA GPUs are unavailable")
def test_out_of_core_engine_trims_idle_hbm_workers():
    program, batches = make_destination_inputs(torch.float16)
    selected = batches[:1]
    destination = HBMKVUpdateDestination(("cuda:0", "cuda:1"))
    engine = OutOfCoreKVUpdateEngine(
        program,
        devices=("cuda:0", "cuda:1"),
        destination=destination,
    )

    report = engine.run("job-trim-hbm", selected)

    assert report.metrics.device_count == 1
    assert report.metrics.program_replica_bytes == program.nbytes
    assert report.manifest.record_ids == selected[0].record_ids
    assert all(extent.device == "cuda:0" for extent in report.manifest.extents)
    assert_destination_matches(
        destination,
        report.manifest,
        expected_by_record(program, selected),
        atol=2e-2,
        rtol=2e-2,
    )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA GPUs are unavailable")
def test_out_of_core_engine_publishes_two_gpu_host_staged_version():
    program, batches = make_destination_inputs(torch.float16)
    destination = DRAMKVUpdateDestination()
    engine = OutOfCoreKVUpdateEngine(
        program,
        devices=("cuda:0", "cuda:1"),
        destination=destination,
        wave_batch_limit=2,
    )

    report = engine.run("job-two-gpu", batches)

    assert report.metrics.device_count == 2
    assert report.metrics.wave_count == 1
    assert report.manifest.record_ids == (200, 201, 202, 203)
    assert_destination_matches(
        destination,
        report.manifest,
        expected_by_record(program, batches),
        atol=2e-2,
        rtol=2e-2,
    )
