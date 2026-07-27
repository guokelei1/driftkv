import pytest
import torch

from hstu_kvcache.migration import (
    CohortStreamingExecutor,
    DeviceJaggedKVOutputPool,
    FullRecomputeStreamingExecutor,
    FusedJaggedMigrationOperator,
    FusedMigrationOperator,
    JaggedCohortStreamingExecutor,
    JaggedMigratedKVBatch,
    JaggedMigrationCapsuleBatch,
    MigrationCapsuleBatch,
    MultiGPUCohortExecutor,
    MultiGPUHBMJaggedCohortExecutor,
    MultiGPUJaggedCohortExecutor,
    PackedJaggedMigrationOperator,
    PackedMigrationOperator,
    PinnedJaggedKVOutputPool,
    PinnedKVOutputPool,
    RawHistoryBatch,
    ReferenceMigrationOperator,
    build_contiguous_cohort_plan,
    build_length_bucketed_cohort_plan,
    capture_layerwise_state,
    compile_migration_program,
    execute_migration_program,
    migrate_compiled_low_rank_cache,
    profile_packed_operator_stages,
    profile_reference_operator_stages,
    validate_contiguous_output_extent,
)
from hstu_kvcache.models import HSTU, HSTUConfig


def make_runtime_inputs():
    torch.manual_seed(23)
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
    item_ids = torch.randint(1, 101, (5, 6))
    behaviors = torch.randint(1, 9, (5, 6))
    time_deltas = torch.rand(5, 6) * 100
    lengths = torch.tensor([6, 5, 4, 3, 2])
    valid = torch.arange(6).unsqueeze(0) < lengths.unsqueeze(1)
    item_ids = item_ids * valid
    behaviors = behaviors * valid
    time_deltas = time_deltas * valid
    state = capture_layerwise_state(
        model,
        item_ids,
        behaviors,
        time_deltas,
        lengths,
    )
    capsule = MigrationCapsuleBatch.from_layerwise_state(
        state,
        migration_anchor_version="theta-0",
        record_ids=(100, 101, 102, 103, 104),
    )
    program = compile_migration_program(
        model,
        source_version="theta-0",
        target_version="theta-1",
    )
    return state, capsule, program


def concatenate_results(report):
    return (
        torch.cat([batch.cache.k for batch in report.batches], dim=1),
        torch.cat([batch.cache.v for batch in report.batches], dim=1),
        tuple(record_id for batch in report.batches for record_id in batch.record_ids),
    )


def make_jagged_capsule(capsule):
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


def make_output_extent(capsule, program, device="cpu"):
    lengths = capsule.lengths.to(device)
    offsets = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=device),
            lengths.long().cumsum(dim=0),
        )
    )
    shape = (
        program.num_layers,
        int(offsets[-1]),
        program.kv_width,
    )
    return JaggedMigratedKVBatch(
        record_ids=capsule.record_ids,
        migration_anchor_version=capsule.migration_anchor_version,
        served_kv_target=program.target_version,
        k=torch.empty(shape, dtype=capsule.normed.dtype, device=device),
        v=torch.empty(shape, dtype=capsule.normed.dtype, device=device),
        lengths=lengths,
        offsets=offsets,
    )


def test_capsule_split_preserves_contiguous_records_and_anchor():
    _, capsule, _ = make_runtime_inputs()

    batches = capsule.split(2)

    assert [batch.batch_size for batch in batches] == [2, 2, 1]
    assert [batch.record_ids for batch in batches] == [(100, 101), (102, 103), (104,)]
    assert all(batch.migration_anchor_version == "theta-0" for batch in batches)
    assert torch.equal(
        torch.cat([batch.normed for batch in batches], dim=1),
        capsule.normed,
    )
    assert torch.equal(
        torch.cat([batch.lengths for batch in batches]),
        capsule.lengths,
    )


def test_program_execution_matches_existing_compiled_operator():
    state, capsule, program = make_runtime_inputs()

    result = execute_migration_program(program, capsule)
    expected = migrate_compiled_low_rank_cache(state, program.adapter)

    assert result.migration_anchor_version == "theta-0"
    assert result.served_kv_target == "theta-1"
    assert result.record_ids == capsule.record_ids
    assert torch.allclose(result.cache.k, expected.k, atol=1e-5, rtol=1e-5)
    assert torch.allclose(result.cache.v, expected.v, atol=1e-5, rtol=1e-5)


def test_reference_and_packed_write_the_same_unpadded_extent():
    _, capsule, program = make_runtime_inputs()
    reference_destination = make_output_extent(capsule, program)
    packed_destination = make_output_extent(capsule, program)
    reference = ReferenceMigrationOperator()
    packed = PackedMigrationOperator(torch.float32)

    reference_result = reference.execute_into(
        reference.prepare_program(program, "cpu"),
        capsule,
        reference_destination,
    )
    packed_result = packed.execute_into(
        packed.prepare_program(program, "cpu"),
        capsule,
        packed_destination,
    )

    validate_contiguous_output_extent(
        program,
        capsule,
        reference_result,
        check_metadata_values=True,
    )
    assert reference_result.k.shape[1] == int(capsule.lengths.sum())
    assert reference_result.k.is_contiguous()
    assert reference_result.v.is_contiguous()
    assert torch.allclose(reference_result.k, packed_result.k)
    assert torch.allclose(reference_result.v, packed_result.v)


def test_output_extent_rejects_changed_offsets_and_aliased_storage():
    _, capsule, program = make_runtime_inputs()
    destination = make_output_extent(capsule, program)
    destination.offsets[1] = 0

    with pytest.raises(ValueError, match="offsets"):
        validate_contiguous_output_extent(
            program,
            capsule,
            destination,
            check_metadata_values=True,
        )

    offsets = torch.cat(
        (
            torch.zeros(1, dtype=torch.long),
            capsule.lengths.long().cumsum(0),
        )
    )
    shape = (
        program.num_layers,
        int(offsets[-1]),
        program.kv_width,
    )
    shared = torch.empty(shape)
    aliased = JaggedMigratedKVBatch(
        record_ids=capsule.record_ids,
        migration_anchor_version=capsule.migration_anchor_version,
        served_kv_target=program.target_version,
        k=shared,
        v=shared,
        lengths=capsule.lengths,
        offsets=offsets,
    )

    with pytest.raises(ValueError, match="share storage"):
        validate_contiguous_output_extent(program, capsule, aliased)


def test_program_rejects_capsule_from_wrong_anchor():
    _, capsule, program = make_runtime_inputs()
    wrong = MigrationCapsuleBatch(
        record_ids=capsule.record_ids,
        migration_anchor_version="theta-other",
        normed=capsule.normed,
        lengths=capsule.lengths,
    )

    with pytest.raises(ValueError, match="anchor"):
        execute_migration_program(program, wrong)


def test_cpu_streaming_executor_preserves_order_and_matches_reference():
    state, capsule, program = make_runtime_inputs()
    executor = CohortStreamingExecutor(
        program,
        device="cpu",
        max_inflight_batches=2,
    )

    report = executor.run(capsule.split(2))
    actual_k, actual_v, record_ids = concatenate_results(report)
    expected = migrate_compiled_low_rank_cache(state, program.adapter)

    assert record_ids == capsule.record_ids
    assert torch.allclose(actual_k, expected.k, atol=1e-5, rtol=1e-5)
    assert torch.allclose(actual_v, expected.v, atol=1e-5, rtol=1e-5)
    assert report.metrics.batch_count == 3
    assert report.metrics.record_count == capsule.batch_size
    assert report.metrics.token_count == int(capsule.lengths.sum())
    assert report.metrics.input_bytes == capsule.nbytes
    assert report.metrics.output_bytes == sum(
        batch.nbytes for batch in report.batches
    )
    assert report.metrics.elapsed_seconds > 0


def test_streaming_executor_rejects_duplicate_record_ids():
    _, capsule, program = make_runtime_inputs()
    batch = capsule.split(2)[0]
    executor = CohortStreamingExecutor(program, device="cpu")

    with pytest.raises(ValueError, match="unique"):
        executor.run((batch, batch))


def test_streaming_executor_dispatches_multiple_version_cohorts():
    state, capsule, program = make_runtime_inputs()
    second_program = type(program)(
        source_version="theta-other",
        target_version=program.target_version,
        adapter=program.adapter,
    )
    batches = list(capsule.split(2))
    batches[1] = MigrationCapsuleBatch(
        record_ids=batches[1].record_ids,
        migration_anchor_version="theta-other",
        normed=batches[1].normed,
        lengths=batches[1].lengths,
    )
    executor = CohortStreamingExecutor(
        (program, second_program),
        device="cpu",
    )

    report = executor.run(batches)
    actual_k, actual_v, record_ids = concatenate_results(report)
    expected = migrate_compiled_low_rank_cache(state, program.adapter)

    assert record_ids == capsule.record_ids
    assert torch.allclose(actual_k, expected.k, atol=1e-5, rtol=1e-5)
    assert torch.allclose(actual_v, expected.v, atol=1e-5, rtol=1e-5)


def test_length_bucketed_plan_reduces_padding_and_restores_logical_order():
    state, capsule, program = make_runtime_inputs()
    contiguous = build_contiguous_cohort_plan(capsule, max_records=2)
    bucketed = build_length_bucketed_cohort_plan(
        capsule,
        max_records=2,
        bucket_width=2,
    )
    executor = CohortStreamingExecutor(program, device="cpu")

    report = executor.run(bucketed.batches)
    restored = bucketed.restore_logical_order(report.batches)
    expected = migrate_compiled_low_rank_cache(state, program.adapter)

    assert bucketed.payload_nbytes < contiguous.payload_nbytes
    assert bucketed.padded_tokens < contiguous.padded_tokens
    assert restored.record_ids == capsule.record_ids
    assert torch.allclose(restored.cache.k, expected.k, atol=1e-5, rtol=1e-5)
    assert torch.allclose(restored.cache.v, expected.v, atol=1e-5, rtol=1e-5)


def test_packed_float32_operator_matches_reference():
    state, capsule, program = make_runtime_inputs()
    executor = CohortStreamingExecutor(
        program,
        device="cpu",
        operator=PackedMigrationOperator(torch.float32),
    )

    report = executor.run(capsule.split(2))
    actual_k, actual_v, _ = concatenate_results(report)
    expected = migrate_compiled_low_rank_cache(state, program.adapter)

    assert torch.allclose(actual_k, expected.k, atol=1e-5, rtol=1e-5)
    assert torch.allclose(actual_v, expected.v, atol=1e-5, rtol=1e-5)


def test_packed_jagged_operator_matches_dense_reference_without_padding():
    state, capsule, program = make_runtime_inputs()
    jagged = make_jagged_capsule(capsule)
    operator = PackedJaggedMigrationOperator(torch.float32)
    prepared = operator.prepare_program(program, "cpu")

    actual = operator.execute(prepared, jagged)
    expected = migrate_compiled_low_rank_cache(state, program.adapter)

    assert jagged.token_count == int(capsule.lengths.sum())
    assert jagged.token_count < capsule.batch_size * capsule.seq_len
    assert actual.k.is_contiguous()
    assert actual.v.is_contiguous()
    for row, record_id in enumerate(capsule.record_ids):
        length = int(capsule.lengths[row])
        actual_k, actual_v = actual.record_kv(record_id)
        assert torch.allclose(actual_k, expected.k[:, row, :length])
        assert torch.allclose(actual_v, expected.v[:, row, :length])


def test_cpu_jagged_executor_preserves_records_and_token_count():
    state, capsule, program = make_runtime_inputs()
    jagged = make_jagged_capsule(capsule)
    executor = JaggedCohortStreamingExecutor(
        program,
        device="cpu",
        operator=PackedJaggedMigrationOperator(torch.float32),
    )

    report = executor.run((jagged,))
    expected = migrate_compiled_low_rank_cache(state, program.adapter)

    assert report.metrics.batch_count == 1
    assert report.metrics.record_count == capsule.batch_size
    assert report.metrics.token_count == int(capsule.lengths.sum())
    for row, record_id in enumerate(capsule.record_ids):
        length = int(capsule.lengths[row])
        actual_k, actual_v = report.batches[0].record_kv(record_id)
        assert torch.allclose(actual_k, expected.k[:, row, :length])
        assert torch.allclose(actual_v, expected.v[:, row, :length])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_streaming_executor_matches_cpu_reference():
    state, capsule, program = make_runtime_inputs()
    executor = CohortStreamingExecutor(
        program,
        device="cuda",
        max_inflight_batches=3,
    )

    report = executor.run(capsule.split(2))
    actual_k, actual_v, record_ids = concatenate_results(report)
    expected = migrate_compiled_low_rank_cache(state, program.adapter)

    assert record_ids == capsule.record_ids
    assert torch.allclose(actual_k, expected.k, atol=1e-5, rtol=1e-5)
    assert torch.allclose(actual_v, expected.v, atol=1e-5, rtol=1e-5)
    assert report.metrics.auto_pinned_batches == 3
    assert all(batch.cache.k.is_pinned() for batch in report.batches)
    assert all(batch.cache.v.is_pinned() for batch in report.batches)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_packed_float16_operator_stays_close_to_fp32_reference():
    state, capsule, program = make_runtime_inputs()
    capsule = MigrationCapsuleBatch(
        record_ids=capsule.record_ids,
        migration_anchor_version=capsule.migration_anchor_version,
        normed=capsule.normed.half(),
        lengths=capsule.lengths,
    )
    executor = CohortStreamingExecutor(
        program,
        device="cuda",
        operator=PackedMigrationOperator(torch.float16),
    )

    report = executor.run(capsule.split(2))
    actual_k, actual_v, _ = concatenate_results(report)
    expected = migrate_compiled_low_rank_cache(state, program.adapter)

    assert torch.allclose(actual_k.float(), expected.k, atol=2e-2, rtol=2e-2)
    assert torch.allclose(actual_v.float(), expected.v, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_fused_operator_matches_packed_float16():
    _, capsule, program = make_runtime_inputs()
    capsule = MigrationCapsuleBatch(
        record_ids=capsule.record_ids,
        migration_anchor_version=capsule.migration_anchor_version,
        normed=capsule.normed.half(),
        lengths=capsule.lengths,
    ).to("cuda")
    packed_program = PackedMigrationOperator(torch.float16).prepare_program(
        program,
        "cuda",
    )
    fused_program = FusedMigrationOperator().prepare_program(program, "cuda")

    packed = PackedMigrationOperator(torch.float16).execute(
        packed_program,
        capsule,
    )
    fused = FusedMigrationOperator().execute(fused_program, capsule)

    assert torch.allclose(fused.cache.k, packed.cache.k, atol=2e-2, rtol=2e-2)
    assert torch.allclose(fused.cache.v, packed.cache.v, atol=2e-2, rtol=2e-2)
    positions = torch.arange(capsule.seq_len, device="cuda").unsqueeze(0)
    invalid = positions >= capsule.lengths.unsqueeze(1)
    mask = invalid.unsqueeze(0).unsqueeze(-1)
    assert float(fused.cache.k.masked_select(mask).abs().max()) == 0.0
    assert float(fused.cache.v.masked_select(mask).abs().max()) == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_dense_operators_write_identical_contiguous_extents():
    _, capsule, program = make_runtime_inputs()
    capsule = MigrationCapsuleBatch(
        record_ids=capsule.record_ids,
        migration_anchor_version=capsule.migration_anchor_version,
        normed=capsule.normed.half(),
        lengths=capsule.lengths,
    ).to("cuda")
    operators = (
        ReferenceMigrationOperator(),
        PackedMigrationOperator(torch.float16),
        FusedMigrationOperator(),
    )
    outputs = {}
    pointers = {}
    for operator in operators:
        destination = make_output_extent(capsule, program, "cuda")
        prepared = operator.prepare_program(program, "cuda")
        pointers[operator.name] = (
            destination.k.data_ptr(),
            destination.v.data_ptr(),
        )
        outputs[operator.name] = operator.execute_into(
            prepared,
            capsule,
            destination,
        )
        validate_contiguous_output_extent(
            prepared,
            capsule,
            outputs[operator.name],
            check_metadata_values=True,
        )

    reference = outputs["reference_fp32"]
    for name, output in outputs.items():
        assert pointers[name] == (output.k.data_ptr(), output.v.data_ptr())
        assert output.k.shape[1] == int(capsule.lengths.sum())
        assert torch.isfinite(output.k).all()
        assert torch.isfinite(output.v).all()
        assert torch.allclose(output.k, reference.k, atol=2e-2, rtol=2e-2)
        assert torch.allclose(output.v, reference.v, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_fused_jagged_operator_matches_packed_jagged():
    _, capsule, program = make_runtime_inputs()
    capsule = MigrationCapsuleBatch(
        record_ids=capsule.record_ids,
        migration_anchor_version=capsule.migration_anchor_version,
        normed=capsule.normed.half(),
        lengths=capsule.lengths,
    )
    jagged = make_jagged_capsule(capsule).to("cuda")
    packed_operator = PackedJaggedMigrationOperator(torch.float16)
    fused_operator = FusedJaggedMigrationOperator()
    packed_program = packed_operator.prepare_program(program, "cuda")
    fused_program = fused_operator.prepare_program(program, "cuda")

    packed = packed_operator.execute(packed_program, jagged)
    fused = fused_operator.execute(fused_program, jagged)

    assert torch.allclose(fused.k, packed.k, atol=2e-2, rtol=2e-2)
    assert torch.allclose(fused.v, packed.v, atol=2e-2, rtol=2e-2)
    assert fused.k.is_contiguous()
    assert fused.v.is_contiguous()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_fused_jagged_operator_writes_into_device_extent():
    _, capsule, program = make_runtime_inputs()
    capsule = MigrationCapsuleBatch(
        record_ids=capsule.record_ids,
        migration_anchor_version=capsule.migration_anchor_version,
        normed=capsule.normed.half(),
        lengths=capsule.lengths,
    )
    jagged = make_jagged_capsule(capsule).pin_memory()
    pool = DeviceJaggedKVOutputPool.allocate(
        (jagged,),
        served_kv_target=program.target_version,
        num_layers=program.num_layers,
        kv_width=program.kv_width,
        device="cuda",
        dtype=torch.float16,
    )
    device_capsule = jagged.to("cuda")
    destination = pool.acquire(jagged)
    operator = FusedJaggedMigrationOperator()
    prepared = operator.prepare_program(program, "cuda")
    pointers = (destination.k.data_ptr(), destination.v.data_ptr())

    result = operator.execute_into(prepared, device_capsule, destination)

    assert pointers == (result.k.data_ptr(), result.v.data_ptr())
    expected = PackedJaggedMigrationOperator(torch.float16).execute(
        PackedJaggedMigrationOperator(torch.float16).prepare_program(
            program,
            "cuda",
        ),
        device_capsule,
    )
    assert torch.allclose(result.k, expected.k, atol=2e-2, rtol=2e-2)
    assert torch.allclose(result.v, expected.v, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_streaming_executor_reuses_preallocated_outputs():
    _, capsule, program = make_runtime_inputs()
    capsule = MigrationCapsuleBatch(
        record_ids=capsule.record_ids,
        migration_anchor_version=capsule.migration_anchor_version,
        normed=capsule.normed.half(),
        lengths=capsule.lengths,
    )
    batches = tuple(batch.pin_memory() for batch in capsule.split(2))
    pool = PinnedKVOutputPool.allocate(
        batches,
        served_kv_target=program.target_version,
        num_layers=program.num_layers,
        kv_width=program.kv_width,
        dtype=torch.float16,
    )
    executor = CohortStreamingExecutor(
        program,
        device="cuda",
        operator=PackedMigrationOperator(torch.float16),
        output_pool=pool,
    )

    first = executor.run(batches)
    pointers = [
        (batch.cache.k.data_ptr(), batch.cache.v.data_ptr())
        for batch in first.batches
    ]
    second = executor.run(batches)

    assert second.metrics.preallocated_output_batches == len(batches)
    assert pointers == [
        (batch.cache.k.data_ptr(), batch.cache.v.data_ptr())
        for batch in second.batches
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_jagged_executor_reuses_preallocated_outputs():
    _, capsule, program = make_runtime_inputs()
    capsule = MigrationCapsuleBatch(
        record_ids=capsule.record_ids,
        migration_anchor_version=capsule.migration_anchor_version,
        normed=capsule.normed.half(),
        lengths=capsule.lengths,
    )
    jagged = make_jagged_capsule(capsule).pin_memory()
    pool = PinnedJaggedKVOutputPool.allocate(
        (jagged,),
        served_kv_target=program.target_version,
        num_layers=program.num_layers,
        kv_width=program.kv_width,
        dtype=torch.float16,
    )
    executor = JaggedCohortStreamingExecutor(
        program,
        device="cuda",
        operator=FusedJaggedMigrationOperator(),
        output_pool=pool,
    )

    first = executor.run((jagged,))
    pointers = (first.batches[0].k.data_ptr(), first.batches[0].v.data_ptr())
    second = executor.run((jagged,))

    assert second.metrics.preallocated_output_batches == 1
    assert pointers == (
        second.batches[0].k.data_ptr(),
        second.batches[0].v.data_ptr(),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_full_recompute_streaming_matches_direct_compute():
    torch.manual_seed(41)
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
    ).to("cuda")
    lengths = torch.tensor([6, 4])
    valid = torch.arange(6).unsqueeze(0) < lengths.unsqueeze(1)
    raw = RawHistoryBatch(
        record_ids=(200, 201),
        migration_anchor_version="theta-0",
        item_ids=torch.randint(1, 101, (2, 6)) * valid,
        behaviors=torch.randint(1, 9, (2, 6)) * valid,
        time_deltas=torch.rand(2, 6) * 100 * valid,
        lengths=lengths,
    ).pin_memory()
    pool = PinnedKVOutputPool.allocate(
        (raw,),
        served_kv_target="theta-1",
        num_layers=2,
        kv_width=16,
        dtype=torch.float16,
    )
    device_raw = raw.to("cuda")
    expected = model.compute_kv(
        device_raw.item_ids,
        device_raw.behaviors,
        device_raw.time_deltas,
        lengths=device_raw.lengths,
    )
    executor = FullRecomputeStreamingExecutor(
        model=model,
        source_version="theta-0",
        target_version="theta-1",
        device="cuda",
        execution_dtype=None,
        output_pool=pool,
    )

    report = executor.run((raw,))

    assert report.batches[0].record_ids == raw.record_ids
    assert report.metrics.preallocated_output_batches == 1
    assert torch.allclose(
        report.batches[0].cache.k,
        expected.k.half().cpu(),
        atol=1e-3,
        rtol=1e-3,
    )
    assert torch.allclose(
        report.batches[0].cache.v,
        expected.v.half().cpu(),
        atol=1e-3,
        rtol=1e-3,
    )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA GPUs are unavailable")
def test_multi_gpu_executor_preserves_global_batch_order():
    state, capsule, program = make_runtime_inputs()
    with MultiGPUCohortExecutor(
        program,
        devices=("cuda:0", "cuda:1"),
        max_inflight_batches=2,
    ) as executor:
        report = executor.run(capsule.split(2))
    actual_k = torch.cat([batch.cache.k for batch in report.batches], dim=1)
    actual_v = torch.cat([batch.cache.v for batch in report.batches], dim=1)
    record_ids = tuple(
        record_id
        for batch in report.batches
        for record_id in batch.record_ids
    )
    expected = migrate_compiled_low_rank_cache(state, program.adapter)

    assert record_ids == capsule.record_ids
    assert torch.allclose(actual_k, expected.k, atol=1e-5, rtol=1e-5)
    assert torch.allclose(actual_v, expected.v, atol=1e-5, rtol=1e-5)
    assert report.metrics.device_count == 2
    assert report.metrics.batch_count == 3
    assert report.metrics.record_count == capsule.batch_size
    assert len(report.metrics.devices) == 2


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA GPUs are unavailable")
def test_multi_gpu_jagged_executor_preserves_records():
    state, capsule, program = make_runtime_inputs()
    capsule = MigrationCapsuleBatch(
        record_ids=capsule.record_ids,
        migration_anchor_version=capsule.migration_anchor_version,
        normed=capsule.normed.half(),
        lengths=capsule.lengths,
    )
    first_dense, second_dense = capsule.split(3)
    first = make_jagged_capsule(first_dense).pin_memory()
    second = make_jagged_capsule(second_dense).pin_memory()
    pool = PinnedJaggedKVOutputPool.allocate(
        (first, second),
        served_kv_target=program.target_version,
        num_layers=program.num_layers,
        kv_width=program.kv_width,
        dtype=torch.float16,
    )
    with MultiGPUJaggedCohortExecutor(
        program,
        devices=("cuda:0", "cuda:1"),
        operator=FusedJaggedMigrationOperator(),
        output_pool=pool,
    ) as executor:
        report = executor.run((first, second))
    expected = migrate_compiled_low_rank_cache(state, program.adapter)

    assert report.metrics.device_count == 2
    assert report.metrics.batch_count == 2
    assert report.metrics.record_count == capsule.batch_size
    by_record = {
        record_id: batch.record_kv(record_id)
        for batch in report.batches
        for record_id in batch.record_ids
    }
    for row, record_id in enumerate(capsule.record_ids):
        length = int(capsule.lengths[row])
        actual_k, actual_v = by_record[record_id]
        assert torch.allclose(
            actual_k.float(),
            expected.k[:, row, :length],
            atol=2e-2,
            rtol=2e-2,
        )
        assert torch.allclose(
            actual_v.float(),
            expected.v[:, row, :length],
            atol=2e-2,
            rtol=2e-2,
        )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA GPUs are unavailable")
def test_multi_gpu_hbm_jagged_executor_publishes_device_extents():
    state, capsule, program = make_runtime_inputs()
    capsule = MigrationCapsuleBatch(
        record_ids=capsule.record_ids,
        migration_anchor_version=capsule.migration_anchor_version,
        normed=capsule.normed.half(),
        lengths=capsule.lengths,
    )
    first_dense, second_dense = capsule.split(3)
    batches = (
        make_jagged_capsule(first_dense).pin_memory(),
        make_jagged_capsule(second_dense).pin_memory(),
    )
    with MultiGPUHBMJaggedCohortExecutor(
        program,
        devices=("cuda:0", "cuda:1"),
        batches=batches,
        operator=FusedJaggedMigrationOperator(),
    ) as executor:
        report = executor.run()
    expected = migrate_compiled_low_rank_cache(state, program.adapter)

    assert report.metrics.device_count == 2
    assert sum(
        device.execution.preallocated_output_batches
        for device in report.metrics.devices
    ) == 2
    by_record = {
        record_id: batch.record_kv(record_id)
        for batch in report.batches
        for record_id in batch.record_ids
    }
    assert {batch.k.device.index for batch in report.batches} == {0, 1}
    for row, record_id in enumerate(capsule.record_ids):
        length = int(capsule.lengths[row])
        actual_k, actual_v = by_record[record_id]
        assert torch.allclose(
            actual_k.float().cpu(),
            expected.k[:, row, :length],
            atol=2e-2,
            rtol=2e-2,
        )
        assert torch.allclose(
            actual_v.float().cpu(),
            expected.v[:, row, :length],
            atol=2e-2,
            rtol=2e-2,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_stage_profiler_covers_reference_and_packed_paths():
    _, capsule, program = make_runtime_inputs()
    capsule = MigrationCapsuleBatch(
        record_ids=capsule.record_ids,
        migration_anchor_version=capsule.migration_anchor_version,
        normed=capsule.normed.half(),
        lengths=capsule.lengths,
    ).to("cuda")

    reference = profile_reference_operator_stages(program, capsule, repeats=2)
    packed = profile_packed_operator_stages(
        program,
        capsule,
        torch.float16,
        repeats=2,
    )

    assert set(reference.stages) == {
        "input_cast",
        "bmm",
        "bias",
        "mask",
        "output_cast",
    }
    assert set(packed.stages) == {
        "input_cast",
        "baddbmm_bias",
        "mask_inplace",
        "output_cast",
    }
    assert reference.total.median_ms > 0
    assert packed.total.median_ms > 0
