import pytest
import torch

from hstu_kvcache.migration import (
    CohortStreamingExecutor,
    MigrationCapsuleBatch,
    MultiGPUCohortExecutor,
    PackedMigrationOperator,
    build_contiguous_cohort_plan,
    build_length_bucketed_cohort_plan,
    capture_layerwise_state,
    compile_migration_program,
    execute_migration_program,
    migrate_compiled_low_rank_cache,
    profile_packed_operator_stages,
    profile_reference_operator_stages,
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
