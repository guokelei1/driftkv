import copy
import gc
from pathlib import Path

import pytest
import torch

from hstu_kvcache.migration import (
    CompiledStage4Transform,
    ExactStage4Transform,
    LazyStage4SourceReader,
    NoTransformStage4Transform,
    PackedMigrationOperator,
    ResidualPStage4Transform,
    SelectiveStage4Transform,
    SourceRecordDescriptor,
    Stage4CoreEngine,
    Stage4RuntimeConfig,
    Stage4SourceManifest,
    build_stage4_extents,
    capture_layerwise_state,
    compile_migration_program,
    place_stage4_extents_lpt,
    stage4_capacity_preflight,
    write_source_shard,
)
from hstu_kvcache.models import HSTU, HSTUConfig

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Stage 4 core engine requires CUDA",
)


def make_stage4_engine_fixture(root: Path):
    cfg = HSTUConfig(
        num_items=100,
        num_behaviors=8,
        hidden_size=16,
        num_layers=2,
        num_heads=2,
        head_dim=8,
        max_seq_len=16,
        input_dropout=0.0,
        attn_dropout=0.0,
    )
    torch.manual_seed(41)
    source_model = HSTU(cfg).eval()
    torch.manual_seed(43)
    target_model = HSTU(cfg).eval()
    records = []
    lengths = (5, 8, 10)
    for record_id, length in enumerate(lengths):
        generator = torch.Generator().manual_seed(100 + record_id)
        item_ids = torch.randint(
            1,
            cfg.num_items + 1,
            (1, length),
            generator=generator,
        )
        behaviors = torch.randint(
            1,
            cfg.num_behaviors + 1,
            (1, length),
            generator=generator,
        )
        time_deltas = torch.rand(
            1,
            length,
            generator=generator,
        )
        state = capture_layerwise_state(
            source_model,
            item_ids,
            behaviors,
            time_deltas,
            torch.tensor([length]),
        )
        normalized = write_source_shard(
            root,
            f"normalized/{record_id:06d}.pt",
            "normalized_capsule_fp16",
            record_id,
            "theta0",
            "theta11",
            length,
            {
                "normed": torch.stack(state.normed_states)[:, 0].to(
                    torch.float16
                )
            },
        )
        old_kv = write_source_shard(
            root,
            f"old_kv/{record_id:06d}.pt",
            "old_kv_fp16",
            record_id,
            "theta0",
            "theta11",
            length,
            {
                "k": state.kv.k[:, 0].to(torch.float16),
                "v": state.kv.v[:, 0].to(torch.float16),
            },
        )
        raw = write_source_shard(
            root,
            f"raw/{record_id:06d}.pt",
            "raw_history",
            record_id,
            "theta0",
            "theta11",
            length,
            {
                "item_ids": item_ids[0],
                "behaviors": behaviors[0],
                "time_deltas": time_deltas[0],
            },
        )
        residual = write_source_shard(
            root,
            f"residual/{record_id:06d}.pt",
            "residual_hidden_suffix_bf16",
            record_id,
            "theta0",
            "theta11",
            length,
            {
                "hidden_states": torch.stack(state.hidden_states[1:])[
                    :, 0
                ].to(torch.bfloat16)
            },
            {"start_layer": 1, "num_layers": 2},
        )
        records.append(
            SourceRecordDescriptor(
                record_id=record_id,
                user_id=record_id + 1,
                evaluation_role="program_selection",
                source_version="theta0",
                target_version="theta11",
                prefix_tokens=length,
                shards=(normalized, old_kv, raw, residual),
            )
        )
    manifest = Stage4SourceManifest(
        workload_content_sha256="c" * 64,
        workload_file_sha256="d" * 64,
        num_layers=cfg.num_layers,
        hidden_size=cfg.hidden_size,
        kv_width=cfg.num_heads * cfg.head_dim,
        records=tuple(records),
        creation={"kind": "test"},
    )
    path = root / "source_manifest.json"
    manifest.write(path)
    return path, target_model, sum(lengths)


def release_stage4_result(result):
    del result
    gc.collect()
    torch.cuda.empty_cache()


def assert_stage4_result(result, tokens, destination):
    report = result.report
    assert report.record_count == 3
    assert report.prefix_tokens == tokens
    assert report.destination == destination
    assert report.manifest.record_count == 3
    assert report.manifest.token_count == tokens
    assert report.correctness is not None
    assert report.correctness.finite
    assert report.correctness.allclose
    assert report.correctness.max_abs_error == 0
    assert report.correctness.valid_element_count == 2 * 2 * tokens * 16
    assert report.correctness.record_order_valid
    assert report.correctness.lengths_offsets_valid
    assert report.logical_output_bytes == 2 * 2 * tokens * 16 * 2
    assert report.physical_output_bytes > report.logical_output_bytes
    assert sum(value.record_count for value in report.devices) == 3
    assert sum(value.prefix_tokens for value in report.devices) == tokens


def test_stage4_exact_dram_job_is_pinned_complete_and_validated(tmp_path):
    path, target_model, tokens = make_stage4_engine_fixture(tmp_path)
    transform = ExactStage4Transform(
        copy.deepcopy(target_model).to("cuda:0"),
        "theta11",
        None,
    )
    engine = Stage4CoreEngine(
        path,
        (transform,),
        "dram",
        Stage4RuntimeConfig(
            batch_size=2,
            length_bucket_width=16,
            max_inflight=2,
            exact_compute="float32",
        ),
        expected_workload_content_sha256="c" * 64,
    )

    result = engine.run(validate=True)

    assert_stage4_result(result, tokens, "dram")
    reader = LazyStage4SourceReader(path)
    extents = build_stage4_extents(
        reader.manifest,
        tuple(range(3)),
        {"theta0": ("raw_history",)},
        batch_size=2,
        bucket_width=16,
    )
    first_batch, _ = reader.read_extent(extents[0], pin_memory=False)
    _, second_metrics = reader.read_extent(extents[1], pin_memory=False)
    expected_source_wave = (
        first_batch.nbytes + second_metrics.peak_source_resident_bytes
    )
    assert result.report.peak_source_resident_bytes == expected_source_wave
    assert result.report.peak_staging_bytes == result.report.physical_output_bytes
    assignments = place_stage4_extents_lpt(extents, 1)
    capacity = stage4_capacity_preflight(
        assignments,
        (transform,),
        "dram",
        (0,),
        max_inflight=2,
        calibration_assignments=assignments,
    )
    assert capacity["required_peak_host_bytes"] == (
        result.report.physical_output_bytes + expected_source_wave
    )
    assert result.report.peak_host_bytes == capacity["required_peak_host_bytes"]
    device_capacity = capacity["per_gpu"][0]
    assert device_capacity["full_device_wave_bytes"] > 0
    assert (
        device_capacity["calibration_device_wave_bytes"]
        == device_capacity["full_device_wave_bytes"]
    )
    assert device_capacity["shared_compute_slack_bytes"] == 0
    for extent in result.report.manifest.extents:
        output = result.destination.load_extent("theta11", extent.extent_id)
        assert all(
            value.is_pinned()
            for value in (output.k, output.v, output.lengths, output.offsets)
        )
    release_stage4_result(result)


def test_stage4_compiled_hbm_job_uses_common_extent_contract(tmp_path):
    path, target_model, tokens = make_stage4_engine_fixture(tmp_path)
    program = compile_migration_program(
        target_model,
        source_version="theta0",
        target_version="theta11",
    )
    transform = CompiledStage4Transform(
        {"theta0": program},
        PackedMigrationOperator(torch.float16),
        "cuda:0",
    )
    engine = Stage4CoreEngine(
        path,
        (transform,),
        "hbm",
        Stage4RuntimeConfig(
            batch_size=2,
            length_bucket_width=16,
            max_inflight=2,
            compiled_operator="packed_fp16",
        ),
    )

    result = engine.run(validate=True)

    assert_stage4_result(result, tokens, "hbm")
    assert all(
        result.destination.load_extent("theta11", extent.extent_id).k.is_cuda
        for extent in result.report.manifest.extents
    )
    release_stage4_result(result)


@pytest.mark.parametrize(
    ("destination", "factory"),
    [
        (
            "hbm",
            lambda model: SelectiveStage4Transform(
                model,
                "theta11",
                0,
                1,
            ),
        ),
        (
            "hbm",
            lambda model: ResidualPStage4Transform(
                model,
                "theta11",
                1,
                ("theta0",),
            ),
        ),
        (
            "dram",
            lambda model: NoTransformStage4Transform(
                "cuda:0",
                "theta11",
            ),
        ),
    ],
)
def test_stage4_secondary_transforms_share_transaction_path(
    tmp_path,
    destination,
    factory,
):
    path, target_model, tokens = make_stage4_engine_fixture(tmp_path)
    transform = factory(copy.deepcopy(target_model).to("cuda:0"))
    engine = Stage4CoreEngine(
        path,
        (transform,),
        destination,
        Stage4RuntimeConfig(
            batch_size=2,
            length_bucket_width=16,
            max_inflight=2,
        ),
    )

    result = engine.run(validate=True)

    assert_stage4_result(result, tokens, destination)
    release_stage4_result(result)


def test_stage4_runtime_variant_mismatch_rejects_before_job(tmp_path):
    path, target_model, _ = make_stage4_engine_fixture(tmp_path)
    transform = ExactStage4Transform(
        target_model.to("cuda:0"),
        "theta11",
        None,
    )

    with pytest.raises(ValueError, match="variants differ"):
        Stage4CoreEngine(
            path,
            (transform,),
            "hbm",
            Stage4RuntimeConfig(
                batch_size=1,
                length_bucket_width=16,
                max_inflight=2,
                exact_compute="bfloat16",
            ),
        )


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="multi-GPU Stage 4 test requires two CUDA devices",
)
def test_stage4_two_gpu_lpt_and_concurrent_publication_are_complete(tmp_path):
    path, _, tokens = make_stage4_engine_fixture(tmp_path)
    transforms = (
        NoTransformStage4Transform("cuda:0", "theta11"),
        NoTransformStage4Transform("cuda:1", "theta11"),
    )
    engine = Stage4CoreEngine(
        path,
        transforms,
        "hbm",
        Stage4RuntimeConfig(
            batch_size=1,
            length_bucket_width=16,
            max_inflight=2,
        ),
    )

    result = engine.run(validate=True)

    assert_stage4_result(result, tokens, "hbm")
    assert len(result.report.devices) == 2
    assert sum(value.record_count for value in result.report.devices) == 3
    assert sum(value.prefix_tokens for value in result.report.devices) == tokens
    assert all(value.record_count > 0 for value in result.report.devices)
    release_stage4_result(result)
