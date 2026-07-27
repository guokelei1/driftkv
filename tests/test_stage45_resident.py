import copy
import gc
import json
from pathlib import Path

import pytest
import torch

from hstu_kvcache.migration import (
    CompiledStage4Transform,
    ExactStage4Transform,
    PackedMigrationOperator,
    SourceRecordDescriptor,
    Stage4RuntimeConfig,
    Stage4SourceManifest,
    capture_layerwise_state,
    compile_migration_program,
    write_source_shard,
)
from hstu_kvcache.migration.cohort_jagged import JaggedMigratedKVBatch
from hstu_kvcache.migration.stage45_oldkv import (
    DirectOldKVFusedOperator,
    DirectOldKVTransform,
    Stage45OldKVEngine,
    build_stage45_oldkv_plan,
    compile_direct_oldkv_program,
    execute_direct_oldkv_reference,
    load_direct_oldkv_program,
    stage45_oldkv_preflight,
    write_direct_oldkv_program,
)
from hstu_kvcache.migration.stage45_reclaim import (
    Stage45ReclaimingEngine,
    allocate_reclaimable_old_kv,
    stage45_reclaim_preflight,
)
from hstu_kvcache.migration.stage45_resident import (
    Stage45ResidentEngine,
    build_stage45_resident_plan,
    materialize_stage45_resident_source,
    stage45_resident_preflight,
)
from hstu_kvcache.models import HSTU, HSTUConfig

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Stage 4.5 resident engine requires CUDA",
)


def make_resident_fixture(root: Path):
    config = HSTUConfig(
        num_items=80,
        num_behaviors=8,
        hidden_size=16,
        num_layers=2,
        num_heads=2,
        head_dim=8,
        max_seq_len=16,
        input_dropout=0.0,
        attn_dropout=0.0,
    )
    torch.manual_seed(101)
    source_model = HSTU(config).eval()
    torch.manual_seed(103)
    target_model = HSTU(config).eval()
    records = []
    for record_id, length in enumerate((5, 7, 10)):
        generator = torch.Generator().manual_seed(200 + record_id)
        item_ids = torch.randint(
            1,
            config.num_items + 1,
            (1, length),
            generator=generator,
        )
        behaviors = torch.randint(
            1,
            config.num_behaviors + 1,
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
        records.append(
            SourceRecordDescriptor(
                record_id=record_id,
                user_id=record_id + 1,
                evaluation_role="program_selection",
                source_version="theta0",
                target_version="theta11",
                prefix_tokens=length,
                shards=(normalized, old_kv, raw),
            )
        )
    manifest = Stage4SourceManifest(
        workload_content_sha256="a" * 64,
        workload_file_sha256="b" * 64,
        num_layers=config.num_layers,
        hidden_size=config.hidden_size,
        kv_width=config.num_heads * config.head_dim,
        records=tuple(records),
        creation={"kind": "stage45_test"},
    )
    path = root / "source_manifest.json"
    manifest.write(path)
    return path, source_model, target_model, sum((5, 7, 10))


def release_resident_values(*values):
    del values
    gc.collect()
    torch.cuda.empty_cache()


def test_stage45_hbm_compiled_ceiling_is_complete_and_validated(tmp_path):
    path, source_model, target_model, tokens = make_resident_fixture(tmp_path)
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
    config = Stage4RuntimeConfig(
        batch_size=2,
        length_bucket_width=16,
        max_inflight=2,
        compiled_operator="packed_fp16",
    )
    plan = build_stage45_resident_plan(
        path,
        (transform,),
        "hbm_resident",
        config,
        expected_workload_content_sha256="a" * 64,
    )
    preflight = stage45_resident_preflight(plan, (transform,))

    assert preflight["passed"]
    assert not stage45_resident_preflight(
        plan,
        (transform,),
        allocator_margin_bytes=10**15,
    )["passed"]
    source = materialize_stage45_resident_source(plan, (transform,))
    assert source.preload.standing_hbm_bytes == plan.resident_source_bytes
    assert source.preload.standing_host_bytes == 0
    assert all(
        value.batch.device == torch.device("cuda:0")
        for value in source.assignments[0]
    )

    engine = Stage45ResidentEngine(source, (transform,))
    result = engine.run(validate=True)

    assert result.report.record_count == 3
    assert result.report.prefix_tokens == tokens
    assert result.report.manifest.record_count == 3
    assert result.report.manifest.token_count == tokens
    assert result.report.correctness is not None
    assert result.report.correctness.finite
    assert result.report.correctness.allclose
    assert result.report.correctness.max_abs_error == 0
    assert result.report.correctness.valid_element_count == (
        2 * 2 * tokens * 16
    )
    assert result.report.devices[0].h2d_traffic_bytes == 0
    assert result.report.devices[0].standing_source_hbm_bytes == (
        plan.resident_source_bytes
    )
    engine.close()
    release_resident_values(
        result,
        engine,
        source,
        plan,
        transform,
        source_model,
        target_model,
    )


def test_stage45_pinned_dram_exact_ceiling_counts_timed_h2d(tmp_path):
    path, source_model, target_model, tokens = make_resident_fixture(tmp_path)
    transform = ExactStage4Transform(
        copy.deepcopy(target_model).to("cuda:0"),
        "theta11",
        torch.bfloat16,
    )
    config = Stage4RuntimeConfig(
        batch_size=2,
        length_bucket_width=16,
        max_inflight=2,
        exact_compute="bfloat16",
    )
    plan = build_stage45_resident_plan(
        path,
        (transform,),
        "dram_resident",
        config,
    )
    source = materialize_stage45_resident_source(plan, (transform,))

    assert source.preload.standing_host_bytes == plan.resident_source_bytes
    assert source.preload.standing_hbm_bytes == 0
    assert all(value.batch.is_pinned for value in source.assignments[0])

    engine = Stage45ResidentEngine(source, (transform,))
    result = engine.run(validate=True)

    assert result.report.prefix_tokens == tokens
    assert result.report.correctness is not None
    assert result.report.correctness.allclose
    assert result.report.devices[0].h2d_traffic_bytes == (
        plan.resident_source_bytes
    )
    assert result.report.devices[0].h2d_seconds > 0
    assert result.report.devices[0].standing_source_host_bytes == (
        plan.resident_source_bytes
    )
    assert result.report.timing_breakdown()["elapsed"] == (
        result.report.elapsed_seconds
    )
    engine.close()
    release_resident_values(
        result,
        engine,
        source,
        plan,
        transform,
        source_model,
        target_model,
    )


def test_stage45_extent_reclaim_replaces_complete_old_hbm_footprint(
    tmp_path,
):
    path, source_model, target_model, tokens = make_resident_fixture(tmp_path)
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
    config = Stage4RuntimeConfig(
        batch_size=2,
        length_bucket_width=16,
        max_inflight=2,
        compiled_operator="packed_fp16",
    )
    plan = build_stage45_resident_plan(
        path,
        (transform,),
        "dram_resident",
        config,
    )
    preflight = stage45_reclaim_preflight(plan, (transform,))
    assert preflight["passed"]
    assert not stage45_reclaim_preflight(
        plan,
        (transform,),
        allocator_margin_bytes=10**15,
    )["passed"]
    source = materialize_stage45_resident_source(plan, (transform,))
    old_cache = allocate_reclaimable_old_kv(plan, (transform,))
    initial = old_cache.metrics()
    assert initial.initial_old_kv_bytes == plan.physical_output_bytes
    assert initial.final_old_kv_bytes == plan.physical_output_bytes
    assert initial.final_new_kv_bytes == 0

    engine = Stage45ReclaimingEngine(
        source,
        (transform,),
        old_cache,
    )
    result = engine.run(validate=True)

    reclamation = result.reclamation
    assert result.report.prefix_tokens == tokens
    assert result.report.correctness is not None
    assert result.report.correctness.allclose
    assert reclamation.initial_old_kv_bytes == plan.physical_output_bytes
    assert reclamation.retired_old_kv_bytes == plan.physical_output_bytes
    assert reclamation.final_old_kv_bytes == 0
    assert reclamation.final_new_kv_bytes == plan.physical_output_bytes
    assert reclamation.peak_old_plus_new_kv_bytes > (
        reclamation.initial_old_kv_bytes
    )
    assert reclamation.retired_extent_count == len(plan.extents)
    del result
    gc.collect()
    torch.cuda.empty_cache()
    baselines = []
    for repeat in range(2):
        old_cache = allocate_reclaimable_old_kv(plan, (transform,))
        engine.install_old_cache(old_cache)
        repeated = engine.run(
            job_id=f"stage45-reclaim-repeat-{repeat}",
        )
        baselines.append(repeated.report.devices[0].baseline_hbm_bytes)
        del repeated, old_cache
        gc.collect()
        torch.cuda.empty_cache()
    assert baselines[0] == baselines[1]
    engine.close()
    release_resident_values(
        engine,
        source,
        plan,
        transform,
        source_model,
        target_model,
    )


def test_stage45_hbm_exact_reclaim_counts_resident_raw_source(tmp_path):
    path, source_model, target_model, tokens = make_resident_fixture(
        tmp_path
    )
    transform = ExactStage4Transform(
        copy.deepcopy(target_model).to("cuda:0"),
        "theta11",
        torch.bfloat16,
    )
    config = Stage4RuntimeConfig(
        batch_size=2,
        length_bucket_width=16,
        max_inflight=2,
        exact_compute="bfloat16",
    )
    plan = build_stage45_resident_plan(
        path,
        (transform,),
        "hbm_resident",
        config,
    )
    preflight = stage45_reclaim_preflight(plan, (transform,))
    assert preflight["passed"]
    assert preflight["per_gpu"][0]["standing_source_hbm_bytes"] > 0
    assert (
        preflight["per_gpu"][0]["maximum_source_movement_wave_bytes"]
        == 0
    )
    source = materialize_stage45_resident_source(
        plan,
        (transform,),
        require_capacity=False,
    )
    old_cache = allocate_reclaimable_old_kv(plan, (transform,))
    engine = Stage45ReclaimingEngine(source, (transform,), old_cache)
    result = engine.run(validate=True)

    assert result.report.prefix_tokens == tokens
    assert result.report.correctness is not None
    assert result.report.correctness.allclose
    assert result.report.devices[0].standing_source_hbm_bytes == (
        plan.resident_source_bytes
    )
    assert result.report.devices[0].standing_source_host_bytes == 0
    assert result.report.devices[0].h2d_traffic_bytes == 0
    assert result.reclamation.final_old_kv_bytes == 0
    engine.close()
    release_resident_values(
        result,
        engine,
        old_cache,
        source,
        plan,
        transform,
        source_model,
        target_model,
    )


def test_stage45_direct_oldkv_program_round_trip(tmp_path):
    _, source_model, target_model, _ = make_resident_fixture(tmp_path)
    compiled = compile_migration_program(
        target_model,
        source_version="theta0",
        target_version="theta11",
    )
    direct, metrics = compile_direct_oldkv_program(
        source_model.to("cuda:0"),
        compiled,
    )

    assert direct.weights.shape == (2, 32, 32)
    assert direct.biases.shape == (2, 32)
    assert direct.weights.dtype == torch.float16
    assert metrics.to_dict()["condition_number_max"] < 100
    assert metrics.to_dict()["float16_weight_residual_max"] < 0.01

    path = tmp_path / "direct_oldkv.pt"
    descriptor = write_direct_oldkv_program(
        direct,
        path,
        {"labels_used": False},
        metrics,
    )
    loaded, metadata = load_direct_oldkv_program(
        path,
        expected_sha256=descriptor["sha256"],
        expected_source_version="theta0",
        expected_target_version="theta11",
        expected_num_layers=2,
        expected_kv_width=16,
    )

    assert torch.equal(loaded.weights, direct.weights)
    assert torch.equal(loaded.biases, direct.biases)
    assert metadata["provenance"]["labels_used"] is False
    release_resident_values(
        loaded,
        direct,
        compiled,
        source_model,
        target_model,
    )


def test_stage45_direct_oldkv_fused_matches_deployed_affine(tmp_path):
    path, source_model, target_model, _ = make_resident_fixture(tmp_path)
    compiled = compile_migration_program(
        target_model,
        source_version="theta0",
        target_version="theta11",
    )
    direct, _ = compile_direct_oldkv_program(
        source_model.to("cuda:0"),
        compiled,
    )
    manifest = json.loads(path.read_text())
    record = manifest["records"][0]
    shards = {
        value["representation"]: value
        for value in record["shards"]
    }
    old = torch.load(
        tmp_path / shards["old_kv_fp16"]["path"],
        map_location="cpu",
        weights_only=True,
    )["tensors"]
    normalized = torch.load(
        tmp_path / shards["normalized_capsule_fp16"]["path"],
        map_location="cpu",
        weights_only=True,
    )["tensors"]["normed"]
    length = record["prefix_tokens"]
    lengths = torch.tensor([length], device="cuda:0")
    offsets = torch.tensor([0, length], device="cuda:0")
    source = JaggedMigratedKVBatch(
        record_ids=(record["record_id"],),
        migration_anchor_version="theta0",
        served_kv_target="theta0",
        k=old["k"].cuda(),
        v=old["v"].cuda(),
        lengths=lengths,
        offsets=offsets,
    )

    def output():
        shape = source.k.shape
        return JaggedMigratedKVBatch(
            record_ids=source.record_ids,
            migration_anchor_version="theta0",
            served_kv_target="theta11",
            k=torch.empty(shape, dtype=torch.float16, device="cuda:0"),
            v=torch.empty(shape, dtype=torch.float16, device="cuda:0"),
            lengths=lengths.clone(),
            offsets=offsets.clone(),
        )

    operator = DirectOldKVFusedOperator(
        block_m=16,
        block_n=32,
        block_k=16,
        num_warps=4,
        num_stages=2,
    )
    deployed = operator.prepare_program(direct, "cuda:0")
    fused = operator.execute_into(deployed, source, output())
    reference = execute_direct_oldkv_reference(
        deployed,
        source,
        output(),
    )
    compiled_gpu = compiled.to("cuda:0", dtype=torch.float16)
    expected = torch.baddbmm(
        compiled_gpu.adapter.biases[:, None, :].expand(
            compiled_gpu.num_layers,
            length,
            2 * compiled_gpu.kv_width,
        ),
        normalized.cuda(),
        compiled_gpu.adapter.weights,
    )
    actual = torch.cat((fused.k, fused.v), dim=-1)

    assert torch.allclose(fused.k, reference.k, atol=0.02, rtol=0.02)
    assert torch.allclose(fused.v, reference.v, atol=0.02, rtol=0.02)
    assert torch.allclose(actual, expected, atol=0.02, rtol=0.02)
    release_resident_values(
        actual,
        expected,
        compiled_gpu,
        reference,
        fused,
        source,
        deployed,
        direct,
        compiled,
        source_model,
        target_model,
    )


def test_stage45_direct_oldkv_engine_reclaims_its_source(tmp_path):
    path, source_model, target_model, tokens = make_resident_fixture(
        tmp_path
    )
    compiled = compile_migration_program(
        target_model,
        source_version="theta0",
        target_version="theta11",
    )
    direct, _ = compile_direct_oldkv_program(
        source_model.to("cuda:0"),
        compiled,
    )
    transform = DirectOldKVTransform(
        {"theta0": direct},
        DirectOldKVFusedOperator(
            block_m=16,
            block_n=32,
            block_k=16,
            num_warps=4,
            num_stages=2,
        ),
        "cuda:0",
    )
    config = Stage4RuntimeConfig(
        batch_size=2,
        length_bucket_width=16,
        max_inflight=2,
        compiled_operator="fused_fp16",
    )
    plan = build_stage45_oldkv_plan(
        path,
        (transform,),
        config,
        expected_workload_content_sha256="a" * 64,
    )

    assert stage45_oldkv_preflight(plan, (transform,))["passed"]
    assert not stage45_oldkv_preflight(
        plan,
        (transform,),
        allocator_margin_bytes=10**15,
    )["passed"]
    old_cache = allocate_reclaimable_old_kv(
        plan,
        (transform,),
        zero=True,
    )
    engine = Stage45OldKVEngine(plan, (transform,))
    engine.install_old_cache(old_cache)
    result = engine.run(validate_zero_source=True)

    assert result.report.record_count == 3
    assert result.report.prefix_tokens == tokens
    assert result.report.correctness is not None
    assert result.report.correctness.finite
    assert result.report.correctness.allclose
    assert result.report.correctness.max_abs_error == 0
    assert result.report.correctness.valid_element_count == (
        2 * 2 * tokens * 16
    )
    assert result.report.devices[0].h2d_traffic_bytes == 0
    assert result.reclamation.final_old_kv_bytes == 0
    assert result.reclamation.retired_extent_count == len(plan.extents)
    assert result.reclamation.final_new_kv_bytes == (
        plan.physical_output_bytes
    )
    engine.close()
    release_resident_values(
        result,
        engine,
        old_cache,
        plan,
        transform,
        direct,
        compiled,
        source_model,
        target_model,
    )
